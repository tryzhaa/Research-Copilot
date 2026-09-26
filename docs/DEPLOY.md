# Deploying to Google Cloud Run

How the [live demo](https://research-copilot-1014589976171.asia-south1.run.app) is deployed, step
by step. Steps 1-6 are one-time setup; after that, Step 7 is the whole redeploy.

## What gets deployed

- **`Dockerfile`**: builds the image with the Papers with Code index (287k paper→code links) and
  the `bge-small` embedding model baked in, so a cold start doesn't redownload them. Runs as an
  unprivileged user with `DEMO_MODE=1`.
- **Demo mode** (`copilot/demo.py`): each visitor's library (saves, ratings, folders,
  summaries) lives in their own browser's localStorage, and the server's write endpoints return
  403, so visitors never share state. It's rate-limited: 5 searches and 5 summaries per visitor per hour, 150 model
  calls a day in total. Limits are configurable with `DEMO_SEARCHES_PER_HOUR`,
  `DEMO_SUMMARIES_PER_HOUR` and `DEMO_DAILY_LIMIT`. The owner's private signals (CPU/GPU,
  recruiter score and reason, similarity, preference) still shape the ranking but are blanked in
  every response, so visitors never receive them.
- **`.dockerignore` / `.gcloudignore`**: keep `.env` (API keys), `library.json` and `data/` out of
  the image and off Google's servers. The demo starts with an empty library of its own.

The ranking model is whatever the committed `preferences.yaml` names (a hosted provider such as
Groq; a local Ollama model can't run on Cloud Run).

## 1. Install the CLI and log in

```bash
brew install --cask gcloud-cli     # or see https://cloud.google.com/sdk/docs/install
gcloud auth login                   # opens a browser to sign in
gcloud config set project <project-id>
```

## 2. Billing

Cloud Run needs a billing account linked to the project, even if usage stays inside the free
tier. In the [Cloud Console](https://console.cloud.google.com) under **Billing**:

1. Create (or reactivate) a billing account and link it to the project.
2. Under **Budgets & alerts**, add a small budget (e.g. $1) so any charge reaches you by email.

```bash
gcloud billing projects describe <project-id>    # expect billingEnabled: true
```

## 3. Enable the services

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com secretmanager.googleapis.com
```

Cloud Run runs the app, Cloud Build builds the image, Artifact Registry stores it, and Secret
Manager holds the API keys.

## 4. Store the API keys as secrets

Read from `.env` and piped in, so the values never appear on screen or in shell history:

```bash
grep '^GROQ_API_KEY=' .env | cut -d= -f2- | tr -d '\n' | \
    gcloud secrets create groq-api-key --data-file=- --replication-policy=automatic
grep '^OPENALEX_API_KEY=' .env | cut -d= -f2- | tr -d '\n' | \
    gcloud secrets create openalex-api-key --data-file=- --replication-policy=automatic
```

The OpenAlex key is optional (a free one from openalex.org avoids anonymous rate limits); drop it
here and from `--set-secrets` in Step 7 if you don't use one.

## 5. Let the service read them

Cloud Run runs as the project's default compute service account:

```bash
PN=$(gcloud projects describe <project-id> --format='value(projectNumber)')
SA="${PN}-compute@developer.gserviceaccount.com"
for s in groq-api-key openalex-api-key; do
  gcloud secrets add-iam-policy-binding $s \
      --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor
done
```

## 6. Check what will be uploaded

```bash
gcloud meta list-files-for-upload
```

`.env`, `library.json` and `data/` must not be in the list; only code, tests and config should be
(about 50 files).

## 7. Deploy

```bash
gcloud run deploy research-copilot --source . --region asia-south1 \
    --allow-unauthenticated --memory 1Gi --cpu 1 --cpu-boost \
    --min-instances 0 --max-instances 1 \
    --set-secrets GROQ_API_KEY=groq-api-key:latest,OPENALEX_API_KEY=openalex-api-key:latest
```

| Flag | Why |
|---|---|
| `--source .` | Uploads the source; Cloud Build builds the Dockerfile on Google's side, so no local Docker is needed |
| `--region asia-south1` | Mumbai. Pick a region near your visitors |
| `--allow-unauthenticated` | Public: anyone with the link can open it |
| `--memory 1Gi --cpu 1` | Measured peak is 650-780 MB, so 1 GB leaves headroom |
| `--cpu-boost` | Extra CPU while starting, for faster cold starts |
| `--min-instances 0` | Scales to zero when idle, so an unused demo costs nothing |
| `--max-instances 1` | Caps spend, and keeps the demo's in-memory rate limits in one place |
| `--set-secrets` | Exposes the secrets to the app as environment variables |

The first deploy takes about 12 minutes, mostly building the image, and prints the service URL.

## 8. Check that it works

```bash
U=https://<your-service-url>
curl $U/api/prefs              # includes "demo": {...}, so demo mode is on
curl -X POST $U/api/search -H 'Content-Type: application/json' \
     -d '{"query":"diffusion models for music generation","fields":["ml","art"]}'   # ~40 s
curl -X POST $U/api/save -H 'Content-Type: application/json' \
     -d '{"paper":{"title":"T"},"saved":true}'             # 403: read-only
```

## Day to day

- **Redeploy after a change:** re-run Step 7.
- **Rotate a key:** add a new version, then redeploy so the service picks it up:
  ```bash
  grep '^GROQ_API_KEY=' .env | cut -d= -f2- | tr -d '\n' | \
      gcloud secrets versions add groq-api-key --data-file=-
  ```
- **Logs:** `gcloud run services logs read research-copilot --region asia-south1`
- **Take it offline:** `gcloud run services delete research-copilot --region asia-south1`

## Why the setup looks like this

Problems found while testing the container under Cloud Run's limits (1 GB, 1 CPU), and how they
were fixed:

1. **Hosting choice.** Hugging Face Docker Spaces need a paid plan, and 512 MB free tiers
   (Render, Koyeb) are too small: the app peaks at 650-780 MB. Cloud Run scales to zero and has a
   free monthly allowance.
2. **Permissions.** `WORKDIR` creates `/app` owned by root, so the unprivileged user couldn't create
   `data/` and the build failed. The Dockerfile hands `/app` to that user.
3. **arXiv failed only in Docker.** Every request got an empty 406. The container's OpenSSL 3.5
   offers the post-quantum `X25519MLKEM768` key share by default and arXiv's edge rejects those
   handshakes; the same request from a host on OpenSSL 3.0 worked. The app offers classic X25519
   for arxiv.org only (`copilot/sources.py`).
4. **Memory and speed.** Embeddings run in batches of 8 to bound attention memory, and ONNX Runtime
   uses as many threads as the container's CPU quota: 8 threads on 1 CPU had made embedding 7x
   slower (53.6 s vs 7.6 s for 45 abstracts).
5. **Slow sources.** OpenAlex can take 10-50 s per query, so all sources share a 25 s deadline
   (`source_timeout_seconds`); a straggler is reported and skipped.
6. **Free-tier token limits.** Summaries shrink to fit `request_token_limit` (Groq's free tier caps
   each request) instead of being rejected.
7. **Rebuild time.** The index and model download sits in an early layer that doesn't depend on app
   code, so a code change rebuilds in seconds instead of redownloading for ~10 minutes.
