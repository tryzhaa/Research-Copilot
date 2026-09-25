"""Research Copilot. Importing the package loads API keys from a git-ignored .env (KEY=value per line)."""
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Real environment variables win over .env."""
    path = Path(__file__).resolve().parent.parent / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and not name.strip().startswith("#"):
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


_load_dotenv()
