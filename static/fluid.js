// Background smoke that the cursor stirs: a small GPU fluid simulation ("stable fluids":
// advect, add vorticity, project to divergence-free with a Jacobi pressure solve).
// Monochrome and faint so text stays readable. Skipped for prefers-reduced-motion or
// without WebGL2; the loop sleeps once the smoke has faded and while the tab is hidden.
(() => {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const CONFIG = {
    SIM_RES: 128,              // velocity grid (short side)
    DYE_RES: 512,              // smoke texture (short side)
    DENSITY_DISSIPATION: 1.8,  // how fast smoke fades
    VELOCITY_DISSIPATION: 0.35,
    PRESSURE: 0.8,
    PRESSURE_ITERATIONS: 20,
    CURL: 22,                  // swirliness
    SPLAT_RADIUS: 0.18,
    SPLAT_FORCE: 5200,
    COLOR: [0.05, 0.048, 0.045],  // --fg (#ecebe6) at ~5%: a faint warm grey
    MAX_BRIGHTNESS: 0.13,      // dense smoke eases toward this instead of clipping to a flat blob
    IDLE_MS: 6000,             // after this long without movement, finish fading and sleep
  };

  const canvas = document.createElement("canvas");
  canvas.id = "fluid";
  canvas.setAttribute("aria-hidden", "true");
  document.body.prepend(canvas);

  const gl = canvas.getContext("webgl2", { alpha: false, depth: false, stencil: false, antialias: false });
  if (!gl || !gl.getExtension("EXT_color_buffer_float")) { canvas.remove(); return; }

  // ---------- shaders ----------

  const VERT = `
    precision highp float;
    attribute vec2 aPosition;
    varying vec2 vUv, vL, vR, vT, vB;
    uniform vec2 texelSize;
    void main () {
      vUv = aPosition * 0.5 + 0.5;
      vL = vUv - vec2(texelSize.x, 0.0);
      vR = vUv + vec2(texelSize.x, 0.0);
      vT = vUv + vec2(0.0, texelSize.y);
      vB = vUv - vec2(0.0, texelSize.y);
      gl_Position = vec4(aPosition, 0.0, 1.0);
    }`;

  const FRAG = {
    splat: `
      uniform sampler2D uTarget; uniform float aspectRatio, radius; uniform vec3 color; uniform vec2 point;
      void main () {
        vec2 p = vUv - point; p.x *= aspectRatio;
        gl_FragColor = vec4(texture2D(uTarget, vUv).xyz + exp(-dot(p, p) / radius) * color, 1.0);
      }`,
    advection: `
      uniform sampler2D uVelocity, uSource; uniform vec2 texelSize; uniform float dt, dissipation;
      void main () {
        vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;
        gl_FragColor = texture2D(uSource, coord) / (1.0 + dissipation * dt);
      }`,
    divergence: `
      uniform sampler2D uVelocity;
      void main () {
        float L = texture2D(uVelocity, vL).x, R = texture2D(uVelocity, vR).x;
        float T = texture2D(uVelocity, vT).y, B = texture2D(uVelocity, vB).y;
        vec2 C = texture2D(uVelocity, vUv).xy;
        if (vL.x < 0.0) L = -C.x;  if (vR.x > 1.0) R = -C.x;   // walls reflect
        if (vT.y > 1.0) T = -C.y;  if (vB.y < 0.0) B = -C.y;
        gl_FragColor = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
      }`,
    curl: `
      uniform sampler2D uVelocity;
      void main () {
        float L = texture2D(uVelocity, vL).y, R = texture2D(uVelocity, vR).y;
        float T = texture2D(uVelocity, vT).x, B = texture2D(uVelocity, vB).x;
        gl_FragColor = vec4(0.5 * (R - L - T + B), 0.0, 0.0, 1.0);
      }`,
    vorticity: `
      uniform sampler2D uVelocity, uCurl; uniform float curl, dt;
      void main () {
        float L = texture2D(uCurl, vL).x, R = texture2D(uCurl, vR).x;
        float T = texture2D(uCurl, vT).x, B = texture2D(uCurl, vB).x, C = texture2D(uCurl, vUv).x;
        vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
        force = force / (length(force) + 0.0001) * curl * C;
        force.y *= -1.0;
        vec2 v = texture2D(uVelocity, vUv).xy + force * dt;
        gl_FragColor = vec4(clamp(v, -1000.0, 1000.0), 0.0, 1.0);
      }`,
    pressure: `
      uniform sampler2D uPressure, uDivergence;
      void main () {
        float L = texture2D(uPressure, vL).x, R = texture2D(uPressure, vR).x;
        float T = texture2D(uPressure, vT).x, B = texture2D(uPressure, vB).x;
        gl_FragColor = vec4((L + R + B + T - texture2D(uDivergence, vUv).x) * 0.25, 0.0, 0.0, 1.0);
      }`,
    gradientSubtract: `
      uniform sampler2D uPressure, uVelocity;
      void main () {
        float L = texture2D(uPressure, vL).x, R = texture2D(uPressure, vR).x;
        float T = texture2D(uPressure, vT).x, B = texture2D(uPressure, vB).x;
        vec2 v = texture2D(uVelocity, vUv).xy - vec2(R - L, T - B);
        gl_FragColor = vec4(v, 0.0, 1.0);
      }`,
    scale: `
      uniform sampler2D uTexture; uniform float value;
      void main () { gl_FragColor = value * texture2D(uTexture, vUv); }`,
    display: `
      uniform sampler2D uTexture; uniform float maxBrightness;
      void main () {
        vec3 c = max(texture2D(uTexture, vUv).rgb, 0.0);
        gl_FragColor = vec4(maxBrightness * (1.0 - exp(-c / maxBrightness)), 1.0);  // soft shoulder, no plateau
      }`,
  };

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }

  const vert = compile(gl.VERTEX_SHADER, VERT);
  function program(frag) {
    const p = gl.createProgram();
    gl.attachShader(p, vert);
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, `precision highp float; precision highp sampler2D;
      varying vec2 vUv, vL, vR, vT, vB;\n${frag}`));
    gl.bindAttribLocation(p, 0, "aPosition");
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    for (let i = 0; i < gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i++) {
      const name = gl.getActiveUniform(p, i).name;
      u[name] = gl.getUniformLocation(p, name);
    }
    return { use: () => gl.useProgram(p), u };
  }

  let P;
  try {
    P = Object.fromEntries(Object.entries(FRAG).map(([k, src]) => [k, program(src)]));
  } catch (e) {
    console.warn("fluid background disabled:", e);
    canvas.remove();
    return;
  }

  // Full-screen quad
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(0);

  function blit(target) {
    if (target) {
      gl.viewport(0, 0, target.w, target.h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    } else {
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }

  // ---------- render targets ----------

  function fbo(w, h) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return {
      fbo: fb, w, h, texel: [1 / w, 1 / h],
      attach(unit) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); return unit; },
    };
  }

  function doubleFbo(w, h) {
    let a = fbo(w, h), b = fbo(w, h);
    return { get read() { return a; }, get write() { return b; }, swap() { [a, b] = [b, a]; } };
  }

  function resolution(res) {
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    const aspect = w > h ? w / h : h / w;
    const min = Math.round(res), max = Math.round(res * aspect);
    return w > h ? [max, min] : [min, max];
  }

  let velocity, dye, pressure, divergence, curl;
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5); // smoke is soft: full retina res is wasted work
    canvas.width = Math.round(innerWidth * dpr);
    canvas.height = Math.round(innerHeight * dpr);
    const [sw, sh] = resolution(CONFIG.SIM_RES), [dw, dh] = resolution(CONFIG.DYE_RES);
    velocity = doubleFbo(sw, sh);
    dye = doubleFbo(dw, dh);
    pressure = doubleFbo(sw, sh);
    divergence = fbo(sw, sh);
    curl = fbo(sw, sh);
  }
  resize();

  // ---------- simulation ----------

  function step(dt) {
    const texel = velocity.read.texel;

    P.curl.use();
    gl.uniform2fv(P.curl.u.texelSize, texel);
    gl.uniform1i(P.curl.u.uVelocity, velocity.read.attach(0));
    blit(curl);

    P.vorticity.use();
    gl.uniform2fv(P.vorticity.u.texelSize, texel);
    gl.uniform1i(P.vorticity.u.uVelocity, velocity.read.attach(0));
    gl.uniform1i(P.vorticity.u.uCurl, curl.attach(1));
    gl.uniform1f(P.vorticity.u.curl, CONFIG.CURL);
    gl.uniform1f(P.vorticity.u.dt, dt);
    blit(velocity.write); velocity.swap();

    P.divergence.use();
    gl.uniform2fv(P.divergence.u.texelSize, texel);
    gl.uniform1i(P.divergence.u.uVelocity, velocity.read.attach(0));
    blit(divergence);

    P.scale.use();
    gl.uniform1i(P.scale.u.uTexture, pressure.read.attach(0));
    gl.uniform1f(P.scale.u.value, CONFIG.PRESSURE);
    blit(pressure.write); pressure.swap();

    P.pressure.use();
    gl.uniform2fv(P.pressure.u.texelSize, texel);
    gl.uniform1i(P.pressure.u.uDivergence, divergence.attach(0));
    for (let i = 0; i < CONFIG.PRESSURE_ITERATIONS; i++) {
      gl.uniform1i(P.pressure.u.uPressure, pressure.read.attach(1));
      blit(pressure.write); pressure.swap();
    }

    P.gradientSubtract.use();
    gl.uniform2fv(P.gradientSubtract.u.texelSize, texel);
    gl.uniform1i(P.gradientSubtract.u.uPressure, pressure.read.attach(0));
    gl.uniform1i(P.gradientSubtract.u.uVelocity, velocity.read.attach(1));
    blit(velocity.write); velocity.swap();

    P.advection.use();
    gl.uniform2fv(P.advection.u.texelSize, texel);
    gl.uniform1f(P.advection.u.dt, dt);
    gl.uniform1i(P.advection.u.uVelocity, velocity.read.attach(0));
    gl.uniform1i(P.advection.u.uSource, velocity.read.attach(0));
    gl.uniform1f(P.advection.u.dissipation, CONFIG.VELOCITY_DISSIPATION);
    blit(velocity.write); velocity.swap();

    gl.uniform1i(P.advection.u.uVelocity, velocity.read.attach(0));
    gl.uniform1i(P.advection.u.uSource, dye.read.attach(1));
    gl.uniform1f(P.advection.u.dissipation, CONFIG.DENSITY_DISSIPATION);
    blit(dye.write); dye.swap();
  }

  function splat(x, y, dx, dy, color) {
    const aspect = canvas.width / canvas.height;
    P.splat.use();
    gl.uniform1f(P.splat.u.aspectRatio, aspect);
    gl.uniform2f(P.splat.u.point, x, y);
    gl.uniform1f(P.splat.u.radius, (CONFIG.SPLAT_RADIUS / 100) * (aspect > 1 ? aspect : 1));

    gl.uniform1i(P.splat.u.uTarget, velocity.read.attach(0));
    gl.uniform3f(P.splat.u.color, dx, dy, 0);
    blit(velocity.write); velocity.swap();

    gl.uniform1i(P.splat.u.uTarget, dye.read.attach(0));
    gl.uniform3fv(P.splat.u.color, color);
    blit(dye.write); dye.swap();
  }

  function render() {
    P.display.use();
    gl.uniform1f(P.display.u.maxBrightness, CONFIG.MAX_BRIGHTNESS);
    gl.uniform1i(P.display.u.uTexture, dye.read.attach(0));
    blit(null);
  }

  // ---------- input and loop ----------

  const pointer = { x: 0, y: 0, dx: 0, dy: 0, moved: false, seen: false };
  let lastMove = 0, lastFrame = 0, raf = 0;

  addEventListener("pointermove", e => {
    const x = e.clientX / innerWidth, y = 1 - e.clientY / innerHeight;
    if (pointer.seen) {
      const aspect = innerWidth / innerHeight;
      pointer.dx += (x - pointer.x) * CONFIG.SPLAT_FORCE * (aspect < 1 ? aspect : 1);
      pointer.dy += (y - pointer.y) * CONFIG.SPLAT_FORCE / (aspect > 1 ? aspect : 1);
      pointer.moved = true;
    }
    pointer.x = x; pointer.y = y; pointer.seen = true;
    lastMove = performance.now();
    wake();
  }, { passive: true });

  function frame(now) {
    const dt = Math.min((now - lastFrame) / 1000, 1 / 60);
    lastFrame = now;
    if (pointer.moved) {
      // Faster strokes leave a little more smoke, capped so it never washes out the text.
      const speed = Math.min(Math.hypot(pointer.dx, pointer.dy) / 60, 1.6);
      splat(pointer.x, pointer.y, pointer.dx, pointer.dy, CONFIG.COLOR.map(c => c * (0.4 + speed)));
      pointer.dx = pointer.dy = 0;
      pointer.moved = false;
    }
    step(dt);
    const idle = now - lastMove >= CONFIG.IDLE_MS;
    if (idle) {
      // Whatever smoke is left is invisible by now; clear it so nothing freezes on screen.
      gl.bindFramebuffer(gl.FRAMEBUFFER, dye.read.fbo);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    render();
    raf = !idle && !document.hidden ? requestAnimationFrame(frame) : 0;
  }

  function wake() {
    if (!raf && !document.hidden) {
      lastFrame = performance.now();
      raf = requestAnimationFrame(frame);
    }
  }

  addEventListener("resize", () => { resize(); render(); });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) { lastMove = performance.now(); wake(); }  // finish fading what was left mid-stroke
  });
  render();
})();
