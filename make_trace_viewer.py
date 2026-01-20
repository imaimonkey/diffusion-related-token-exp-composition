#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


STEP_RE = re.compile(r"^step(?P<step>\d+):\s*(?P<body>.*?)(?:\s+\(unmasked=(?P<unmasked>\d+)\))?\s*$")
SAMPLE_RE = re.compile(r"^\[(?P<k>\d+)/(?P<n>\d+)\]\s+index=(?P<index>\d+)\s*$")
GOLD_RE = re.compile(r"^gold=(?P<gold>\S+)\s+pred=(?P<pred>\S+)\s*$")
FINAL_RE = re.compile(r"^final_pred=(?P<final_pred>\S+)\s+correct=(?P<correct>\S+)\s*$")
PROMPT_RE = re.compile(r"^prompt_text:\s*(?P<prompt>.*)\s*$")

def normalize_trace_text(s: str) -> str:
    # Make tokenizer artifacts less noisy in the viewer.
    # - GPT2/byte-level BPE uses "Ġ" for a leading space and "Ċ" for newline.
    # - SentencePiece often uses "▁" for a leading space.
    return s.replace("Ġ", "").replace("▁", "").replace("Ċ", "↵")


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Trace Viewer</title>
  <style>
    :root {{
      --bg: #0b0f14;
      --panel: #111827;
      --panel2: #0f172a;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #60a5fa;
      --border: #1f2937;
      --mask: #374151;
      --changed: #fbbf24;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: linear-gradient(180deg, rgba(11,15,20,0.98), rgba(11,15,20,0.85));
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(10px);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 14px 16px;
    }}
    .row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }}
    .btn {{
      border: 1px solid var(--border);
      background: var(--panel2);
      color: var(--text);
      padding: 8px 10px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 600;
    }}
    .btn:hover {{ border-color: #334155; }}
    select, input[type="range"] {{
      border-radius: 10px;
    }}
    select {{
      border: 1px solid var(--border);
      background: var(--panel2);
      color: var(--text);
      padding: 8px 10px;
      font-weight: 600;
    }}
    .meta {{
      display: grid;
      gap: 8px;
      padding: 14px 16px;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: var(--panel);
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      margin-top: 14px;
    }}
    .card {{
      border: 1px solid var(--border);
      border-radius: 16px;
      background: var(--panel);
      padding: 14px 16px;
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .value {{
      font-size: 14px;
      line-height: 1.4;
      word-break: break-word;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--border);
      background: var(--panel2);
      padding: 6px 10px;
      border-radius: 999px;
      font-weight: 700;
      color: var(--text);
    }}
    .pill b {{ color: var(--accent); }}
    .tokens {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: flex-start;
      line-height: 1.1;
      user-select: text;
    }}
    .tok {{
      border: 1px solid var(--border);
      background: var(--panel2);
      padding: 4px 6px;
      border-radius: 10px;
      font-size: 13px;
    }}
    .tok.mask {{
      color: var(--muted);
      border-color: #1f2937;
      background: rgba(55,65,81,0.25);
    }}
    .tok.changed {{
      border-color: rgba(251,191,36,0.7);
      box-shadow: 0 0 0 2px rgba(251,191,36,0.18) inset;
    }}
    .hint {{
      color: var(--muted);
      font-size: 12px;
    }}
    .range {{
      min-width: 360px;
      max-width: 520px;
      width: 42vw;
    }}
    .range input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .footer {{
      padding: 18px 16px 30px;
      color: var(--muted);
      font-size: 12px;
      text-align: center;
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <div class="row">
        <div class="controls">
          <button class="btn" id="prevSample">◀ Sample</button>
          <select id="sampleSelect"></select>
          <button class="btn" id="nextSample">Sample ▶</button>
        </div>
        <div class="controls">
          <button class="btn" id="prevStep">◀ Step</button>
          <div class="range">
            <div class="hint"><span id="stepLabel"></span></div>
            <input type="range" min="0" max="0" value="0" id="stepRange" />
          </div>
          <button class="btn" id="nextStep">Step ▶</button>
          <button class="btn" id="playPause">Play</button>
        </div>
      </div>
      <div class="hint" style="margin-top:8px">
        Highlight: <span class="tok changed mono">changed</span> <span class="tok mask mono">[mask]</span>
      </div>
    </div>
  </header>

  <main class="wrap">
    <div class="grid">
      <div class="meta">
        <div class="row">
          <div class="pill">Sample <b id="sampleCounter">-</b></div>
          <div class="pill">Index <b id="sampleIndex">-</b></div>
          <div class="pill">Gold <b id="sampleGold">-</b></div>
          <div class="pill">Pred <b id="samplePred">-</b></div>
          <div class="pill">Final <b id="sampleFinal">-</b></div>
          <div class="pill">Correct <b id="sampleCorrect">-</b></div>
        </div>
      </div>

      <div class="card">
        <div class="label">Question</div>
        <div class="value" id="question"></div>
      </div>

      <div class="card">
        <div class="row">
          <div>
            <div class="label">Step State</div>
            <div class="hint mono" id="stepMeta"></div>
          </div>
          <div class="hint">Tip: drag the step slider</div>
        </div>
        <div class="tokens mono" id="tokens"></div>
      </div>
    </div>
  </main>

  <div class="footer">
    Generated by make_trace_viewer.py
  </div>

  <script id="trace-data" type="application/json">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('trace-data').textContent);
    const samples = data.samples || [];

    const elSampleSelect = document.getElementById('sampleSelect');
    const elStepRange = document.getElementById('stepRange');
    const elTokens = document.getElementById('tokens');
    const elStepLabel = document.getElementById('stepLabel');
    const elStepMeta = document.getElementById('stepMeta');

    const elSampleCounter = document.getElementById('sampleCounter');
    const elSampleIndex = document.getElementById('sampleIndex');
    const elSampleGold = document.getElementById('sampleGold');
    const elSamplePred = document.getElementById('samplePred');
    const elSampleFinal = document.getElementById('sampleFinal');
    const elSampleCorrect = document.getElementById('sampleCorrect');
    const elQuestion = document.getElementById('question');

    let sampleIdx = 0;
    let stepIdx = 0;
    let playing = false;
    let timer = null;

    function clamp(v, lo, hi) {{
      return Math.max(lo, Math.min(hi, v));
    }}

    function stopPlay() {{
      playing = false;
      if (timer) window.clearInterval(timer);
      timer = null;
      document.getElementById('playPause').textContent = 'Play';
    }}

    function startPlay() {{
      if (playing) return;
      playing = true;
      document.getElementById('playPause').textContent = 'Pause';
      timer = window.setInterval(() => {{
        const s = samples[sampleIdx];
        const maxStep = (s.steps?.length || 1) - 1;
        if (stepIdx >= maxStep) {{
          stopPlay();
          return;
        }}
        setStep(stepIdx + 1);
      }}, 120);
    }}

    function buildSampleSelect() {{
      elSampleSelect.innerHTML = '';
      samples.forEach((s, i) => {{
        const opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = `#${{i+1}} index=${{s.index}} gold=${{s.gold}} pred=${{s.pred}}`;
        elSampleSelect.appendChild(opt);
      }});
    }}

    function renderTokens(tokens, prevTokens) {{
      elTokens.innerHTML = '';
      for (let i = 0; i < tokens.length; i++) {{
        const t = tokens[i];
        const span = document.createElement('span');
        span.className = 'tok mono';
        if (t === '[mask]') span.classList.add('mask');
        if (prevTokens && prevTokens[i] !== undefined && prevTokens[i] !== t) span.classList.add('changed');
        span.textContent = t;
        elTokens.appendChild(span);
      }}
    }}

    function setSample(i) {{
      stopPlay();
      sampleIdx = clamp(i, 0, samples.length - 1);
      const s = samples[sampleIdx];
      elSampleSelect.value = String(sampleIdx);

      elSampleCounter.textContent = `${{sampleIdx + 1}}/${{samples.length}}`;
      elSampleIndex.textContent = String(s.index ?? '-');
      elSampleGold.textContent = String(s.gold ?? '-');
      elSamplePred.textContent = String(s.pred ?? '-');
      elSampleFinal.textContent = String(s.final_pred ?? '-');
      elSampleCorrect.textContent = String(s.correct ?? '-');
      elQuestion.textContent = String(s.prompt_text ?? '');

      const maxStep = (s.steps?.length || 1) - 1;
      elStepRange.max = String(maxStep);
      setStep(0);
    }}

    function setStep(i) {{
      const s = samples[sampleIdx];
      const steps = s.steps || [];
      const maxStep = steps.length - 1;
      stepIdx = clamp(i, 0, maxStep);
      elStepRange.value = String(stepIdx);

      const cur = steps[stepIdx] || {{ step: stepIdx, state: '' }};
      const prev = stepIdx > 0 ? steps[stepIdx - 1] : null;

      elStepLabel.textContent = `Step ${{stepIdx}}/${{maxStep}}`;
      const unmasked = cur.unmasked == null ? '-' : cur.unmasked;
      elStepMeta.textContent = `unmasked=${{unmasked}}`;

      const tokens = (cur.state || '').trim().split(/\\s+/).filter(Boolean);
      const prevTokens = prev ? (prev.state || '').trim().split(/\\s+/).filter(Boolean) : null;
      renderTokens(tokens, prevTokens);
    }}

    document.getElementById('prevSample').addEventListener('click', () => setSample(sampleIdx - 1));
    document.getElementById('nextSample').addEventListener('click', () => setSample(sampleIdx + 1));
    elSampleSelect.addEventListener('change', (e) => setSample(parseInt(e.target.value, 10)));

    document.getElementById('prevStep').addEventListener('click', () => setStep(stepIdx - 1));
    document.getElementById('nextStep').addEventListener('click', () => setStep(stepIdx + 1));
    elStepRange.addEventListener('input', (e) => setStep(parseInt(e.target.value, 10)));

    document.getElementById('playPause').addEventListener('click', () => {{
      if (playing) stopPlay();
      else startPlay();
    }});

    // Keyboard: left/right for step, up/down for sample.
    window.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowLeft') setStep(stepIdx - 1);
      if (e.key === 'ArrowRight') setStep(stepIdx + 1);
      if (e.key === 'ArrowUp') setSample(sampleIdx - 1);
      if (e.key === 'ArrowDown') setSample(sampleIdx + 1);
    }});

    buildSampleSelect();
    setSample(0);
  </script>
</body>
</html>
"""


def parse_trace_log(path: Path):
    samples = []
    current = None

    def finish_current():
        nonlocal current
        if not current:
            return
        steps = current.get("steps", [])
        steps.sort(key=lambda x: x["step"])
        current["steps"] = steps
        samples.append(current)
        current = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            m = SAMPLE_RE.match(line)
            if m:
                finish_current()
                current = {
                    "k": int(m.group("k")),
                    "n": int(m.group("n")),
                    "index": int(m.group("index")),
                    "steps": [],
                }
                continue

            if current is None:
                continue

            m = GOLD_RE.match(line)
            if m:
                current["gold"] = m.group("gold")
                current["pred"] = m.group("pred")
                continue

            m = PROMPT_RE.match(line)
            if m:
                current["prompt_text"] = m.group("prompt")
                continue

            m = FINAL_RE.match(line)
            if m:
                current["final_pred"] = m.group("final_pred")
                current["correct"] = m.group("correct")
                continue

            m = STEP_RE.match(line)
            if m:
                step = int(m.group("step"))
                body = normalize_trace_text(m.group("body"))
                unmasked = m.group("unmasked")
                current["steps"].append(
                    {
                        "step": step,
                        "state": body,
                        "unmasked": int(unmasked) if unmasked is not None else None,
                    }
                )

    finish_current()
    return {"samples": samples}


def main():
    parser = argparse.ArgumentParser(description="Build a self-contained HTML viewer for trace_gsm8k_wrong logs.")
    parser.add_argument("log_path", help="Path to trace .out log (e.g., logs/trace_gsm8k_wrong_10445.out)")
    parser.add_argument("--output", default=None, help="Output HTML path (default: <log>.html)")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")

    output = Path(args.output) if args.output else log_path.with_suffix(".html")
    data = parse_trace_log(log_path)
    data_json = json.dumps(data, ensure_ascii=False)
    data_json = data_json.replace("</", "<\\/")  # avoid breaking out of <script>

    html = HTML_TEMPLATE.format(data_json=data_json)
    output.write_text(html, encoding="utf-8")
    print(f"Wrote: {output}")
    print("Open it in a browser (Chrome/Firefox).")


if __name__ == "__main__":
    main()
