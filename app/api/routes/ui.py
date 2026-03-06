from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["UI"])

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>RAG UI</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; max-width: 980px; }
    h1 { margin: 0 0 6px 0; }
    .muted { color: #666; margin-top: 0; }
    textarea { width: 100%; height: 90px; padding: 10px; border-radius: 10px; border: 1px solid #ddd; }
    input { padding: 10px; border-radius: 10px; border: 1px solid #ddd; width: 100%; }
    button { padding: 10px 16px; margin-top: 10px; border-radius: 10px; border: 1px solid #111; background: #111; color: #fff; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .row { display:flex; gap:12px; margin-top: 10px; }
    .col { flex:1; }
    .panel { margin-top: 16px; }
    .bar { display:flex; gap:12px; flex-wrap: wrap; font-size: 14px; background: #f6f7f9; border: 1px solid #e6e8ee; padding: 10px; border-radius: 12px; }
    .chip { background:#fff; border:1px solid #e6e8ee; padding: 6px 10px; border-radius: 999px; }
    .card { border:1px solid #e6e8ee; border-radius: 14px; padding: 12px; margin: 10px 0; background: #fff; }
    .meta { display:flex; gap:10px; flex-wrap:wrap; font-size: 13px; color:#222; margin-bottom: 8px; }
    .meta b { color:#000; }
    .preview { white-space: pre-wrap; color:#111; line-height: 1.35; }
    details { margin-top: 10px; }
    summary { cursor: pointer; color: #0b5fff; }
    .error { background:#fff3f3; border:1px solid #ffd1d1; padding: 10px; border-radius: 12px; white-space: pre-wrap; }
    .loading { color:#666; }
    .footer { margin-top: 14px; color:#777; font-size: 12px; }
  </style>
</head>
<body>
  <h1>RAG UI</h1>
  <p class="muted">Uses <code>/query</code> (retrieval). Later you’ll switch to <code>/ask</code> for grounded generation.</p>

  <label><b>Question</b></label>
  <textarea id="q" placeholder="Type your question..."></textarea>

  <div class="row">
    <div class="col">
      <label><b>top_k</b></label>
      <input id="k" type="number" value="5" min="1" max="20" />
    </div>
    <div class="col">
      <label><b>source (optional)</b></label>
      <input id="src" type="text" placeholder="flashattention.pdf" />
    </div>
  </div>

  <button id="btn" onclick="run()">Ask</button>

  <div class="panel" id="out">
    <div class="loading">Ready.</div>
  </div>

<script>
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;"
  }[c]));
}

function fmt(x) {
  return (x === null || x === undefined) ? "-" : x;
}

async function run() {
  const btn = document.getElementById("btn");
  const root = document.getElementById("out");

  const query = document.getElementById("q").value.trim();
  const top_k = parseInt(document.getElementById("k").value || "5", 10);
  const source = document.getElementById("src").value.trim();

  if (!query) { alert("Type a question"); return; }

  const payload = { query, top_k };
  if (source) payload.source = source;

  btn.disabled = true;
  root.innerHTML = `<div class="loading">Loading...</div>`;

  try {
    const r = await fetch("/query", {
      method: "POST",
      headers: {"content-type":"application/json"},
      body: JSON.stringify(payload),
    });

    if (!r.ok) {
      const t = await r.text();
      root.innerHTML = `<div class="error"><b>HTTP ${r.status}</b>\\n${escapeHtml(t)}</div>`;
      return;
    }

    const data = await r.json();

    // header bar
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.innerHTML = `
      <div class="chip"><b>Query</b>: ${escapeHtml(data.query)}</div>
      <div class="chip"><b>TopK</b>: ${fmt(data.top_k)}</div>
      <div class="chip"><b>embed_ms</b>: ${fmt(data.timings_ms?.embed_ms)}</div>
      <div class="chip"><b>search_ms</b>: ${fmt(data.timings_ms?.search_ms)}</div>
      <div class="chip"><b>total_ms</b>: ${fmt(data.timings_ms?.total_ms)}</div>
    `;

    root.innerHTML = "";
    root.appendChild(bar);

    const results = data.results || [];
    if (results.length === 0) {
      const empty = document.createElement("div");
      empty.className = "card";
      empty.textContent = "No results.";
      root.appendChild(empty);
      return;
    }

    results.forEach((x, i) => {
      const card = document.createElement("div");
      card.className = "card";

      const score = (typeof x.score === "number") ? x.score.toFixed(4) : fmt(x.score);

      card.innerHTML = `
        <div class="meta">
          <div><b>#${i+1}</b></div>
          <div><b>score</b> ${score}</div>
          <div><b>source</b> ${escapeHtml(fmt(x.source))}</div>
          <div><b>page</b> ${fmt(x.page)}</div>
          <div><b>chunk_idx</b> ${fmt(x.chunk_idx)}</div>
          <div><b>id</b> <code>${escapeHtml(fmt(x.id))}</code></div>
        </div>

        <div class="preview">${escapeHtml(fmt(x.text_preview))}</div>

        <details>
          <summary>Show full chunk text</summary>
          <div class="preview" style="margin-top:8px">${escapeHtml(fmt(x.text))}</div>
        </details>
      `;

      root.appendChild(card);
    });

    const footer = document.createElement("div");
    footer.className = "footer";
    footer.innerHTML = `Tip: add <code>source</code> filter (e.g. <code>flashattention.pdf</code>) to debug retrieval.`;
    root.appendChild(footer);

  } catch (e) {
    root.innerHTML = `<div class="error"><b>Error</b>\\n${escapeHtml(e)}</div>`;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return HTML

@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return HTML