/* AI 金融智能投研 Agent — 前端交互（无框架依赖）
 *
 * 只做三件事：提问 → 调用 /api/chat → 把「执行轨迹 / 量化结果 / 报告 / 证据来源」渲染出来。
 * 报告为 Markdown，本地渲染（不依赖 CDN，离线演示可用）。
 */
"use strict";

const $ = (sel) => document.querySelector(sel);

/* ---------------------------------------------------------------- markdown */
function escapeHtml(text) {
  return String(text == null ? "" : text)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
             '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/(^|\s)(https?:\/\/[^\s<]+)/g,
             '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const out = [];
  let list = null;      // "ul" | "ol"
  let inQuote = false;
  let table = null;     // { header: [], rows: [] }

  const flushList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  const flushQuote = () => { if (inQuote) { out.push("</blockquote>"); inQuote = false; } };
  const flushTable = () => {
    if (!table) return;
    const head = table.header.map((c) => `<th>${inline(c)}</th>`).join("");
    const body = table.rows
      .map((row) => `<tr>${row.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`)
      .join("");
    out.push(`<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`);
    table = null;
  };

  for (const raw of lines) {
    const line = raw.trimEnd();

    if (/^\s*\|.*\|\s*$/.test(line)) {
      const cells = line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      if (cells.every((c) => /^:?-{2,}:?$/.test(c))) continue;   // separator row
      flushList(); flushQuote();
      if (!table) table = { header: cells, rows: [] };
      else table.rows.push(cells);
      continue;
    }
    flushTable();

    if (!line.trim()) { flushList(); flushQuote(); continue; }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      flushList(); flushQuote();
      out.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`);
      continue;
    }
    if (/^(-{3,}|\*{3,})$/.test(line.trim())) { flushList(); flushQuote(); out.push("<hr/>"); continue; }

    const quote = line.match(/^>\s?(.*)$/);
    if (quote) {
      flushList();
      if (!inQuote) { out.push("<blockquote>"); inQuote = true; }
      out.push(`<p>${inline(quote[1])}</p>`);
      continue;
    }
    flushQuote();

    const ul = line.match(/^\s*[-*+]\s+(.*)$/);
    if (ul) {
      if (list !== "ul") { flushList(); out.push("<ul>"); list = "ul"; }
      out.push(`<li>${inline(ul[1])}</li>`);
      continue;
    }
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ol) {
      if (list !== "ol") { flushList(); out.push("<ol>"); list = "ol"; }
      out.push(`<li>${inline(ol[1])}</li>`);
      continue;
    }
    flushList();
    out.push(`<p>${inline(line)}</p>`);
  }
  flushTable(); flushList(); flushQuote();
  return out.join("\n");
}

/* -------------------------------------------------------------- rendering */
function fmt(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "—";
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return Math.abs(num) >= 1000 ? num.toLocaleString("zh-CN") : num.toFixed(digits);
}

function pct(value, digits = 2) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  return `${(num * 100).toFixed(digits)}%`;
}

function renderTrace(data, elapsedMs) {
  const rows = [];
  rows.push(`<div class="kv">
    <div><span>意图</span><span>${escapeHtml(data.intent)} <span class="muted">(${escapeHtml(data.intent_source)})</span></span></div>
    <div><span>股票</span><span>${escapeHtml((data.stock_codes || []).join(", ") || "—")}</span></div>
    <div><span>计划工具</span><span>${escapeHtml((data.plan || []).join(" → ") || "—")}</span></div>
    <div><span>证据校验</span><span>${data.evidence_ok ? "通过" : "不足"}</span></div>
    <div><span>模型</span><span>${escapeHtml(data.model)}</span></div>
    <div><span>耗时</span><span>${(elapsedMs / 1000).toFixed(1)}s</span></div>
  </div>`);

  for (const call of data.tool_calls || []) {
    rows.push(`<div class="tool">
      <span class="tag ${call.ok ? "ok" : "err"}">${call.ok ? "ok" : "fail"}</span>
      <strong>${escapeHtml(call.name)}</strong>
      <code>${escapeHtml(JSON.stringify(call.arguments || {}))}</code>
    </div>`);
  }
  for (const err of data.tool_errors || []) {
    rows.push(`<div class="tool"><span class="tag err">${escapeHtml(err.code)}</span>${escapeHtml(err.message)}</div>`);
  }
  for (const note of data.validator_notes || []) {
    rows.push(`<div class="tool"><span class="tag">note</span>${escapeHtml(note)}</div>`);
  }
  $("#trace").innerHTML = rows.join("");
  $("#trace-box").classList.remove("hidden");
}

function renderQuant(quant) {
  const html = [];
  const prediction = quant.predict_stock;
  const factors = quant.get_stock_factors;
  const ranking = quant.rank_stocks;
  const profile = quant.get_stock_profile;

  const cards = [];
  if (prediction) {
    const cls = Number(prediction.pred_return_5d) >= 0 ? "pos" : "neg";
    cards.push(`<div class="card">
      <h3>模型预测 · ${escapeHtml(prediction.name || prediction.code || "")}</h3>
      <div class="big ${cls}">${pct(prediction.pred_return_5d)}</div>
      <div class="meta">5 日预测超额收益 · 基准日 ${escapeHtml(prediction.as_of || "")}</div>
      <div class="meta">综合评分 ${fmt(prediction.score, 2)} · 风险 ${fmt(prediction.risk_score, 3)} ·
        strong_prob ${fmt(prediction.strong_prob, 3)}（弱信号）</div>
      <div class="meta">模型版本 ${escapeHtml(prediction.model_version || "")}</div>
    </div>`);
  }
  if (factors) {
    cards.push(`<div class="card">
      <h3>因子快照 · ${escapeHtml(factors.name || factors.code || "")}</h3>
      <div class="meta">基准日 ${escapeHtml(factors.as_of || "")} · 板块 ${escapeHtml(factors.board || "—")}</div>
      <table>
        <tr><th>ret_5d</th><td>${pct(factors.ret_5d)}</td><th>ret_20d</th><td>${pct(factors.ret_20d)}</td></tr>
        <tr><th>ma20_bias</th><td>${pct(factors.ma20_bias)}</td><th>波动率20d</th><td>${pct(factors.volatility_20d)}</td></tr>
        <tr><th>量比5d</th><td>${fmt(factors.volume_ratio_5d, 3)}</td><th>20日回撤</th><td>${pct(factors.drawdown_20d)}</td></tr>
      </table>
    </div>`);
  }
  if (profile && !prediction && !factors) {
    cards.push(`<div class="card">
      <h3>股票信息</h3>
      <div class="big">${escapeHtml(profile.name || "")}</div>
      <div class="meta">${escapeHtml(profile.code || "")} · ${escapeHtml(profile.market || "")} ·
        本地量化数据：${profile.has_factor_data ? "有" : "无"}</div>
    </div>`);
  }
  if (cards.length) html.push(`<div class="cards">${cards.join("")}</div>`);

  if (ranking && (ranking.items || []).length) {
    const rows = ranking.items
      .map((item) => `<tr>
        <td>${item.rank}</td><td>${escapeHtml(item.code)}</td><td>${escapeHtml(item.name)}</td>
        <td>${fmt(item.score, 2)}</td><td>${pct(item.pred_return_5d)}</td>
        <td>${fmt(item.risk_score, 3)}</td><td>${fmt(item.strong_prob, 3)}</td>
        <td>${escapeHtml(item.main_factor || "")}</td></tr>`)
      .join("");
    html.push(`<h3 style="font-size:14px">选股排名 · 基准日 ${escapeHtml(ranking.as_of || "")}</h3>
      <table><thead><tr><th>#</th><th>代码</th><th>名称</th><th>评分</th><th>5日预测</th>
      <th>风险</th><th>strong_prob</th><th>主因子</th></tr></thead><tbody>${rows}</tbody></table>`);
  }

  if (!html.length) { $("#quant-box").classList.add("hidden"); return; }
  $("#quant").innerHTML = html.join("");
  $("#quant-box").classList.remove("hidden");
}

function renderEvidence(evidence) {
  if (!evidence || !evidence.length) { $("#evidence-box").classList.add("hidden"); return; }
  $("#evidence-count").textContent = `（${evidence.length} 条，均已标注来源）`;
  $("#evidence").innerHTML = evidence.map((item) => {
    const scores = [
      item.dense_score != null ? `dense ${fmt(item.dense_score, 3)}` : null,
      item.bm25_score != null ? `bm25 ${fmt(item.bm25_score, 2)}` : null,
      item.rrf_score != null ? `rrf ${fmt(item.rrf_score, 4)}` : null,
      item.rerank_score != null ? `rerank ${fmt(item.rerank_score, 3)}` : null,
    ].filter(Boolean).join(" · ");
    const link = item.url
      ? `<a href="${escapeHtml(item.url)}" target="_blank" rel="noopener">${escapeHtml(item.title || item.url)}</a>`
      : escapeHtml(item.title || "");
    return `<div class="evidence">
      <div>${link}</div>
      <div class="scores">
        <span class="tag">${escapeHtml(item.doc_type || "")}</span>
        ${escapeHtml(item.stock_code || "")} ${escapeHtml(item.stock_name || "")} ·
        ${escapeHtml((item.publish_time || "").slice(0, 19))} · ${escapeHtml(item.source || "")}
      </div>
      <div class="scores">${scores}</div>
      <p>${escapeHtml((item.content || "").slice(0, 260))}${(item.content || "").length > 260 ? "…" : ""}</p>
    </div>`;
  }).join("");
  $("#evidence-box").classList.remove("hidden");
}

/* -------------------------------------------------------------------- flow */
function showError(message) {
  $("#error").textContent = message;
  $("#error").classList.remove("hidden");
}

function reset() {
  $("#error").classList.add("hidden");
  for (const id of ["trace-box", "quant-box", "answer-box", "evidence-box"]) {
    $(`#${id}`).classList.add("hidden");
  }
}

async function ask(query) {
  reset();
  const button = $("#submit");
  button.disabled = true;
  $("#status").textContent = "分析中…（首次请求需加载向量与重排模型，约 30s）";
  $("#status").className = "status busy";
  const started = Date.now();
  const timer = setInterval(() => {
    $("#elapsed").textContent = `${((Date.now() - started) / 1000).toFixed(0)}s`;
  }, 500);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = body.detail || body;
      throw new Error(`[${detail.code || response.status}] ${detail.message || "请求失败"}`);
    }

    const elapsedMs = Date.now() - started;
    renderTrace(body, elapsedMs);
    renderQuant(body.quant || {});
    $("#answer").innerHTML = renderMarkdown(body.answer || "");
    $("#answer-box").classList.remove("hidden");
    renderEvidence(body.evidence || []);

    $("#status").textContent = body.evidence_ok ? "完成" : "完成（证据不足，见轨迹）";
    $("#status").className = "status";
  } catch (err) {
    $("#status").textContent = "出错";
    $("#status").className = "status err";
    showError(`请求失败：${err.message}\n\n若提示 LLMUnavailable，请检查 .env 中的 LLM_API_KEY。`);
  } finally {
    clearInterval(timer);
    $("#elapsed").textContent = "";
    button.disabled = false;
  }
}

$("#submit").addEventListener("click", () => {
  const query = $("#query").value.trim();
  if (query.length < 2) { showError("请输入至少 2 个字符的问题。"); return; }
  ask(query);
});
$("#query").addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") $("#submit").click();
});
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    $("#query").value = chip.dataset.q;
    ask(chip.dataset.q);
  });
});
