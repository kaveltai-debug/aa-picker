"""Local web UI and JSON API for Artificial Analysis model picker."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .db import get_connection, get_db_age_hours, get_meta, init_db
from .fetch_and_store import ensure_fresh_db
from .picker import format_telegram_text, pick_models

HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Подбор ИИ-моделей (AA + Gemini Decision)</title>
<style>
  :root {
    --bg: #0f172a; --card: #1e293b; --muted: #94a3b8; --text: #f1f5f9;
    --accent: #38bdf8; --accent2: #22c55e; --warn: #f59e0b; --border: #334155;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); }
  .wrap { max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 22px; margin: 0 0 6px; }
  .sub { color: var(--muted); font-size: 14px; margin-bottom: 18px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
  .badge { display: inline-block; padding: 3px 9px; border-radius: 999px; font-size: 12px; background: #0b2538; color: var(--accent); border: 1px solid #1e4b6e; }
  .badge.warn { background: #3b2606; color: var(--warn); border-color: #78480a; }
  .panel { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 16px; }
  label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 6px; }
  textarea, input, select {
    width: 100%; background: #0f172a; color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 12px; font-size: 14px;
  }
  textarea { min-height: 74px; resize: vertical; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 12px; }
  .actions { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 14px; }
  button {
    background: var(--accent); color: #082032; border: 0; border-radius: 8px;
    padding: 10px 16px; font-weight: 600; cursor: pointer; font-size: 14px;
  }
  button.sec { background: #334155; color: var(--text); }
  button:disabled { opacity: 0.6; cursor: wait; }
  .chk { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text); margin-left: auto; }
  .chk input { width: auto; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; margin-bottom: 12px; }
  .card.top1 { border-color: var(--accent); }
  .card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .card-title { font-size: 17px; font-weight: 700; }
  .score { font-size: 15px; font-weight: 700; color: var(--accent2); }
  .meta { display: flex; flex-wrap: wrap; gap: 14px; font-size: 13px; color: var(--muted); margin-top: 8px; }
  .meta b { color: var(--text); font-weight: 600; }
  .expl { margin-top: 8px; font-size: 13px; color: #cbd5e1; }
  pre { background: #0b1220; border: 1px solid var(--border); border-radius: 8px; padding: 12px; white-space: pre-wrap; font-size: 13px; margin: 8px 0 0; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Подбор ИИ-модели под задачу</h1>
  <div class="sub">
    <span id="dbStatus" class="badge">Загрузка статуса базы...</span>
    <span id="decStatus" class="badge" style="display:none"></span>
  </div>

  <div class="panel">
    <label for="task">Описание задачи</label>
    <textarea id="task" placeholder="Например: код на Python, дёшево"></textarea>
    <div class="grid">
      <div>
        <label for="priority">Приоритет</label>
        <select id="priority">
          <option value="auto">Авто (Gemini Decision)</option>
          <option value="quality">Качество</option>
          <option value="price">Цена</option>
          <option value="speed">Скорость</option>
        </select>
      </div>
      <div>
        <label for="max_price">Макс. цена input ($/1M)</label>
        <input id="max_price" type="number" step="0.01" min="0" placeholder="без ограничения">
      </div>
      <div>
        <label for="min_speed">Мин. скорость (tok/s)</label>
        <input id="min_speed" type="number" step="1" min="0" placeholder="без ограничения">
      </div>
      <div>
        <label for="max_ttft">Макс. TTFT (сек)</label>
        <input id="max_ttft" type="number" step="0.1" min="0" placeholder="без ограничения">
      </div>
      <div>
        <label for="min_context">Мин. контекст (токенов)</label>
        <input id="min_context" type="number" step="1000" min="0" placeholder="без ограничения">
      </div>
    </div>
    <div class="actions">
      <button id="pickBtn" onclick="runPick()">Подобрать</button>
      <button id="refreshBtn" class="sec" onclick="runRefresh()">Обновить базу AA</button>
      <label class="chk"><input id="show_all" type="checkbox"> Показать все модели (вне парка)</label>
    </div>
  </div>

  <div id="results"></div>
</div>
<script>
async function loadStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const el = document.getElementById('dbStatus');
    const age = d.age_hours !== null && d.age_hours !== undefined ? d.age_hours.toFixed(1) + ' ч' : 'н/д';
    el.textContent = `База: ${d.models_count} моделей, ${d.endpoints_count} эндпоинтов | возраст: ${age}`;
    el.className = 'badge' + (d.stale ? ' warn' : '');
    if (d.stale) el.textContent += ' (устарела >24ч)';
  } catch (e) {
    document.getElementById('dbStatus').textContent = 'Ошибка статуса БД';
  }
}

function numOrNull(id) {
  const v = document.getElementById(id).value.trim();
  return v === '' ? null : Number(v);
}

async function runPick() {
  const task = document.getElementById('task').value.trim();
  if (!task) { alert('Введите описание задачи'); return; }
  const btn = document.getElementById('pickBtn');
  btn.disabled = true;
  try {
    const body = {
      task,
      priority: document.getElementById('priority').value,
      max_price: numOrNull('max_price'),
      min_speed: numOrNull('min_speed'),
      max_ttft: numOrNull('max_ttft'),
      min_context: numOrNull('min_context'),
      show_all: document.getElementById('show_all').checked
    };
    const r = await fetch('/api/pick', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const d = await r.json();
    if (!r.ok) { alert(d.error || 'Ошибка подбора'); return; }
    renderResult(d);
    loadStatus();
  } finally {
    btn.disabled = false;
  }
}

async function runRefresh() {
  const btn = document.getElementById('refreshBtn');
  btn.disabled = true;
  btn.textContent = 'Обновляю...';
  try {
    const r = await fetch('/api/refresh', {method: 'POST'});
    const d = await r.json();
    if (!r.ok) { alert(d.error || 'Ошибка обновления'); return; }
    loadStatus();
  } finally {
    btn.disabled = false;
    btn.textContent = 'Обновить базу AA';
  }
}

function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function renderResult(d) {
  const decEl = document.getElementById('decStatus');
  decEl.style.display = 'inline-block';
  decEl.textContent = `Decision: ${d.decision.source} | приоритет: ${d.priority} | теги: ${(d.decision.tags || []).join(', ')}`;

  const box = document.getElementById('results');
  let html = '';
  if (d.db && d.db.stale) {
    const age = d.db.age_hours !== null ? d.db.age_hours.toFixed(1) + ' ч' : 'н/д';
    html += `<div class="panel" style="border-color:#78480a;color:#fbbf24">Внимание: база моделей устарела (возраст ${age} &gt; 24 ч).</div>`;
  }
  const top3 = d.top3 || [];
  if (!top3.length) {
    html += `<div class="card">Подходящие модели с ненулевой ценой и скоростью под заданные фильтры не найдены.</div>`;
  } else {
    top3.forEach((m, i) => {
      const park = m.park_engines && m.park_engines.length ? m.park_engines.join(', ') : 'вне парка';
      const ttft = m.time_to_first_token !== null && m.time_to_first_token !== undefined ? m.time_to_first_token.toFixed(2) + ' с' : 'н/д';
      const ctx = m.context_window_tokens ? Math.round(m.context_window_tokens / 1000) + 'k' : 'н/д';
      html += `
        <div class="card ${i === 0 ? 'top1' : ''}">
          <div class="card-head">
            <div class="card-title">${i + 1}. ${esc(m.name)}</div>
            <div class="score">${m.task_score.toFixed(1)} / 100</div>
          </div>
          <div class="meta">
            <span>Провайдер: <b>${esc(m.provider)}</b></span>
            <span>Парк: <b>${esc(park)}</b></span>
            <span>Effort: <b>${esc(m.effort)}</b></span>
            <span>Цена (in/out): <b>$${m.price_1m_input.toFixed(2)} / $${m.price_1m_output.toFixed(2)}</b></span>
            <span>Скорость: <b>${m.output_speed.toFixed(1)} tok/s</b></span>
            <span>TTFT: <b>${ttft}</b></span>
            <span>Контекст: <b>${ctx}</b></span>
          </div>
          <div class="expl">${esc(m.explanation)}</div>
        </div>
      `;
    });
  }
  html += `<div class="panel"><label>Ответ для Telegram (идентично CLI)</label><pre>${esc(d.telegram_text || '')}</pre></div>`;
  box.innerHTML = html;
}

loadStatus();
</script>
</body>
</html>
"""


def _read_status(db_path: Path | str | None = None) -> dict[str, Any]:
    path = init_db(db_path)
    conn = get_connection(path)
    try:
        m_cnt = int(conn.execute("SELECT COUNT(*) FROM models").fetchone()[0])
        ep_cnt = int(conn.execute("SELECT COUNT(*) FROM provider_endpoints").fetchone()[0])
        age = get_db_age_hours(conn)
        last_upd = get_meta(conn, "last_updated_utc")
    finally:
        conn.close()
    return {
        "db_path": str(path),
        "models_count": m_cnt,
        "endpoints_count": ep_cnt,
        "last_updated_utc": last_upd,
        "age_hours": age,
        "stale": age is None or age > 24.0,
    }


def _opt_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    return float(val)


def _opt_int(val: Any) -> int | None:
    if val is None or val == "":
        return None
    return int(val)


def make_handler(db_path: Path | str | None = None):
    class PickerHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, code: int, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(200, HTML_PAGE)
                return
            if parsed.path == "/health":
                self._send_json(200, {"ok": True, "db": _read_status(db_path)})
                return
            if parsed.path == "/api/status":
                self._send_json(200, _read_status(db_path))
                return
            if parsed.path == "/api/pick":
                qs = parse_qs(parsed.query)
                task = (qs.get("task") or [""])[0].strip()
                if not task:
                    self._send_json(400, {"error": "Параметр task обязателен"})
                    return
                res = pick_models(
                    task,
                    priority=(qs.get("priority") or ["auto"])[0],
                    max_price=_opt_float((qs.get("max_price") or [None])[0]),
                    min_speed=_opt_float((qs.get("min_speed") or [None])[0]),
                    max_ttft=_opt_float((qs.get("max_ttft") or [None])[0]),
                    min_context=_opt_int((qs.get("min_context") or [None])[0]),
                    show_all=(qs.get("show_all") or ["false"])[0].lower() in ("1", "true", "yes"),
                    db_path=db_path,
                )
                res["telegram_text"] = format_telegram_text(res)
                self._send_json(200, res)
                return
            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                self._send_json(400, {"error": "Некорректный JSON"})
                return

            if parsed.path == "/api/refresh":
                try:
                    info = ensure_fresh_db(db_path=db_path, force=True)
                    self._send_json(200, info)
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            if parsed.path == "/api/pick":
                task = str(body.get("task") or "").strip()
                if not task:
                    self._send_json(400, {"error": "Поле task обязательно"})
                    return
                try:
                    res = pick_models(
                        task,
                        priority=str(body.get("priority") or "auto"),
                        max_price=_opt_float(body.get("max_price")),
                        min_speed=_opt_float(body.get("min_speed")),
                        max_ttft=_opt_float(body.get("max_ttft")),
                        min_context=_opt_int(body.get("min_context")),
                        show_all=bool(body.get("show_all", False)),
                        refresh=bool(body.get("refresh", False)),
                        db_path=db_path,
                    )
                    res["telegram_text"] = format_telegram_text(res)
                    self._send_json(200, res)
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})
                return

            self._send_json(404, {"error": "Not found"})

    return PickerHandler


def create_server(host: str = "127.0.0.1", port: int = 8742, db_path: Path | str | None = None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(db_path))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Локальный веб-сервер подбора ИИ-моделей")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8742)
    p.add_argument("--db", default=None)
    args = p.parse_args(argv)

    srv = create_server(host=args.host, port=args.port, db_path=args.db)
    print(f"aa_picker web listening on http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
