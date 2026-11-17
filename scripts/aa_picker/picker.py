"""Unified model selection engine for CLI and Web UI."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .db import get_connection, get_db_age_hours, get_meta
from .decision import BENCHMARK_LABELS, analyze_task
from .fetch_and_store import ensure_fresh_db

PARK_ENGINE_LABELS = {
    "claude": "Claude Code",
    "codex": "Codex",
    "antigravity": "Antigravity",
    "openrouter": "OpenRouter",
}

OPENROUTER_CREATORS = {
    "Anthropic",
    "OpenAI",
    "Google",
    "DeepSeek",
    "Meta",
    "Mistral",
    "Alibaba",
    "Qwen",
    "xAI",
    "Zhipu AI",
    "Z AI",
    "Z.ai",
    "Moonshot AI",
    "Kimi",
    "MiniMax",
    "Cohere",
    "Perplexity",
    "Amazon",
    "NVIDIA",
    "Xiaomi",
    "StepFun",
    "ByteDance",
    "Baidu",
    "01.AI",
    "Microsoft",
}


def normalize_effort(effort_suffix: str | None, is_reasoning: int) -> str:
    s = (effort_suffix or "").strip().lower()
    if "xhigh" in s:
        return "xhigh"
    if "max" in s:
        return "max"
    if "high" in s:
        return "high"
    if "medium" in s:
        return "medium"
    if "low" in s:
        return "low"
    if "non-reasoning" in s:
        return "low"
    if "reasoning" in s or is_reasoning:
        return "high"
    return "medium"


def format_effort_by_engine(effort: str, engines: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for eng in engines:
        if eng == "claude":
            mapping["claude"] = "high" if effort == "xhigh" else effort
        elif eng == "codex":
            mapping["codex"] = "xhigh" if effort == "max" else effort
        elif eng == "antigravity":
            mapping["antigravity"] = "high" if effort in ("xhigh", "max") else effort
        elif eng == "openrouter":
            mapping["openrouter"] = "high" if effort == "max" else effort
    if not mapping:
        mapping["default"] = effort
    return mapping


def detect_park_engines(
    creator_name: str | None,
    model_name: str,
    is_open_weights: int,
    endpoint_hosts: list[str],
) -> list[str]:
    creator = (creator_name or "").strip()
    mname = (model_name or "").lower()
    hosts = {h.lower() for h in endpoint_hosts if h}
    engines: list[str] = []

    if creator == "Anthropic" or "claude" in mname or "anthropic" in hosts:
        engines.append("claude")
    if creator == "OpenAI" or mname.startswith(("gpt-", "o1", "o3", "o4", "codex")) or "openai" in hosts:
        engines.append("codex")
    if (creator == "Google" and "gemini" in mname) or ("google" in hosts and "gemini" in mname):
        engines.append("antigravity")

    if hosts and (is_open_weights or creator in OPENROUTER_CREATORS or len(engines) > 0):
        engines.append("openrouter")

    return engines


def _pick_best_endpoint(
    endpoints: list[dict[str, Any]],
    priority: str,
    max_price: float | None,
    min_speed: float | None,
    max_ttft: float | None,
    min_context: int | None,
) -> dict[str, Any] | None:
    valid: list[dict[str, Any]] = []
    for ep in endpoints:
        if ep.get("deprecated"):
            continue
        pin = ep.get("price_1m_input")
        pout = ep.get("price_1m_output")
        spd = ep.get("output_speed")
        ttft = ep.get("time_to_first_token")
        ctx = ep.get("context_window_tokens")

        if pin is None or pin <= 0 or spd is None or spd <= 0:
            continue
        if max_price is not None and pin > max_price:
            continue
        if min_speed is not None and spd < min_speed:
            continue
        if max_ttft is not None and (ttft is None or ttft > max_ttft):
            continue
        if min_context is not None and (ctx is None or ctx < min_context):
            continue
        valid.append(ep)

    if not valid:
        return None

    if priority == "speed":
        valid.sort(key=lambda e: (-float(e["output_speed"]), float(e["time_to_first_token"] or 999.0), float(e["price_1m_input"])))
    elif priority == "price":
        valid.sort(key=lambda e: (float(e["price_1m_input"]), float(e["price_1m_output"] or 999.0), -float(e["output_speed"])))
    else:
        official = {"anthropic", "openai", "google", "deepseek", "mistral", "xai", "alibaba_cloud"}
        valid.sort(
            key=lambda e: (
                0 if str(e.get("host_slug") or "").lower() in official else 1,
                -float(e["output_speed"]),
                float(e["price_1m_input"]),
            )
        )
    return valid[0]


def _compute_task_score(
    scores: dict[str, float],
    intelligence_index: float | None,
    weights: dict[str, float],
) -> tuple[float, list[dict[str, Any]]]:
    weighted_sum = 0.0
    covered_weight = 0.0
    used: list[dict[str, Any]] = []

    for bslug, w in weights.items():
        if bslug == "intelligence_index":
            if intelligence_index is not None:
                norm = max(0.0, min(1.0, float(intelligence_index) / 100.0))
                weighted_sum += norm * w
                covered_weight += w
                used.append({"slug": bslug, "label": BENCHMARK_LABELS.get(bslug, bslug), "score": round(norm * 100, 1), "weight": w})
        elif bslug in scores:
            val = max(0.0, min(1.0, float(scores[bslug])))
            weighted_sum += val * w
            covered_weight += w
            used.append({"slug": bslug, "label": BENCHMARK_LABELS.get(bslug, bslug), "score": round(val * 100, 1), "weight": w})

    if covered_weight > 0:
        base_score = weighted_sum / covered_weight
        # Small penalty if model only covers a tiny fraction of requested benchmarks
        coverage_factor = 0.85 + 0.15 * min(1.0, covered_weight)
        final_score = base_score * coverage_factor
    elif intelligence_index is not None:
        final_score = max(0.0, min(1.0, float(intelligence_index) / 100.0)) * 0.75
    else:
        final_score = 0.0

    return round(final_score * 100.0, 2), used


def pick_models(
    task: str,
    *,
    priority: str = "auto",
    max_price: float | None = None,
    min_speed: float | None = None,
    max_ttft: float | None = None,
    min_context: int | None = None,
    show_all: bool = False,
    refresh: bool = False,
    db_path: Path | str | None = None,
    top_n: int = 3,
) -> dict[str, Any]:
    fresh_info = ensure_fresh_db(db_path=db_path, max_age_hours=24.0, force=refresh)
    decision_info = analyze_task(task, priority_override=priority)
    eff_priority = decision_info["priority"]
    weights = decision_info["benchmarks"]

    conn = get_connection(db_path)
    try:
        age_hours = get_db_age_hours(conn)
        last_updated = get_meta(conn, "last_updated_utc")
        stale = age_hours is None or age_hours > 24.0

        models_rows = conn.execute("SELECT * FROM models WHERE deprecated = 0").fetchall()
        score_rows = conn.execute("SELECT model_id, benchmark_slug, score FROM model_scores").fetchall()
        ep_rows = conn.execute("SELECT * FROM provider_endpoints WHERE deprecated = 0").fetchall()
    finally:
        conn.close()

    scores_by_model: dict[str, dict[str, float]] = {}
    for sr in score_rows:
        scores_by_model.setdefault(sr["model_id"], {})[sr["benchmark_slug"]] = float(sr["score"])

    eps_by_release: dict[str, list[dict[str, Any]]] = {}
    eps_by_model: dict[str, list[dict[str, Any]]] = {}
    for er in ep_rows:
        ed = dict(er)
        if ed.get("release_slug"):
            eps_by_release.setdefault(ed["release_slug"], []).append(ed)
        if ed.get("matched_model_id"):
            eps_by_model.setdefault(ed["matched_model_id"], []).append(ed)

    candidates: list[dict[str, Any]] = []
    for mr in models_rows:
        m = dict(mr)
        rel_slug = m.get("release_slug")
        eps = eps_by_release.get(rel_slug, []) if rel_slug else eps_by_model.get(m["id"], [])
        all_hosts = [str(e.get("host_slug") or "") for e in eps]

        best_ep = _pick_best_endpoint(eps, eff_priority, max_price, min_speed, max_ttft, min_context)
        if best_ep:
            pin = float(best_ep["price_1m_input"])
            pout = float(best_ep["price_1m_output"] or 0.0)
            spd = float(best_ep["output_speed"])
            ttft = float(best_ep["time_to_first_token"]) if best_ep.get("time_to_first_token") is not None else None
            ctx = best_ep.get("context_window_tokens") or m.get("context_window_tokens")
            provider_name = str(best_ep.get("host_name") or best_ep.get("host_slug") or "Unknown")
            provider_slug = str(best_ep.get("host_slug") or "")
        else:
            # Fallback to model-level aggregated metrics if no endpoint row matched
            pin = m.get("price_1m_input")
            pout = m.get("price_1m_output")
            spd = m.get("output_speed")
            ttft = m.get("time_to_first_token")
            ctx = m.get("context_window_tokens")
            if pin is None or pin <= 0 or spd is None or spd <= 0:
                continue
            if max_price is not None and pin > max_price:
                continue
            if min_speed is not None and spd < min_speed:
                continue
            if max_ttft is not None and (ttft is None or ttft > max_ttft):
                continue
            if min_context is not None and (ctx is None or ctx < min_context):
                continue
            provider_name = str(m.get("creator_name") or "Direct")
            provider_slug = provider_name.lower()

        park_engines = detect_park_engines(
            m.get("creator_name"),
            m["name"],
            int(m.get("is_open_weights") or 0),
            all_hosts,
        )
        in_park = len(park_engines) > 0
        effort = normalize_effort(m.get("effort_suffix"), int(m.get("is_reasoning") or 0))
        effort_map = format_effort_by_engine(effort, park_engines)

        m_scores = scores_by_model.get(m["id"], {})
        task_score, used_benchmarks = _compute_task_score(m_scores, m.get("intelligence_index"), weights)
        if task_score <= 0:
            continue

        candidates.append(
            {
                "id": m["id"],
                "release_slug": rel_slug or m["id"],
                "name": m["name"],
                "short_name": m.get("short_name") or m["name"],
                "creator": m.get("creator_name") or "",
                "provider": provider_name,
                "provider_slug": provider_slug,
                "in_park": in_park,
                "park_engines": park_engines,
                "effort": effort,
                "effort_by_engine": effort_map,
                "task_score": task_score,
                "intelligence_index": m.get("intelligence_index"),
                "price_1m_input": round(float(pin), 4),
                "price_1m_output": round(float(pout or 0.0), 4),
                "output_speed": round(float(spd), 2),
                "time_to_first_token": round(float(ttft), 3) if ttft is not None else None,
                "context_window_tokens": int(ctx) if ctx is not None else None,
                "used_benchmarks": used_benchmarks,
            }
        )

    # Filter or prioritize park models
    park_candidates = [c for c in candidates if c["in_park"]]
    pool = candidates if (show_all or not park_candidates) else park_candidates

    # For price/speed priority, keep models above a sensible quality floor so weak models don't dominate
    if eff_priority in ("price", "speed") and pool:
        max_q = max(c["task_score"] for c in pool)
        quality_floor = max(35.0, max_q * 0.55) if eff_priority == "price" else max(25.0, max_q * 0.40)
        filtered_q = [c for c in pool if c["task_score"] >= quality_floor]
        if filtered_q:
            pool = filtered_q

    def sort_key(c: dict[str, Any]) -> tuple[Any, ...]:
        park_rank = 0 if (not show_all and c["in_park"]) else 1
        if eff_priority == "price":
            return (park_rank, c["price_1m_input"], c["price_1m_output"], -c["task_score"])
        if eff_priority == "speed":
            ttft_val = c["time_to_first_token"] if c["time_to_first_token"] is not None else 999.0
            return (park_rank, -c["output_speed"], ttft_val, -c["task_score"])
        return (park_rank, -c["task_score"], c["price_1m_input"], -c["output_speed"])

    pool.sort(key=sort_key)

    # Deduplicate by release_slug so top-3 shows 3 distinct models
    top_items: list[dict[str, Any]] = []
    seen_releases: set[str] = set()
    for c in pool:
        rslug = c["release_slug"]
        if rslug in seen_releases:
            continue
        seen_releases.add(rslug)
        top_items.append(c)
        if len(top_items) >= top_n:
            break

    top_bench_str = ", ".join(
        f"{BENCHMARK_LABELS.get(b, b)} ({int(round(w * 100))}%)"
        for b, w in list(weights.items())[:4]
    )
    for item in top_items:
        eng_str = ", ".join(PARK_ENGINE_LABELS.get(e, e) for e in item["park_engines"]) if item["park_engines"] else "вне парка"
        eff_str = ", ".join(f"{k}: {v}" for k, v in item["effort_by_engine"].items())
        item["explanation"] = (
            f"Оценка под задачу {item['task_score']:.1f}/100 по [{top_bench_str}]; "
            f"приоритет: {eff_priority}; доступность: {eng_str}; effort ({eff_str})."
        )

    return {
        "task": task,
        "priority": eff_priority,
        "show_all": show_all,
        "filters": {
            "max_price": max_price,
            "min_speed": min_speed,
            "max_ttft": max_ttft,
            "min_context": min_context,
        },
        "decision": {
            "source": decision_info["source"],
            "priority": eff_priority,
            "tags": decision_info["tags"],
            "benchmarks": decision_info["benchmarks"],
            "benchmark_labels": decision_info["benchmark_labels"],
        },
        "db": {
            "path": fresh_info.get("db_path"),
            "models_count": len(models_rows),
            "endpoints_count": len(ep_rows),
            "last_updated_utc": last_updated,
            "age_hours": age_hours,
            "stale": stale,
            "refreshed_now": bool(fresh_info.get("refreshed")),
            "refresh_error": fresh_info.get("error"),
        },
        "total_matching_candidates": len(pool),
        "top3": top_items,
    }


def format_telegram_text(result: dict[str, Any]) -> str:
    lines: list[str] = []
    db_info = result["db"]
    if db_info.get("stale"):
        age = db_info.get("age_hours")
        age_txt = f"{age:.1f} ч" if age is not None else "неизвестен"
        lines.append(f"Внимание: база моделей устарела (возраст: {age_txt} > 24 ч).")

    dec = result["decision"]
    top_b = ", ".join(
        f"{dec['benchmark_labels'].get(b, b)} ({int(round(w * 100))}%)"
        for b, w in list(dec["benchmarks"].items())[:3]
    )
    lines.append(f"Подбор под задачу (приоритет: {result['priority']}, Decision: {dec['source']})")
    lines.append(f"Бенчмарки: {top_b}")

    top3 = result.get("top3") or []
    if not top3:
        lines.append("Подходящие модели с ненулевой ценой и скоростью под заданные фильтры не найдены.")
        return "\n".join(lines)

    for idx, m in enumerate(top3, 1):
        park_tag = ", ".join(m["park_engines"]) if m["park_engines"] else "вне парка"
        ttft_s = f"{m['time_to_first_token']:.2f}с" if m["time_to_first_token"] is not None else "н/д"
        ctx_k = f"{m['context_window_tokens'] // 1000}k" if m["context_window_tokens"] else "н/д"
        lines.append(
            f"{idx}. {m['name']} | {m['provider']} [{park_tag}] | effort: {m['effort']} | "
            f"${m['price_1m_input']:.2f}/${m['price_1m_output']:.2f} за 1M | "
            f"{m['output_speed']:.1f} tok/s, TTFT {ttft_s}, ctx {ctx_k} | "
            f"балл {m['task_score']:.1f}/100"
        )
    return "\n".join(lines)
