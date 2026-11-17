"""Fetch Artificial Analysis leaderboards (models + providers) and store in SQLite."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from .db import get_connection, get_db_age_hours, init_db, resolve_db_path, set_meta, utc_now_iso

MODELS_URL = "https://artificialanalysis.ai/leaderboards/models"
PROVIDERS_URL = "https://artificialanalysis.ai/leaderboards/providers"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)


def fetch_rsc(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    pushes = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', html, re.S)
    chunks: list[str] = []
    for p in pushes:
        try:
            chunks.append(json.loads('"' + p + '"'))
        except Exception:
            pass
    return "".join(chunks)


def extract_json_field(rsc: str, key: str) -> Any:
    needle = f'"{key}":'
    idx = rsc.find(needle)
    if idx == -1:
        raise ValueError(f"Key {key!r} not found in RSC payload")
    dec = json.JSONDecoder()
    val, _ = dec.raw_decode(rsc[idx + len(needle) :])
    return val


def _to_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _to_int(val: Any) -> int | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _to_bool_int(val: Any) -> int:
    return 1 if bool(val) else 0


def parse_evaluations(rsc: str) -> list[dict[str, Any]]:
    try:
        ev = extract_json_field(rsc, "evaluations")
    except Exception:
        return []
    if not isinstance(ev, dict):
        return []
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for k, v in ev.items():
        if isinstance(v, dict) and v.get("slug"):
            slug = str(v["slug"])
            if slug not in seen:
                seen.add(slug)
                kw = v.get("keywords")
                items.append(
                    {
                        "slug": slug,
                        "name": str(v.get("name") or slug),
                        "keywords": kw if isinstance(kw, list) else [],
                        "featured": 1,
                    }
                )
        elif isinstance(v, list):
            for entry in v:
                if isinstance(entry, dict) and entry.get("slug"):
                    slug = str(entry["slug"])
                    if slug not in seen:
                        seen.add(slug)
                        kw = entry.get("keywords")
                        items.append(
                            {
                                "slug": slug,
                                "name": str(entry.get("name") or slug),
                                "keywords": kw if isinstance(kw, list) else [],
                                "featured": 1 if k == "generalEvaluations" else 0,
                            }
                        )
    return items


def fetch_and_store(db_path: Path | str | None = None, timeout: int = 30) -> dict[str, Any]:
    path = init_db(db_path)
    rsc_models = fetch_rsc(MODELS_URL, timeout=timeout)
    rsc_providers = fetch_rsc(PROVIDERS_URL, timeout=timeout)

    models = extract_json_field(rsc_models, "initialModels")
    rows = extract_json_field(rsc_providers, "initialRows")
    evals = parse_evaluations(rsc_models)

    now = utc_now_iso()
    conn = get_connection(path)
    try:
        conn.execute("DELETE FROM model_scores")
        conn.execute("DELETE FROM endpoint_scores")
        conn.execute("DELETE FROM provider_endpoints")
        conn.execute("DELETE FROM models")
        conn.execute("DELETE FROM benchmarks")

        for b in evals:
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmarks
                (slug, name, summary, skills_json, knowledge_json, attributes_json, featured, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    b["slug"],
                    b["name"],
                    "",
                    json.dumps(b["keywords"], ensure_ascii=False),
                    "[]",
                    "{}",
                    b["featured"],
                    now,
                ),
            )

        release_to_model_ids: dict[str, list[str]] = {}
        scores_count = 0
        for m in models:
            mid = str(m["id"])
            rel = m.get("release") or {}
            rel_slug = rel.get("slug") if isinstance(rel, dict) else None
            rel_name = rel.get("name") if isinstance(rel, dict) else None
            if rel_slug:
                release_to_model_ids.setdefault(str(rel_slug), []).append(mid)

            conn.execute(
                """
                INSERT OR REPLACE INTO models
                (id, release_slug, release_name, name, short_name, effort_suffix,
                 deprecated, is_reasoning, is_open_weights, param_class, price_class,
                 creator_name, context_window_tokens, intelligence_index, omniscience,
                 price_1m_input, price_1m_output, cache_hit_price, cache_write_price,
                 output_speed, time_to_first_token, time_to_first_answer_token, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
                """,
                (
                    mid,
                    rel_slug,
                    rel_name,
                    str(m.get("name") or mid),
                    m.get("shortName"),
                    m.get("suffix"),
                    _to_bool_int(m.get("deprecated")),
                    _to_bool_int(m.get("isReasoning")),
                    _to_bool_int(m.get("isOpenWeights")),
                    m.get("paramClass"),
                    m.get("priceClass"),
                    m.get("modelCreatorName"),
                    _to_int(m.get("contextWindowTokens")),
                    _to_float(m.get("intelligenceIndex")),
                    _to_float(m.get("omniscience")),
                    _to_float(m.get("price1mInputTokens")),
                    _to_float(m.get("price1mOutputTokens")),
                    _to_float(m.get("cacheHitPrice")),
                    _to_float(m.get("cacheWritePrice")),
                    now,
                ),
            )
            scores = m.get("scores")
            if isinstance(scores, dict):
                for bslug, sval in scores.items():
                    fval = _to_float(sval)
                    if fval is not None:
                        conn.execute(
                            "INSERT OR REPLACE INTO model_scores (model_id, benchmark_slug, score) VALUES (?, ?, ?)",
                            (mid, str(bslug), fval),
                        )
                        scores_count += 1

        # Insert provider endpoints and aggregate speed/price back to models if missing
        rel_best_perf: dict[str, dict[str, Any]] = {}
        ep_count = 0
        for r in rows:
            eid = str(r["id"])
            host = r.get("host") or {}
            model_obj = r.get("model") or {}
            rel = model_obj.get("release") or {}
            rel_slug = rel.get("slug") if isinstance(rel, dict) else None
            creator = model_obj.get("creator") or {}
            features = r.get("features") or {}
            pricing = r.get("pricing") or {}
            perf = r.get("performance") or {}

            matched_ids = release_to_model_ids.get(str(rel_slug), []) if rel_slug else []
            matched_id = matched_ids[0] if matched_ids else None

            pin = _to_float(pricing.get("price1mInputTokens"))
            pout = _to_float(pricing.get("price1mOutputTokens"))
            spd = _to_float(perf.get("outputSpeed"))
            ttft = _to_float(perf.get("timeToFirstToken"))
            ttfat = _to_float(perf.get("timeToFirstAnswerToken"))

            conn.execute(
                """
                INSERT OR REPLACE INTO provider_endpoints
                (id, label, host_api_id, host_name, host_slug, release_slug, matched_model_id,
                 is_open_weights, deprecated, reasoning_model, size_class, creator_name,
                 context_window_tokens, function_calling, json_mode, openai_compatible,
                 price_class, price_1m_input, price_1m_output, cache_hit_price, cache_write_price,
                 output_speed, time_to_first_token, time_to_first_answer_token, endpoint_error_rate,
                 intelligence_index, omniscience, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eid,
                    str(r.get("label") or eid),
                    r.get("hostApiId"),
                    str(host.get("name") or ""),
                    str(host.get("slug") or ""),
                    rel_slug,
                    matched_id,
                    _to_bool_int(model_obj.get("isOpenWeights")),
                    _to_bool_int(model_obj.get("deprecated")),
                    _to_bool_int(model_obj.get("reasoningModel")),
                    model_obj.get("sizeClass"),
                    creator.get("name") if isinstance(creator, dict) else None,
                    _to_int(features.get("contextWindowTokens")),
                    _to_bool_int(features.get("functionCalling")),
                    _to_bool_int(features.get("jsonMode")),
                    _to_bool_int(features.get("openaiCompatible")),
                    pricing.get("priceClass"),
                    pin,
                    pout,
                    _to_float(pricing.get("cacheHitPrice")),
                    _to_float(pricing.get("cacheWritePrice")),
                    spd,
                    ttft,
                    ttfat,
                    _to_float(perf.get("endpointErrorRate")),
                    _to_float(model_obj.get("intelligenceIndex")),
                    _to_float(model_obj.get("omniscience")),
                    now,
                ),
            )
            ep_count += 1

            ep_scores = model_obj.get("scores")
            if isinstance(ep_scores, dict):
                for bslug, sval in ep_scores.items():
                    fval = _to_float(sval)
                    if fval is not None:
                        conn.execute(
                            "INSERT OR REPLACE INTO endpoint_scores (endpoint_id, benchmark_slug, score) VALUES (?, ?, ?)",
                            (eid, str(bslug), fval),
                        )

            if rel_slug and not model_obj.get("deprecated"):
                cur = rel_best_perf.setdefault(str(rel_slug), {"speeds": [], "ttfts": [], "pins": [], "pouts": []})
                if spd is not None and spd > 0:
                    cur["speeds"].append(spd)
                if ttft is not None and ttft > 0:
                    cur["ttfts"].append(ttft)
                if pin is not None and pin >= 0:
                    cur["pins"].append(pin)
                if pout is not None and pout >= 0:
                    cur["pouts"].append(pout)

        # Backfill models with median/best endpoint speed & price where missing
        for rel_slug, agg in rel_best_perf.items():
            spd = max(agg["speeds"]) if agg["speeds"] else None
            ttft = min(agg["ttfts"]) if agg["ttfts"] else None
            pin = min(agg["pins"]) if agg["pins"] else None
            pout = min(agg["pouts"]) if agg["pouts"] else None
            conn.execute(
                """
                UPDATE models
                SET output_speed = COALESCE(output_speed, ?),
                    time_to_first_token = COALESCE(time_to_first_token, ?),
                    price_1m_input = COALESCE(price_1m_input, ?),
                    price_1m_output = COALESCE(price_1m_output, ?)
                WHERE release_slug = ?
                """,
                (spd, ttft, pin, pout, rel_slug),
            )

        set_meta(conn, "last_updated_utc", now)
        set_meta(conn, "models_count", str(len(models)))
        set_meta(conn, "endpoints_count", str(ep_count))
        conn.commit()
    finally:
        conn.close()

    return {
        "refreshed": True,
        "db_path": str(path),
        "models_count": len(models),
        "endpoints_count": ep_count,
        "benchmarks_count": len(evals),
        "scores_count": scores_count,
        "updated_at": now,
        "age_hours": 0.0,
        "stale": False,
    }


def ensure_fresh_db(
    db_path: Path | str | None = None,
    max_age_hours: float = 24.0,
    force: bool = False,
) -> dict[str, Any]:
    path = resolve_db_path(db_path)
    init_db(path)
    conn = get_connection(path)
    try:
        cnt_row = conn.execute("SELECT COUNT(*) AS c FROM models").fetchone()
        cnt = int(cnt_row["c"]) if cnt_row else 0
        age = get_db_age_hours(conn)
    finally:
        conn.close()

    needs_refresh = force or cnt == 0 or age is None or age > max_age_hours
    if not needs_refresh:
        return {
            "refreshed": False,
            "db_path": str(path),
            "models_count": cnt,
            "age_hours": age,
            "stale": False,
        }

    try:
        return fetch_and_store(path)
    except Exception as exc:
        if cnt > 0:
            return {
                "refreshed": False,
                "db_path": str(path),
                "models_count": cnt,
                "age_hours": age,
                "stale": True if (age is None or age > max_age_hours) else False,
                "error": str(exc),
            }
        raise
