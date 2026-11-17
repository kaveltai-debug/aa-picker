"""Gemini Decision module for selecting relevant AA benchmarks and priority."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

GEMINI_DECISION_ENV_PATH = Path(r"C:\Users\user\.claude\.secrets\gemini_api.env")
GEMINI_DECISION_URL = "https://generativelanguage.googleapis.com/v1beta/decisions"
GEMINI_DECISION_MODEL = "gemini-decision-1"
GEMINI_DECISION_TIMEOUT_SEC = 6.0

_DECISION_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_COOLDOWN_UNTIL = 0.0

BENCHMARK_GROUPS: dict[str, dict[str, float]] = {
    "coding": {
        "frontiercode-2-0": 0.25,
        "frontiercode": 0.20,
        "livecodebench": 0.25,
        "terminalbench-hard": 0.15,
        "terminalbench-2-1": 0.15,
    },
    "reasoning": {
        "gpqa-diamond": 0.35,
        "humanitys-last-exam": 0.30,
        "aime-2025": 0.20,
        "mmmu-pro": 0.15,
    },
    "long_context": {
        "artificial-analysis-long-context-reasoning": 0.70,
        "ifbench": 0.15,
        "gpqa-diamond": 0.15,
    },
    "agentic": {
        "tau2-bench": 0.35,
        "ifbench": 0.30,
        "terminalbench-hard": 0.20,
        "aa-analyst-agent": 0.15,
    },
    "accuracy": {
        "omniscience-accuracy": 0.45,
        "omniscience-non-hallucination": 0.35,
        "ifbench": 0.20,
    },
}

BENCHMARK_LABELS: dict[str, str] = {
    "frontiercode-2-0": "FrontierCode 2.0",
    "frontiercode": "FrontierCode",
    "livecodebench": "LiveCodeBench",
    "terminalbench-hard": "TerminalBench Hard",
    "terminalbench-2-1": "TerminalBench 2.1",
    "gpqa-diamond": "GPQA Diamond",
    "humanitys-last-exam": "Humanity's Last Exam",
    "aime-2025": "AIME 2025",
    "mmmu-pro": "MMMU-Pro",
    "artificial-analysis-long-context-reasoning": "AA Long Context Reasoning",
    "ifbench": "IFBench",
    "tau2-bench": "Tau2-Bench",
    "aa-analyst-agent": "AA Analyst Agent",
    "omniscience-accuracy": "Omniscience Accuracy",
    "omniscience-non-hallucination": "Omniscience Non-Hallucination",
    "intelligence_index": "AA Intelligence Index",
}


def load_gemini_api_key(env_path: Path | str | None = None) -> str | None:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"].strip()
    path = Path(env_path) if env_path else GEMINI_DECISION_ENV_PATH
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "GEMINI_API_KEY":
                    val = v.strip().strip('"').strip("'")
                    if val:
                        return val
    except OSError:
        return None
    return None


def _rule_based_tags_and_priority(task: str) -> tuple[str, list[str]]:
    t = task.lower()
    priority = "quality"
    if any(w in t for w in ["дёшев", "дешев", "бюджет", "эконом", "цена", "cheap", "budget", "cost"]):
        priority = "price"
    elif any(w in t for w in ["быстр", "скорост", "задерж", "ttft", "fast", "speed", "latency", "чат"]):
        priority = "speed"

    tags: list[str] = []
    if any(w in t for w in ["код", "python", "программ", "скрипт", "рефактор", "баг", "code", "dev", "sql", "eplan"]):
        tags.append("coding")
    if any(w in t for w in ["контекст", "книг", "том", "500к", "500k", "100к", "100k", "200к", "длинн", "long context"]):
        tags.append("long_context")
    if any(w in t for w in ["агент", "инструмент", "вызов", "парсинг", "автомат", "agent", "tool", "швейцар"]):
        tags.append("agentic")
    if any(w in t for w in ["анализ", "сложн", "качеств", "архитект", "матем", "расчёт", "расчет", "логик", "reason"]):
        tags.append("reasoning")
    if any(w in t for w in ["факт", "точн", "галлюцин", "документ", "правов", "договор"]):
        tags.append("accuracy")

    if not tags:
        tags = ["reasoning"] if priority == "quality" else ["agentic"]
    return priority, tags


def _build_weights(tags: list[str], tag_confidences: dict[str, float] | None = None) -> dict[str, float]:
    weights: dict[str, float] = {}
    confs = tag_confidences or {}
    for tag in tags:
        group = BENCHMARK_GROUPS.get(tag, {})
        mult = max(0.5, float(confs.get(tag, 1.0)))
        for bslug, w in group.items():
            weights[bslug] = weights.get(bslug, 0.0) + w * mult

    if not weights:
        weights = {"intelligence_index": 1.0}
    else:
        weights["intelligence_index"] = weights.get("intelligence_index", 0.0) + 0.15

    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 4) for k, v in sorted(weights.items(), key=lambda x: -x[1])}
    return weights


def _call_gemini_decision(task: str, api_key: str) -> dict[str, Any]:
    payload = {
        "model": GEMINI_DECISION_MODEL,
        "input": f"Описание задачи пользователя для подбора ИИ-модели:\n{task}",
        "questions": {
            "priority": {
                "type": "choice",
                "instructions": "Какой главный приоритет для этой задачи?",
                "criteria": {
                    "quality": "Максимальное качество, точность, глубина анализа (или приоритет не указан)",
                    "price": "Минимальная цена, экономия бюджета",
                    "speed": "Максимальная скорость ответа, минимальная задержка (TTFT)",
                },
            },
            "coding": {
                "type": "choice",
                "instructions": "Связана ли задача с программированием, кодом, отладкой или CLI?",
                "criteria": {"yes": "Да, это задача на код/разработку", "no": "Нет"},
            },
            "reasoning": {
                "type": "choice",
                "instructions": "Требует ли задача сложного рассуждения, архитектуры, математики или глубокого анализа?",
                "criteria": {"yes": "Да, нужны сильные рассуждения", "no": "Нет"},
            },
            "long_context": {
                "type": "choice",
                "instructions": "Требует ли задача работы с большим объёмом текста / длинным контекстом?",
                "criteria": {"yes": "Да, длинный контекст или большой документ", "no": "Нет"},
            },
            "agentic": {
                "type": "choice",
                "instructions": "Предполагает ли задача агентное поведение, вызов инструментов или строгое следование инструкциям?",
                "criteria": {"yes": "Да, агентная задача или строгие инструкции", "no": "Нет"},
            },
            "accuracy": {
                "type": "choice",
                "instructions": "Критична ли фактологическая точность и отсутствие галлюцинаций по документам?",
                "criteria": {"yes": "Да, критична точность без выдумывания", "no": "Нет"},
            },
        },
    }
    req = urllib.request.Request(
        GEMINI_DECISION_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=GEMINI_DECISION_TIMEOUT_SEC) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("answers") or {}


def analyze_task(task: str, priority_override: str = "auto") -> dict[str, Any]:
    global _COOLDOWN_UNTIL
    task_clean = (task or "").strip()
    cache_key = (task_clean.lower(), priority_override)
    if cache_key in _DECISION_CACHE:
        return dict(_DECISION_CACHE[cache_key])

    rule_priority, rule_tags = _rule_based_tags_and_priority(task_clean)
    api_key = load_gemini_api_key()
    now = time.time()

    source = "fallback_rules"
    chosen_priority = rule_priority
    tags: list[str] = list(rule_tags)
    tag_confidences: dict[str, float] = {}
    raw_answers: dict[str, Any] = {}

    if not api_key:
        source = "fallback_no_key"
    elif now < _COOLDOWN_UNTIL:
        source = "fallback_cooldown"
    else:
        try:
            answers = _call_gemini_decision(task_clean, api_key)
            raw_answers = answers
            p_ans = (answers.get("priority") or {}).get("choice")
            if p_ans in ("quality", "price", "speed"):
                chosen_priority = p_ans
            g_tags: list[str] = []
            for tag in ("coding", "reasoning", "long_context", "agentic", "accuracy"):
                a = answers.get(tag) or {}
                if a.get("choice") == "yes":
                    g_tags.append(tag)
                    conf = a.get("confidence")
                    if isinstance(conf, (int, float)):
                        tag_confidences[tag] = float(conf)
            if g_tags:
                tags = g_tags
            # Combine with explicit keywords from task if any strong hint was missed
            for rt in rule_tags:
                if rt not in tags and rt in ("coding", "long_context"):
                    tags.append(rt)
            source = "gemini-decision-1"
        except urllib.error.HTTPError as he:
            _COOLDOWN_UNTIL = now + (60.0 if he.code == 429 else 20.0)
            source = f"fallback_http_{he.code}"
        except Exception:
            _COOLDOWN_UNTIL = now + 20.0
            source = "fallback_error"

    if priority_override in ("quality", "price", "speed"):
        chosen_priority = priority_override

    weights = _build_weights(tags, tag_confidences)
    result = {
        "task": task_clean,
        "priority": chosen_priority,
        "tags": tags,
        "benchmarks": weights,
        "benchmark_labels": {k: BENCHMARK_LABELS.get(k, k) for k in weights},
        "source": source,
        "raw_answers": raw_answers,
    }
    _DECISION_CACHE[cache_key] = result
    return dict(result)
