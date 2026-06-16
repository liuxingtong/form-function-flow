from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request

from .prompt_templates import build_audience_messages, build_floor_stack_messages
from .rules_skill import load_rule_skill


@dataclass
class AgentConfig:
    api_key: str
    model: str
    base_url: str
    timeout_sec: float


def load_agent_config() -> AgentConfig:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("SITE_AI_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("SITE_AI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    timeout_sec = float(os.getenv("SITE_AI_TIMEOUT_SEC", "45"))
    return AgentConfig(api_key=api_key, model=model, base_url=base_url, timeout_sec=timeout_sec)


def _extract_json_content(obj: dict[str, Any]) -> str:
    choices = obj.get("choices", [])
    if not choices:
        raise RuntimeError("AI response missing choices")
    msg = choices[0].get("message", {})
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("AI response content is empty")
    return content


def _validate_agent_output(out: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(out, dict):
        raise RuntimeError("AI output must be JSON object")
    if not isinstance(out.get("outputs"), list):
        raise RuntimeError("AI output missing outputs[]")
    out.setdefault("summary", "AI floor stack planning finished.")
    if not isinstance(out.get("zoneInsights"), list):
        out["zoneInsights"] = []
    return out


def infer_floor_stack(prompt: str, blocks_fc: dict[str, Any], parcels_fc: dict[str, Any], audience_profile: str = "") -> dict[str, Any]:
    cfg = load_agent_config()
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured in .env")

    skill = load_rule_skill()
    body = {
        "model": cfg.model,
        "messages": build_floor_stack_messages(prompt, blocks_fc, parcels_fc, audience_profile, skill),
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    req = request.Request(
        url=f"{cfg.base_url}/chat/completions",
        method="POST",
        headers={
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
    )
    with request.urlopen(req, timeout=cfg.timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    api_obj = json.loads(raw)
    content = _extract_json_content(api_obj)
    out = json.loads(content)
    return _validate_agent_output(out)


def fallback_audience_from_prompt(prompt: str) -> dict[str, Any]:
    t = str(prompt or "")
    tags: list[str] = []
    if any(k in t for k in ["商务", "办公", "金融", "总部", "效率"]):
        tags.append("商务办公人群")
    if any(k in t for k in ["年轻", "潮流", "夜间", "消费", "餐饮"]):
        tags.append("年轻消费人群")
    if any(k in t for k in ["创意", "文创", "艺术", "设计", "展览"]):
        tags.append("创意产业人群")
    if any(k in t for k in ["居住", "宜居", "家庭", "社区", "安静"]):
        tags.append("家庭居住人群")
    if any(k in t for k in ["旅游", "地标", "游客", "门户"]):
        tags.append("游客与外来访客")
    if not tags:
        tags = ["商务办公人群", "家庭居住人群"]
    return {
        "audiences": tags[:3],
        "source": "fallback_template",
        "note": "agent failed, fallback template applied",
    }


def infer_audience_profile(prompt: str) -> dict[str, Any]:
    cfg = load_agent_config()
    if not cfg.api_key:
        return fallback_audience_from_prompt(prompt)

    body = {
        "model": cfg.model,
        "messages": build_audience_messages(prompt),
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    try:
        req = request.Request(
            url=f"{cfg.base_url}/chat/completions",
            method="POST",
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        )
        with request.urlopen(req, timeout=cfg.timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
        api_obj = json.loads(raw)
        content = _extract_json_content(api_obj)
        out = json.loads(content)
        audiences = out.get("audiences", [])
        if not isinstance(audiences, list) or not audiences:
            return fallback_audience_from_prompt(prompt)
        return {"audiences": [str(x) for x in audiences[:3]], "source": "agent", "note": str(out.get("reason", ""))}
    except Exception:
        return fallback_audience_from_prompt(prompt)
