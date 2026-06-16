from __future__ import annotations

import json
from typing import Any

from .rules_skill import RuleSkill


def build_floor_stack_system_prompt(rule_skill: RuleSkill) -> str:
    return (
        "You are an urban design mixed-use floor allocation planner. Return strict JSON only. "
        "Follow zoning-first, hard-conflict-first, and floor-slice-by-height logic. "
        "Do not reallocate excluded infrastructure/open-space layers. "
        "Each output block must include dominant, dominantCode, and detailed segments. "
        "Use the following compact rule skill as hard guidance: "
        + rule_skill.render_compact_instruction()
    )


def build_floor_stack_user_payload(
    prompt: str,
    blocks_fc: dict[str, Any],
    parcels_fc: dict[str, Any],
    audience_profile: str,
    rule_skill: RuleSkill,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "audience_profile": audience_profile,
        "blocks": blocks_fc,
        "parcels": parcels_fc,
        "constraints": {
            "exclude_from_allocation": rule_skill.exclude_from_allocation,
            "max_outputs": 3000,
        },
        "output_schema_hint": {
            "summary": "string",
            "outputs": [
                {
                    "id": "string",
                    "zone": "CBD|LEISURE|RESIDENTIAL|CREATIVE|UNKNOWN",
                    "height": "number",
                    "dominant": "string",
                    "dominantCode": "OFFICE|RESIDENTIAL|CENTER_COMMERCIAL|LEISURE_COMMERCIAL|PUBLIC|GREEN",
                    "segments": [
                        {
                            "segment": "string",
                            "primary": "string",
                            "primaryCode": "OFFICE|RESIDENTIAL|CENTER_COMMERCIAL|LEISURE_COMMERCIAL|PUBLIC|GREEN",
                            "secondary": "string|null",
                            "score": "number",
                            "reason": "string",
                        }
                    ],
                }
            ],
            "zoneInsights": [
                {
                    "zone": "CBD|LEISURE|RESIDENTIAL|CREATIVE|UNKNOWN",
                    "headline": "string",
                    "narrativeName": "string",
                    "ratios": [{"key": "OFFICE", "ratio": 0.5}],
                }
            ],
        },
    }


def build_floor_stack_messages(
    prompt: str,
    blocks_fc: dict[str, Any],
    parcels_fc: dict[str, Any],
    audience_profile: str,
    rule_skill: RuleSkill,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_floor_stack_system_prompt(rule_skill)},
        {
            "role": "user",
            "content": json.dumps(
                build_floor_stack_user_payload(prompt, blocks_fc, parcels_fc, audience_profile, rule_skill),
                ensure_ascii=False,
            ),
        },
    ]


def build_audience_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Return strict JSON only. Infer audience groups for urban design from user vision.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "prompt": prompt,
                    "allowed": ["商务办公人群", "年轻消费人群", "创意产业人群", "家庭居住人群", "游客与外来访客"],
                    "output_schema": {"audiences": ["商务办公人群"], "reason": "string"},
                },
                ensure_ascii=False,
            ),
        },
    ]
