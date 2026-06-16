from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RuleSkill:
    data: dict[str, Any]

    @property
    def exclude_from_allocation(self) -> list[str]:
        return [str(x).upper() for x in self.data.get("exclude_from_allocation", [])]

    def render_compact_instruction(self) -> str:
        zones = self.data.get("zones", {})
        zone_lines = []
        for k, v in zones.items():
            role = v.get("role", "")
            preferred = ",".join(v.get("preferred", []))
            zone_lines.append(f"{k}: role={role}; preferred={preferred}")

        floor_rules = self.data.get("floor_rules", {})
        floor_lines = [f"{k}=>{','.join(v)}" for k, v in floor_rules.items()]

        hard_conflicts = self.data.get("hard_conflicts", [])
        scoring = self.data.get("scoring", {})

        return (
            "RuleSkill(site_function_allocation_rules): "
            f"principles={','.join(self.data.get('principles', []))}; "
            f"exclude={','.join(self.exclude_from_allocation)}; "
            f"zones=[{' | '.join(zone_lines)}]; "
            f"floors=[{' | '.join(floor_lines)}]; "
            f"hard_conflicts=[{' | '.join(hard_conflicts)}]; "
            f"scoring={json.dumps(scoring, ensure_ascii=False)}"
        )


_RULE_SKILL_CACHE: RuleSkill | None = None


def load_rule_skill() -> RuleSkill:
    global _RULE_SKILL_CACHE
    if _RULE_SKILL_CACHE is not None:
        return _RULE_SKILL_CACHE

    p = Path(__file__).resolve().parent / "skills" / "site_function_allocation_rules.json"
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    _RULE_SKILL_CACHE = RuleSkill(data=data)
    return _RULE_SKILL_CACHE
