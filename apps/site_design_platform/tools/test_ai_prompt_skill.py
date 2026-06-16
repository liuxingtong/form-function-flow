import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from apps.site_design_platform.backend.ai_agent.rules_skill import load_rule_skill
from apps.site_design_platform.backend.ai_agent.prompt_templates import build_floor_stack_system_prompt

skill = load_rule_skill()
s = build_floor_stack_system_prompt(skill)
assert "RuleSkill(site_function_allocation_rules)" in s
assert "exclude=GROUND,GREEN,WALKWAY,HIGHWAY" in s
assert "zoning-first" in s
print("PASS test_ai_prompt_skill")
