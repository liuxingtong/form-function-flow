"""
Canonical time-slice ids: 4 weekday + 4 weekend clock partitions (independent Q1–Q4 per day type).

Weekday: WD_AM, WD_PM, WD_EVE, WD_NT
Weekend: WE_AM, WE_MD, WE_EVE, WE_NT

Linear narrative order (unit×day-type slices for long tables / transitions):
  WD_AM→WD_PM→WD_EVE→WD_NT→WE_AM→WE_MD→WE_EVE→WE_NT
"""
from __future__ import annotations

T_IDS_WEEKDAY: tuple[str, ...] = ("WD_AM", "WD_PM", "WD_EVE", "WD_NT")
T_IDS_WEEKEND: tuple[str, ...] = ("WE_AM", "WE_MD", "WE_EVE", "WE_NT")
T_IDS: tuple[str, ...] = T_IDS_WEEKDAY + T_IDS_WEEKEND

T_ORDER: list[str] = list(T_IDS)

# Adjacent pairs along the linear chain (7 segments; no wrap by default)
T_ADJ_PAIRS: tuple[tuple[str, str], ...] = tuple((T_ORDER[i], T_ORDER[i + 1]) for i in range(len(T_ORDER) - 1))

# Cyclic pairs including WE_NT → WD_AM (weekly wrap); used where a closed loop is intended
T_CYCL_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (T_ORDER[i], T_ORDER[(i + 1) % len(T_ORDER)]) for i in range(len(T_ORDER))
)

# Maps / top-N figures anchor (evening–night emphasis on weekend last slice)
DEFAULT_ANCHOR_TID: str = "WE_NT"
