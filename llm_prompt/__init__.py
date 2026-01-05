"""
LLM Prompt Templates Package

This package contains prompt generation functions for all 6 planning agents.
Also includes single agent mode for comparison experiments.
"""

from .pre_grasp_prompt import get_pre_grasp_prompt, PRE_GRASP_REPLY_TEMPLATE
from .descend_prompt import get_descend_prompt, DESCEND_REPLY_TEMPLATE
from .close_prompt import get_close_prompt, CLOSE_REPLY_TEMPLATE
from .lift_prompt import get_lift_prompt, LIFT_REPLY_TEMPLATE
from .place_prompt import get_place_prompt, PLACE_REPLY_TEMPLATE
from .release_prompt import get_release_prompt, RELEASE_REPLY_TEMPLATE
from .single_agent_prompt import get_single_agent_prompt, SINGLE_AGENT_REPLY_TEMPLATE
from .push_prompt import (
    get_batch_pose_check_prompt,
    get_push_plan_prompt,
    BATCH_POSE_CHECK_REPLY_TEMPLATE,
    PUSH_PLAN_REPLY_TEMPLATE,
    AREA_THRESHOLD
)

__all__ = [
    # Multi-agent mode (6 separate agents)
    'get_pre_grasp_prompt',
    'get_descend_prompt',
    'get_close_prompt',
    'get_lift_prompt',
    'get_place_prompt',
    'get_release_prompt',
    'PRE_GRASP_REPLY_TEMPLATE',
    'DESCEND_REPLY_TEMPLATE',
    'CLOSE_REPLY_TEMPLATE',
    'LIFT_REPLY_TEMPLATE',
    'PLACE_REPLY_TEMPLATE',
    'RELEASE_REPLY_TEMPLATE',
    # Single-agent mode (unified planning)
    'get_single_agent_prompt',
    'SINGLE_AGENT_REPLY_TEMPLATE',
    # Push/Pose check (batch mode)
    'get_batch_pose_check_prompt',
    'get_push_plan_prompt',
    'BATCH_POSE_CHECK_REPLY_TEMPLATE',
    'PUSH_PLAN_REPLY_TEMPLATE',
    'AREA_THRESHOLD',
]
