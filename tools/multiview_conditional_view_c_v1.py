"""GT-free conditional activation policy for the View A/C arm selector.

View A is always evaluated.  View C is requested only when View A cannot
provide reliable measured K2 positions for both registered right-arm joints.
No visibility threshold is introduced: the rule reuses the frozen sensitive
depth-reliability semantics and therefore has no fitted parameter.
"""

from __future__ import annotations

from typing import Any

import multiview_best_view_measured_first_v1 as best_view


METHOD_VERSION = "view_ac_conditional_c_sensitive_v1_20260802"


def activation_decision(view_a: dict[str, Any]) -> dict[str, Any]:
    """Return a causal View-C request using View-A values only."""
    measured_count = int(view_a["score"][0])
    complete = bool(view_a["complete"])
    triggered = (not complete) or measured_count < len(best_view.ARM_JOINTS)
    if not complete:
        reason = "view_a_incomplete"
    elif measured_count == 0:
        reason = "view_a_no_reliable_measured_arm_joint"
    elif measured_count == 1:
        reason = "view_a_partial_reliable_measured_arm"
    else:
        reason = "view_a_all_arm_joints_reliably_measured"
    return {
        "trigger_view_c": bool(triggered),
        "reason": reason,
        "view_a_measured_count": measured_count,
        "view_a_complete": complete,
        "method_version": METHOD_VERSION,
    }


def select_conditionally(
    view_a: dict[str, Any], view_c: dict[str, Any] | None
) -> dict[str, Any]:
    """Select A alone or invoke the frozen A/C best-view rule after a trigger."""
    decision = activation_decision(view_a)
    if decision["trigger_view_c"]:
        if view_c is None:
            raise ValueError("View C is required after a positive activation decision.")
        selected = best_view.select_arm(view_a, view_c)
    else:
        selected = {
            "selected_view": "a" if view_a["complete"] else "",
            "score_a": tuple(view_a["score"]),
            "score_c": (),
            "joints": {
                joint: view_a["joints"][joint]
                for joint in best_view.ARM_JOINTS
            },
            "method_version": METHOD_VERSION,
        }
    return {**selected, **decision, "method_version": METHOD_VERSION}
