"""Adaptive paired metrics: all planned outcomes, actual Dynamic model responses."""
from __future__ import annotations


def summarize_pair(case, control, intervention):
    events = intervention.get("intervention_events", [])
    event_step = events[0]["cosmos_query_boundary_step"] if events else None
    base_queries = [q for q in control["policy_queries"] if q["source"] == "model"]
    dynamic_queries = [q for q in intervention["policy_queries"] if q["source"] == "model"]
    response = next((q["policy_step"] for q in dynamic_queries
                     if event_step is not None and q["policy_step"] >= event_step), None)
    base, dynamic = control["success"], intervention["success"]
    if type(base) is not bool or type(dynamic) is not bool:
        raise RuntimeError("paired outcomes must be measured booleans")
    scenario = case["scenario"]
    row = {"pair_id": case["case_id"], "control_correct": base, "intervention_correct": dynamic,
           "control_success": base, "intervention_success": dynamic,
           "paired_outcome": ("preserved_capability" if base and dynamic else
                              "intervention_side_gain" if dynamic else
                              "regression_under_change" if base else "persistent_failure"),
           "terminal_status": ("trigger_unreached" if event_step is None else
                               "response_query_unreached" if response is None else "complete"),
           "trigger_reached": event_step is not None, "response_query_reached": response is not None,
           "event_step": event_step, "intervention_policy_step": event_step,
           "next_query_step": response, "policy_response_query_step": response,
           "query_delay_steps": None if response is None else response-event_step,
           "open_loop_exposure_steps": None if response is None else response-event_step,
           "base_model_queries": len(base_queries), "dynamic_model_queries": len(dynamic_queries),
           "dynamic_replay_queries": len(intervention["policy_queries"])-len(dynamic_queries),
           "base_model_query_elapsed_s": sum(q.get("query_elapsed_s",0) for q in base_queries),
           "dynamic_model_query_elapsed_s": sum(q.get("query_elapsed_s",0) for q in dynamic_queries),
           "prefix_replay_steps": intervention["prefix_replay_steps"],
           "pre_change_executed_actions_match": intervention.get("prefix_verified", True),
           "change_type": scenario.get("change_type",scenario["change"].get("type")),
           "change_family": scenario.get("change_family",scenario["change"].get("family")),
           "severity": scenario.get("severity",scenario["change"].get("severity"))}
    return row


def aggregate_pairs(cases, rows):
    planned = len(cases)
    if (not planned or len(rows) != planned or
            len({row["pair_id"] for row in rows}) != planned or
            {row["pair_id"] for row in rows} != {case["case_id"] for case in cases}):
        raise RuntimeError("paired coverage is incomplete or duplicated")
    base = sum(row["control_correct"] for row in rows)
    dynamic = sum(row["intervention_correct"] for row in rows)
    diagnostics = {"repaired_pairs": sum(not r["control_correct"] and r["intervention_correct"] for r in rows),
                   "regressed_pairs": sum(r["control_correct"] and not r["intervention_correct"] for r in rows)}
    for field in ("base_model_queries","dynamic_model_queries","dynamic_replay_queries",
                  "base_model_query_elapsed_s","dynamic_model_query_elapsed_s"):
        diagnostics[field] = sum(row[field] for row in rows)
    delays = [row["query_delay_steps"] for row in rows if row["query_delay_steps"] is not None]
    diagnostics["mean_query_delay_steps"] = sum(delays)/len(delays) if delays else None
    return {"coverage": {"planned": planned, "completed": len(rows), "execution_complete": True,
                         "trigger_unreached": sum(not r["trigger_reached"] for r in rows),
                         "response_query_unreached": sum(r["trigger_reached"] and not r["response_query_reached"] for r in rows)},
            "end_to_end_metrics": {"complete": True,
                "control": {"measured": planned,"correct":base,"accuracy_on_planned":base/planned},
                "intervention": {"measured":planned,"correct":dynamic,"accuracy_on_planned":dynamic/planned},
                "paired_robustness_delta_on_planned": (dynamic-base)/planned},
            "diagnostics":diagnostics}
