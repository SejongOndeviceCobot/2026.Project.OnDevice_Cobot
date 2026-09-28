"""Preserve genuine trajectory constraints before cuRobo's top-k discards them."""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def capture_optimizer_constraints(solver, records, limit=8):
    """Observe bounded pre-selection results without changing solver decisions."""
    from depallet.motion.curobo_diagnostics import solver_summary
    original = getattr(solver, "_get_result", None)
    if original is None:
        yield
        return
    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(records) < limit:
            entry = solver_summary(result)
            state = getattr(result, "js_solution", None)
            metrics = getattr(result, "metrics", None)
            if state is not None and metrics is not None:
                constraints = metrics.costs_and_constraints.constraints
                if "self_collision" in constraints.names:
                    costs = constraints.values[constraints.names.index("self_collision")]
                    positions = state.position
                    if positions.numel() // 6 == costs.numel():
                        q = positions.detach().reshape(-1, positions.shape[-2], 6)
                        c = costs.detach().reshape(q.shape[:2])
                        samples = []
                        for seed in range(min(8, q.shape[0])):
                            index = int(c[seed].argmax().item())
                            samples.append({"seed": seed, "index": index,
                                "self_collision_cost": float(c[seed, index].item()),
                                "joint_position_rad": q[seed, index].cpu().tolist()})
                        entry["diagnostic_self_collision_samples"] = samples
            records.append(entry)
        return result
    solver._get_result = record
    try:
        yield
    finally:
        solver._get_result = original


def trajectory_arrays(state):
    import numpy as np
    arrays={}
    for name in ("position","velocity","acceleration","jerk","dt"):
        value=getattr(state,name,None)
        if value is not None:
            arrays[name]=value.detach().cpu().numpy().astype(np.float64)
    return arrays


def summarize_arrays(arrays,dt=None):
    import numpy as np
    result={}
    for name,a in arrays.items():
        item={"shape":list(a.shape),"all_finite":bool(np.isfinite(a).all())}
        if a.size and np.isfinite(a).all():
            item.update(maximum_absolute=float(np.max(np.abs(a))),minimum=float(a.min()),maximum=float(a.max()))
            if name!="dt":
                flat=a.reshape(-1,a.shape[-1])
                item.update(per_joint_maximum_absolute=np.max(np.abs(flat),axis=0).tolist(),
                            first=flat[0].tolist(),last=flat[-1].tolist())
        result[name]=item
    if dt is not None and "position" in arrays:
        delta=np.diff(arrays["position"],axis=-2)/dt
        result["finite_difference_velocity_rad_s"]={
            "maximum_absolute":float(np.max(np.abs(delta))),
            "per_joint_maximum_absolute":np.max(np.abs(delta.reshape(-1,6)),axis=0).tolist()}
    return result


def capture_motion(planner,target,current,request,output,label):
    """One real full-collision attempt, with no executable trajectory exported."""
    import numpy as np
    from curobo._src.util.trajectory import calculate_dt_no_clamp
    from depallet.motion.curobo_diagnostics import compact,solver_summary
    solver=planner.trajopt_solver
    original=solver._get_result
    captured=[]
    def record(*args,**kwargs):
        result=original(*args,**kwargs)
        entry=solver_summary(result)
        for name in ("metrics","interpolated_metrics"):
            metric=getattr(result,name,None)
            if metric is not None:
                entry[name+"_official_feasible"]=compact(
                    metric.costs_and_constraints.get_feasible(include_all_hybrid=False,sum_horizon=True))
        if len(captured)<8:
            captured.append(entry)
        return result
    solver._get_result=record
    try:
        result=planner.plan_pose(target,current,max_attempts=1)
    finally:
        solver._get_result=original
    report={"selected":solver_summary(result),"optimization_iterations_before_topk":captured,
            "all_world_and_self_collisions_enabled":True,"motion_exported":False}
    if result is None or result.js_solution is None:
        return report
    state=result.js_solution
    raw=trajectory_arrays(state)
    report["selected_raw_trajectory"]=summarize_arrays(raw)
    transition=solver.transition_model
    limits={"velocity":transition.max_velocity,"acceleration":transition.max_acceleration,
            "jerk":transition.max_jerk}
    report["actual_joint_limits"]={k:compact(v) for k,v in limits.items()}
    broadcast=[]
    for value in limits.values():
        limit=value.view(1,-1)
        while limit.ndim<state.position.ndim-1:
            limit=limit.unsqueeze(0)
        broadcast.append(limit)
    scale=calculate_dt_no_clamp(state.velocity,state.acceleration,state.jerk,*broadcast,epsilon=1e-3)
    required=scale*state.dt
    report["unclamped_required_trajectory_dt_s"]=compact(required)
    report["unclamped_required_duration_s"]=compact(required*(state.position.shape[-2]-1))
    report["configured_maximum_trajectory_dt_s"]=float(solver.config.maximum_trajectory_dt)
    report["required_dt_exceeds_configured_maximum"]=bool((required>solver.config.maximum_trajectory_dt).any().item())
    trimmed=result.get_interpolated_plan()
    if trimmed is not None:
        interpolated=trajectory_arrays(trimmed)
        report["selected_interpolated_trajectory"]=summarize_arrays(interpolated,request["interpolation_dt_s"])
        # Deliberately different filename and diagnostic result mode: the
        # execution loader cannot mistake this failed/untested path for a plan.
        path=output/f"diagnostic-motion-{label}.npz"
        np.savez_compressed(path,**{f"raw_{k}":v for k,v in raw.items()},
                            **{f"interpolated_{k}":v for k,v in interpolated.items()})
        report["diagnostic_trajectory_arrays"]=str(path)
    return report
