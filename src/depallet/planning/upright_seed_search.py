"""Bounded H2017 path seeds; sampled checks never authorize execution."""
import math
import numpy as np
from depallet.motion.contact_escape import RobotModel, pose_matrix


def make_seed_search(request, robot):
    kin=robot.get("robot_cfg",robot)["kinematics"]
    model=RobotModel(kin,request["joint_names"])
    original=np.asarray(request["start_position_rad"],float)
    relative=np.linalg.inv(model.transforms(original)["suction_tcp"])@pose_matrix(request["payload"]["pose_base_wxyz"])
    up=np.asarray(request["payload_orientation_policy"]["world_up_base"],float)
    limit=request["payload_orientation_policy"]["maximum_nominal_tilt_rad"]
    def search(planner, current, goals):
        import torch
        start=current.position.reshape(6)
        horizon=planner.trajopt_solver.action_horizon
        u=torch.linspace(0,1,horizon,device=start.device,dtype=start.dtype)
        bump=torch.sin(math.pi*u)**2
        # Keep the current goal endpoints exact; each perturbed seed is checked
        # by independent FK rather than assuming this coupling preserves tilt.
        paths=[];unique=[]
        for goal in goals.reshape(-1,6):
            if any(bool((goal-other).abs().max()<1e-4) for other in unique):continue
            unique.append(goal)
            early=torch.clamp(u/.5,0.,1.)
            late=torch.clamp((u-.5)/.5,0.,1.)
            early=early.square()*(3-2*early)
            late=late.square()*(3-2*late)
            for turn,arm in ((u,u),(early,late),(late,early)):
                delta=goal-start
                base=start[None,:]+arm[:,None]*delta
                base[:,0]=start[0]+turn*delta[0]
                base[:,5]=start[5]+turn*delta[0]+arm*(delta[5]-delta[0])
                for a in (-.6,-.3,0.,.3,.6):
                    for b in (-.6,-.3,0.,.3,.6):
                        offset=torch.tensor([0.,a,b,0.,a+b,0.],device=start.device,dtype=start.dtype)
                        path=base+bump[:,None]*offset
                        path[0]=start;path[-1]=goal
                        paths.append(path)
        paths=torch.stack(paths)
        bounds=torch.tensor([model.lower.tolist(),model.upper.tolist()],device=start.device,dtype=start.dtype)
        bounded=((paths>=bounds[0])&(paths<=bounds[1])).all(dim=-1).all(dim=-1)
        tested=int(paths.shape[0]);paths=paths[bounded]
        accepted=[]
        sample_indices=np.unique(np.linspace(0,horizon-1,min(17,horizon)).astype(int))
        for path in paths:
            samples=path[sample_indices.tolist()].detach().cpu().double().numpy()
            if all(np.arccos(np.clip(np.dot((model.transforms(q)["suction_tcp"]@relative)[:3,2],up),-1,1))<=limit*.8 for q in samples):
                accepted.append(path)
        receipt={"generated_paths":tested,"sampled_upright_paths":len(accepted),
                 "sampled_collision_free_paths":0,"continuous_path_certified":False}
        if not accepted:return None,receipt
        candidates=torch.stack(accepted)
        feasible=[]
        # Bound temporary GPU collision buffers to one path at a time.
        best_failed=None
        for path in candidates:
            mask=planner.graph_planner.check_samples_feasibility(path.contiguous()).reshape(-1)
            passed=bool(mask.all());feasible.append(passed)
            invalid=int((~mask).sum().item())
            if not passed and (best_failed is None or invalid<best_failed[0]):
                best_failed=(invalid,path.detach().cpu().tolist(),(~mask).nonzero().reshape(-1).cpu().tolist())
        if best_failed is not None:
            receipt['closest_rejected_seed']={'invalid_sample_count':best_failed[0],
                'position_rad':best_failed[1],'invalid_sample_indices':best_failed[2],
                'diagnostic_only':True}
        candidates=candidates[torch.tensor(feasible,device=start.device)]
        receipt["sampled_collision_free_paths"]=int(candidates.shape[0])
        if not len(candidates):return None,receipt
        costs=(candidates[:,1:]-candidates[:,:-1]).norm(dim=-1).sum(dim=-1)
        candidates=candidates[costs.argsort()]
        count=goals.shape[1]
        selected=candidates.repeat((math.ceil(count/len(candidates)),1,1))[:count]
        return selected.unsqueeze(0).contiguous(),receipt
    return search
