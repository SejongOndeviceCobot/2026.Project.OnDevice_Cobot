# SPDX-License-Identifier: Apache-2.0
# Camera lifecycle derives from run_p0.py / official NVIDIA examples; see LICENSE-NVIDIA.
"""One continuous guarded scene: observe, plan, grasp, transfer and verify every carton."""
from __future__ import annotations
import argparse
from collections import deque
import copy,hashlib,faulthandler,json,math,os,sys,time,traceback
from pathlib import Path

MIN_WHOLE_PALLET_SURVEY_FRAMES = 16


def resolve_transfer_limit(requested, box_count):
    """Resolve an optional transfer prefix after the scenario catalog is loaded."""
    if type(box_count) is not int or box_count < 1:
        raise ValueError('Scenario must declare at least one box')
    if requested is None:
        return box_count
    if type(requested) is not int or not 1 <= requested <= box_count:
        raise ValueError(f'Bounded prefix must contain 1..{box_count} transfers')
    return requested


def validate_runtime_plan_catalog(spec, packing, order):
    """Validate the complete scenario-sized execution catalog before Isaac starts."""
    from depallet.scene.scenario_suite import validate_goal_layout
    boxes = spec.get('boxes') if isinstance(spec, dict) else None
    if not isinstance(boxes, list) or not boxes:
        raise ValueError('A nonempty scenario box catalog is required')
    box_count = len(boxes)
    ids = [box.get('id') for box in boxes if isinstance(box, dict)]
    if (len(ids) != box_count or None in ids or len(set(ids)) != box_count
            or not isinstance(order, list) or len(order) != box_count
            or set(order) != set(ids)
            or validate_goal_layout(packing, spec).get('passed') is not True):
        raise ValueError(
            f'A validated complete {box_count}-box goal layout is required')
    return box_count


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)

class _ObserverClockReadError(RuntimeError):
    def __init__(self,receipt):
        self.receipt=receipt
        super().__init__('Observer RGB clock did not converge after bounded immediate rereads: '+json.dumps(receipt,allow_nan=False))


def _read_observer_frame(clock,read_clocks,read_rgb,physics_time,last_accepted,max_attempts=8):
    """Read one observer RGB frame between a coherent pair of strict clocks."""
    if type(max_attempts) is not int or max_attempts<1:raise ValueError('Positive integer observer read attempts required')
    mismatch='image and reference clocks must advance together and strictly increase'
    observations=[]
    for attempt in range(1,max_attempts+1):
        before=read_clocks()
        rgb,info=read_rgb()
        after=read_clocks()
        observation={'attempt':attempt,'before':before,'after':after}
        if rgb is None:
            observation['outcome']='rgb_unavailable';observations.append(observation)
            receipt=None
            if attempt>1:
                receipt={'schema':'depallet.observer_clock_reread.v1','policy_input':False,
                    'status':'no_rgb_after_retry','physics_time':float(physics_time),
                    'last_accepted':last_accepted,'attempt_count':attempt,'observations':observations}
            return None,info,None,receipt
        if before!=after:
            observation['outcome']='clock_changed_around_rgb_read';observations.append(observation);continue
        try:stamp=clock.accept(*after,physics_time)
        except ValueError as error:
            if str(error)!=mismatch:
                if not observations:raise
                observation.update(outcome='nonretryable_strict_clock_error',error=str(error));observations.append(observation)
                receipt={'schema':'depallet.observer_clock_reread.v1','policy_input':False,
                    'status':'aborted_on_nonretryable_clock_error','physics_time':float(physics_time),
                    'last_accepted':last_accepted,'attempt_count':attempt,'observations':observations}
                raise _ObserverClockReadError(receipt) from error
            observation.update(outcome='independent_clock_annotators_did_not_advance_together',error=str(error))
            observations.append(observation);continue
        observation['outcome']='strict_clock_accepted' if stamp is not None else 'strict_clock_duplicate'
        observations.append(observation)
        receipt=None
        if attempt>1:
            receipt={'schema':'depallet.observer_clock_reread.v1','policy_input':False,
                'status':'coherent_read_after_retry' if stamp is not None else 'coherent_duplicate_after_retry',
                'physics_time':float(physics_time),'last_accepted':last_accepted,
                'attempt_count':attempt,'observations':observations}
        return rgb,info,stamp,receipt
    receipt={'schema':'depallet.observer_clock_reread.v1','policy_input':False,
        'status':'aborted_without_recording_frame','physics_time':float(physics_time),
        'last_accepted':last_accepted,'attempt_count':max_attempts,'observations':observations}
    raise _ObserverClockReadError(receipt)


def _read_policy_frame(clock, read_clocks, read_rgb, physics_time, last_accepted,
                       recovery, max_deferred=3):
    """Bounded cross-update recovery; never record or consume an incoherent RGB-D.

    Missing frames remain explicit gaps, not synthetic 30Hz training samples.
    Only a fresh coherent frame clears the budget, never a duplicate/no-data read.
    """
    if type(max_deferred) is not int or not 1<=max_deferred<=3:
        raise ValueError('Policy clock deferral budget must be 1..3')
    try:
        result=_read_observer_frame(clock,read_clocks,read_rgb,physics_time,last_accepted)
    except _ObserverClockReadError as error:
        if error.receipt.get('status')!='aborted_without_recording_frame':raise
        count=recovery.get('deferred_updates',0)+1
        recovery['deferred_updates']=count
        if count>max_deferred:raise
        receipt=copy.deepcopy(error.receipt)
        receipt.update(status='policy_frame_deferred',policy_input=True,recorded=False,
                       admitted_to_policy=False,deferred_updates=count,
                       maximum_deferred_updates=max_deferred,missing_frame_is_not_valid_dataset_sample=True)
        return None,None,None,receipt
    if result[2] is not None:recovery['deferred_updates']=0
    elif recovery.get('deferred_updates',0):
        recovery['deferred_updates']+=1
        if recovery['deferred_updates']>max_deferred:
            raise RuntimeError('Policy RGB-D did not recover a fresh coherent frame within bounded updates')
    return result


def _read_monitor_frame(*args, **kwargs):
    """Observer-only: discard an incoherent frame without admitting false timestamps."""
    try:
        return _read_observer_frame(*args, **kwargs)
    except _ObserverClockReadError as error:
        if error.receipt.get('status')!='aborted_without_recording_frame':raise
        receipt=copy.deepcopy(error.receipt)
        receipt.update(status='monitor_frame_dropped',recorded=False,policy_input=False)
        return None,None,None,receipt


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--robot-usd',type=Path)
    p.add_argument('--robot-collision-profile',choices=['baseline','link2_hull12_margin10mm','link24_hull_margin10mm'],default='baseline')
    p.add_argument('--recording-compression-level',type=int,choices=[0,1],default=1)
    p.add_argument('--recording-workers',type=int,choices=range(0,9),default=0,help='Bounded per-policy-camera RGB-D writer threads; 0 preserves synchronous recording')
    p.add_argument('--transport-overhead-m',type=float,default=.08)
    p.add_argument('--payload-cover-profile',choices=['legacy64','grid12x12x3','grid8x8x8','grid10x7x7'],default='legacy64')
    p.add_argument('--scenario',type=Path,required=True,help='Prepared scenario.json under this project data root')
    p.add_argument('--cutamp-plan',type=Path,help='Guarded learned-grasp cuTAMP horizon')
    p.add_argument('--cutamp-input',type=Path,help='cuTAMP input directory associated with that plan')
    p.add_argument('--grasp-candidates',type=Path,help='Guarded GraspGen suction result for the observed surface packet')
    p.add_argument('--observed-surfaces',type=Path,help='Explicit saved SAM3.1 hybrid integration; one transfer only')
    p.add_argument('--max-transfers',type=int,default=None,
        help='1..scenario box count; omitted means the complete loaded scenario')
    p.add_argument('--max-steps',type=int)
    p.add_argument('--physics-hz',type=int,choices=[240],default=240)
    p.add_argument('--vacuum-policy',choices=['aggregate_compliant_si'],default='aggregate_compliant_si')
    p.add_argument('--vacuum-damping-ratio',type=float,choices=[1.,2.],default=2.)
    p.add_argument('--vacuum-lateral-compliance-m',type=float,choices=[.004],default=.004)
    p.add_argument('--contact-anchor-policy',choices=['project_measured_surface'],default='project_measured_surface')
    p.add_argument('--camera-rig',choices=['legacy_v1','overhead_wrist_v2'],default='overhead_wrist_v2')
    p.add_argument('--motion-profile',choices=['baseline','brisk','brisk34','brisk36'],default='brisk')
    p.add_argument('--preview-hz',type=int,choices=range(1,31),default=30,help='latest.jpg refresh only; native RGB-D and observer remain 30Hz')
    p.add_argument('--inspection-ai',choices=['off','shadow_plan','apply_plan'],default='off',help='Run SAM3.1/Point2Pose/GraspGen/VLM/cuTAMP during a frozen wrist survey; shadow or fresh motion-checked application')
    p.add_argument('--inspection-view',choices=['off','wrist_top_v1','wrist_oblique_v1','wrist_pallet_v1'],default='off')
    p.add_argument('--survey-scope',choices=['all_exposed','highest_layer'],default='all_exposed',help='PF3 may require only the highest layer; full pallet frustum and selected-target visibility remain required')
    p.add_argument('--survey-candidate',choices=['oblique','top_0','top_90','top_180','top_270','top_180_clear'],default='oblique',help='Whole-pallet geometric view; continuous planner and measured depth gates remain required')
    p.add_argument('--frames',type=int)
    p.add_argument('--record-rollout',action='store_true',help='Trace native commands and exact post-physics states for independent dataset audit')
    p.add_argument('--live',action='store_true')
    p.add_argument('--port',type=int,default=18766)
    a=p.parse_args()
    if a.vacuum_lateral_compliance_m and a.vacuum_policy!='aggregate_compliant_si':
        p.error('Lateral compliance requires aggregate_compliant_si')
    if a.inspection_ai=='apply_plan' and (a.cutamp_plan or a.observed_surfaces or a.grasp_candidates):p.error('Online apply_plan obtains fresh artifacts; do not supply replay inputs')
    if a.inspection_ai!='off' and a.inspection_view!='wrist_pallet_v1':p.error('Inspection AI requires whole-pallet wrist survey')
    if a.inspection_view!='off' and a.camera_rig!='overhead_wrist_v2':p.error('Inspection requires overhead+wrist native rig')
    a.execute_plan=True
    if not .08<=a.transport_overhead_m<=.5:p.error('Transport overhead must be within 0.08..0.5m')
    if a.max_steps is None:a.max_steps=1800*a.physics_hz
    camera_hz=30 if a.camera_rig=='overhead_wrist_v2' else 10
    if a.frames is None:a.frames=1800*camera_hz
    if not 600<=a.max_steps<=1800*a.physics_hz or not 100<=a.frames<=1800*camera_hz:
        p.error('Task recording limit: 1800 simulated seconds')
    if not os.environ.get('ISAAC_P0_OUTPUT') or not os.environ.get('ISAAC_P0_GPU_UUID'):p.error('Use guarded_run.py')
    output=Path(os.environ['ISAAC_P0_OUTPUT']).resolve()
    runroot=Path(os.environ['ISAAC_P0_RUNS']).resolve()
    cache=Path(os.environ['ISAAC_P0_CACHE']).resolve()
    data=Path(os.environ['ISAAC_P0_DATA']).resolve()
    if not output.is_relative_to(runroot) or output==runroot:raise ValueError('dedicated run required')
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    os.environ['OMNI_ENV_PRIVACY_CONSENT']='0'
    os.environ['CUDA_VISIBLE_DEVICES']=os.environ['ISAAC_P0_GPU_UUID']
    runtime_cache=Path(os.environ.get('ISAAC_P0_RUNTIME_CACHE',str(cache))).resolve()
    sys.argv=[sys.argv[0],'--portable-root',str(runtime_cache/'kit')]
    write_json(output/'worker-isolation.json',{'parallel_slot':os.environ.get('ISAAC_P0_PARALLEL_SLOT',''),
        'runtime_cache':str(runtime_cache),'kit_portable_root':str(runtime_cache/'kit'),
        'cpu_affinity':sorted(os.sched_getaffinity(0)), 'pid':os.getpid(), 'process_group':os.getpgrp(),
        'live_server_enabled':a.live})
    faulthandler.enable()
    # The external guard owns the wall timeout. Avoid asynchronous periodic
    # Python thread dumps inside Kit/native numerical code; use phase receipts.
    if a.scenario.is_symlink() or not a.scenario.resolve().is_relative_to(data):
        raise ValueError('Scenario must be a regular file within project data')
    scenario=json.loads(a.scenario.read_text());spec=scenario['spec'];cell=scenario['cell']
    planning=json.loads(a.scenario.with_name('rule-plan.json').read_text());packing=planning['packing'];order=planning['order']
    box_count=validate_runtime_plan_catalog(spec,packing,order)
    a.max_transfers=resolve_transfer_limit(a.max_transfers,box_count)
    if scenario['capabilities']['unsupported_reasons']:
        raise ValueError('Unsupported physical scenario: '+repr(scenario['capabilities']['unsupported_reasons']))
    placements={x['box_id']:x for x in packing['placements']}
    source_pose=cell['source_pose'];goal_pose=cell['goal_pose']
    if a.grasp_candidates and not a.observed_surfaces:raise ValueError('GraspGen requires observed surfaces')
    observed_packet=None
    if a.observed_surfaces:
        if a.max_transfers!=1:raise ValueError('Observed replay integration is bounded to one transfer')
        from depallet.integration.modular_pick_bridge import load_surfaces
        observed_packet=load_surfaces(a.observed_surfaces,runroot,a.grasp_candidates)
        write_json(output/'modular-pipeline.json',observed_packet)
    if bool(a.cutamp_plan)!=bool(a.cutamp_input):raise ValueError('Both cuTAMP plan and input are required')
    if a.cutamp_plan:
        if a.observed_surfaces or a.grasp_candidates:raise ValueError('cuTAMP supplies its verified model artifacts')
        from depallet.integration.cutamp_execution_bridge import load_execution
        planning,observed_packet,horizon=load_execution(a.cutamp_plan,a.cutamp_input,scenario,planning,runroot,a.max_transfers)
        packing=planning['packing'];order=planning['order']
        placements={x['box_id']:x for x in packing['placements']}
        write_json(output/'cutamp-execution-horizon.json',horizon)
        write_json(output/'modular-pipeline.json',observed_packet)
    completed=[];transfer_results=[];cycle_output=output/'transfers'/('01_'+order[0]);cycle_output.mkdir(parents=True)
    assembly_path=data/'h2017-vgp20-v2/manifest.json'
    template=json.loads((data/'grasp-requests-v2/rear_upper_3.request.json').read_text())
    robot_config=template['robot_config']
    from depallet.motion.robot_collision_profiles import resolve_robot_collision_profile
    robot_config,robot_collision_receipt=resolve_robot_collision_profile(a.robot_collision_profile,robot_config,workspace=Path(os.environ['JCLEE_WORKSPACE']))
    write_json(output/'robot-collision-profile.json',robot_collision_receipt)
    from depallet.planning.planning_requests import slow_robot_config
    _,motion_receipt=slow_robot_config(robot_config,output/'robot-motion-profile.yml',a.motion_profile,
        native_urdf=data/'curobo-assets-v1/h2017_vgp20_planner.urdf')
    write_json(output/'motion-profile.json',motion_receipt)
    robot_config=motion_receipt['robot_config']
    if a.robot_usd is None:a.robot_usd=data/'h2017-vgp20-v2/h2017_vgp20.usda'
    wall_budget=float(os.environ.get('ISAAC_P0_WALL_SECONDS','600'))
    write_json(output/'scenario.json',scenario);write_json(output/'task-plan.json',planning)
    snapshot_dir=output/'code-snapshot';snapshot_dir.mkdir()
    source_files=sorted((Path(os.environ['ISAAC_P0_PROJECT'])/'src').rglob('*.py'))
    source_files += [Path(os.environ['ISAAC_P0_PROJECT'])/'scripts'/name for name in ['run_task.py','guarded_run.py','curobo_worker.py']]
    receipts=[]
    for source_file in source_files:
        relative=source_file.relative_to(Path(os.environ['ISAAC_P0_PROJECT']))
        destination=snapshot_dir/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        raw=source_file.read_bytes();destination.write_bytes(raw)
        receipts.append({'path':str(relative),'sha256':hashlib.sha256(raw).hexdigest()})
    write_json(snapshot_dir/'manifest.json',{'files':receipts,'captured_before_simulation_start':True})

    runtime_contract={'schema':'depallet.continuous_task_contract.v3',
        'prerelease_stability_required':True,'resolve_goal_on_measured_support_required':True,
        'preclose_stability_required':True,'prerelease_payload_compensation_required':True,
        'prerelease_pose_check_required':True,'perception_source':'simulation_oracle'}
    runtime_contract.update(robot_collision_profile=a.robot_collision_profile,
        robot_collision_profile_receipt='robot-collision-profile.json',
        robot_config_sha256=robot_collision_receipt['selected_robot_config_sha256'])
    runtime_contract['payload_cover_profile']=a.payload_cover_profile
    runtime_contract['transport_overhead_m']=a.transport_overhead_m
    runtime_contract['recording_compression_level']=a.recording_compression_level
    runtime_contract['recording_workers']=a.recording_workers
    runtime_contract['inspection_view']=a.inspection_view
    runtime_contract['inspection_ai_mode']=a.inspection_ai
    runtime_contract['survey_scope']=a.survey_scope
    runtime_contract['survey_candidate']=a.survey_candidate if a.inspection_view=='wrist_pallet_v1' else None
    runtime_contract['live_preview_hz']=a.preview_hz
    runtime_contract.update(camera_rig=a.camera_rig,motion_profile=a.motion_profile,camera_hz=camera_hz,observer_is_policy_input=False,motion_profile_receipt='motion-profile.json')
    write_json(output/'runtime-contract.json',runtime_contract)

    from isaacsim import SimulationApp
    app=viewer=stepper=None;execution_plan=None;code=1;started=time.monotonic()
    from depallet.runtime.runtime_timing import BestEffortTimingJournal
    timing=BestEffortTimingJournal(output,origin=started)
    from depallet.runtime.runtime_costs import RuntimeCosts,PreviewCadence
    costs=RuntimeCosts();preview_cadence=PreviewCadence(a.preview_hz)
    task_started=None;task_ended=None;task_failed_at=None;task_end_sim=None
    rollout=None;rollout_callback=None;rollout_error=None;observer=None
    try:
        assembly=json.loads(assembly_path.read_text())
        if not a.robot_usd.is_file() or a.robot_usd.resolve()!=Path(assembly['usd']).resolve():
            raise ValueError('Physical USD must be the exact reviewed H2017/VGP20 assembly')
        write_json(output/'assembly-input.json',{'path':str(a.robot_usd.resolve()),'sha256':hashlib.sha256(a.robot_usd.read_bytes()).hexdigest(),
            'manifest_path':str(assembly_path),'manifest_sha256':hashlib.sha256(assembly_path.read_bytes()).hexdigest()})
        first=placements[order[0]];yaw=goal_pose[3]+first['yaw_goal_rad'];c=math.cos(goal_pose[3]);sn=math.sin(goal_pose[3]);local=first['position_goal_m']
        execution_plan={'box_id':order[0],'goal_position_m':[goal_pose[0]+c*local[0]-sn*local[1],goal_pose[1]+sn*local[0]+c*local[1],goal_pose[2]+local[2]],
            'goal_quaternion_wxyz':[math.cos(yaw/2),0,0,math.sin(yaw/2)],'goal_support_id':first['support_id'],
            'perception_source':'simulation_oracle','attachment_uncertainty':{'translation_m':.004,'rotation_rad':.01,'payload_padding_m':.0075}}
        app=SimulationApp({'headless':True,'hide_ui':True,'width':640,'height':480,
            'active_cuda_gpus':[0],'multi_gpu':False,'max_gpu_count':1,'physics_gpu':0,
            'limit_cpu_threads':4,'renderer':'RealTimePathTracing','anti_aliasing':0,
            'samples_per_pixel_per_frame':1,'max_bounces':3,'enable_crashreporter':False,
            'disable_viewport_updates':True,'fast_shutdown':True,'shutdown_watchdog_timeout':15.,
            'extra_args':['--/app/settings/persistent=false','--/app/settings/loadUserConfig=false',
                '--/telemetry/enabled=false','--/app/runLoops/main/rateLimitEnabled=true',
                '--/app/runLoops/main/rateLimitFrequency=60',f'--/log/file={output}/kit.log']})
        import numpy as np
        import omni.usd
        import omni.replicator.core as rep
        import isaacsim.core.experimental.utils.app as app_utils
        from pxr import Gf,Sdf,UsdGeom,UsdPhysics,UsdShade,UsdLux,PhysxSchema,PhysicsSchemaTools
        from PIL import Image,ImageDraw,ImageFont
        from isaacsim.core.experimental.prims import RigidPrim,Articulation
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.sensors.experimental.rtx import CameraSensor,RtxCamera
        from omni.physx import get_physx_simulation_interface
        from depallet.scene.photo_scene import build_scene
        from depallet.runtime.recording import Recorder
        from depallet.runtime.frame_clock import RenderClock
        from depallet.observation.camera_rig import camera_specs,intrinsics,world_camera_pose
        from depallet.planning.motion_profiles import resolve_motion_profile
        motion_profile=resolve_motion_profile(a.motion_profile)
        from depallet.scene.depallet_scene_export import object_masks,summarize_settle,initial_settle_decision
        if execution_plan:
            from depallet.manipulation.h2017_execution import Trajectory,SingleBoxStepper
            from depallet.planning.depallet_execution_plan import measured_scene_check,static_scene_check,goal_on_empty_pallet,payload_request,transform,pose_from_transform,contact_classification,frozen_state_check,attachment_rigidity_check
            from depallet.motion.nested_curobo import run_curobo_child as _run_curobo_child
            def run_curobo_child(*args,**kwargs):
                with timing.span('motion_planning',transfer_id=execution_plan.get('box_id'),
                        simulation_clock=lambda:float(SimulationManager.get_simulation_time())):
                    return costs.call('motion_planning_subprocess',_run_curobo_child,*args,**kwargs)
            from depallet.manipulation.vgp20_surface import create_vgp20_surface,sample_attachment_joint_schema
            enabled=omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('isaacsim.robot.surface_gripper',True)
            if not enabled:raise RuntimeError('Surface Gripper extension did not enable')
        print('DEPALLET_STAGE imports_ready',flush=True)
        SimulationManager.setup_simulation(dt=1/a.physics_hz,device='cpu')
        # Local CPU physics writes articulated transforms to USD for child cameras.
        SimulationManager.enable_fabric(False)
        stage=omni.usd.get_context().get_stage()
        physics_solver_settings=[]
        for scene_prim in stage.Traverse():
            if scene_prim.IsA(UsdPhysics.Scene):
                scene_api=PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
                scene_api.CreateSolverTypeAttr('TGS')
                scene_api.CreateEnableExternalForcesEveryIterationAttr(True)
                physics_solver_settings.append({'path':str(scene_prim.GetPath()),'solver':'TGS',
                    'enable_external_forces_every_iteration':bool(scene_api.GetEnableExternalForcesEveryIterationAttr().Get()),
                    'physics_hz':a.physics_hz})
        if not physics_solver_settings:raise RuntimeError('Expected local physics scene missing')
        write_json(output/'physics-solver-settings.json',physics_solver_settings)
        UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
        gravity_readback=[list(item.get_gravity()) for item in SimulationManager.get_physics_scenes()]
        if len(gravity_readback)!=1 or not np.allclose(gravity_readback[0],[0.,0.,-9.81],atol=1e-5,rtol=0.):
            raise RuntimeError('Payload compensation requires one measured -9.81m/s2 physics scene')
        write_json(output/'gravity-readback.json',{'gravity_world_m_s2':gravity_readback[0],
            'readback_source':'SimulationManager PhysicsScene.get_gravity','stage_meters_per_unit':1.,
            'payload_model_gravity_world_m_s2':[0.,0.,-9.81]})
        scene=build_scene(stage,spec,source_pose=source_pose,goal_pose=goal_pose)
        write_json(output/'scene-spec.json',spec);write_json(output/'scene-build.json',scene)
        boxes={box['id']:box for box in scene['transformed_boxes']}
        paths=scene['box_paths'];rigid=RigidPrim(list(paths.values()))
        by_path={path:key for key,path in paths.items()}
        box_ids=[by_path[path] for path in rigid.paths]
        authored_positions=[boxes[key]['position_world_m'] for key in box_ids]
        authored_quaternions=[boxes[key]['quaternion_world_wxyz'] for key in box_ids]
        for path in paths.values():
            api=PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(path))
            api.CreateSolverPositionIterationCountAttr(32);api.CreateSolverVelocityIterationCountAttr(1)
        def material(name,color):
            mat=UsdShade.Material.Define(stage,'/World/SceneLooks/'+name)
            shader=UsdShade.Shader.Define(stage,str(mat.GetPath())+'/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.9)
            shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
            shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.))
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            return mat
        def solid(path,position,size,mat):
            geom=UsdGeom.Cube.Define(stage,path);geom.CreateSizeAttr(1.)
            geom.AddTranslateOp().Set(Gf.Vec3d(*position));geom.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat)
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim()).CreateCollisionEnabledAttr(True)
        solid('/World/Ground',(0,0,-.025),(7,7,.05),material('Floor',(.16,.18,.19)))
        solid('/World/BackWall',(0,2.4,1.5),(7,.08,3),material('Wall',(.64,.65,.64)))
        if a.camera_rig=='overhead_wrist_v2':
            solid('/World/Ceiling',(0,0,4.9),(7,7,.06),material('Ceiling',(.65,.67,.68)))
            seam=material('CeilingSeams',(.24,.27,.29))
            for i in range(-3,4):
                solid(f'/World/CeilingSeamX{i+3}',(i,0,4.86),(.025,7,.012),seam)
                solid(f'/World/CeilingSeamY{i+3}',(0,i,4.86),(7,.025,.012),seam)
        base=cell['robot_base_position_m']
        solid('/World/RobotPedestal',(base[0],base[1],base[2]/2),(.55,.55,base[2]),material('Pedestal',(.11,.12,.13)))
        robot=model=None
        robot_report={'requested_usd':str(a.robot_usd) if a.robot_usd else None,'loaded':False,
                      'grasp_executed':False,'surface_gripper_authored':False}
        if a.robot_usd is not None and a.robot_usd.is_file():
            root=UsdGeom.Xform.Define(stage,'/World/H2017')
            root.GetPrim().GetReferences().AddReference(str(a.robot_usd.resolve()))
            root.AddTranslateOp(opSuffix='workcell').Set(Gf.Vec3d(*base))
            joint=UsdPhysics.FixedJoint(stage.GetPrimAtPath('/World/H2017/root_joint'))
            if not joint:raise RuntimeError('H2017 root fixed joint missing')
            joint.CreateLocalPos0Attr(Gf.Vec3f(*base))
            if execution_plan:
                root_solver=PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath('/World/H2017/root_joint'))
                root_solver.CreateSolverPositionIterationCountAttr(32);root_solver.CreateSolverVelocityIterationCountAttr(1)
                root_solver.CreateEnabledSelfCollisionsAttr(True)
            model=json.loads((data/'h2017-dynamics-v1/independent-validation.json').read_text())
            robot=Articulation('/World/H2017')
            if list(robot.dof_names)!=model['joint_names']:raise RuntimeError('H2017 joint order mismatch')
            robot.set_default_state(dof_positions=[model['home_positions_rad']],dof_velocities=[[0.]*6],dof_efforts=[[0.]*6])
            robot.set_dof_max_efforts([model['max_efforts_nm']]);robot.set_dof_max_velocities([[.15]*6])
            robot.set_dof_gains(stiffnesses=[model['drive_stiffnesses_nm_per_rad']],dampings=[model['drive_dampings_nm_s_per_rad']])
            robot.set_dof_position_targets([model['home_positions_rad']]);robot.set_dof_velocity_targets([[0.]*6])
            robot_report.update(loaded=True,base_link='/World/H2017/base_link',base_world_m=list(base),
                joint_policy='one home reset, then constant drive hold',self_collision_certified=False)
        else:robot_report['missing_reason']='optional robot USD absent; scene observation without robot'
        if execution_plan:
            if robot is None:raise RuntimeError('Execution requires the reviewed H2017 assembly')
            if execution_plan['box_id'] not in boxes:raise ValueError('Execution target is not a scene box')
            gripper_info=create_vgp20_surface(stage,'/World/H2017/link_6',assembly,attachment_policy=a.vacuum_policy,
                payload_mass_kg=boxes[execution_plan['box_id']]['physical']['mass_kg'],
                payload_dimensions_m=boxes[execution_plan['box_id']]['dimensions_m'],physics_dt_s=1/a.physics_hz,damping_ratio=a.vacuum_damping_ratio,lateral_compliance_m=a.vacuum_lateral_compliance_m)
            flange=RigidPrim('/World/H2017/link_6')
            execution_velocity_limits=np.minimum(model['max_velocities_rad_s'],1.2*motion_profile['maximum_velocity_rad_s']).tolist()
            robot.set_dof_max_velocities([execution_velocity_limits])
            articulation_root=stage.GetPrimAtPath('/World/H2017/root_joint')
            if not articulation_root.HasAPI(UsdPhysics.ArticulationRootAPI):raise RuntimeError('Reviewed articulation root API missing')
            solver_api=PhysxSchema.PhysxArticulationAPI.Apply(articulation_root)
            solver_api.CreateSolverPositionIterationCountAttr(32);solver_api.CreateSolverVelocityIterationCountAttr(1)
            self_collision_api=PhysxSchema.PhysxArticulationAPI.Apply(articulation_root)
            self_collision_api.CreateEnabledSelfCollisionsAttr(True)
            if self_collision_api.GetEnabledSelfCollisionsAttr().Get() is not True:raise RuntimeError('Runtime self collisions did not enable')
            robot_report['self_collisions_enabled_for_execution']=True
            robot_report.update(surface_gripper_authored=True,joint_policy='one initial home reset; checked trajectory drive targets only')
            write_json(output/'surface-gripper.json',gripper_info)
        write_json(output/'robot-scene.json',robot_report)
        disabled=[]
        for prim in stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI):
                UsdLux.LightAPI(prim).CreateIntensityAttr(0.);disabled.append(str(prim.GetPath()))
            for name in ['inputs:emissiveColor','inputs:enable_emission','inputs:emissive_intensity']:
                attr=prim.GetAttribute(name)
                if attr and attr.Get() is not None:attr.Set(Gf.Vec3f(0.) if name.endswith('Color') else (False if name.endswith('emission') else 0.))
        light=UsdLux.DomeLight.Define(stage,'/World/SceneLights/Ambient');light.CreateIntensityAttr(250.)
        for name,pos,intensity,size in [('CeilingKey',(0.,-1.,4.6),2200.,(4.,3.)),('CeilingFill',(1.,1.,4.5),800.,(3.,2.))]:
            light=UsdLux.RectLight.Define(stage,'/World/SceneLights/'+name)
            light.CreateWidthAttr(size[0]);light.CreateHeightAttr(size[1]);light.CreateIntensityAttr(intensity)
            light.CreateColorAttr(Gf.Vec3f(1.,.96,.91));light.AddTranslateOp().Set(Gf.Vec3d(*pos))
        write_json(output/'appearance.json',{'lights_attached_to_robot':False,'disabled_imported_lights':disabled,
            'lighting':'unmeasured neutral indoor ceiling illumination','materials':'green lattice plastic, rough cardboard, packing tape'})
        def plain(value):
            if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
            if isinstance(value,(tuple,list)):return [plain(v) for v in value]
            if isinstance(value,np.ndarray):return plain(value.tolist())
            if isinstance(value,np.generic):return plain(value.item())
            if isinstance(value,float) and not math.isfinite(value):return {'nonfinite':str(value)}
            return value if value is None or isinstance(value,(str,int,float,bool)) else str(value)
        def make_camera(specification):
            folder=output/specification['root'];folder.mkdir(parents=True,exist_ok=True)
            path=specification['parent']+'/'+specification['name']
            prim=UsdGeom.Camera.Define(stage,path)
            local=np.asarray(specification['T_parent_camera_cv'])@np.diag([1.,-1.,-1.,1.])
            UsdGeom.Xformable(prim).AddTransformOp().Set(Gf.Matrix4d(local.T.tolist()))
            camera=RtxCamera(path,tick_rate=float(specification['fps']),reset_xform_op_properties=False)
            height,width=specification['resolution_hw'];ha=specification['horizontal_aperture_mm']
            prim.GetFocalLengthAttr().Set(specification['focal_mm'])
            prim.GetHorizontalApertureAttr().Set(ha);prim.GetVerticalApertureAttr().Set(ha*height/width)
            camera.camera.set_clipping_ranges(.02,20.)
            policy=specification['role']=='policy'
            annotators=['rgb']+(['distance_to_image_plane'] if policy else [])+(['instance_id_segmentation'] if specification['seg'] else [])
            sensor=CameraSensor(camera,resolution=(height,width),annotators=annotators)
            rp=str(sensor.render_product.GetPath());product=stage.GetPrimAtPath(rp)
            product.GetAttribute('omni:rtx:rendermode').Set('RealTimePathTracing')
            product.GetAttribute('omni:rtx:rtpt:maxBounces').Set(3)
            ft=rep.AnnotatorRegistry.get_annotator('IsaacReadSimulationTime');ft.initialize(resetOnStop=True);ft.attach([rp])
            rt=rep.AnnotatorRegistry.get_annotator('ReferenceTime');rt.attach([rp])
            calibration={'camera_id':specification['id'],'K':intrinsics(specification).tolist(),
                'resolution_hw':[height,width],'camera_hz':specification['fps'],'camera_axes':'OpenCV +X right,+Y down,+Z forward',
                'synthetic_calibration':True,'policy_input':policy,'renderer':str(product.GetAttribute('omni:rtx:rendermode').Get())}
            if policy:calibration['depth_kind']='optical_z_m'
            if specification['id']=='wrist':
                calibration.update(T_flange_camera_cv=specification['T_parent_camera_cv'],parent_prim_path=specification['parent'],
                    mount='Synthetic flange-side optical mount; housing, mass, cable and real hand-eye calibration pending')
            else:calibration['T_world_camera_cv']=specification['T_parent_camera_cv']
            write_json(folder/'camera.json',calibration)
            if policy:recorder=Recorder(folder,width=width,height=height,png_compress_level=a.recording_compression_level,depth_compress_level=a.recording_compression_level,writer_workers=a.recording_workers)
            else:
                from depallet.runtime.observer_recording import ObserverRecorder
                recorder=ObserverRecorder(folder,width=width,height=height,fps=specification['fps'],allow_frame_gaps=True)
            return dict(sensor=sensor,time=ft,reference=rt,clock=RenderClock(),recorder=recorder,
                        folder=folder,camera=calibration,spec=specification,path=path,id=specification['id'])
        rig_specs=camera_specs(a.camera_rig)
        all_cameras=[make_camera(specification) for specification in rig_specs]
        cameras=[b for b in all_cameras if b['spec']['role']=='policy']
        overview=cameras[0];source=next(b for b in cameras if b['spec']['seg'])
        observer=next((b for b in all_cameras if b['spec']['role']=='observer_only'),None)
        write_json(output/'camera-rig.json',{'schema':'depallet.camera_rig.v2','profile':a.camera_rig,'cameras':rig_specs,
            'policy_camera_ids':[b['id'] for b in cameras],'observer_is_policy_input':False,'fps_time_base':'simulation_time',
            'depth_noise_model':'ideal optical Z; no calibrated RealSense noise model',
            'flange_usd_writeback_enabled':True,'synthetic_mount_not_hardware_selection':True})
        camera_pose_history=deque(maxlen=a.physics_hz*3)
        inspection_frames=None
        observer_clock_last_accepted=None;observer_clock_rereads=[]
        camera_follow_check={'frames':0,'maximum_transform_error':0.,'passed':True}
        iteration_settings=[]
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                api.CreateSolverPositionIterationCountAttr(32)
                api.CreateSolverVelocityIterationCountAttr(1)
                iteration_settings.append({'path':str(prim.GetPath()),'type':'rigid_body',
                    'position_iterations':api.GetSolverPositionIterationCountAttr().Get(),
                    'velocity_iterations':api.GetSolverVelocityIterationCountAttr().Get()})
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                api=PhysxSchema.PhysxArticulationAPI.Apply(prim)
                api.CreateSolverPositionIterationCountAttr(32)
                api.CreateSolverVelocityIterationCountAttr(1)
                iteration_settings.append({'path':str(prim.GetPath()),'type':'articulation',
                    'position_iterations':api.GetSolverPositionIterationCountAttr().Get(),
                    'velocity_iterations':api.GetSolverVelocityIterationCountAttr().Get()})
        write_json(output/'solver-iteration-readback.json',iteration_settings)
        stage.GetRootLayer().Export(str(output/'scene.usda'))
        if a.live:
            from depallet.runtime.live_view import LiveView
            viewer=LiveView(output,port=a.port).start();print('LIVE_VIEW',viewer.url,flush=True)
        contacts={'headers_with_points':0,'box_headers':0,'robot_scene_headers':0,'pairs':{},'callback_error':None}
        observing=False;execution_active=False
        operation_contacts={'classes':{},'phase_counts':{},'unexpected':{},'runtime_self_collision_certified':False}
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        def contact_event(headers,data):
            try:
                for header in headers:
                    if header.num_contact_data<=0:continue
                    pair=sorted(str(PhysicsSchemaTools.intToSdfPath(v)) for v in [header.collider0,header.collider1])
                    key=' | '.join(pair);contacts['pairs'][key]=contacts['pairs'].get(key,0)+1
                    contacts['headers_with_points']+=1
                    if any(path.startswith(scene['root_path']+'/Boxes/') for path in pair):contacts['box_headers']+=1
                    if observing and any(path.startswith('/World/H2017/') for path in pair) and any(path.startswith(scene['root_path']+'/') for path in pair):contacts['robot_scene_headers']+=1
                    if execution_plan:
                        target_id=execution_plan['box_id'];support_id=boxes[target_id]['support_id']
                        support_path=scene['source_pallet_path'] if support_id=='source_pallet' else paths[support_id]
                        contact_phase=stepper.state if stepper is not None else 'INITIALIZING'
                        goal_support=execution_plan['goal_support_id']
                        goal_support_path=scene['goal_pallet_path'] if goal_support=='goal_pallet' else paths[goal_support]
                        classification=contact_classification(pair,paths[target_id],support_path,goal_support_path,
                            phase=contact_phase,lift_confirmed=bool(stepper is not None and stepper.lift_confirmed))
                        phase_counts=operation_contacts['phase_counts'].setdefault(contact_phase,{})
                        phase_counts[classification]=phase_counts.get(classification,0)+1
                        operation_contacts['classes'][classification]=operation_contacts['classes'].get(classification,0)+1
                        if classification.startswith('unexpected_'):operation_contacts['unexpected'][key]=classification
            except Exception as error:contacts['callback_error']=repr(error)
        subscription=get_physx_simulation_interface().subscribe_contact_report_events(contact_event)
        print('DEPALLET_STAGE scene_and_cameras_ready',flush=True)
        app.update();app_utils.play()
        if robot is not None:
            robot.reset_to_default_state();robot.set_dof_position_targets([model['home_positions_rad']]);robot.set_dof_velocity_targets([[0.]*6])
        history=deque();last_step=-1;physics_start=SimulationManager.get_num_physics_steps();max_joint_error=0.
        def snapshot():
            pos,quat=rigid.get_world_poses();linear,angular=rigid.get_velocities()
            state={'sim_time':float(SimulationManager.get_simulation_time()),'physics_step':int(SimulationManager.get_num_physics_steps()),
                'positions_m':pos.numpy().tolist(),'quaternions_wxyz':quat.numpy().tolist(),
                'linear_velocities_m_s':linear.numpy().tolist(),'angular_velocities_rad_s':angular.numpy().tolist()}
            if any(not np.isfinite(state[key]).all() for key in ['positions_m','quaternions_wxyz','linear_velocities_m_s','angular_velocities_rad_s']):raise RuntimeError('nonfinite PhysX state')
            return state
        for warmup in range(600):
            app.update();okay=[]
            if execution_plan and (contacts['callback_error'] or operation_contacts['unexpected']):
                raise RuntimeError('Initial warmup contact validation failed: '+repr(contacts['callback_error'] or operation_contacts['unexpected']))
            for bundle in cameras:
                rgb,_=bundle['sensor'].get_data('rgb');depth,_=bundle['sensor'].get_data('distance_to_image_plane')
                valid=False
                if rgb is not None and depth is not None:
                    ri=rgb.numpy()[...,:3];di=depth.numpy().reshape(bundle['spec']['resolution_hw']);mask=np.isfinite(di)&(di>0)
                    valid=np.mean(mask)>=.05 and float(ri[mask].mean())>5 and np.any(np.var(ri.astype(float),axis=(0,1))>0)
                okay.append(valid)
            if all(okay):break
        else:raise TimeoutError('both cameras failed RGB-D warmup')
        task_started=time.monotonic()
        print('DEPALLET_STAGE cameras_warm',flush=True);observing=True
        sequence={'camera':source['camera'],'gravity_world':[0,0,-1],'objects':[],'frames':[],
            'source':'Isaac RGB-D; explicit scenario assumed geometry','initialization_policy':'manual diagnostic subset, not VLM planning',
            'ground_truth_object_poses_in_input':False}
        masks_written=False;all_paths={**paths,'source_pallet':scene['source_pallet_path']}
        pre_execution_settle=None;done_frame_counts=None;last_execution_state=None
        initial_settle_started=None;initial_report_time=-math.inf
        actual_escape_watchdog=None
        release_gap_m=None;payload_gravity_evaluator=None
        attachment_reference=None;attachment_rigidity_peak={"position_error_m":0.,"orientation_error_rad":0.}
        upright_peak_rad=0.
        if execution_plan:
            target_index=box_ids.index(execution_plan['box_id'])
            def measure_box():
                measured=snapshot()
                return {key:measured[source_key][target_index] for key,source_key in [('position_m','positions_m'),('quaternion_wxyz','quaternions_wxyz'),('linear_velocity_m_s','linear_velocities_m_s'),('angular_velocity_rad_s','angular_velocities_rad_s')]}
            contact_projection=None
            def check_vacuum_patch():
                nonlocal contact_projection
                from depallet.manipulation.vacuum_patch import validate_vacuum_patch
                pp,pq=flange.get_world_poses();actual=measure_box()
                proof=validate_vacuum_patch(transform(pp.numpy().reshape(-1),pq.numpy().reshape(-1)),assembly,
                    transform(actual['position_m'],actual['quaternion_wxyz']),boxes[execution_plan['box_id']]['dimensions_m'])
                if proof['passed'] and a.contact_anchor_policy=='project_measured_surface':
                    flange_q=pq.numpy().reshape(-1).astype(float);flange_q/=np.linalg.norm(flange_q)
                    box_q=np.asarray(actual['quaternion_wxyz'],float);box_q/=np.linalg.norm(box_q)
                    T_flange=transform(pp.numpy().reshape(-1),flange_q)
                    T_box=transform(actual['position_m'],box_q)
                    if contact_projection is None:
                        from depallet.manipulation.contact_anchor import project_contact_anchor
                        if gripper_info['constraint_count']!=1 or stepper.last_evidence.get('detached') is not True:
                            raise RuntimeError('Contact projection requires one open aggregate attachment')
                        nominal=np.mean(np.asarray(assembly['pads_flange_m']),axis=0).tolist()
                        contact_projection=project_contact_anchor(T_flange,T_box,nominal,boxes[execution_plan['box_id']]['dimensions_m'])
                        anchor=contact_projection['new_anchor_flange_m']
                        joint=UsdPhysics.Joint(stage.GetPrimAtPath(gripper_info['joint_paths'][0]))
                        joint_prim=joint.GetPrim()
                        clearance=joint_prim.GetAttribute('isaac:clearanceOffset').Get()
                        axis=str(joint_prim.GetAttribute('isaac:forwardAxis').Get()).lower()
                        rotation=joint.GetLocalRot0Attr().Get()
                        if (clearance!=0. or axis!='z' or abs(float(rotation.GetReal())-1.)>1e-8
                                or np.linalg.norm(list(rotation.GetImaginary()))>1e-8):
                            raise RuntimeError('Contact projection native formula assumptions differ from authored joint')
                        joint.GetLocalPos0Attr().Set(Gf.Vec3f(*anchor))
                        readback=list(joint.GetLocalPos0Attr().Get())
                        if not np.allclose(readback,anchor,rtol=0.,atol=2e-8):raise RuntimeError('Contact anchor USD readback differs')
                        contact_projection['calculation_only_receipt']=dict(contact_projection)
                        contact_projection.pop('execution_authorized',None)
                        contact_projection.update(joint_state_changed=True,geometry_computed_on_cpu=True,
                            scope='open D6 local anchor authored; original CAD/TCP/actor poses unchanged; native joint frame not measured',
                            actual_usd_readback_flange_m=readback,
                            nominal_pad_patch_before_projection=proof,measurement_source='isaac_runtime_simulation_oracle',
                            simulation_time_s=snapshot()['sim_time'],native_anchor_readback_available=False,
                            measurement_quaternions_normalized_for_SE3=True,joint_local_anchor_authored=True,
                            T_world_flange_before_anchor_change=T_flange.tolist(),T_world_box_before_anchor_change=T_box.tolist())
                        write_json(cycle_output/'contact-anchor-projection.json',contact_projection)
                        proof['requires_physics_update_before_close']=True
                    else:
                        joint=UsdPhysics.Joint(stage.GetPrimAtPath(gripper_info['joint_paths'][0]))
                        anchor=np.asarray(joint.GetLocalPos0Attr().Get())
                        rotation=joint.GetLocalRot0Attr().Get()
                        if (stepper.last_evidence.get('detached') is not True
                                or not np.allclose(anchor,contact_projection['new_anchor_flange_m'],rtol=0.,atol=2e-8)
                                or joint.GetPrim().GetAttribute('isaac:clearanceOffset').Get()!=0.
                                or str(joint.GetPrim().GetAttribute('isaac:forwardAxis').Get()).lower()!='z'
                                or abs(float(rotation.GetReal())-1.)>1e-8
                                or np.linalg.norm(list(rotation.GetImaginary()))>1e-8):
                            raise RuntimeError('Contact preparation assumptions changed after physics update')
                        origin=np.linalg.inv(T_box)@T_flange@np.r_[anchor,1.]
                        ray=T_box[:3,:3].T@T_flange[:3,2]
                        distance=(boxes[execution_plan['box_id']]['dimensions_m'][2]/2-origin[2])/ray[2]
                        post_update={'passed':bool(ray[2]<-.9986 and 0.<=distance<=.0001),
                            'sim_time_s':snapshot()['sim_time'],'ray_distance_m':float(distance),
                            'T_world_flange_measured':T_flange.tolist(),'T_world_box_measured':T_box.tolist(),
                            'nominal_anchor_world_m':(T_flange@np.r_[np.mean(np.asarray(assembly['pads_flange_m']),axis=0),1.])[:3].tolist(),
                            'authored_anchor_flange_m':anchor.tolist(),'native_anchor_readback_available':False,
                            'measurement_source':'isaac_runtime_simulation_oracle','allowed_ray_distance_m':[0.,.0001]}
                        write_json(cycle_output/'contact-anchor-post-update.json',post_update)
                        if not post_update['passed']:
                            raise RuntimeError('Projected actual contact ray left 0..0.1mm gate: '+str(distance))
                        extended=dict(assembly)
                        delta=anchor-np.mean(np.asarray(assembly['pads_flange_m']),axis=0)
                        extended['pads_flange_m']=(np.asarray(assembly['pads_flange_m'])+delta).tolist()
                        extended_patch=validate_vacuum_patch(T_flange,extended,T_box,boxes[execution_plan['box_id']]['dimensions_m'])
                        if not extended_patch['passed']:raise RuntimeError('Extended nominal pad patch failed')
                        contact_projection.update(verified_after_physics_update=True,ray_distance_before_close_m=float(distance),
                            extended_pad_patch=extended_patch,actual_pad_extension_measured=False,
                            extended_pad_patch_is_strict_nonpenetration_check=False,
                            close_geometry_gate_passed=True,full_pad_patch_verified=True)
                        write_json(cycle_output/'contact-anchor-projection.json',contact_projection)
                        proof['contact_anchor_projection_verified']=True
                write_json(cycle_output/'pre-grasp-patch.json',proof)
                write_json(cycle_output/'pre-grasp-joint-schema.json',sample_attachment_joint_schema(stage,gripper_info))
                return proof
            def measure_suction_tcp():
                pp,qq=flange.get_world_poses()
                pose=pose_from_transform(transform(pp.numpy().reshape(-1),qq.numpy().reshape(-1))@np.asarray(assembly['T_flange_tcp']))
                return {'position_m':pose[:3],'quaternion_wxyz':pose[3:]}
            stepper=SingleBoxStepper(robot,stage,gripper_info,measure_box,box_path=paths[execution_plan['box_id']],box_id=execution_plan['box_id'],
                goal_position_m=execution_plan['goal_position_m'],goal_quaternion_wxyz=execution_plan['goal_quaternion_wxyz'],
                limits=model['limits_rad'],max_velocities=execution_velocity_limits,measurement_source='isaac_runtime',
                gravity_compensation=True,max_efforts_nm=model['max_efforts_nm'],contact_escape=True,pre_grasp_check=check_vacuum_patch,
                preclose_stability=a.contact_anchor_policy=='project_measured_surface',measure_tool_pose=measure_suction_tcp)
            def measured_base_pose():
                pos,quat=robot.get_world_poses()
                return pos.numpy().reshape(-1).tolist()+quat.numpy().reshape(-1).tolist()
            def measure_payload_request(state):
                # Planning is synchronous: physics does not advance while CPU/GPU planners run.
                pos,quat=flange.get_world_poses()
                T_world_flange=transform(pos.numpy().reshape(-1),quat.numpy().reshape(-1))
                tcp_pose=pose_from_transform(T_world_flange@np.asarray(assembly['T_flange_tcp']))
                q=robot.get_dof_positions().numpy().reshape(-1).tolist();v=robot.get_dof_velocities().numpy().reshape(-1).tolist()
                base_pose=measured_base_pose()
                from depallet.planning.multi_transfer_planning import make_payload_request
                request,evidence=make_payload_request(execution_plan,state,box_ids,q,v,base_pose,tcp_pose,attachment_confirmed=stepper.grasp_confirmed,departure_completed=stepper.escape_confirmed)
                return request,evidence,q,v,base_pose,tcp_pose
            def verify_frozen(state,q,v,base_pose,tcp_pose):
                after=snapshot();after_pos,after_quat=flange.get_world_poses()
                after_tcp=pose_from_transform(transform(after_pos.numpy().reshape(-1),after_quat.numpy().reshape(-1))@np.asarray(assembly['T_flange_tcp']))
                return frozen_state_check(state,after,q,robot.get_dof_positions().numpy().reshape(-1),v,robot.get_dof_velocities().numpy().reshape(-1),
                    before_base=base_pose,after_base=measured_base_pose(),before_tcp=tcp_pose,after_tcp=after_tcp)
            def plan_measured_escape(state):
                nonlocal attachment_reference,actual_escape_watchdog
                from depallet.motion.contact_escape import export_escape
                from depallet.validation.source_stack_integrity import validate_remaining_source_stack
                write_json(cycle_output/'post-grasp-joint-schema.json',sample_attachment_joint_schema(stage,gripper_info))
                support_check=validate_remaining_source_stack(spec,box_ids,state,source_pose=source_pose,excluded_ids=completed+[execution_plan['box_id']])
                write_json(cycle_output/'pre-escape-source-integrity.json',support_check)
                task_check=whole_scene_check(state,completed,execution_plan['box_id'])
                write_json(cycle_output/'pre-escape-task-gate.json',task_check)
                if not support_check['passed'] or not task_check['passed']:raise RuntimeError('Source or committed goal support failed before contact escape')
                request,evidence,q,v,base_pose,tcp_pose=measure_payload_request(state)
                request['contact_escape_policy']={'numerical_support_overlap_m':.0001}
                from depallet.motion.actual_contact_escape import ActualContactEscapeWatchdog
                T_world_base=transform(base_pose[:3],base_pose[3:])
                static_obstacles={key:{'dimensions_m':item['dims'],
                    'T_world_object':(T_world_base@transform(item['pose'][:3],item['pose'][3:])).tolist()}
                    for key,item in request['scene']['cuboid'].items() if key not in box_ids}
                actual_escape_watchdog=ActualContactEscapeWatchdog(spec,box_ids,state,
                    target_id=execution_plan['box_id'],support_id=boxes[execution_plan['box_id']]['support_id'],
                    static_obstacles=static_obstacles,measurement_source='isaac_runtime',
                    initial_overlap_allowance_m=.0001,jitter_allowance_m=.0001,support_geometry_policy='obb_sat')
                write_json(cycle_output/'actual-contact-escape-result.json',actual_escape_watchdog.result())
                with (cycle_output/'actual-contact-escape-samples.jsonl').open('x') as log:
                    log.write(json.dumps(actual_escape_watchdog.result(),allow_nan=False)+'\n')
                directory=cycle_output/'contact-escape-plan';directory.mkdir()
                request_file=directory/'request.json';write_json(request_file,request)
                write_json(directory/'measured-attachment.json',evidence)
                try:
                    export_escape(request,directory,support_box_id=boxes[execution_plan['box_id']]['support_id'],lift_m=.08,
                        numerical_support_overlap_m=.0001)
                finally:
                    frozen=verify_frozen(state,q,v,base_pose,tcp_pose)
                    write_json(directory/'pause-verification.json',frozen)
                if not frozen['passed']:raise RuntimeError('Physical state changed during contact escape planning')
                trajectory=Trajectory.load(directory/'trajectory.npz',directory/'result.json',model['limits_rad'],execution_velocity_limits,request_file)
                attachment_reference=evidence['T_tcp_box_measured']
                stepper.start_escape(trajectory,state['sim_time'])
            def plan_measured_transport(state):
                nonlocal attachment_reference,release_gap_m
                if execution_plan.get('observation_only_placeholder'):
                    raise RuntimeError('Observation placeholder cannot authorize payload transport')
                request,evidence,q,v,base_pose,tcp_pose=measure_payload_request(state)
                directory=cycle_output/'confirmed-payload-plan';directory.mkdir()
                request_file=directory/'request.json';write_json(request_file,request);write_json(directory/'measured-attachment.json',evidence)
                child_env=os.environ.copy();child_env['ISAAC_P0_OUTPUT']=str(directory)
                if child_env.get('ISAAC_P0_GUARDED')!='1':raise RuntimeError('Payload worker requires inherited active guard')
                remaining=wall_budget-30.-(time.monotonic()-started)
                if remaining<15:raise TimeoutError('Insufficient remaining guarded wall time for payload planner')
                outcome=None;attempts=[];retry_preparation=None
                selection_path=cycle_output/'confirmed-payload-plan-selection.json'
                try:
                    with (directory/'worker.log').open('x') as log:
                        outcome=run_curobo_child(request_file,directory,child_env,min(180.,remaining),log)
                    first_seed=request.get('planner_random_seed',42)
                    attempts.append({'attempt':1,'directory':str(directory),'request_path':str(request_file),
                        'planner_random_seed':first_seed,'outcome':copy.deepcopy(outcome)})
                    result_path=directory/'result.json'
                    result=json.loads(result_path.read_text()) if result_path.is_file() else {}
                    if result and result.get('planner_random_seed')!=first_seed:
                        raise RuntimeError('Payload planner result seed differs from attempt 1 request')
                    remaining=wall_budget-30.-(time.monotonic()-started)
                    from depallet.planning.motion_retry import create_payload_retry_attempt,retryable_planning_failure
                    if retryable_planning_failure(outcome,result,remaining):
                        # Preserve attempt 1 exactly; attempt 2 must be a fresh sibling output.
                        directory,retry_preparation=create_payload_retry_attempt(directory)
                        request_file=directory/'request.json'
                        child_env=dict(child_env,ISAAC_P0_OUTPUT=str(directory))
                        with (directory/'worker.log').open('x') as log:
                            outcome=run_curobo_child(request_file,directory,child_env,min(180.,remaining),log)
                        attempts.append({'attempt':2,'directory':str(directory),'request_path':str(request_file),
                            'planner_random_seed':retry_preparation['retry_planner_random_seed'],
                            'outcome':copy.deepcopy(outcome)})
                finally:
                    frozen=verify_frozen(state,q,v,base_pose,tcp_pose)
                    frozen.update(subprocess=outcome,inherited_guard=True,
                        planner_attempt_directory=str(directory),planner_request_path=str(request_file))
                    write_json(directory/'pause-verification.json',frozen)
                    attempt_receipt={'schema':'depallet.payload_planning_attempt_selection.v1',
                        'attempts_limit':2,'retry_performed':retry_preparation is not None,
                        'attempts':attempts,'active_attempt_path':str(directory),
                        'active_request_path':str(request_file),'retry_preparation':retry_preparation,
                        'physical_state_frozen_during_planning':frozen.get('passed') is True,
                        'selected_for_downstream':False}
                    write_json(selection_path,attempt_receipt)
                if not frozen['passed']:raise RuntimeError('Physical state changed while planning the confirmed payload: '+repr(frozen['failed_components']))
                if outcome['returncode']!=0 or outcome['timed_out'] or not outcome['child_exit_verified']:raise RuntimeError('Confirmed-payload cuRobo planning failed or timed out; see '+str(directory/'worker.log'))
                active_result=json.loads((directory/'result.json').read_text())
                expected_seed=attempts[-1]['planner_random_seed']
                if active_result.get('planner_random_seed')!=expected_seed:
                    raise RuntimeError('Selected payload planner result seed differs from active request')
                trajectory=Trajectory.load(directory/'trajectory.npz',directory/'result.json',model['limits_rad'],execution_velocity_limits,request_file)
                write_json(directory/'execution-upright-revalidation.json',trajectory.result['execution_upright_revalidation'])
                attempt_receipt.update(selected_for_downstream=True,
                    selected_attempt_path=str(directory),selected_request_path=str(request_file),
                    selected_trajectory_path=str(directory/'trajectory.npz'),
                    selected_result_path=str(directory/'result.json'))
                write_json(selection_path,attempt_receipt)
                attachment_reference=evidence['T_tcp_box_measured']
                release_gap_m=float(evidence['release_above_physical_goal_m'])
                write_json(cycle_output/'prerelease-pose-policy.json',{
                    'schema':'depallet.prerelease_pose_policy.v1',
                    'physical_goal_position_m':execution_plan['goal_position_m'],
                    'goal_quaternion_wxyz':execution_plan['goal_quaternion_wxyz'],
                    'release_above_physical_goal_m':release_gap_m,
                    'expected_release_box_position_m':(np.asarray(execution_plan['goal_position_m'])+[0.,0.,release_gap_m]).tolist(),
                    'thresholds':{'xy_m':.00075,'z_m':.004,'world_tilt_rad':.001,'yaw_rad':.003},
                    'checked_payload_request':str(request_file),
                    'checked_payload_request_sha256':hashlib.sha256(request_file.read_bytes()).hexdigest(),
                    'landing_pose_certified':False})
                stepper.start_transport(trajectory,state['sim_time'])
        from depallet.validation.task_runtime_checks import validate_task_state,commit_transfer,module_progress,measured_upright_check,measured_path_progress
        def whole_scene_check(state,committed,active=None):
            return validate_task_state(spec,packing,box_ids,state,source_pose=source_pose,goal_pose=goal_pose,
                completed_ids=committed,active_id=active)
        def plan_current_approach(state, *, already_inspected=False):
            nonlocal execution_plan,stepper,target_index,cycle_output,contact_projection,actual_escape_watchdog
            nonlocal attachment_reference,attachment_rigidity_peak,upright_peak_rad,operation_contacts,last_execution_state,execution_active,done_frame_counts,release_gap_m,payload_gravity_evaluator
            from depallet.planning.multi_transfer_planning import build_cycle_plan,validate_current_request
            name=order[len(completed)];target_index=box_ids.index(name)
            if completed and not already_inspected:
                if stepper.surface.get_gripped_objects(stepper.gripper) or stepper.surface.get_gripper_status(stepper.gripper)!=stepper.open_status:
                    raise RuntimeError('Next cycle requires measured Open and no attached objects')
                cycle_output=output/'transfers'/(f'{len(completed)+1:02d}_'+name);cycle_output.mkdir()
                stepper=None  # The previous committed controller must never become this cycle's failure evidence.
            before=whole_scene_check(state,completed)
            write_json(cycle_output/'before-cycle-state.json',state);write_json(cycle_output/'before-cycle-gate.json',before)
            if not before['passed']:raise RuntimeError('Whole-scene support/goal gate before next approach failed')
            # Update only the already Open joint material model and nominal local anchor.
            # No actor or joint-angle resets; native body1 remains managed by Surface Gripper.
            from depallet.manipulation.vacuum_compliance import build_vacuum_compliance
            box=boxes[name]
            compliance=build_vacuum_compliance(box['physical']['mass_kg'],box['dimensions_m'],box['dimensions_m'][2]/2+.004,
                physics_dt_s=1/a.physics_hz,damping_ratio=a.vacuum_damping_ratio,lateral_compliance_m=a.vacuum_lateral_compliance_m)
            for path in gripper_info['joint_paths']:
                joint=UsdPhysics.Joint(stage.GetPrimAtPath(path))
                joint.GetLocalPos0Attr().Set(Gf.Vec3f(*np.mean(np.asarray(assembly['pads_flange_m']),axis=0)))
                for axis,params in compliance['axes'].items():
                    limit=UsdPhysics.LimitAPI.Apply(joint.GetPrim(),axis)
                    limit.CreateLowAttr(params['low']);limit.CreateHighAttr(params['high'])
                    drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),axis)
                    drive.CreateStiffnessAttr(params['stiffness_usd']);drive.CreateDampingAttr(params['damping_usd'])
            write_json(cycle_output/'vacuum-compliance.json',compliance)
            contact_projection=None;actual_escape_watchdog=None;attachment_reference=None;release_gap_m=None;payload_gravity_evaluator=None
            attachment_rigidity_peak={'position_error_m':0.,'orientation_error_rad':0.}
            upright_peak_rad=0.
            operation_contacts={'classes':{},'phase_counts':{},'unexpected':{},'runtime_self_collision_certified':False}
            q=robot.get_dof_positions().numpy().reshape(-1).tolist();v=robot.get_dof_velocities().numpy().reshape(-1).tolist()
            base_pose=measured_base_pose();tcp=measure_suction_tcp();tcp_pose=tcp['position_m']+tcp['quaternion_wxyz']
            approach_placement,approach_packing=placements[name],packing
            if a.inspection_ai=='apply_plan' and not already_inspected:
                from depallet.observation.survey_planning import observation_planning_inputs
                approach_placement,approach_packing=observation_planning_inputs(spec,placements[name],packing)
            execution_plan,request=build_cycle_plan(spec=spec,box_ids=box_ids,state=state,measured_q=q,measured_v=v,
                base_pose=base_pose,source_pose=source_pose,goal_pose=goal_pose,robot_config=robot_config,
                assembly_manifest=assembly_path,source_scene_run=output,box_id=name,placement=approach_placement,
                gripper_detached=True,completed_ids=completed,
                retreat_tcp_world_pose=tcp_pose if completed and not already_inspected else None,
                resolve_goal_on_measured_support=True,goal_packing=approach_packing,motion_profile=a.motion_profile,
                observation_only=bool(approach_placement.get('observation_only_placeholder')))
            execution_plan['observation_only_placeholder']=bool(approach_placement.get('observation_only_placeholder'))
            if observed_packet is not None and (a.inspection_ai!='apply_plan' or already_inspected):
                from depallet.integration.modular_pick_bridge import apply_surface
                execution_plan,request,bridge_receipt=apply_surface(request,execution_plan,observed_packet)
                write_json(cycle_output/'observed-surface-bridge.json',bridge_receipt)
            if a.inspection_view!='off' and not already_inspected:
                from depallet.observation.wrist_inspection import inspection_request
                request=inspection_request(request,assembly['T_flange_tcp'],policy=a.inspection_view,survey_candidate_id=a.survey_candidate)
                execution_plan['approach_request_data']=copy.deepcopy(request)
                execution_plan['inspection_view']=copy.deepcopy(request['inspection_view'])
            execution_plan.update(actual_contact_escape_policy='obb_sat',contact_anchor_policy=a.contact_anchor_policy,preclose_stability_required=True,prerelease_stability_required=True,
                prerelease_payload_compensation_required=True,prerelease_pose_check_required=True)
            execution_plan['payload_cover_profile']=a.payload_cover_profile
            execution_plan.update(transport_overhead_m=a.transport_overhead_m,payload_orientation_policy={'schema':'depallet.payload_upright.v1',
                'maximum_nominal_tilt_rad':.02,'maximum_actual_tilt_rad':.03})
            if a.cutamp_plan or already_inspected:execution_plan['payload_orientation_policy']['allow_transit_yaw']=True
            directory=cycle_output/('approach-plan-ai' if already_inspected else 'approach-plan');directory.mkdir();request_file=directory/'request.json';write_json(request_file,request)
            execution_plan.update(approach_run=str(directory),request_path=str(request_file))
            write_json(cycle_output/'execution-plan.json',execution_plan)
            proof=validate_current_request(request,spec=spec,box_ids=box_ids,state=state,base_pose=base_pose,
                source_pose=source_pose,goal_pose=goal_pose)
            write_json(cycle_output/'approach-scene-check.json',proof)
            child_env=os.environ.copy();child_env['ISAAC_P0_OUTPUT']=str(directory)
            remaining=wall_budget-30.-(time.monotonic()-started)
            if remaining<15:raise TimeoutError('Insufficient bounded time for next approach planner')
            with (directory/'worker.log').open('x') as log:
                outcome=run_curobo_child(request_file,directory,child_env,min(180.,remaining),log)
            frozen=verify_frozen(state,q,v,base_pose,tcp_pose);frozen['subprocess']=outcome
            write_json(directory/'pause-verification.json',frozen)
            if not frozen['passed'] or outcome['returncode']!=0 or outcome['timed_out'] or not outcome['child_exit_verified']:
                raise RuntimeError('Fresh full-world approach plan failed; see current transfer approach-plan')
            if any(g.get('id')=='contact' for g in request.get('goals',[])):
                from depallet.validation.pregrasp_exit_preview import preview as preview_exit
                with np.load(directory/'trajectory.npz') as planned_approach:
                    nominal_contact_q=planned_approach['position_rad'][-1]
                preview_request=copy.deepcopy(request)
                preview_request['completed_box_ids']=list(completed)
                write_json(cycle_output/'pregrasp-exit-preview.json',preview_exit(preview_request,nominal_contact_q))
            from depallet.motion.payload_gravity import PayloadGravityEvaluator
            payload_gravity_evaluator=PayloadGravityEvaluator(robot_config,base_pose,
                box['physical']['mass_kg'],box['dimensions_m'],box['physical']['center_of_mass_local_m'],
                model['max_efforts_nm'],include_articulation_gravity=True)
            def measured_payload_compensation(q_actual,box_actual):
                tcp_actual=measure_suction_tcp()
                return payload_gravity_evaluator.evaluate(q_actual,box_actual,
                    tcp_world_pose_wxyz=tcp_actual['position_m']+tcp_actual['quaternion_wxyz'])
            def measured_prerelease_pose(q_actual,box_actual):
                from depallet.validation.task_runtime_checks import measured_release_pose_check
                if release_gap_m is None:raise RuntimeError('Checked transport release pose missing')
                return measured_release_pose_check(box_actual,execution_plan['goal_position_m'],
                    execution_plan['goal_quaternion_wxyz'],release_gap_m,measurement_source='isaac_runtime')
            stepper=SingleBoxStepper(robot,stage,gripper_info,measure_box,box_path=paths[name],box_id=name,
                goal_position_m=execution_plan['goal_position_m'],goal_quaternion_wxyz=execution_plan['goal_quaternion_wxyz'],
                limits=model['limits_rad'],max_velocities=execution_velocity_limits,measurement_source='isaac_runtime',
                gravity_compensation=True,max_efforts_nm=model['max_efforts_nm'],contact_escape=True,pre_grasp_check=check_vacuum_patch,
                preclose_stability=True,prerelease_stability=True,measure_tool_pose=measure_suction_tcp,
                prerelease_payload_compensation=measured_payload_compensation,prerelease_pose_check=measured_prerelease_pose)
            trajectory=Trajectory.load(directory/'trajectory.npz',directory/'result.json',model['limits_rad'],execution_velocity_limits,request_file)
            if rollout is not None:
                stepper.surface=rollout.wrap_surface(stepper.surface)
                rollout.skill('INSPECTION_MOVE' if a.inspection_view!='off' and not already_inspected else 'APPROACH',name,'Move the selected exposed carton from the source pallet to its reserved target-pallet position.')
            inspection_index=None
            if a.inspection_view!='off' and not already_inspected:
                from depallet.observation.wrist_inspection import inspection_boundary
                inspection_index=inspection_boundary(trajectory,request)
            stepper.start_approach(trajectory,state['sim_time'],inspection_index=inspection_index);execution_active=True
            last_execution_state=None;done_frame_counts=None
            print('TASK_CYCLE',len(completed)+1,'OF',len(order),name,flush=True)
        inspection_attempts={}
        def inspect_with_ai(state):
            nonlocal planning,packing,order,placements,observed_packet,cycle_output
            from depallet.observation.inspection_ai import run_inspection_proposal
            from depallet.planning.multi_transfer_planning import build_cycle_plan
            q=robot.get_dof_positions().numpy().reshape(-1).tolist()
            v=robot.get_dof_velocities().numpy().reshape(-1).tolist()
            base_pose=measured_base_pose();tcp=measure_suction_tcp()
            tcp_pose=tcp['position_m']+tcp['quaternion_wxyz'];name=execution_plan['box_id']
            observation_placement,observation_packing=placements[name],packing
            if a.inspection_ai=='apply_plan':
                from depallet.observation.survey_planning import observation_planning_inputs
                observation_placement,observation_packing=observation_planning_inputs(spec,placements[name],packing)
            _,request=build_cycle_plan(spec=spec,box_ids=box_ids,state=state,measured_q=q,measured_v=v,
                base_pose=base_pose,source_pose=source_pose,goal_pose=goal_pose,robot_config=robot_config,
                assembly_manifest=assembly_path,source_scene_run=output,box_id=name,placement=observation_placement,
                gripper_detached=True,completed_ids=completed,retreat_tcp_world_pose=tcp_pose if completed else None,
                resolve_goal_on_measured_support=True,goal_packing=observation_packing,motion_profile=a.motion_profile,
                observation_only=bool(observation_placement.get('observation_only_placeholder')))
            request['completed_box_ids']=list(completed)
            attempt=inspection_attempts.get(len(completed),0)+1
            inspection_attempts[len(completed)]=attempt
            stem='inspection-ai' if attempt==1 else f'inspection-ai-{attempt:02d}'
            request_path=cycle_output/(stem+'-request.json');write_json(request_path,request)
            outcome=None
            try:
                outcome=run_inspection_proposal(output,request_path,cycle_output/stem,
                    wall_budget-30.-(time.monotonic()-started),os.environ,
                    center_goal_init=False)
            finally:
                frozen=verify_frozen(state,q,v,base_pose,tcp_pose)
                write_json(cycle_output/(stem+'-pause-verification.json'),frozen)
            if not frozen['passed']:raise RuntimeError('Scene moved during inspection AI proposal')
            write_json(cycle_output/'inspection-ai-result.json',outcome)
            write_json(cycle_output/(stem+'-attempt-result.json'),outcome)
            if outcome['pipeline_status']=='reobserve_required' and attempt<3:
                return outcome
            if not outcome['complete_chain_ran']:raise RuntimeError('Inspection AI chain failed; proposal not applied')
            if a.inspection_ai=='apply_plan':
                from depallet.integration.cutamp_execution_bridge import load_execution
                ai_root=Path(outcome['output'])
                # One newly observed target per hold; the next cycle must observe again.
                updated,packet,horizon=load_execution(Path(outcome.get('task_plan') or ai_root/'task-plan/task-plan.json'),
                    Path(outcome.get('task_input') or ai_root/'task-input'),
                    scenario,planning,runroot,1,completed_ids=completed,current_request=request)
                if not horizon.get('current_world_check',{}).get('passed'):
                    raise RuntimeError('Online plan lacks a current full-world check')
                survey_output=cycle_output
                write_json(survey_output/'survey-execution-plan.json',execution_plan)
                selected=horizon['order'][0]
                selected_output=output/'transfers'/(f'{len(completed)+1:02d}_'+selected)
                if selected_output!=cycle_output:
                    selected_output.mkdir()
                    cycle_output=selected_output
                write_json(cycle_output/'observation-handoff.json',{
                    'survey_cycle_directory':str(survey_output),'ai_output':str(ai_root),
                    'request_path':str(request_path),'frozen_state_path':str(survey_output/(stem+'-pause-verification.json')),
                    'selected_box_id':selected,'selection_changed':selected!=execution_plan['box_id'],
                    'world_check':horizon['current_world_check']})
                planning,observed_packet=updated,packet
                # Preserve the authored baseline and record every adopted online revision separately.
                write_json(cycle_output/'adopted-task-plan.json',planning)
                write_json(output/'latest-online-task-plan.json',planning)
                packing=planning['packing'];order=planning['order']
                placements={x['box_id']:x for x in packing['placements']}
                write_json(cycle_output/'cutamp-execution-horizon.json',horizon)
                write_json(cycle_output/'modular-pipeline.json',packet)
                plan_current_approach(state,already_inspected=True)
                outcome.update(mode='apply_plan',proposal_applied_to_robot=True,
                    selected_box_id=execution_plan['box_id'],fresh_motion_plan=execution_plan['approach_run'])
                write_json(cycle_output/'inspection-ai-result.json',outcome)
            return outcome

        def complete_cycle(state):
            nonlocal completed
            result=stepper.result()
            result.update(perception_source=execution_plan['perception_source'],end_to_end_perception_pipeline_validated=False,
                attachment_rigidity_peak=attachment_rigidity_peak,attachment_uncertainty=execution_plan['attachment_uncertainty'],
                actual_payload_upright_passed=upright_peak_rad<=.03,maximum_actual_payload_world_tilt_rad=upright_peak_rad,
                actual_contact_escape_passed=bool(actual_escape_watchdog and actual_escape_watchdog.result()['passed'] and actual_escape_watchdog.result()['endpoint_check']),
                external_contact_check_passed=not operation_contacts['unexpected'],
                contact_anchor_policy=a.contact_anchor_policy,joint_anchor_reconfigured_before_close=bool(contact_projection),
                contact_anchor_extension_is_assumed=True,physical_parameters_measured=False)
            check=whole_scene_check(state,completed+[execution_plan['box_id']])
            write_json(cycle_output/'after-cycle-state.json',state);write_json(cycle_output/'after-cycle-gate.json',check)
            write_json(cycle_output/'final-source-integrity.json',check['remaining_source'])
            write_json(cycle_output/'contact-report.json',operation_contacts)
            result['passed']=bool(result['passed'] and result['actual_contact_escape_passed'] and result['external_contact_check_passed'] and check['passed'])
            result['physics_grasp_validated']=result['passed'];result['remaining_source_support_passed']=check['remaining_source']['passed']
            result['placed_goal_support_passed']=check['placed_goal']['passed']
            write_json(cycle_output/'execution-result.json',result)
            if not result['passed']:
                raise RuntimeError('Cycle completion gate failed: '+repr({
                    'box_id':execution_plan['box_id'],
                    'source_failed_ids':check['remaining_source'].get('failed_box_ids',[]),
                    'goal_failed_ids':check['placed_goal'].get('failed_box_ids',[]),
                    'goal_com_passed':check['placed_goal_com']['passed'],
                    'actual_contact_escape_passed':result['actual_contact_escape_passed'],
                    'external_contact_check_passed':result['external_contact_check_passed']}))
            completed=commit_transfer(completed,order,result,check,box_id=execution_plan['box_id'])
            transfer_results.append({'box_id':execution_plan['box_id'],'passed':True,'cycle_directory':str(cycle_output),
                'execution_sha256':hashlib.sha256((cycle_output/'execution-result.json').read_bytes()).hexdigest(),
                'snapshot_sha256':hashlib.sha256((cycle_output/'after-cycle-state.json').read_bytes()).hexdigest(),
                'sim_time_s':state['sim_time'],'physics_step':state['physics_step']})
            write_json(output/'task-progress.json',{'completed_ids':completed,'total_boxes':len(order),
                'task_complete':len(completed)==len(order),'transfers':transfer_results,'resume_authorized':False,
                'scores':module_progress('IDLE',len(completed),len(order))})
            timing.phase('COMMITTED',execution_plan['box_id'],state['sim_time'])
            print('TASK_COMMIT',len(completed),execution_plan['box_id'],flush=True)
        if a.record_rollout:
            from depallet.runtime.rollout_trace import RolloutTrace
            from isaacsim.core.simulation_manager import SimulationEvent
            def rollout_clock():
                return float(SimulationManager.get_simulation_time()),int(SimulationManager.get_num_physics_steps())
            def rollout_context():
                return {'phase':stepper.state if stepper is not None else 'INITIALIZING',
                    'transfer_id':rollout.transfer_id if rollout is not None else None}
            rollout=RolloutTrace(output,clock=rollout_clock,joint_names=tuple(robot.dof_names),context=rollout_context,camera_profile='overview_source_v1' if a.camera_rig=='legacy_v1' else a.camera_rig)
            robot=rollout.wrap_robot(robot)
            stepper.robot=robot
            stepper.surface=rollout.wrap_surface(stepper.surface)
            rollout.skill('INITIALIZING',None,'Initialization and camera warmup; excluded from collection hours.')
        if a.record_rollout or a.camera_rig=='overhead_wrist_v2':
            from isaacsim.core.simulation_manager import SimulationEvent
            def capture_rollout_physics(dt=None,context=None):
                nonlocal rollout_error
                if rollout_error is not None:return
                try:
                    rt=float(SimulationManager.get_simulation_time());rs=int(SimulationManager.get_num_physics_steps())
                    pp,pq=flange.get_world_poses();fp=pp.numpy().reshape(-1).tolist();fq=pq.numpy().reshape(-1).tolist()
                    camera_pose_history.append({'sim_time':rt,'physics_step':rs,'T_world_flange':transform(fp,fq)})
                    if rollout is not None:
                        tcp=measure_suction_tcp()
                        rollout.physics(sim_time=rt,physics_step=rs,
                            q_rad=robot.get_dof_positions().numpy().reshape(-1).tolist(),
                            dq_rad_s=robot.get_dof_velocities().numpy().reshape(-1).tolist(),
                            tcp_pose_wxyz=tcp['position_m']+tcp['quaternion_wxyz'],flange_pose_wxyz=fp+fq,
                            gripper_status=stepper.surface.get_gripper_status(stepper.gripper))
                except Exception as error:
                    rollout_error=repr(error)
                    if rollout is not None:rollout.callback_error=repr(error)
            capture_rollout_physics()
            rollout_callback=SimulationManager.register_callback(capture_rollout_physics,
                event=SimulationEvent.PHYSICS_POST_STEP,order=1000)
        preferred=['source_pallet']+order[:4]
        with (output/'evaluation_ground_truth.jsonl').open('x') as gt:
            while app.is_running():
                costs.call('app_update_physics_render_callbacks',app.update);step=int(SimulationManager.get_num_physics_steps())
                if step==last_step or not SimulationManager.is_simulating():continue
                last_step=step
                if rollout_error is not None:raise RuntimeError('Physics capture failed: '+rollout_error)
                if step-physics_start>a.max_steps:raise TimeoutError('bounded physics-step limit')
                if contacts['callback_error']:raise RuntimeError(contacts['callback_error'])
                state=snapshot();history.append(state)
                if rollout is not None:rollout.check_current_physics(state['sim_time'],state['physics_step'])
                while len(history)>1 and history[1]['sim_time']<state['sim_time']-2.:history.popleft()
                joints=None if robot is None else robot.get_dof_positions().numpy().reshape(-1).tolist()
                if joints is not None and not execution_active:max_joint_error=max(max_joint_error,float(np.max(np.abs(np.asarray(joints)-model['home_positions_rad']))))
                phase='SCENE_SETTLE_OBSERVE'
                if execution_plan:
                    if operation_contacts['unexpected'] and stepper.state not in ('IDLE','FAILED'):stepper.abort('Unexpected external contact: '+repr(operation_contacts['unexpected']))
                    if stepper.state=='IDLE':
                        if initial_settle_started is None:initial_settle_started=state['sim_time']
                        if not np.isfinite(joints).all():raise RuntimeError('Nonfinite initial robot joint measurement')
                        pre_execution_settle=summarize_settle(list(history),authored_positions,authored_quaternions)
                        gate=initial_settle_decision(pre_execution_settle,elapsed_s=state['sim_time']-initial_settle_started,
                            home_error_rad=max_joint_error,robot_scene_headers=contacts['robot_scene_headers'],
                            unexpected_contacts=operation_contacts['unexpected'],timeout_s=8.)
                        pre_execution_settle.update(box_ids=box_ids,final_physics_state=state,initialization_gate=gate)
                        if state['sim_time']-initial_report_time>=.25 or gate['decision']!='WAIT':
                            write_json(output/'pre-execution-settle.json',pre_execution_settle)
                            write_json(output/'initialization-gate.json',gate)
                            robot_report['initialization_gate']=gate;write_json(output/'robot-scene.json',robot_report)
                            initial_report_time=state['sim_time']
                        if gate['decision']=='ABORT':raise RuntimeError('Initial validation failed: '+', '.join(gate['failed_conditions']))
                        if gate['decision']=='READY':
                            plan_current_approach(state)
                    elif stepper.state not in ('IDLE','FAILED'):
                        if actual_escape_watchdog is not None and stepper.state=='ESCAPE':
                            observed_escape=actual_escape_watchdog.observe(state)
                            write_json(cycle_output/'actual-contact-escape-result.json',observed_escape)
                            with (cycle_output/'actual-contact-escape-samples.jsonl').open('a') as log:
                                log.write(json.dumps(observed_escape,allow_nan=False)+'\n')
                            if not observed_escape['passed']:
                                stepper.abort('Actual contact escape failed: '+repr(observed_escape['failure_reasons']))
                        if attachment_reference is not None and stepper.state in ('ESCAPE','ESCAPED','TRANSPORT','PRE_RELEASE_SETTLE'):
                            p_tcp,q_tcp=flange.get_world_poses()
                            tcp_now=pose_from_transform(transform(p_tcp.numpy().reshape(-1),q_tcp.numpy().reshape(-1))@np.asarray(assembly['T_flange_tcp']))
                            rigidity=attachment_rigidity_check(attachment_reference,tcp_now,state['positions_m'][target_index],state['quaternions_wxyz'][target_index],
                                attachment_uncertainty=execution_plan.get('attachment_uncertainty'))
                            for key in attachment_rigidity_peak:attachment_rigidity_peak[key]=max(attachment_rigidity_peak[key],rigidity[key])
                            upright=measured_upright_check(state['quaternions_wxyz'][target_index])
                            upright_peak_rad=max(upright_peak_rad,upright['world_tilt_rad'])
                            with (cycle_output/'payload-pose-samples.jsonl').open('a') as payload_log:
                                payload_log.write(json.dumps({'sim_time_s':state['sim_time'],'physics_step':state['physics_step'],
                                    'phase':stepper.state,'box_id':execution_plan['box_id'],'box_position_m':state['positions_m'][target_index],
                                    'box_quaternion_wxyz':state['quaternions_wxyz'][target_index],'tcp_world_pose':tcp_now,
                                    'upright':upright,'attachment_rigidity':rigidity},allow_nan=False)+'\n')
                            if not upright['passed']:stepper.abort('Actual payload lost upright orientation: '+repr(upright))
                            if not rigidity['passed']:stepper.abort('Measured payload slipped relative to TCP: '+repr(rigidity))
                        result=costs.call('controller_step',stepper.step,state['sim_time'])
                        if result['requires_contact_escape']:plan_measured_escape(state)
                        elif result['requires_transport_plan']:
                            observed_escape=actual_escape_watchdog.finish() if actual_escape_watchdog is not None else {}
                            write_json(cycle_output/'actual-contact-escape-result.json',observed_escape)
                            if not (observed_escape.get('passed') is True and observed_escape.get('endpoint_check') is True and observed_escape.get('monitoring_finalized') is True):
                                stepper.abort('Actual contact escape endpoint evidence missing')
                            plan_measured_transport(state)
                    phase=stepper.state
                    if phase!=last_execution_state:
                        print('DEPALLET_EXECUTION',phase,flush=True);last_execution_state=phase
                        timing.phase(phase,execution_plan['box_id'],state['sim_time'])
                        if rollout is not None:rollout.skill(phase,rollout.transfer_id,'Execute the current transfer skill; labels come from the scripted expert.')
                        with (output/'task-events.jsonl').open('a') as event_log:
                            event_log.write(json.dumps({'box_id':execution_plan['box_id'],'phase':phase,'sim_time_s':state['sim_time'],
                                'physics_step':state['physics_step'],'scores':module_progress(phase,len(completed),len(order))},allow_nan=False)+'\n')
                        write_json(cycle_output/'execution-state.json',stepper.result())
                    if phase=='DONE' and done_frame_counts is None:done_frame_counts=[bundle['recorder'].count for bundle in cameras]
                if phase=='INSPECTION_HOLD' and inspection_frames is None:
                    from depallet.observation.wrist_inspection import InspectionFrames
                    inspection_frames=InspectionFrames(stepper.state_since)
                    survey_depth_checks=[]
                if rollout is not None and execution_active:rollout.control()
                dense_progress=None
                if stepper is not None and stepper.trajectory is not None and phase in ('INSPECTION_MOVE','APPROACH','ESCAPE','TRANSPORT'):
                    dense_progress=measured_path_progress(joints,stepper.trajectory.positions)
                    dense_progress.update(module=phase,source='actual articulation joint state projected onto current checked path')
                for bundle in cameras:
                    if bundle['recorder'].count>=a.frames:continue
                    sensor=bundle['sensor']
                    def read_rgbd():
                        rgb,rinfo=sensor.get_data('rgb');depth,dinfo=sensor.get_data('distance_to_image_plane')
                        return (None if rgb is None or depth is None else (rgb,depth)),(rinfo,dinfo)
                    read_clocks=lambda:(plain(bundle['time'].get_data()),plain(bundle['reference'].get_data()))
                    try:
                        pair,infos,stamp,retry_receipt=_read_policy_frame(
                            bundle['clock'],read_clocks,read_rgbd,state['sim_time'],None,
                            bundle.setdefault('clock_recovery',{}))
                    except _ObserverClockReadError as error:
                        receipt=dict(error.receipt,schema='depallet.rgbd_clock_reread.v1',camera_id=bundle['id'],policy_input=True)
                        write_json(output/('rgbd-clock-failure-'+bundle['id']+'.json'),receipt)
                        raise
                    if retry_receipt is not None:
                        receipt=dict(retry_receipt,schema='depallet.rgbd_clock_reread.v1',camera_id=bundle['id'],policy_input=True)
                        with (output/'rgbd-clock-rereads.jsonl').open('a') as logfile:logfile.write(json.dumps(receipt)+'\n')
                    if pair is None or stamp is None:continue
                    rgb,depth=pair;rinfo,dinfo=infos
                    captured=None;camera_pose=None
                    if a.camera_rig=='overhead_wrist_v2':
                        captured=next((entry for entry in reversed(camera_pose_history) if abs(entry['sim_time']-stamp['image_sim_time'])<1e-6),None)
                        if captured is None:
                            if bundle['recorder'].count==0:continue
                            raise RuntimeError('No exact capture-time flange state for RGB-D frame')
                        camera_pose={'T_world_camera_cv':world_camera_pose(bundle['spec'],captured['T_world_flange']).tolist(),
                            'pose_sim_time':stamp['image_sim_time'],'pose_source':'post_physics_measured_flange_at_capture'}
                        if bundle['id']=='wrist':
                            pp,pq=flange.get_world_poses()
                            expected=world_camera_pose(bundle['spec'],transform(pp.numpy().reshape(-1),pq.numpy().reshape(-1)))
                            actual=np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(bundle['path'])).ComputeLocalToWorldTransform(0.)).T@np.diag([1.,-1.,-1.,1.])
                            error=float(np.max(np.abs(expected-actual)))
                            camera_follow_check['frames']+=1
                            camera_follow_check['maximum_transform_error']=max(camera_follow_check['maximum_transform_error'],error)
                            if error>1e-4:raise RuntimeError('USD wrist camera does not follow measured flange: '+str(error))
                    ri=rgb.numpy();di=depth.numpy().reshape(bundle['spec']['resolution_hw']);valid=np.isfinite(di)&(di>0)
                    if np.mean(valid)<.05 or float(ri[...,:3][valid].mean())<=5:raise RuntimeError('invalid/black rendered geometry')
                    if bundle is source and not masks_written:
                        seg,info=sensor.get_data('instance_id_segmentation')
                        if seg is None or not info.get('idToLabels'):continue
                        masks=object_masks(seg.numpy(),info['idToLabels'],all_paths)
                        available=[key for key in preferred if int(masks[key].sum())>=64]
                        if 'source_pallet' not in available or len(available)<2:raise RuntimeError('camera must cover pallet and a candidate carton')
                        folder=bundle['folder']/'masks';folder.mkdir()
                        for key in available:
                            Image.fromarray(masks[key].astype(np.uint8)*255).save(folder/f'{key}.png')
                            item={'id':key,'kind':'pallet' if key=='source_pallet' else 'box','init_mask':f'masks/{key}.png','mask_source':'simulation_oracle'}
                            if key!='source_pallet':item.update(dimensions_m=boxes[key]['dimensions_m'],dimensions_source='assumed_scene_dimensions')
                            sequence['objects'].append(item)
                        sequence['missing_requested_initial_instances']=[key for key in preferred if key not in available]
                        image=Image.fromarray(ri[...,:3]);draw=ImageDraw.Draw(image)
                        try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
                        except OSError:font=ImageFont.load_default()
                        observed=[]
                        for index,key in enumerate(paths,1):
                            yy,xx=np.nonzero(masks[key]);bbox=None
                            if len(xx):
                                bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]
                                draw.rectangle(bbox,outline=(240,210,60),width=2)
                                draw.text((bbox[0]+2,bbox[1]+2),str(index),font=font,fill='white',stroke_width=2,stroke_fill='black')
                            observed.append({'visual_index':index,'instance_id':key,'bbox_xyxy':bbox,'visible_pixels':len(xx)})
                        image.save(output/'vlm-layout.png')
                        write_json(output/'vlm-layout-request.json',{'image':'vlm-layout.png','image_source':'actual render with oracle ID overlay',
                            'instances':observed,'provider_executed':False,'output_is_not_a_motion_command':True,
                            'request':'Infer visible support and occlusion relationships, then top-first candidate_order by visual_index. Mark invisible/ambiguous boxes unknown. Do not invent poses, dimensions, masses or hidden objects. Return JSON with candidate_order, support_relationships, uncertainty, reasoning.'})
                        write_json(output/'initial-instance-id-mapping.json',plain(info));masks_written=True
                    row={'sim_time':stamp['image_sim_time'],'sim_time_kind':'rendered_image_simulation_time','physics_step':step,
                        'physics_snapshot_sim_time':state['sim_time'],'phase':phase,'box_position_m':state['positions_m'][target_index if execution_plan else 0],
                        'module_dense_progress':dense_progress,'completed_box_count':len(completed),'representative_box_id':execution_plan['box_id'] if execution_plan else box_ids[0],'gripped_objects':stepper.last_evidence.get('gripped_objects',[]) if stepper else [],'joints_rad':joints,'robot_present':robot is not None,
                        'robot_state_available':robot is not None,'joint_state_source':'articulation_sensor' if robot is not None else 'unavailable_in_static_render',
                        'image_state_time_alignment_verified':False,'render_reference':stamp,'annotator_info':{'rgb':plain(rinfo),'depth':plain(dinfo)}}
                    if camera_pose is not None:row['camera_pose']=camera_pose
                    index=costs.call('policy_rgbd_append',bundle['recorder'].append,ri,di,row)
                    if rollout is not None:
                        prefix='' if bundle['spec']['root']=='.' else bundle['spec']['root']+'/'
                        rollout.observation(bundle['id'],index,stamp,
                            rgb_path=prefix+f'rgb/{index:06d}.png',depth_path=prefix+f'depth/{index:06d}.npz',camera_pose=camera_pose)
                    if bundle is overview and observer is None and preview_cadence.due(stamp['image_sim_time']):
                        tmp=output/'latest.tmp.jpg';Image.fromarray(ri[...,:3]).save(tmp,quality=85);tmp.replace(output/'latest.jpg')
                        write_json(output/'live-state.json',{'camera_id':bundle['id'],'frame_id':index,'sim_time':stamp['image_sim_time'],
                            'sensor_sim_hz':camera_hz,'phase':phase,'completed_boxes':len(completed),'total_boxes':len(order)})
                    if bundle is source:
                        sequence['frames'].append({'frame_id':index,'sim_time':stamp['image_sim_time'],'rgb':f'rgb/{index:06d}.png','depth':f'depth/{index:06d}.npz'})
                        gt.write(json.dumps({'source_frame_id':index,'image_sim_time':stamp['image_sim_time'],
                            'physics_snapshot_sim_time':state['sim_time'],'image_state_time_alignment_verified':False,
                            'source':'simulator_ground_truth_evaluation_only','box_ids':box_ids,**state},allow_nan=False)+'\n');gt.flush()
                    if phase=='INSPECTION_HOLD' and inspection_frames is not None:
                        admitted=inspection_frames.append(bundle['id'],index,stamp['image_sim_time'])
                        if admitted and bundle['id']=='wrist' and a.inspection_view=='wrist_pallet_v1':
                            from depallet.observation.pallet_survey import measured_survey_coverage
                            survey_check=measured_survey_coverage(execution_plan['inspection_view'],camera_pose['T_world_camera_cv'],di,
                                scope=a.survey_scope,required_box_id=execution_plan['box_id'])
                            survey_check.update(frame_index=index,image_sim_time=stamp['image_sim_time'])
                            survey_depth_checks.append(survey_check)
                    if (index+1)%25==0:print('DEPALLET_CAPTURE',bundle['folder'].name,index+1,flush=True)
                if observer is not None:
                    def read_observer_clocks():
                        return (plain(observer['time'].get_data()),plain(observer['reference'].get_data()))
                    try:
                        rgb,info,stamp,reread=_read_monitor_frame(observer['clock'],read_observer_clocks,
                            lambda:observer['sensor'].get_data('rgb'),state['sim_time'],observer_clock_last_accepted)
                    except _ObserverClockReadError as error:
                        write_json(output/'observer-clock-rereads.json',{
                            'schema':'depallet.observer_clock_rereads.v1','policy_input':False,
                            'receipts':observer_clock_rereads+[error.receipt]})
                        raise
                    if reread is not None:
                        observer_clock_rereads.append(reread)
                        write_json(output/'observer-clock-rereads.json',{
                            'schema':'depallet.observer_clock_rereads.v1','policy_input':False,
                            'receipts':observer_clock_rereads})
                    if rgb is not None and stamp is not None:
                        observer_clock_last_accepted=stamp
                        index=costs.call('observer_video_append',observer['recorder'].append,rgb.numpy()[...,:3],{
                                'sim_time':stamp['image_sim_time'],'physics_step':step,'phase':phase,
                                'render_reference':stamp,'completed_boxes':len(completed),'policy_input':False})
                        if preview_cadence.due(stamp['image_sim_time']):
                            with costs.measure('live_preview_publish'):
                                tmp=output/'latest.tmp.jpg';Image.fromarray(rgb.numpy()[...,:3]).save(tmp,quality=85);tmp.replace(output/'latest.jpg')
                                write_json(output/'live-state.json',{'camera_id':'observer','policy_input':False,'frame_id':index,
                                    'sim_time':stamp['image_sim_time'],'sensor_sim_hz':camera_hz,'preview_hz':a.preview_hz,'phase':phase,
                                    'completed_boxes':len(completed),'total_boxes':len(order)})
                if inspection_frames is not None and phase=='INSPECTION_HOLD':
                    inspection_receipt=inspection_frames.result()
                    if inspection_receipt['ready']:
                        inspection_receipt.update(planning=execution_plan['inspection_view'],
                            state_before_resume=stepper.result(),perception_source='simulation_oracle',
                            vision_inference_executed=False,grasp_authorized_by_vision=False)
                        if a.inspection_view=='wrist_pallet_v1':
                            accepted_indices=[r['frame_index'] for r in inspection_receipt['cameras']['wrist']]
                            survey_ok=(accepted_indices==[r['frame_index'] for r in survey_depth_checks]
                                and len(survey_depth_checks)>=MIN_WHOLE_PALLET_SURVEY_FRAMES
                                and all(r['passed'] for r in survey_depth_checks))
                            inspection_receipt.update(survey_geometry_passed=survey_ok,survey_depth_checks=survey_depth_checks,
                                semantic_full_survey_validated=False)
                            write_json(cycle_output/'inspection-view.json',inspection_receipt)
                            if not survey_ok:raise RuntimeError('Whole-pallet wrist coverage failed; approach stays blocked')
                        if a.inspection_ai in ('shadow_plan','apply_plan'):
                            inspection_receipt['ai_proposal']=inspect_with_ai(state)
                            if inspection_receipt['ai_proposal']['pipeline_status']=='reobserve_required':
                                write_json(cycle_output/f'reobservation-{inspection_attempts[len(completed)]:02d}.json',inspection_receipt)
                                inspection_frames=None
                                continue  # Advance physics/cameras and capture a new stable frame batch at the held pose.
                            inspection_receipt['vision_inference_executed']=True
                            inspection_receipt['grasp_authorized_by_vision']=bool(inspection_receipt['ai_proposal']['proposal_applied_to_robot'])
                        if a.inspection_ai!='apply_plan':
                            if rollout is not None:rollout.skill('APPROACH',rollout.transfer_id,'Resume checked approach after stable RGB-D inspection; oracle controls the robot.')
                            stepper.resume_after_inspection(state['sim_time'])
                        inspection_receipt['resumed_state']=stepper.state
                        write_json(cycle_output/'inspection-view.json',inspection_receipt)
                        inspection_frames=None
                if execution_plan:
                    if done_frame_counts is not None and all(bundle['recorder'].count>=count+int(round(.2*camera_hz)) for bundle,count in zip(cameras,done_frame_counts)):
                        complete_cycle(state)
                        if len(completed)>=a.max_transfers:break
                        plan_current_approach(state)
                    if any(bundle['recorder'].count>=a.frames for bundle in cameras):raise TimeoutError('Execution recording budget reached before DONE frames')
                elif all(bundle['recorder'].count>=a.frames for bundle in cameras):break
        if not masks_written:raise RuntimeError('no initial masks exported')
        write_json(source['folder']/'sequence.json',sequence)
        final_gate=whole_scene_check(state,completed)
        task_ended=time.monotonic();task_end_sim=state['sim_time']
        prefix_passed=bool(len(completed)==a.max_transfers and final_gate['passed'] and all(x['passed'] for x in transfer_results))
        task_complete=bool(prefix_passed and len(completed)==len(order))
        execution_result={'schema':'depallet.continuous_task_execution.v1','state':'COMPLETE' if task_complete else 'PREFIX_COMPLETE',
            'passed':task_complete,'requested_prefix_passed':prefix_passed,'task_complete':task_complete,
            'requested_transfers':a.max_transfers,'total_boxes':len(order),'completed_ids':completed,
            'transfer_results':transfer_results,'final_whole_scene_gate':final_gate,'runtime_contract':runtime_contract,
            'physics_grasp_validated':prefix_passed,'perception_source':'simulation_oracle',
            'continuous_simulation':True,'scene_resets_after_initialization':0,'joint_or_object_teleport_used':False,
            'end_to_end_perception_pipeline_validated':False,'valid_sim_hours':0}
        execution_result.update(physical_task_complete=task_complete,finalized=False,rgbd_integrity_passed=False,passed=False)
        write_json(output/'physical-task-result.json',execution_result)
        settle={'passed':final_gate['passed'],'box_ids':box_ids,'final_physics_state':state,'whole_scene_gate':final_gate}
        write_json(output/'settle-report.json',settle);write_json(output/'contact-report.json',contacts)
        robot_report['maximum_home_joint_error_rad']=max_joint_error
        if robot is not None:
            pos,quat=robot.get_world_poses()
            robot_report.update(measured_base_position_m=pos.numpy().reshape(-1).tolist(),
                measured_base_quaternion_wxyz=quat.numpy().reshape(-1).tolist(),final_joints_rad=joints)
        write_json(output/'robot-scene.json',robot_report)
        metadata={'mode':'photo_scene_observation','isaac_version':'6.1.0.0','renderer':'RealTimePathTracing',
            'render_device':'GPU','physics_device':'cpu','physics_dt':1/a.physics_hz,'physics_solver_settings':physics_solver_settings,'camera_fps':camera_hz,'box_count':len(box_ids),'solver_position_iterations':32,'solver_velocity_iterations':1,'vacuum_attachment_policy':a.vacuum_policy if execution_plan else None,'contact_anchor_policy':a.contact_anchor_policy if execution_plan else None,
            'overview_frames':overview['recorder'].count,'source_frames':source['recorder'].count,'box_settle_passed':settle['passed'],
            'robot_loaded':robot is not None,'robot_scene_contact_headers':contacts['robot_scene_headers'],
            'initialization_mask_source':'simulation_oracle','pose_estimation_executed':False,'vlm_executed':False,'grasp_executed':False,
            'physical_parameters_measured':False,'real2sim_reconstruction_verified':False,'image_state_time_alignment_verified':False,
            'valid_sim_hours':0,'training_executed':False,'real_robot_executed':False,'hf_uploaded':False,'wall_seconds':time.monotonic()-started}
        metadata.update(mode='continuous_pallet_task',scenario_id=scenario['scenario_id'],scenario_level=scenario['level'],
            runtime_contract=runtime_contract,
            perception_source='simulation_oracle',grasp_executed=bool(completed),
            task_complete=task_complete,requested_prefix_passed=prefix_passed,completed_boxes=len(completed),
            requested_transfers=a.max_transfers,physics_grasp_validated=prefix_passed,
            simulation_oracle_diagnostic=True,end_to_end_perception_pipeline_validated=False,
            continuous_simulation=True,scene_resets_after_initialization=0)
        write_json(output/'wrist-camera-follow-check.json',camera_follow_check)
        metadata.update(camera_rig=a.camera_rig,motion_profile=a.motion_profile,policy_camera_frames={b['id']:b['recorder'].count for b in cameras})
        if observer is not None:metadata['observer_summary']=costs.call('observer_finalize',observer['recorder'].finish,metadata)
        summaries=[costs.call('policy_rgbd_finalize',bundle['recorder'].finish,metadata) for bundle in cameras]
        metadata['rgbd_integrity_passed']=all(summary['passed'] for summary in summaries)
        execution_result.update(finalized=True,rgbd_integrity_passed=metadata['rgbd_integrity_passed'],
            passed=bool(task_complete and metadata['rgbd_integrity_passed']),task_complete=bool(task_complete and metadata['rgbd_integrity_passed']))
        metadata['task_complete']=execution_result['task_complete']
        write_json(output/'task-result.json',execution_result);write_json(output/'execution-result.json',execution_result)
        metadata['scene_ready_for_perception']=bool(metadata['rgbd_integrity_passed'] and settle['passed'] and not contacts['robot_scene_headers'])
        write_json(output/'experiment-result.json',metadata);print('DEPALLET_RESULT',json.dumps(metadata),flush=True)
        code=0 if prefix_passed and metadata['rgbd_integrity_passed'] else 2
        if rollout is not None:rollout.finish(task_complete=execution_result['task_complete'],requested_prefix_passed=prefix_passed)
    except BaseException as error:
        task_failed_at=time.monotonic()
        rollout_error=repr(error)
        traceback.print_exc()
        if stepper is not None and stepper.box_id not in completed:
            if stepper.state not in ('DONE','FAILED'):
                try:stepper.abort(str(error))
                except Exception:pass
            result=stepper.result();result.update(passed=False,physics_grasp_validated=False,perception_source=execution_plan['perception_source'],end_to_end_perception_pipeline_validated=False)
            if not (cycle_output/'execution-result.json').exists():
                write_json(cycle_output/'execution-result.json',result)
        write_json(cycle_output/'failure.json',{'message':str(error),'completed_ids':completed,
            'controller_box_id':None if stepper is None else stepper.box_id,'committed_results_preserved':True})
        write_json(output/'task-result.json',{'passed':False,'task_complete':False,'completed_ids':completed,'total_boxes':len(order),'transfers':transfer_results,'failure':str(error),'perception_source':'simulation_oracle'})
        if 'contacts' in locals():
            if execution_plan:contacts['operation']=operation_contacts
            write_json(output/'contact-report.json',contacts)
        write_json(output/'failure.json',{'error':type(error).__name__,'message':str(error),'success':False,'wall_seconds':time.monotonic()-started})
    finally:
        faulthandler.cancel_dump_traceback_later()
        if rollout_callback is not None:SimulationManager.deregister_callback(rollout_callback)
        if task_started is not None:
            clock_receipt={
                'schema':'depallet.task_clock.v1','clock':'monotonic_wall',
                'start_boundary':'cameras_warm_before_first_observation_and_planning',
                'end_boundary':'final_whole_scene_check' if task_ended is not None else 'runtime_failure',
                'task_wall_s':(task_ended if task_ended is not None else (task_failed_at if task_failed_at is not None else time.monotonic()))-task_started,
                'completed_ledger_ids':list(completed),'final_state_sim_time':task_end_sim,
                'final_verification_reached':task_ended is not None,
                'includes_planning_reobservation_and_recovery':True,
                'excludes_app_initialization_and_recording_finalize':True}
            try:write_json(output/'task-clock.json',clock_receipt)
            except Exception as clock_error:
                print('TASK_CLOCK_WRITE_FAILED',repr(clock_error),flush=True)
        if rollout is not None and not rollout.closed:rollout.finish(error=rollout_error or 'Runtime did not finalize')
        if observer is not None:observer['recorder'].close()
        if viewer is not None:viewer.close()
        try:
            timing.finish('success' if code==0 else 'failed')
            costs.finish(output)
        except Exception as timing_error:
            print('TIMING_FINALIZATION_FAILED',repr(timing_error),flush=True)
        finally:
            if app is not None:app.close(exit_code=code)
    return code



