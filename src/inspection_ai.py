"""Run a complete AI proposal while the caller keeps the simulator frozen."""
import json
import os
from pathlib import Path
import subprocess
import time
from nested_ai import validate_parent


def run_inspection_proposal(source_run,request,output,remaining_seconds,environment,*,center_goal_init=False):
    source=validate_parent(environment,source_run)
    request,output=Path(request).resolve(),Path(output).resolve()
    if not request.is_relative_to(source) or not request.is_file() or not output.is_relative_to(source) or output==source or output.exists():
        raise ValueError('Fresh request and new inference output inside current run required')
    if remaining_seconds<240:raise TimeoutError('Insufficient parent guard time for an observation proposal')
    project=Path(environment['ISAAC_P0_PROJECT']);cache=Path(environment['ISAAC_P0_CACHE'])
    command=[str(cache/'venv/bin/python'),str(project/'scripts/run_observation_pipeline.py'),
        '--source-run',str(source),'--output',str(output),'--planning-request',str(request),
        '--with-graspgen','--inherited-guard','--grasp-workers','2','--grasp-batch-seeds','--execute']
    if center_goal_init:command.append('--center-goal-init')
    start=time.monotonic()
    with output.with_suffix('.launcher.log').open('x') as log:
        # The outer guard owns the total deadline and cleanup of every descendant.
        result=subprocess.run(command,env=dict(environment),stdout=log,stderr=subprocess.STDOUT,start_new_session=False)
    report=json.loads((output/'pipeline.json').read_text()) if (output/'pipeline.json').is_file() else {}
    return dict(mode='shadow_plan',returncode=result.returncode,pipeline_status=report.get('status','missing'),
        elapsed_wall_seconds=time.monotonic()-start,output=str(output),
        task_plan=report.get('selected_task_plan'),task_input=report.get('selected_task_input'),
        complete_chain_ran=result.returncode==0 and report.get('status')=='workers_completed',
        proposal_applied_to_robot=False,caller_must_verify_frozen_state=True)
