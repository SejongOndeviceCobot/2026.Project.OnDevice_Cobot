"""Sequential AI inference inside an existing guarded simulator process group."""
import json
import math
import os
from pathlib import Path
import subprocess
import time

WORKERS={'sam31_worker.py':'sam3-venv','point2pose_worker.py':'point2pose-venv',
         'graspgen_worker.py':'graspgen-venv','vlm_worker.py':'vlm-venv','cutamp_worker.py':'cutamp-venv'}


def validate_parent(environment,parent_run):
    parent=Path(parent_run).resolve();runs=Path(environment['ISAAC_P0_RUNS']).resolve()
    if (environment.get('ISAAC_P0_GUARDED')!='1' or not environment.get('ISAAC_P0_GPU_UUID')
            or parent==runs or not parent.is_relative_to(runs)):
        raise ValueError('Inherited project GPU guard required')
    receipt=json.loads((parent/'exit.json').read_text())
    leader=receipt.get('child_pid')
    if receipt.get('status')!='running' or leader!=os.getpgrp() or os.getsid(0)!=leader:
        raise ValueError('Worker is not inside the live guarded simulator session')
    if os.stat('/proc/'+str(leader)).st_uid!=os.getuid():raise ValueError('Guarded process owner differs')
    return parent


def run(module,arguments,output,parent_run,seconds,environment):
    if module not in WORKERS:raise ValueError('Unknown fixed AI worker')
    if isinstance(seconds,bool) or not math.isfinite(seconds) or not 1<=seconds<=180:
        raise ValueError('Nested worker duration must be1..180 seconds')
    parent=validate_parent(environment,parent_run);output=Path(output).resolve()
    if output==parent or not output.is_relative_to(parent) or output.exists():
        raise ValueError('New output directory inside guarded simulator run required')
    project=Path(__file__).resolve().parents[1]
    if Path(environment['ISAAC_P0_PROJECT']).resolve()!=project:raise ValueError('Project mismatch')
    worker=project/'scripts'/module;python=Path(environment['ISAAC_P0_CACHE'])/WORKERS[module]/'bin/python'
    if worker.is_symlink() or not worker.is_file() or not python.is_file():raise ValueError('Fixed worker unavailable')
    output.mkdir(parents=True);env=dict(environment);env['ISAAC_P0_OUTPUT']=str(output)
    started=time.monotonic();receipt=dict(status='running',child_returncode=None,
        guard_scope='inherited_simulator_guard',parent_guard_run=str(parent),
        independent_gpu_admission=False,resource_limits_owned_by_parent=True)
    def save():
        temp=output/'exit.tmp';temp.write_text(json.dumps(receipt,indent=2));temp.replace(output/'exit.json')
    save()
    with (output/'child.log').open('x') as log:
        child=subprocess.Popen([str(python),str(worker),*arguments],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=False)
        receipt['child_pid']=child.pid;save()
        timed_out=False
        try:
            if os.getpgid(child.pid)!=os.getpgrp():raise RuntimeError('Worker escaped parent guard group')
            try:code=child.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                timed_out=True;child.terminate()
                try:code=child.wait(timeout=2)
                except subprocess.TimeoutExpired:child.kill();code=child.wait(timeout=2)
        finally:
            if child.poll() is None:child.kill();child.wait(timeout=2)
        receipt.update(status='timeout' if timed_out else ('success' if code==0 else 'child_failed'),
            child_returncode=code,elapsed_seconds=time.monotonic()-started,
            child_exit_verified=child.poll() is not None)
        save()
    return 124 if timed_out else code
