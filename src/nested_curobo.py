"""Bounded child planner inside the already guarded Isaac process group."""
from __future__ import annotations
import math
import os
from pathlib import Path
import subprocess


def run_curobo_child(request_path, output_dir, environment, timeout_s, log):
    project = Path(__file__).resolve().parents[1]
    if environment.get('ISAAC_P0_GUARDED') != '1' or Path(environment.get('ISAAC_P0_PROJECT', '')).resolve() != project:
        raise ValueError('Nested cuRobo requires this project and an inherited active guard')
    if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or not 1 <= timeout_s <= 180:
        raise ValueError('Nested planner timeout must be between 1 and 180 seconds')
    cache, root = [Path(environment[key]).resolve() for key in ('ISAAC_P0_CACHE', 'ISAAC_P0_RUNS')]
    output, request = Path(output_dir).resolve(), Path(request_path)
    if output == root or not output.is_relative_to(root) or not output.is_dir():
        raise ValueError('Nested output must be an existing dedicated run subdirectory')
    if request.is_symlink() or request.resolve().parent != output or not request.is_file():
        raise ValueError('Nested request must be a regular file directly inside its output')
    if Path(environment.get('ISAAC_P0_OUTPUT', '')).resolve() != output:
        raise ValueError('Nested output environment mismatch')
    python, worker = cache/'curobo-venv/bin/python', project/'scripts/curobo_worker.py'
    if not python.is_file() or not worker.is_file() or worker.is_symlink():
        raise ValueError('Fixed project cuRobo worker or interpreter is unavailable')
    command = [str(python), str(worker), '--request', str(request.resolve()), '--output', str(output)]
    child = subprocess.Popen(command, env=dict(environment), stdout=log, stderr=subprocess.STDOUT, start_new_session=False)
    timed_out, termination = False, 'normal_exit'
    try:
        group, parent_group = os.getpgid(child.pid), os.getpgrp()
        if group != parent_group:
            raise RuntimeError('Nested planner escaped the inherited process group')
        try:
            returncode = child.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out, termination = True, 'terminate'
            child.terminate()
            try:
                returncode = child.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                termination = 'kill'
                child.kill()
                returncode = child.wait(timeout=2.)
        return dict(returncode=returncode, timed_out=timed_out, termination=termination,
                    child_pid=child.pid, child_process_group=group, parent_process_group=parent_group,
                    same_process_group=True, child_exit_verified=child.poll() is not None,
                    descendant_cleanup='outer guard verifies UID/session/starttime and uses pidfd signals')
    finally:
        # Direct Popen child only; never signal a shared process group here.
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2.)
