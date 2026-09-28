"""Bounded retry for an ordinary planner search failure, never a process fault."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def retryable_planning_failure(outcome, result, remaining_seconds):
    return (type(outcome.get('returncode')) is int and outcome['returncode']>0
        and outcome.get('timed_out') is False and outcome.get('child_exit_verified') is True
        and result.get('success') is False and result.get('exception_type')=='RuntimeError'
        and str(result.get('error','')).startswith('Planning failed for ')
        and remaining_seconds>=180)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _manifest(directory):
    files = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink() or not path.is_file():
            if path.is_dir() and not path.is_symlink():
                continue
            raise ValueError('Planner attempt artifacts must be regular files')
        files[str(path.relative_to(directory))] = _sha256(path)
    return files


def create_payload_retry_attempt(first_directory, retry_seed=43):
    """Create a fresh sibling attempt with only the deterministic seed changed."""
    selected = Path(first_directory)
    if selected.is_symlink():
        raise ValueError('First payload attempt must not be a symlink')
    first = selected.resolve()
    if first.name != 'confirmed-payload-plan' or not first.is_dir():
        raise ValueError('First payload attempt must be confirmed-payload-plan')
    required = ('request.json', 'measured-attachment.json', 'result.json', 'worker.log')
    for name in required:
        path = first / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'First payload attempt is missing regular {name}')
    from curobo_bridge import planner_random_seed, validate_request
    first_request = validate_request(json.loads((first / 'request.json').read_text()))
    first_seed = planner_random_seed(first_request)
    if type(retry_seed) is not int or retry_seed == first_seed:
        raise ValueError('Retry planner seed must be a distinct integer')
    retry_request = json.loads(json.dumps(first_request, allow_nan=False))
    retry_request['planner_random_seed'] = retry_seed
    validate_request(retry_request)
    first_problem = dict(first_request);first_problem.pop('planner_random_seed', None)
    retry_problem = dict(retry_request);retry_problem.pop('planner_random_seed', None)
    if first_problem != retry_problem:
        raise RuntimeError('Retry changed fields other than planner_random_seed')
    before = _manifest(first)
    retry = first.with_name('confirmed-payload-plan-attempt2')
    retry.mkdir()
    request_path = retry / 'request.json'
    with request_path.open('x') as stream:
        stream.write(json.dumps(retry_request, indent=2, allow_nan=False) + '\n')
    evidence_source = first / 'measured-attachment.json'
    evidence_path = retry / 'measured-attachment.json'
    with evidence_source.open('rb') as reader, evidence_path.open('xb') as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b''):
            writer.write(block)
    retry_hashes = {
        'request.json': _sha256(request_path),
        'measured-attachment.json': _sha256(evidence_path),
    }
    if retry_hashes['measured-attachment.json'] != before['measured-attachment.json']:
        raise RuntimeError('Retry measured attachment copy differs')
    if _manifest(first) != before:
        raise RuntimeError('First payload attempt changed while preparing retry')
    return retry, {
        'schema': 'depallet.payload_planning_retry.v1',
        'attempts_limit': 2,
        'request_unchanged': False,
        'planning_problem_unchanged_except_random_seed': True,
        'first_planner_random_seed': first_seed,
        'retry_planner_random_seed': retry_seed,
        'first_attempt_path': str(first),
        'retry_attempt_path': str(retry),
        'first_attempt_manifest': before,
        'retry_input_sha256': retry_hashes,
        'first_attempt_preserved': True,
        'retry_output_created_fresh': True,
    }
