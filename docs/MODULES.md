# 모듈 안내

`scripts/run_task.py`가 한 장면의 실행을 시작하고 `src/depallet/runtime/task_runtime.py`가 각 단계를 연결합니다. `tools/run.py`와 `scripts/guarded_run.py`는 실행 준비와 GPU 제한을 맡으며, cuRobo 작업자는 `scripts/curobo_worker.py`에서 별도로 시작합니다. 기능을 바꿀 때는 아래 폴더에서 시작하고 호출 경계와 결과 판정을 함께 확인하세요.

| 단계 | `src/depallet/` 아래 폴더 | 먼저 볼 파일 | 책임 |
| --- | --- | --- | --- |
| 장면 | `scene/` | `scenario_suite.py`, `photo_scene.py` | 시나리오와 상자·팔레트 장면 구성 |
| 관측 | `observation/` | `camera_rig.py`, `sensor_contract.py` | 카메라·RGB-D 입력과 대상 관측 형식 |
| 작업 계획 | `planning/` | `multi_transfer_planning.py`, `planning_requests.py` | 이송 순서, 목표, 계획 요청 |
| 동작 계획 | `motion/` | `curobo_bridge.py`, `contact_escape.py` | cuRobo 경로, 충돌·접촉 탈출 |
| 이송 | `manipulation/` | `h2017_execution.py`, `vgp20_surface.py` | 로봇 실행, 흡착 그리퍼, 상자 접촉 |
| 외부 연결 | `integration/` | `cutamp_execution_bridge.py`, `modular_pick_bridge.py` | 외부 계획·모델의 호출 형식 연결 |
| 판정 | `validation/` | `task_runtime_checks.py`, `source_stack_integrity.py` | 이송 결과와 최종 장면 상태 검사 |
| 전체 순서·기록 | `runtime/` | `task_runtime.py`, `recording.py` | 단계 호출, 상태 전이, 시간·결과 기록 |

기본 흐름은 **장면 → 관측 → 작업 계획 → 동작 계획 → 이송 → 판정**이며 `runtime/`이 이를 반복합니다. `integration/`은 필요한 외부 구현과 단계 사이의 연결부입니다. 예제 입력은 `examples/v1_uniform/`, 실행 결과는 Git 밖의 `runs/`에 있습니다.

개발 시 사용한 로컬 관제판(`http://localhost:18767/pipeline`)의 9개 단계에서 코드를 찾을 때는 아래 대응을 사용하세요. 한 단계가 여러 패키지를 거칠 수 있습니다.

| 관제판 단계 | 주로 볼 패키지 |
| --- | --- |
| 장면·난이도 | `scene/` |
| RGB-D 관측 | `observation/` |
| 인스턴스 분할 | `observation/`, `integration/` |
| 자세·기하 | `observation/`, `motion/` |
| 흡착 후보 | `observation/`, `manipulation/` |
| 순서·배치 계획 | `planning/`, `scene/` |
| 동작 계획·실행 | `motion/`, `manipulation/` |
| 물리 grasp·release | `manipulation/`, `validation/` |
| 평가·데이터 | `validation/`, `runtime/` |

이 대응은 **수정 위치 안내**이지 9단계 성공 판정이 아닙니다. 16/16 기준선은 시뮬레이터 정답값(oracle)으로 대상을 선택했으며, 독립 분할·자세 추정 모델과 학습 데이터 수용은 검증되지 않았습니다. 원본 57개 Python 파일은 `evidence/full16-source.tar.gz`에 보존돼 있고, 현재 폴더 구조의 검증 상태는 [결과 기록](RESULTS.md)에 분리해 적습니다.

변경 후에는 [기여 안내](../CONTRIBUTING.md)의 CPU 검사를 먼저 실행하세요. 실행 경로를 바꿨다면 한 상자 Smoke와 전체 16개 결과의 `task-result.json`을 새로 확인해야 합니다.
