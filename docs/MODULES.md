# 모듈 안내

성공 실행에서 사용한 Python 파일을 그대로 보존했습니다. `src/` 54개와 `scripts/` 3개, 약 1.26만 줄입니다. 따라서 첫 수정은 아래 경계에서 시작하고, 호출부와 결과 검사를 함께 살펴보는 편이 안전합니다.

| 단계 | 주로 볼 파일 | 역할 |
| --- | --- | --- |
| 입력·배치 | `scenario_suite.py`, `task_planning.py` | 상자 시나리오, 이송 순서, 목표 팔레트 배치 |
| 실행 조정 | `task_runtime.py`, `scripts/run_task.py` | 한 장면에서 관측·계획·이송·판정을 연결 |
| 카메라·기록 | `camera_rig.py`, `frame_clock.py`, `recording.py` | RGB-D 관측과 시간·파일 기록 |
| 동작 계획 | `planning_requests.py`, `curobo_bridge.py`, `nested_curobo.py`, `scripts/curobo_worker.py` | 요청 검사와 별도 cuRobo 작업자 실행 |
| 로봇·그리퍼 | `h2017_execution.py`, `vgp20_surface.py`, `vacuum_compliance.py` | H2017 경로 실행과 흡착 접촉 모델 |
| 결과 판정 | `task_runtime_checks.py`, `source_stack_integrity.py` | 실제 시뮬레이션 상태의 적재·잔여 상자 검사 |
| 실행 보호 | `scripts/guarded_run.py`, `tools/run.py` | GPU 및 실행 시간 제한, 출력 디렉터리 관리 |

기본 V1 성공 경로는 oracle 관측을 사용합니다. `inspection_ai.py`, `nested_ai.py`, `modular_pick_bridge.py` 등은 관측 모델 결합을 위한 연구 코드이지만, 이 저장소의 선택 스냅샷만으로 종단 간 hybrid 실행이 완료된 것은 아닙니다. 외부 모델과 일부 작업자 파일이 별도로 필요할 수 있으므로 새 기여는 먼저 이 경계를 명시해 주세요.

실행 입력은 `examples/v1_uniform/`에서 준비하고, 새 결과는 Git 밖의 `runs/`에 씁니다. 결과를 검토할 때는 완료 상자 수와 함께 `task-result.json`의 판정 필드, 실패 이유, 사용한 소스·자산 버전을 확인합니다.
