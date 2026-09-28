# 결과와 검증 범위

| 실행 | 확인된 결과 |
| --- | --- |
| 원본, 2026-09-21 | 재배치 전 평면 코드로 Isaac Sim 연속 이송 16/16 완료 |
| 새 체크아웃, 2026-09-28(KST) | 같은 평면 코드로 16/16 재실행. 당시 두 실행의 Python 파일 57개 해시 일치, 기존 가상환경·외부 자산 재사용 |
| 모듈식 첫 full 시도 | `box_16` 이송 1/16 뒤 `box_04`의 release 전 자세 안정화가 4초 안에 끝나지 않아 종료(종료 코드 1) |
| 모듈식 재시도, 2026-09-29(KST) | `src/depallet/` 코드로 같은 호스트에서 16/16 완료, `COMPLETE`·`passed=true`·종료 코드 0. 최종 장면 배치 검사 16/16, RGB-D 무결성 및 결과 확정 통과. 실행 전 Python 66개 파일의 경로·SHA-256이 현재 모듈 manifest와 일치 |
| 독립 CPU 감사 | 재배치 전 원본 473/473, 새 체크아웃 490/490, 모듈식 재시도 490/490. 각 실행에 대해 별도 평가·해시 기록 |

첫 모듈식 전체 시도는 1/16에서 실패했고 재시도에 16/16을 완료했습니다. 따라서 한 번의 성공만으로 매 실행의 동일한 결과를 보장하지 않습니다.

재배치 전 결과는 `evidence/full16-task-result.json`, `evidence/full16-audit-summary.json`, `evidence/full16-source.tar.gz`와 `evidence/repackaged-full16-*.json`에 보존합니다. 모듈식 재시도는 `evidence/modular-full16-task-result.json`, `evidence/modular-full16-exit.json`, `evidence/modular-full16-source-manifest.json`, `evidence/modular-full16-preflight.json`, `evidence/modular-full16-audit-summary.json`으로 구분합니다. 보존된 과거 57개 소스와 새 모듈식 실행의 66개 소스는 서로 다른 코드 배치입니다.

세 완료 실행의 `passed`와 `physical_task_complete`는 **Isaac 물리 시뮬레이션 완료 판정**입니다. 모두 `perception_source: simulation_oracle`, `end_to_end_perception_pipeline_validated: false`, `valid_sim_hours=0`이며 장면 재설정이나 관절·물체 순간 이동 없이 진행됐습니다. 독립 RGB-D 분할·자세 추정이나 학습 모델 기반 종단 간 인지 성공, 학습 데이터 수용, 실제 로봇 성공을 뜻하지 않습니다. 관련 AI worker 5개와 가중치는 저장소에 포함하지 않습니다.

`python3 tools/verify_evidence.py`는 세 완료 실행의 저장 근거와 결과 계약을 확인합니다. 모듈식 재시도의 독립 CPU 감사는 `COMPLETE_VERIFIED`, 490/490 검사 통과이며 시뮬레이션이나 모델 추론을 다시 실행하지 않았습니다. 원본 감사 파일과 평가기 사본은 Git 밖에 있고 `evidence/modular-full16-audit-summary.json`에 해시와 범위를 기록했습니다. `--check-current`는 편집 가능한 현재 소스 manifest를, `--check-release-source`는 현재 66개 파일과 성공 실행 전 스냅샷의 동일성을 확인합니다. 검증기 자체는 물리 실행이 아닙니다.

모듈식 성공도 **같은 호스트의 기존 Isaac Sim·cuRobo 가상환경과 외부 자산을 재사용한 실행**입니다. 새 설치나 다른 연구실의 전체 실행은 검증되지 않았습니다. 외부 자산 manifest·URDF의 당시 절대 경로와 SHA 고정 때문에 다른 워크스페이스에 단순 복사해도 조건이 맞지 않을 수 있습니다. `check_assets.py`는 외부 자산 59개의 경로·해시를 검사하지만 Doosan H2017 원본 USD의 당시 해시는 고정돼 있지 않습니다. 준비 조건은 [재현 절차](REPRODUCE.md)에 있습니다.
