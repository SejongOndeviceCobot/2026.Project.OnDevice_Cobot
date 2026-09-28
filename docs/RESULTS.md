# 결과와 검증 범위

| 항목 | 확인된 내용 |
| --- | --- |
| 원본 실행 | 2026-09-21, 재배치 전 코드로 Isaac Sim 연속 이송 16/16 완료 |
| 같은 호스트의 새 체크아웃 실행 | 2026-09-28(KST), **재배치 전 코드**로 16/16 재실행. 기존 Isaac Sim·cuRobo 가상환경과 외부 자산 재사용, 종료 코드 0, 약 3795초, 최종 배치 검사 16/16 통과 |
| 현재 모듈 구조 | `src/depallet/{scene,observation,planning,motion,manipulation,integration,validation,runtime}/`로 기능별 재배치. 이 구조의 GPU 전체 실행 결과는 아직 별도로 확인되지 않음 |
| 제어·관측 범위 | 위 두 성공 실행 모두 시뮬레이터 정답값(oracle) 사용, 초기화 후 장면 재설정 0회, 관절·물체 순간 이동 없음, `valid_sim_hours=0` |
| 저장된 근거 | 원본: `evidence/full16-task-result.json`, `evidence/full16-audit-summary.json`, `evidence/full16-source.tar.gz`, `evidence/published-source.json`. 새 체크아웃: `evidence/repackaged-full16-task-result.json`, `evidence/repackaged-full16-exit.json`, `evidence/repackaged-full16-source-manifest.json`, `evidence/repackaged-full16-audit-summary.json`. **당시 두 실행**의 57개 파일 해시가 일치함 |
| 독립 CPU 감사 | 원본 실행 473/473 검사, 새 체크아웃 실행 490/490 검사(COMPLETE_VERIFIED). 평가기 버전이 다른 별도 감사이며 각 원본 감사 파일은 저장소에 포함하지 않음 |
| Hybrid PF2 | 최고 8/16, 최근 v19는 7/16. 전체 16/16 성공 기록 없음 |

두 보존 `task-result.json`의 `passed`와 `physical_task_complete`는 **Isaac 물리 시뮬레이션 완료 판정**입니다. 실제 로봇, 독립 RGB-D 인지, 학습 모델을 통한 종단 간 수행이나 데이터 수용 판정은 아닙니다. 두 기록 모두 `perception_source: simulation_oracle`이고 `end_to_end_perception_pipeline_validated: false`입니다. CPU 감사는 저장 결과를 재계산한 범위이며 모델 추론이나 물리 실행을 새로 수행하지 않았습니다.

`python3 tools/verify_evidence.py`는 보존된 아카이브와 당시 결과·감사 요약을 확인합니다. `--check-current`는 현재 `src/depallet/` 소스와 실행 스크립트의 목록·해시가 `evidence/modular-source-manifest.json`에 맞고 이전 평면 import가 남지 않았는지 확인합니다. 이 manifest는 현재 체크아웃의 편집 가능한 목록이므로 과거 실행 근거나 물리 성공 증명은 아닙니다. 재배치 후의 동작은 CPU 검사와 **새 GPU 실행 결과**로 별도로 확인해야 하며, 이전 16/16을 재배치한 코드의 성과로 표시하지 않습니다.

기존 JSON에는 당시 호스트의 절대 실행 경로가 남아 있습니다. 같은 호스트의 새 체크아웃 실행은 확인됐지만, 새로 설치한 가상환경이나 다른 연구실의 전체 실행은 확인되지 않았습니다. 외부 자산과 전체 원본 로그가 빠진 clone만으로 16/16을 재현할 수 없습니다. 자산 manifest·URDF의 절대 경로와 SHA 고정 때문에 다른 워크스페이스로 단순 이동해도 조건이 맞지 않을 수 있습니다. 필요한 조건은 [재현 절차](REPRODUCE.md)에 있습니다.

`check_assets.py`는 외부 자산 59개 파일의 경로와 해시 연결을 점검합니다. Doosan H2017 원본 USD는 존재 여부·USDC 형식·현재 SHA-256을 보고하지만 당시 값의 고정 해시는 없습니다. 따라서 자산 점검 통과도 원본 USD 바이트의 동일성을 증명하지 않습니다.
