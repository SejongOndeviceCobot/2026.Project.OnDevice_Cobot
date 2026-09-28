# 결과와 검증 범위

| 항목 | 현재 기록 |
| --- | --- |
| 원본 실행 | 2026-09-21, Isaac Sim 연속 실행에서 16/16 완료 |
| 새 체크아웃 실행 | 2026-09-28(KST), 같은 호스트에서 16/16 재실행 성공. 기존 Isaac Sim·cuRobo 가상환경과 외부 자산 재사용, 종료 코드 0, 약 3795초 소요. 최종 배치 검사 16/16 통과 |
| 제어·관측 범위 | 두 실행 모두 시뮬레이터 정답값(oracle) 사용, 초기화 후 장면 재설정 0회, 관절·물체 순간 이동 없음, `valid_sim_hours=0` |
| 저장된 근거 | 원본: `evidence/full16-task-result.json`, `evidence/full16-audit-summary.json`, `evidence/full16-source.tar.gz`, `evidence/published-source.json`. 새 실행: `evidence/repackaged-full16-task-result.json`, `evidence/repackaged-full16-exit.json`, `evidence/repackaged-full16-source-manifest.json`, `evidence/repackaged-full16-audit-summary.json`. 두 소스 목록의 57개 해시가 일치함 |
| 독립 CPU 감사 | 원본 실행은 16/16 이송, 473/473 검사 통과. 새 체크아웃 실행은 16/16 이송, 490/490 검사 통과(COMPLETE_VERIFIED). 서로 다른 평가기 버전의 별도 감사이며, 약 6 MB의 각 원본 감사 파일은 저장소에 포함하지 않음 |
| Hybrid PF2 | 최고 8/16, 최근 v19는 7/16. 전체 16/16 성공 기록 없음 |

두 `task-result.json`의 `passed`, `physical_task_complete`는 **Isaac 물리 시뮬레이션의 완료 판정**입니다. 실제 로봇 성공, 독립 RGB-D 인지 성공, 학습 모델을 통한 종단 간 수행, 데이터 수용 판정으로 해석하면 안 됩니다. 두 기록 모두 `perception_source: simulation_oracle`이고 `end_to_end_perception_pipeline_validated: false`입니다. 새 독립 감사는 저장 결과의 일관성과 CPU 재계산 범위이며 모델 추론이나 물리 실행을 새로 수행하지 않았습니다.

저장된 JSON에는 해당 호스트의 절대 실행 경로가 남아 있습니다. `python3 tools/verify_evidence.py`는 기본적으로 보존 아카이브·manifest·요약 근거를 확인하고, `--check-current`는 현재 작업 파일도 비교합니다. 같은 호스트의 새 체크아웃 재실행은 확인됐지만, 새로 설치한 가상환경이나 다른 연구실에서의 전체 실행은 아직 검증되지 않았습니다. 외부 자산과 전체 원본 로그가 빠진 clone만으로 16/16을 재현할 수 없습니다. 자산 manifest·URDF의 원본 절대 경로와 SHA 고정 때문에 임의의 새 워크스페이스로 옮기는 것만으로도 실행 조건이 맞지 않습니다. 필요한 조건은 [재현 절차](REPRODUCE.md)에 있습니다.

`check_assets.py`는 외부 자산 59개 파일의 경로와 해시 연결을 점검합니다. Doosan H2017 원본 USD는 존재 여부와 USDC 형식을 확인하고 점검 때 계산한 SHA-256을 보고하지만, 당시 값의 공개된 해시 고정은 없습니다. 따라서 자산 점검 통과도 원본 USD 바이트의 동일성을 증명하지 않습니다.
