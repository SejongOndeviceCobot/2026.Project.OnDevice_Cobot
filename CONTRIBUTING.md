# 기여 안내

이 저장소는 여러 연구실이 같은 실행 기준선을 검토하고 개선하기 위한 공간입니다. 먼저 [모듈 안내](docs/MODULES.md)에서 바꾸려는 단계의 `src/depallet/` 폴더를 찾으세요. 단계 사이 형식이 바뀌면 호출하는 쪽과 결과 판정도 함께 살펴봐야 합니다.

1. `main`을 받은 뒤 작업별 브랜치를 만듭니다: `git switch -c <작업-이름>`.
2. 한 Pull Request에는 한 기능 변경을 담고, 바뀐 입력·출력과 실행 조건을 설명합니다. 필요하면 [README](README.md)와 모듈 문서를 함께 고칩니다.
3. 프로젝트 `env.sh`를 불러온 뒤 CPU 검사를 실행합니다.

```bash
source ./env.sh
python3 tools/verify_evidence.py
python3 tools/update_source_manifest.py
python3 tools/verify_evidence.py --check-current
python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tools
python3 tools/run.py full --dry-run
```

`src/depallet/` 또는 `scripts/run_task.py`, `scripts/guarded_run.py`, `scripts/curobo_worker.py`를 고쳤다면 `update_source_manifest.py`로 현재 소스 목록·해시를 갱신한 뒤 `verify_evidence.py --check-current`를 실행하세요. manifest 변경 이유를 PR에 적습니다. 이 검사는 현재 바이트와 모듈 배치를 기록·확인할 뿐, 시뮬레이션 성공을 증명하지 않습니다.

4. 실행 경로가 바뀌었다면 준비된 자산에서 `python3 tools/run.py smoke --gpu 0`로 첫 상자를 확인하고, 전체 성공을 주장할 때는 `python3 tools/run.py full --gpu 0`의 새 `task-result.json`과 코드 커밋을 기록합니다. CPU 검사나 과거 근거만으로 새 코드의 16/16 성공을 주장하지 않습니다.
5. `git push -u origin <작업-이름>` 후 Pull Request에 바뀐 모듈, 검사 결과, 새 실행 조건·출력 경로, 남은 제약을 적습니다. 다른 기여자의 검토 후 병합합니다.

`evidence/`의 기존 JSON과 소스 아카이브는 재배치 전 실행의 불변 기록입니다. 새 구현 결과로 덮어쓰지 말고 별도 근거를 추가해 과거 결과와 구분해 주세요. 외부 H2017/VGP20 자산, Isaac Sim·cuRobo 설치물, 모델 가중치, 원본 녹화, 실행 출력은 커밋하지 않습니다. 배포 권한과 [제3자 고지](THIRD_PARTY_NOTICES.md)를 확인하세요.
