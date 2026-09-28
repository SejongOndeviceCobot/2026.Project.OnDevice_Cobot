# OnDevice Cobot: 팔레트 이송 시뮬레이션

세종대 컨소시엄의 코드 협업을 위한 저장소입니다. 검증된 기준선은 **V1 균일 상자 16개를 Isaac Sim에서 연속 이송한 실행**입니다. 2026-09-21 원본과 2026-09-28(KST) 같은 호스트의 새 체크아웃에서 각각 16/16을 완료했습니다. 두 실행의 당시 Python 파일 57개 해시는 일치하며 결과 요약은 `evidence/`에 있습니다. 두 번째 실행은 기존 Isaac Sim·cuRobo 가상환경과 외부 자산을 재사용했습니다.

현재 소스는 기능별로 `src/depallet/`에 정리했습니다. 위 16/16 기록은 **재배치 전 코드**의 근거입니다. 재배치한 코드의 GPU 실행 결과는 [결과 기록](docs/RESULTS.md)에서 별도로 확인하세요. 기준선의 대상 인식은 **시뮬레이터 정답값(oracle)** 을 사용했습니다. 실제 로봇이나 독립 RGB-D 인지 파이프라인 성공을 뜻하지 않습니다.

## 시작

워크스페이스 루트를 지정하고 저장소를 받습니다.

```bash
export JCLEE_WORKSPACE=/absolute/path/to/workspace
if [[ -f "$JCLEE_WORKSPACE/setup/env.sh" ]]; then
  source "$JCLEE_WORKSPACE/setup/env.sh"
fi
mkdir -p "$JCLEE_WORKSPACE/repos/own"
git clone https://github.com/SejongOndeviceCobot/2026.Project.OnDevice_Cobot.git \
  "$JCLEE_WORKSPACE/repos/own/2026.Project.OnDevice_Cobot"
cd "$JCLEE_WORKSPACE/repos/own/2026.Project.OnDevice_Cobot"
source ./env.sh
python3 tools/verify_evidence.py
python3 tools/verify_evidence.py --check-current
python3 tools/prepare_example.py
python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tools
```

`verify_evidence.py`의 기본 검사는 보존된 과거 실행 자료를 확인합니다. `--check-current`는 현재 모듈 소스·실행 스크립트가 `evidence/modular-source-manifest.json`의 목록·해시와 일치하고, 이전 평면 import가 남지 않았는지 확인합니다. 둘 다 물리 시뮬레이션 재실행은 아닙니다. GPU 실행에는 허가된 H2017·VGP20 외부 자산과 cuRobo checkout이 필요합니다.

```bash
CUROBO_SOURCE="$JCLEE_WORKSPACE/repos/external/github.com/NVlabs/curobo"
python3 tools/install.py --curobo-source "$CUROBO_SOURCE" --dry-run
python3 tools/install.py --curobo-source "$CUROBO_SOURCE" --accept-nvidia-eula
python3 tools/check_assets.py
python3 tools/run.py full --dry-run
python3 tools/run.py smoke --gpu 0
python3 tools/run.py full --gpu 0
```

NVIDIA 약관을 직접 검토하고 동의한 경우에만 설치 명령의 `--accept-nvidia-eula`를 사용하세요. Smoke는 첫 상자 1/1 이송을 확인합니다. 이때 `requested_prefix_passed=true`, `state=PREFIX_COMPLETE`이고 전체 16개가 끝나지 않았으므로 `passed=false`가 정상입니다. 자산 준비, 출력 경로와 판정 방법은 [재현 절차](docs/REPRODUCE.md)에 있습니다.

## 수정할 위치

| 경로 | 맡는 기능 |
| --- | --- |
| `examples/v1_uniform/` | 16개 상자 입력과 규칙 기반 이송 순서 |
| `src/depallet/scene/` | 시나리오와 시뮬레이션 장면 |
| `src/depallet/observation/` | 카메라, RGB-D 관측, 입력 형식 |
| `src/depallet/planning/` | 이송 순서, 목표 배치, 계획 요청 |
| `src/depallet/motion/` | cuRobo 경로, 충돌, 재시도 |
| `src/depallet/manipulation/` | 로봇 동작, 흡착 그리퍼, 상자 접촉 |
| `src/depallet/integration/` | 외부 계획·모델 연결 |
| `src/depallet/validation/` | 적재 상태와 최종 결과 판정 |
| `src/depallet/runtime/` | 전체 실행 순서, 상태·시간·결과 기록 |
| `scripts/`, `tools/` | 실행 진입점과 준비·점검 도구 |

[모듈 안내](docs/MODULES.md)에서 파이프라인 흐름과 수정 범위를 확인하세요. 변경은 별도 브랜치와 Pull Request로 제안합니다. [기여 안내](CONTRIBUTING.md)에 검사 순서가 있습니다. 외부 자산의 배포 범위는 [제3자 고지](THIRD_PARTY_NOTICES.md)를 따르고, 데이터·체크포인트·캐시·로그·실행 출력은 Git 밖의 `data/`, `cache/`, `runs/`에 둡니다.
