# OnDevice Cobot: 팔레트 이송 시뮬레이션

세종대 컨소시엄의 코드 협업을 위한 저장소입니다. V1 균일 상자 16개를 Isaac Sim에서 연속 이송했습니다. **재배치 전 코드**는 2026-09-21 원본과 2026-09-28(KST) 새 체크아웃에서 각각 16/16을 완료했고, 당시 Python 파일 57개 해시가 일치합니다.

현재 소스를 기능별 `src/depallet/`로 정리한 뒤 2026-09-29(KST) 같은 호스트에서 16/16을 다시 완료했습니다. 실행 전 Python 66개 파일의 해시가 현재 manifest와 일치했습니다. 첫 모듈식 전체 시도는 1/16에서 실패했고 재시도에 성공했습니다. 기존 Isaac Sim·cuRobo 가상환경과 외부 자산을 재사용했으며, 세 완료 실행 모두 **시뮬레이터 정답값(oracle)** 으로 대상을 선택했습니다. 다른 연구실의 새 설치, 독립 RGB-D 인지, 실제 로봇 성공을 뜻하지 않습니다. 실행별 근거와 한계는 [결과 기록](docs/RESULTS.md)에 있습니다.

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
python3 tools/verify_evidence.py --check-release-source
python3 tools/prepare_example.py
python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tools
```

`verify_evidence.py`는 세 번의 완료 실행에 저장된 결과와 해시를 확인합니다. `--check-current`는 현재 모듈 소스의 목록·해시·import를, `--check-release-source`는 현재 소스 66개가 모듈식 성공 실행 당시와 같은지를 검사합니다. 이 검사는 물리 시뮬레이션 재실행이 아닙니다. GPU 실행에는 허가된 H2017·VGP20 외부 자산과 cuRobo checkout이 필요합니다.

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
