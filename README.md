# OnDevice Cobot: 팔레트 이송 시뮬레이션

세종대 컨소시엄의 코드 협업을 위한 저장소입니다. 기준선은 **V1 균일 상자 16개를 Isaac Sim에서 연속 이송한 실행**입니다. 2026-09-21 원본 실행과 2026-09-28(KST) 새 체크아웃 재실행에서 각각 16/16을 완료했습니다. 두 실행의 Python 파일 57개 해시는 일치하며 결과 요약은 `evidence/`에 있습니다. 새 실행은 같은 호스트에서 기존 Isaac Sim·cuRobo 가상환경과 외부 자산을 재사용했습니다.

이 결과의 대상 인식은 **시뮬레이터 정답값(oracle)** 을 사용했습니다. 실제 로봇 운전이나 카메라부터 계획까지의 독립 인지 파이프라인 성공을 뜻하지 않습니다. 다른 연구실의 새 설치에서 전체 실행까지 검증된 상태도 아닙니다. 외부 자산의 원본 manifest에는 당시 머신의 절대 경로가 들어 있어 새 워크스페이스에서는 경로를 맞춘 자산 번들과 재검증이 필요합니다. 검증 범위와 남은 조건은 [결과 기록](docs/RESULTS.md)에 있습니다.

## 시작

워크스페이스 루트를 `JCLEE_WORKSPACE`로 지정하고 저장소를 받습니다.

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
```

허가된 외부 자산과 커밋 `78fd485fa82d9b9a063fb4985e371814587e666a`의 cuRobo checkout을 준비합니다. NVIDIA 약관을 검토한 뒤 환경을 설치하고 실행 전 점검을 합니다.

```bash
CUROBO_SOURCE="$JCLEE_WORKSPACE/repos/external/github.com/NVlabs/curobo"
python3 tools/install.py --curobo-source "$CUROBO_SOURCE" --dry-run
python3 tools/install.py --curobo-source "$CUROBO_SOURCE" --accept-nvidia-eula
python3 tools/check_assets.py
python3 tools/run.py full --dry-run
```

기본 근거 검사는 해시로 고정된 불변 아카이브를 확인하고, `--check-current`는 현재 `src/`·`scripts/`도 당시 코드와 비교합니다. H2017, VGP20, Isaac Sim, cuRobo 설치물은 이 저장소에 없습니다. 준비 방법과 출력 경로는 [재현 절차](docs/REPRODUCE.md)를 참고하세요. 점검 후 GPU에서 한 상자 이송을 먼저 확인하고 전체 16개를 실행합니다.

```bash
python3 tools/run.py smoke --gpu 0
python3 tools/run.py full --gpu 0
```

Smoke 결과는 1/1 이송 성공일 때 `requested_prefix_passed=true`, `state=PREFIX_COMPLETE`입니다. 16개 중 1개만 완료하므로 `passed=false`, `task_complete=false`가 정상입니다.

## 저장소 구성

| 경로 | 내용 |
| --- | --- |
| `examples/v1_uniform/` | 16개 상자 시나리오와 규칙 기반 이송 순서 |
| `src/` | 장면, 계획, 동작, 그리퍼, 측정 상태 검사의 프로젝트 코드 |
| `scripts/` | Isaac 실행, cuRobo 작업자, GPU 실행 제한 |
| `tools/` | 예제 준비, 자산 점검, 실행, 근거 확인 진입점 |
| `evidence/` | 당시 결과 요약, 감사 요약, 불변 소스 아카이브와 해시 |
| `requirements/` | 당시 환경의 패키지 버전 기록 |

[모듈 안내](docs/MODULES.md)에서 수정할 위치를 찾을 수 있습니다. 변경 사항은 별도 브랜치와 Pull Request로 제안해 주세요. 절차는 [기여 안내](CONTRIBUTING.md)에 적었습니다.

코드와 외부 자산의 배포 범위는 [제3자 고지](THIRD_PARTY_NOTICES.md)를 확인하세요. 데이터, 체크포인트, 캐시, 로그와 실행 출력은 Git 밖의 `data/`, `cache/`, `runs/`에 둡니다.
