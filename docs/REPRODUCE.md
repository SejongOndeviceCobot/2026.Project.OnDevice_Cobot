# V1 재현 절차

현재 저장소는 성공 실행의 **코드와 요약 근거**를 보존합니다. 새 연구실에서 전체 실행을 시작하려면 NVIDIA GPU가 있는 Isaac Sim 환경과, 별도로 사용 허가를 받은 H2017·VGP20·cuRobo 자산이 필요합니다. Git clone만으로 16/16 재현이 완료되는 구성은 아닙니다.

1. 워크스페이스를 준비하고 이 저장소를 `repos/own/2026.Project.OnDevice_Cobot`에 clone합니다. 다른 이름으로 둔 경우 아래 `cd` 경로만 바꿉니다. 환경을 불러온 뒤 코드 근거와 예제 입력을 확인합니다.

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

`verify_evidence.py`는 기본적으로 불변 `evidence/full16-source.tar.gz`의 57개 파일, 해시 manifest, 결과·감사 요약을 확인합니다. `--check-current`는 현재 `src/`·`scripts/`도 당시 스냅샷과 비교합니다. 두 검사 모두 물리 시뮬레이션을 다시 실행하지 않습니다. `prepare_example.py`는 `examples/v1_uniform/` 입력을 프로젝트 데이터 위치에 준비합니다.

2. 허가된 H2017·VGP20 자산 번들을 `$JCLEE_WORKSPACE/data/depallet_isaac_p0`에 맞추고 외부 Doosan 소스 checkout을 준비합니다. 필요한 파일과 해시는 `check_assets.py`가 알려줍니다. 이 저장소는 해당 CAD/USD, 모델 가중치, 원본 기록을 제공하지 않습니다. 원본 manifest와 URDF에는 당시 `/DATA/jclee/workspace` 등 절대 경로가 들어 있고 파일 해시도 고정되어 있습니다. 다른 루트에 단순 복사하면 점검이 실패합니다. 새 위치용 자산을 재생성·재검증하는 절차는 아직 이 저장소에 없어, 다른 연구실의 독립 전체 재실행은 검증되지 않았습니다.

3. cuRobo 소스를 아래 위치에 두고 지정 커밋으로 고정합니다. 이미 checkout이 있으면 clone은 생략합니다. `tools/install.py`는 Python 3.12, `uv`, 충분한 디스크 공간이 필요하며 `requirements/*.lock` 기준으로 두 가상환경을 설치합니다. NVIDIA Isaac Sim 약관을 직접 읽고 동의한 경우에만 실제 설치 명령의 `--accept-nvidia-eula`를 사용합니다.

```bash
mkdir -p "$JCLEE_WORKSPACE/repos/external/github.com/NVlabs"
git clone https://github.com/NVlabs/curobo.git \
  "$JCLEE_WORKSPACE/repos/external/github.com/NVlabs/curobo"
CUROBO_SOURCE="$JCLEE_WORKSPACE/repos/external/github.com/NVlabs/curobo"
git -C "$CUROBO_SOURCE" checkout --detach 78fd485fa82d9b9a063fb4985e371814587e666a
python3 tools/install.py --curobo-source "$CUROBO_SOURCE" --dry-run
python3 tools/install.py --curobo-source "$CUROBO_SOURCE" --accept-nvidia-eula
```

4. 자산과 실행 명령을 확인합니다.

```bash
python3 tools/check_assets.py
python3 tools/run.py full --dry-run
```

`check_assets.py`는 59개 외부 자산 파일의 누락, 절대 경로 불일치, 해시 불일치를 보고합니다. Doosan H2017 원본 USD는 형식과 존재를 확인하고 SHA-256을 보고하지만, 당시 값으로 고정된 해시는 없습니다. `run.py --dry-run`은 실제 GPU 작업을 시작하지 않고 사용할 명령과 가상환경 존재 여부를 출력합니다.

5. 먼저 한 상자 이송을 실행합니다.

```bash
python3 tools/run.py smoke --gpu 0
```

`$JCLEE_WORKSPACE/runs/<checkout-directory-name>/smoke-<timestamp>/task-result.json`에서 1/1 이송 완료를 확인합니다. `requested_prefix_passed=true`, `state=PREFIX_COMPLETE`가 예상 값입니다. 16개 중 1개만 끝났으므로 `passed=false`, `task_complete=false`가 정상입니다.

6. Smoke를 확인한 뒤 전체 16개 시뮬레이션을 실행합니다.

```bash
python3 tools/run.py full --gpu 0
```

두 명령 모두 권장되는 단독 GPU guard를 기본으로 사용합니다. 당시 전체 성공 실행의 기록은 공유 GPU 슬롯 0이었으며, 작업 인자는 `evidence/original-command.json`에서 확인할 수 있습니다. 전체 실행 출력은 `$JCLEE_WORKSPACE/runs/<checkout-directory-name>/full-<timestamp>`에, 캐시는 `$JCLEE_WORKSPACE/cache/<checkout-directory-name>`에 둡니다. 실행 후 새 `task-result.json`의 완료 상자 수, `passed`, `perception_source`, 실패 이유를 함께 기록하세요. 기존 16/16 결과와 같은 조건을 확인하려면 [결과 기록](RESULTS.md) 및 `evidence/full16-task-result.json`을 비교합니다. 실행 재현 여부는 새 결과로만 판정합니다.
