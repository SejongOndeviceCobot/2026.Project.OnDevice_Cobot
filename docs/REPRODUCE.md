# V1 재현 절차

이 저장소에는 재배치 전 두 번의 16/16 결과와 기능별로 정리한 코드의 16/16 재시도 결과가 있습니다. 모듈식 성공은 **같은 호스트의 기존 가상환경·외부 자산을 재사용한 실행**입니다. 다른 연구실에서 실행하려면 NVIDIA GPU의 Isaac Sim 환경, 허가된 H2017·VGP20 자산, cuRobo가 별도로 필요합니다. Git clone만으로 전체 재현이 완료되지는 않습니다.

1. 워크스페이스를 준비하고 예제 입력과 보존 근거를 확인합니다.

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

`verify_evidence.py` 기본 검사는 보존된 세 완료 실행의 결과·소스 스냅샷·감사 요약을 확인합니다. `--check-current`는 현재 소스의 목록·해시·import를, `--check-release-source`는 현재 소스 66개와 모듈식 성공 실행 당시 소스의 동일성을 검사합니다. 이 검사와 CPU 테스트는 물리 실행을 새로 수행하지 않습니다. `prepare_example.py`는 `examples/v1_uniform/` 입력을 프로젝트 데이터 위치에 준비합니다.

2. H2017·VGP20 자산 번들의 **승인된 접근 경로와 사용 권한을 컨소시엄 조직 담당자에게 확인**한 뒤, 자산을 `$JCLEE_WORKSPACE/data/depallet_isaac_p0`에 맞추고 외부 Doosan 소스 checkout을 준비합니다. 이 저장소는 CAD/USD, 모델 가중치, 원본 기록을 제공하지 않습니다. 원본 자산 manifest·URDF의 절대 경로와 해시가 당시 `/DATA/jclee/workspace` 등을 가리켜 새 워크스페이스에 단순 복사하면 점검이 실패할 수 있습니다. 다른 연구실의 독립 전체 재실행은 아직 확인되지 않았습니다.

3. cuRobo 소스를 지정 커밋으로 고정합니다. 이미 checkout이 있다면 clone은 생략합니다. `tools/install.py`는 Python 3.12, `uv`, 캐시·가상환경용 빈 디스크 공간 **최소 100 GiB**가 필요하며 `requirements/*.lock` 기준으로 두 가상환경을 설치합니다. NVIDIA Isaac Sim 약관을 직접 읽고 동의한 경우에만 `--accept-nvidia-eula`를 사용하세요.

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
command -v nvidia-smi ffmpeg ffprobe
python3 tools/check_assets.py
python3 tools/run.py full --dry-run
```

GPU 실행에는 세 시스템 도구가 모두 필요합니다. `run.py --dry-run`은 이 도구의 존재 여부까지 검사하지 않으므로 위 명령으로 먼저 확인하세요. `check_assets.py`는 외부 자산 59개의 누락, 절대 경로, 해시 불일치를 보고합니다. Doosan H2017 원본 USD는 존재·형식과 현재 SHA-256을 보고하지만 당시 해시는 고정돼 있지 않습니다. `--dry-run`은 GPU 작업을 시작하지 않습니다.

5. 한 상자 이송으로 실행 경로를 확인합니다.

```bash
python3 tools/run.py smoke --gpu 0
```

`$JCLEE_WORKSPACE/runs/<checkout-directory-name>/smoke-<timestamp>/task-result.json`에서 첫 상자 1/1 이송을 확인합니다. 예상 값은 `requested_prefix_passed=true`, `state=PREFIX_COMPLETE`입니다. 전체 작업은 끝나지 않았으므로 `passed=false`, `task_complete=false`가 정상입니다.

6. 전체 16개 시뮬레이션을 실행하고 **현재 코드의 새 결과**를 확인합니다.

```bash
python3 tools/run.py full --gpu 0
```

출력은 `$JCLEE_WORKSPACE/runs/<checkout-directory-name>/full-<timestamp>`, 캐시는 `$JCLEE_WORKSPACE/cache/<checkout-directory-name>`에 둡니다. 새 `task-result.json`의 완료 상자 수, `passed`, `physical_task_complete`, `perception_source`, 실패 이유와 사용한 코드 커밋을 기록하세요. 기존 기록과 비교할 때는 [결과 기록](RESULTS.md)의 조건을 참고합니다. **현재 모듈 구조의 재현 여부는 이 새 실행 결과로만 판정합니다.**
