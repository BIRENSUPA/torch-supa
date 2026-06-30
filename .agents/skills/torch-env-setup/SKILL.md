---
name: torch-env-setup
description: Configure torch-supa development and CI environments from build.sh and docs/user_guider. Use when users ask to set up env, install a specified CPU PyTorch/torchvision version, install a specified full-stack or LKG package, install SUDA, run suda init/setup, compile torch-supa/torch_supa wheels, or troubleshoot missing SUPA libraries after environment changes.
depends_on: []
---

# Env Setup Skill

Use this skill to prepare a local or container environment for `torch-supa` / `torch_supa` development, validation, and CI-style builds.

## Scope

This skill covers:

- installing the required **CPU** `torch` and matching `torchvision` version;
- installing a specified Biren full-stack/LKG package and `suda` wheel;
- sourcing Biren SUPA runtime environment scripts;
- running `suda init` and `suda setup`;
- installing repository OS/Python dependencies;
- compiling `torch-supa` / `torch_supa` into a wheel and installing it;
- verifying that `torch.supa` is available.

Do not use this skill for PyTorch source-version upgrades or compile-error repair. Keep this workflow focused on environment setup after the requested setup command itself is understood.

## Inputs to collect first

Ask for any missing required values before running install commands:

- Target PyTorch version, e.g. `2.8.0`, `2.9.0`, `2.10.0`.
- Python version, usually `3.10` for current docs.
- Target OS/container, e.g. `ubuntu-22.04` or `ubuntu-24.04`.
- Full-stack/LKG source:
  - Docker image/tag, or
  - local unpacked SDK path containing `full-stack/`, or
  - explicit `full-stack` directory path.
- SUDA wheel/version if not obvious from the full-stack directory.
- Device/card to expose to Docker when container setup is requested, e.g. `/dev/biren/card_0`.
- Build mode: `Release` or `Debug`, build tests `sanity`/`regression` if needed, arch such as `br100`/`br2xx` or `arch_20`, and parallel jobs.
- Whether optional build toggles are required: `BUILD_WITHOUT_BCCL=on`, `DEBUG=on`, `CLEAN_TORCH_SUPA=1`, `USE_CCACHE=0`.

## Source references in this repo

- `build.sh`: main build entry; installs dependencies via `.env_setup/deps_install.sh`, then runs `MAX_JOBS=<jobs> python3 setup.py bdist_wheel`.
- `.env_setup/deps_install.sh`: installs OS packages, Python requirements, and CPU torch/torchvision. It reads `PYTORCH_VERSION`, defaulting to `2.8.0`.
- `.env_setup/env_set.sh`: sources `/usr/local/birensupa/all/latest/scripts/brsw_set_env.sh`, runs `suda init`, then evaluates `/usr/local/bin/suda setup`.
- `docs/user_guider/source/chapter02_torch_supa_environments.rst`: official environment, Docker, SUDA, source-build, and verification flow.
- `docs/user_guider/source/chapter08_torch_supa_faq.rst`: missing-library and stale-cache fixes.

## Supported torch/torchvision mapping

Prefer the repository mapping in `.env_setup/deps_install.sh` when it differs from docs. At time of writing:

| torch | torchvision |
| --- | --- |
| 2.6.0 | 0.21.0 |
| 2.7.0 | 0.22.0 |
| 2.8.0 | 0.23.0 |
| 2.9.0 | 0.24.0 |
| 2.9.1 | 0.24.0 |
| 2.10.0 | 0.25.0 |

Important: install CPU torch only. SUPA acceleration is provided by `torch_supa`, not by a CUDA torch wheel.

## Workflow

### 1. Inspect current environment

Run read-only checks first and report mismatches:

```bash
python3 --version
python3 -m pip --version
python3 - <<'PY'
try:
    import torch
    print('torch', torch.__version__, torch.__file__)
except Exception as e:
    print('torch import failed:', repr(e))
PY
```

If using Docker, verify Docker is installed and that the requested Biren device path exists before suggesting `docker run`.

### 2. Install or enter full-stack environment

For a Docker image already loaded locally, use the documented pattern and substitute the requested card, name, OS, and LKG tag:

```bash
docker run -itd --device /dev/biren/card_0 --privileged --name <container-name> \
  <public-full-stack-image>:<lkg-version>
docker exec -it <container-name> bash
```

For an offline image archive, load it first, then verify the image name:

```bash
docker load -i <birensupa-sdk>.tar
docker image list
```

If the user provides a local SDK/LKG directory, identify the `full-stack/` subdirectory and install from there instead of guessing paths.

### 3. Install SUDA and source runtime environment

From the selected full-stack directory:

```bash
cd <your_lkg_pack>/full-stack
python3 -m pip install suda-<suda_version>-linux_x86_64.whl
source /usr/local/birensupa/all/latest/scripts/brsw_set_env.sh
suda init
eval $(/usr/local/bin/suda setup)
```

If `/usr/local/birensupa/all/latest/scripts/brsw_set_env.sh` is absent, do not continue blindly. Ask the user for the installed full-stack location or inspect the SDK install result.

### 4. Install CPU torch and dependencies

Preferred repo-native path:

```bash
export PYTORCH_VERSION=<torch-version>
bash .env_setup/deps_install.sh install
```

`deps_install.sh` handles OS dependencies, `requirements.txt`, and CPU `torch`/`torchvision` through its `install_uninstall_deps` flow. Its supported entrypoints are:

```bash
bash .env_setup/deps_install.sh install
bash .env_setup/deps_install.sh uninstall
```

Do not source `.env_setup/deps_install.sh` and call internal helper functions such as `install_uninstall_torch_torchvision` directly unless you are debugging the script itself.

Alternative public CPU-wheel path when internal wheel access is unavailable:

```bash
python3 -m pip install torch==<torch-version> torchvision==<torchvision-version> \
  --index-url https://download.pytorch.org/whl/cpu
```

After installation, verify that Python imports the intended CPU torch:

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.__file__)
print(getattr(torch.version, 'cuda', None))
PY
```

If `torch.version.cuda` indicates a CUDA wheel, uninstall `torch` and `torchvision` and reinstall CPU wheels.

### 5. Build torch-supa / torch_supa

Use `build.sh` when you want the repo-standard build and test packaging flow:

```bash
PYTORCH_VERSION=<torch-version> bash build.sh -b Release -j 16
```

Useful variants:

```bash
PYTORCH_VERSION=<torch-version> bash build.sh -b Debug -j 16
PYTORCH_VERSION=<torch-version> bash build.sh -a br100 -b Release -t sanity -j 16
PYTORCH_VERSION=<torch-version> bash build.sh -a br100 -b Release -t regression -j 16
```

For direct source build from docs:

```bash
python3 -m pip install -r requirements.txt
suda init
python3 setup.py bdist_wheel
python3 -m pip install dist/torch_supa*.whl
```

Optional toggles from the docs and FAQ:

```bash
DEBUG=on python3 setup.py bdist_wheel
BUILD_WITHOUT_BCCL=on python3 setup.py bdist_wheel
CLEAN_TORCH_SUPA=1 python3 setup.py clean
python3 setup.py bdist_wheel
```

Do not silently disable major components such as BCCL to make a build pass unless the user explicitly wants an isolated/local build.

### 6. Verify installation

Run the minimal verification after installing the built wheel:

```bash
python3 - <<'PY'
import torch
print('torch', torch.__version__)
print('supa available:', torch.supa.is_available())
x = torch.rand(2, 3, device='cuda')
print(x.device)
PY
```

Expected results:

- `torch.supa.is_available()` prints `True`;
- a CUDA device request maps to a `supa:*` device;
- import may print messages that CUDA APIs are replaced by SUPA APIs. Treat those messages as normal initialization output, not a failure.

## Troubleshooting

### Missing `libcu*.so` or SUPA shared libraries

1. Confirm SUDA/full-stack is installed.
2. Source the runtime environment:

```bash
source /usr/local/birensupa/all/latest/scripts/brsw_set_env.sh
```

3. Re-run `suda init` and `eval $(/usr/local/bin/suda setup)` if needed.

### Undefined functions or stale symbols after switching torch version

Clean stale build products before rebuilding:

```bash
CLEAN_TORCH_SUPA=1 python3 setup.py clean
python3 setup.py bdist_wheel
```

### Wrong torch version imported

Print `torch.__file__`, uninstall both user and sudo/system installs if necessary, then reinstall the CPU wheel. The repo script already attempts both normal and sudo uninstall paths.

### Dependency install modifies system packages

`install_uninstall_deps_by_os install` uses `sudo` and changes OS packages. State this clearly before running it. Prefer Python virtual environments for pip packages when possible.

## Done criteria

Environment setup is complete only when:

- the requested CPU torch/torchvision versions are installed and import-verified;
- SUDA is installed, `brsw_set_env.sh` is sourced, and `suda setup` succeeds;
- the repo dependencies are installed;
- `python3 setup.py bdist_wheel` or `bash build.sh ...` succeeds;
- the produced `dist/torch_supa*.whl` is installed;
- `torch.supa.is_available()` returns `True` in the target shell/container.
