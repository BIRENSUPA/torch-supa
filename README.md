# Torch SUPA

Torch SUPA is the Biren PyTorch extension for BR2xx devices. It integrates Biren's SUPA backend with PyTorch through the `PrivateUse1` backend mechanism, so users can run training and inference workloads on Biren hardware while keeping a PyTorch-style programming experience.

Torch SUPA is designed for:

- running existing PyTorch models on Biren devices with minimal code changes;
- mapping CUDA-style device usage to the SUPA backend where compatibility is provided;
- supporting PyTorch native APIs, common training workflows, mixed precision, profiling, and distributed scenarios on Biren platforms;
- adapting selected PyTorch ecosystem libraries and custom operators for Biren hardware.

For complete installation and usage details, see the user guide under [`docs/user_guide/`](docs/user_guide/).

## Supported versions

Torch SUPA wheels are built for specific PyTorch versions. Install the matching CPU-only PyTorch and torchvision versions before installing or building Torch SUPA.

| PyTorch version | Torch SUPA version suffix | torchvision version | Python version |
| --- | --- | --- | --- |
| 2.12.0 | 1.0.0.21200 | 0.27.0 | 3.10 |
| 2.11.0 | 1.0.0.21100 | 0.26.0 | 3.10 |
| 2.10.0 | 1.0.0.21000 | 0.25.0 | 3.10 |
| 2.9.0 | 1.0.0.20900 | 0.24.0 | 3.10 |
| 2.8.0 | 1.0.0.20800 | 0.23.0 | 3.10 |
| 2.6.0 | 1.0.0.20600 | 0.21.0 | 3.10 |

> **Important:** Install the CPU-only PyTorch wheel. Device execution is provided by Torch SUPA and the SUPA software stack, not by the upstream PyTorch CUDA wheel.

Example:

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

## Prerequisites

Before using Torch SUPA, make sure the following components are available:

- a supported Linux environment, such as Ubuntu 22.04 or Ubuntu 24.04;
- a compatible Biren BR2xx device and driver;
- the matching Biren SUPA software stack and SDK from the official release package;
- Python 3.10 and `pip`;
- CPU-only PyTorch and torchvision versions that match the Torch SUPA wheel;
- SUDA runtime/tooling from the same release package when building or adapting native components.

Do not mix driver, SUPA SDK, SUDA, PyTorch, torchvision, and Torch SUPA packages from different release sets unless the release notes explicitly state that the combination is supported.

## Installation

### Option 1: Use an official Docker image

For most users, the recommended way to start is to use an official Docker image or release package that already contains the matching driver runtime libraries, SUPA SDK, SUDA dependencies, PyTorch, and Torch SUPA.

A typical container launch flow is:

```bash
docker load -i <birensupa-sdk-image>.tar

docker run -itd \
  --device /dev/biren/card_0 \
  --privileged \
  --name torch-supa-dev \
  <birensupa-docker-image>:<sdk-version>

docker exec -it torch-supa-dev bash
```

Replace the image name, tag, and device path according to your release package and machine configuration.

### Option 2: Install wheels in an existing environment

Use this method when the SUPA SDK and driver have already been installed on the host or in a container.

1. Install the matching CPU-only PyTorch and torchvision wheels.

   ```bash
   pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
   ```

2. Install SUDA and activate the SUPA SDK environment.

   ```bash
   cd <release-package>/full-stack
   pip install suda-<suda_version>-linux_x86_64.whl

   # Replace the path if your SDK is installed elsewhere.
   source /usr/local/birensupa/all/latest/scripts/brsw_set_env.sh
   ```

3. Install the Torch SUPA wheel that matches your PyTorch version.

   ```bash
   pip install torch_supa-1.0.0.<torch_version>+br2xx-cp310-cp310-linux_x86_64.whl
   ```

   For example, PyTorch 2.10.0 corresponds to the `21000` version suffix.

## Build from source

Source builds are intended for advanced users who need to develop, debug, or customize Torch SUPA. For regular training and inference workloads, prefer the official Docker image or released wheel.

```bash
git clone --recursive <official-torch-supa-repository-url>
cd torch-supa

pip install -r requirements.txt
suda init
python3 setup.py bdist_wheel
pip install dist/torch_supa*.whl
```

Common build options:

```bash
# Build a debug wheel with debug symbols.
DEBUG=on python3 setup.py bdist_wheel

# Disable FlashAttention operator library if it is not needed.
USE_FLASH_ATTENTION=off python3 setup.py bdist_wheel

# Build with isa
TORCH_USE_ISA=1 python3 setup.py bdist_wheel
```

When switching PyTorch versions, clean previous build artifacts before rebuilding:

```bash
CLEAN_TORCH_SUPA_OP=1 python3 setup.py clean
python3 setup.py bdist_wheel
```

## Troubleshooting

If installation or runtime verification fails, check the following items first:

- The PyTorch, torchvision, and Torch SUPA versions match the supported version table.
- The installed PyTorch wheel is the CPU-only wheel.
- The SUPA SDK environment script has been sourced in the current shell.
- The Biren device is visible in the host or container, for example through `/dev/biren/card_X`.
- Docker containers are started with the required device mapping and permissions.
- SUDA, SUPA SDK, driver, and Torch SUPA packages come from the same official release set.
- Source checkouts include all required submodules when building from source.

## License

Torch SUPA is distributed under the terms described in [`LICENSE`](LICENSE).
