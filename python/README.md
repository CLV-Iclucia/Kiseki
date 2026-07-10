# Kiseki Python Bindings

## Prerequisites

- Python 3.8+
- CMake 3.20+
- MSVC (Visual Studio 2022)
- vcpkg with packages: tbb, eigen3, gtest, glfw3, glad, pybind11
- Set `VCPKG_ROOT` environment variable to your vcpkg installation

## Developer Setup (Recommended)

Build in CLion first, then run the dev setup script:

```bash
# 1. Build kiseki target in CLion (Ctrl+F9)

# 2. One-time: link the build output to Python
python dev_setup.py

# 3. Verify
python -c "import kiseki; print(kiseki.__version__)"
```

After this, every CLion Build automatically updates the `.pyd` 鈥?Python picks
up the changes immediately (no reinstall needed).

### How it works

`dev_setup.py` creates a `.pth` file in your Python site-packages that points
to the directory containing `kiseki.pyd`. Python's import system follows
`.pth` files automatically.

## User Install (pip)

Requires a Visual Studio Developer Command Prompt (for MSVC linker access):

```bash
# Open "x64 Native Tools Command Prompt for VS 2022", then:
cd G:\Kiseki
set VCPKG_ROOT=C:\path\to\vcpkg
pip install .
```

This invokes CMake under the hood via scikit-build-core and installs
`kiseki.pyd` into site-packages.

## Verify Installation

```bash
python -c "import kiseki; print(kiseki.__version__)"
# Expected: 0.1.0
```

## Running Examples

```bash
python python/examples/cube_drop.py
python python/examples/bunny_bounce.py
python python/examples/prescribed_motion.py
```

## Uninstall

```bash
# If installed via pip:
pip uninstall kiseki

# If using dev_setup.py: delete the .pth file
python -c "import site; from pathlib import Path; f=Path(site.getsitepackages()[0])/'kiseki-dev.pth'; f.unlink() if f.exists() else None; print('Removed' if not f.exists() else 'Not found')"
```
