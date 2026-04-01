# AGENTS.md — QUICK Quantum Chemistry Package

## Overview

QUICK is a GPU-accelerated quantum chemistry package written primarily in Fortran 90/95,
with C/C++ and CUDA/HIP code for GPU support. The codebase uses two parallel build systems:
a legacy `configure`+`make` system and a modern CMake system.

---

## Build Commands

### CMake Build (recommended)

```bash
mkdir build && cd build
cmake .. -DCOMPILER=GNU -DENABLEF=FALSE -DCMAKE_INSTALL_PREFIX=$PWD/../install
cmake --build . --parallel $(nproc)   # Linux: nproc; macOS: sysctl -n hw.logicalcpu
cmake --install .
source ../install/quick.rc   # sets QUICK_BASIS, PATH, LD_LIBRARY_PATH, LIBRARY_PATH
```

> **macOS + Homebrew GCC note:** On macOS, `/usr/bin/gcc` is an Apple Clang shim, so
> `-DCOMPILER=GNU` will be rejected by the build system's compiler identity check.
> Use `-DCOMPILER=AUTO` and pass the real compiler paths explicitly:
> ```bash
> cmake .. -DCOMPILER=AUTO -DENABLEF=FALSE \
>   -DCMAKE_INSTALL_PREFIX=$PWD/../install \
>   -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-15 \
>   -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-15 \
>   -DCMAKE_Fortran_COMPILER=/opt/homebrew/bin/gfortran
> cmake --build . --parallel $(sysctl -n hw.logicalcpu)
> cmake --install .
> source ../install/quick.rc
> ```
> Adjust `gcc-15`/`g++-15` to match the version installed by Homebrew
> (`ls /opt/homebrew/bin/gcc-*`).

**Key CMake options:**

| Option | Values | Description |
|---|---|---|
| `-DCOMPILER` | `GNU`, `CLANG`, `INTELLLVM`, `ONEAPI`, `PGI`, `AUTO` | Compiler family |
| `-DENABLEF` | `TRUE`/`FALSE` (default: `FALSE`) | Enable f-function ERIs (expensive) |
| `-DCMAKE_BUILD_TYPE` | `Debug`, `Release` | Build type |
| `-DMPI` | `TRUE` | Enable MPI parallel build |
| `-DCUDA` | `TRUE` | Enable NVIDIA GPU build |
| `-DQUICK_USER_ARCH` | `volta`, `turing`, `ampere`, `adalovelace`, `hopper`, etc. | CUDA GPU architecture |
| `-DWARNINGS` | `TRUE` | Enable compiler warnings |

To build with AMD GPUs, add `-DHIP=TRUE` (plus HIP-specific flags like `-DHIP_WARP64=TRUE`). See `quick-cmake/QUICKCudaConfig.cmake` for details.

See `CMake-Options.md` for the full list.

### Legacy Configure+Make Build

```bash
./configure --serial --enablef --shared --prefix $PWD/install gnu
make -j$(nproc) all install
source install/quick.rc   # sets QUICK_BASIS, PATH, LD_LIBRARY_PATH, LIBRARY_PATH
```

MPI variant: replace `--serial` with `--mpi`.
CUDA variant: replace `--serial` with `--cuda --arch volta` (architectures: `kepler`, `maxwell`, `pascal`, `volta`, `turing`, `ampere`, `adalovelace`, `hopper`).
HIP variants use `--hip` (single GPU) or `--hipmpi` (MPI + HIP). Multi-GPU CUDA builds use `--cudampi`.

---

## Test Commands

Tests require QUICK to be **installed** first. The `runtest` script lives in the install
directory (or can be run from the repo root as `tools/runtest`). `QUICK_BASIS` must be set
(done automatically by `source install/quick.rc`).

### Run the full test suite

```bash
cd install
./runtest --serial --full        # 186 CPU tests
```

### Run the short test suite (CI default)

```bash
./runtest --serial               # short set (31 tests)
```

### Run a single test

QUICK has no built-in single-test flag. Run the executable directly and compare via `test/ndiff.awk`:

```bash
export QUICK_BASIS=/path/to/install/basis
/path/to/install/bin/quick test/ene_H2O_rhf_sto3g.in
awk -f test/ndiff.awk test/saved/ene_H2O_rhf_sto3g.out ene_H2O_rhf_sto3g.out
```

### Run tests by category

```bash
./runtest --serial --ene    # energy tests only
./runtest --serial --grad   # gradient tests only
./runtest --serial --opt    # geometry optimization tests only
./runtest --serial --api    # API tests only
./runtest --serial --esp    # ESP tests only
./runtest --serial --chk    # checkpoint/restart tests only
```

### MPI tests

```bash
DO_PARALLEL="mpirun -np 2" ./runtest --mpi --full
```

### Useful environment variables

```bash
QUICK_BASIS            # path to basis set directory (required)
DO_PARALLEL            # MPI launcher, e.g. "mpirun -np 2"
CUDA_VISIBLE_DEVICES   # GPU IDs for CUDA tests (comma-separated)
HIP_VISIBLE_DEVICES    # GPU IDs for HIP tests (comma-separated)
PARALLEL_TEST_COUNT    # number of tests to run in parallel (GNU parallel)
```

---

## Diagnosing Failing Tests

### Test harness tolerances

The `runtest` script compares output against saved references using `test/dacdif`, which
calls `test/ndiff.awk` with a hardcoded absolute-error threshold per check type. There is
no per-test override; the thresholds are set in `tools/runtest`:

| Check function | Absolute-error threshold | Quantity compared |
|---|---|---|
| `check_energy` | `4.0e-5` | Total / electronic / nuclear repulsion energies (Hartree) |
| `check_gradient` | `4.0e-3` | Analytical gradient components (Hartree/Bohr) |
| `check_opt` | `4.0e-3` | Optimized geometry and force elements |
| `check_dipole` | `4.0e-3` | Mulliken/Löwdin charges and dipole (Debye) |
| `check_esp_charge` | `1.0e-5` | ESP-fitted atomic charges |
| `check_vdw_surface` | `2.0e-7` | ESP values on vdW surface |
| `check_esp_grid` | `2.0e-7` | ESP values on external grid point file |
| `check_chk` (density restart) | `1.0e-8` | Total energy after density-matrix restart |
| `check_chk` (xyz restart) | `2.0e-4` | Cartesian coordinates after xyz restart |

### Physics cutoff keywords

The following keywords in `.in` files control integral screening thresholds and directly
affect numerical results across platforms and compiler configurations:

| Keyword | Internal default | Standard test value | Tightest value used |
|---|---|---|---|
| `cutoff=` | `1.0e-7` | `1.0e-9` | `1.0e-12` |
| `gradcutoff=` | `1.0e-7` | *(not set explicitly in any test)* | `1.0e-12` |
| `xccutoff=` | `1.0e-7` | *(not set in most tests)* | `1.0e-12` |
| `basiscutoff=` | `1.0e-6` | *(not set in most tests)* | `1.0e-12` |

`gradcutoff=` applies only to gradient and geometry optimization tests. `xccutoff=` and
`basiscutoff=` apply only to DFT tests. `denserms=` (SCF convergence criterion) is
intentionally excluded — it controls iteration count, not the numerical precision of the
converged result, so tightening it does not help platform-consistency failures.

### Diagnostic workflow for a borderline failure

If a test fails with a numeric deviation slightly above the threshold:

1. **Find the exact deviation.** In `runtest-verbose.log` (or the `test/runs/` directory),
   locate the `ndiff.awk` summary line for the failing test:
   ```
   ### Maximum absolute error in matching lines = X.XXe-YY at line N field M
   ```

2. **Make a scratch copy** of the failing input and tighten the physics cutoffs. Apply
   only the keywords relevant to the test type:
   ```
   # All test types:
   cutoff=1.0e-12

   # Gradient and geometry optimization tests — also add:
   gradcutoff=1.0e-12

   # DFT tests — also add:
   xccutoff=1.0e-12
   basiscutoff=1.0e-12
   ```

3. **Run the binary directly** on the scratch copy:
   ```bash
   export QUICK_BASIS=/path/to/install/basis
   /path/to/install/bin/quick <scratch_copy>.in
   ```

4. **Compare against the saved reference** using `dacdif` with the threshold matching the
   failing check type (see table above). For example, for an energy test:
   ```bash
   test/dacdif -k -a 4.0e-5 test/saved/<stem>.out <scratch_copy>.out
   ```

5. **Interpret the result:**
   - Failure disappears → the production `.in` file's cutoffs are too loose for this
     platform/configuration. Update the cutoffs in `test/<stem>.in` and regenerate the
     reference output in `test/saved/<stem>.out`.
   - Failure persists → the root cause is elsewhere (algorithm change, compiler
     floating-point behaviour, platform ABI). Investigate the logic change or consider
     regenerating the reference output after confirming correctness.

---

## Code Style Guidelines

### Languages

- **Fortran 90/95** — primary language (all SCF, DFT, gradients, MPI, API code)
- **C/C++** — utility subroutines, octree/grid packer, timing code
- **CUDA C/C++** — NVIDIA GPU kernels (`src/gpu/cuda/`)
- **HIP C/C++** — AMD GPU kernels (`src/gpu/hip/`)

### Fortran Naming Conventions

- **Modules:** `quick_<component>_module` — file named `quick_<component>_module.f90`
  - Examples: `quick_method_module`, `quick_basis_module`, `quick_exception_module`
- **Types:** `quick_<component>_type` — e.g., `quick_method_type`, `gpu_calculated_type`
- **Module instances (singletons):** lowercase — e.g., `quick_method`, `quick_molspec`
- **Subroutines/Functions:** No single style is enforced. `snake_case` is the majority
  convention across both old and new code (`raise_exception`, `form_dft_grid`,
  `allocate_quick_scf`). `camelCase` and `PascalCase` appear in legacy files and some
  newer modules (`getEnergy`, `PrtAct`, `gridformSG0`, `printQuickOutput`). When adding
  new code, prefer `snake_case`.
- **Local variables:** lowercase, short abbreviations — e.g., `natom`, `nbasis`, `ierr`
- **Type member variables:** mixed styles coexist — camelCase (`integralCutoff`,
  `analGrad`), ALL_CAPS (`HF`, `DFT`, `MP2`), lowercase (`opt`, `grad`), and snake_case
  (`read_coord`, `esp_charge`). No single rule applies.
- **Constants/Parameters:** `ALL_CAPS` — e.g., `PI`, `BOHR`, `OUTFILEHANDLE`
- **Preprocessor macros:** `ALL_CAPS` — e.g., `RECORD_TIME`, `MPIV`, `ENABLEF`, `DEBUG`

### C/C++/CUDA Naming Conventions

- **Structs/Types:** predominantly `snake_case` with `_type` suffix — e.g.,
  `gpu_timer_type`, `gpu_simulation_type`. Some structs omit the suffix (`gpu_scratch`)
  or use mixed-case prefixes (`XC_quadrature_type`, `ERI_entry`).
- **Functions:** `snake_case` or `camelCase` — both styles appear freely with no clear
  rule (`get_oshell_eri`, `upload_sim_to_constant` vs. `getOEI`, `getGrad`).
- **Macros:** `ALL_CAPS` — e.g., `LOC2`, `LOC3`, `SQR`, `VDIM1`
- **Fortran-callable C functions:** use `extern "C"` with trailing underscore —
  e.g., `gpu_set_device_`, `gpu_upload_method_`

### File Naming

- Fortran modules: `quick_<component>_module.f90`
- Fortran subroutines: predominantly `snake_case.f90`; some legacy files use
  `camelCase.f90` or `PascalCase.f90` (`getEnergy.f90`, `CPHF.f90`)
- CUDA/C++ source: `snake_case.cu`, `snake_case.cpp`, `snake_case.h`
- GPU headers: `gpu_<purpose>.h` or `gpu_<component>_<type>.h`

### Indentation and Formatting

- **Fortran:** 3-space indentation (dominant convention); some legacy files use 2 or 4 spaces
- **C/CUDA:** 4-space indentation; opening `{` on same line as control structure
- **CMake:** 4-space (1-tab) indentation
- New Fortran code should use `implicit none` in every scope. Many legacy subroutines
  use `implicit double precision(a-h,o-z)` instead — do not add new code with implicit
  typing.
- Long lines are allowed (compiler flag `-ffree-line-length-none` is set for GNU Fortran)
- The `tools/amindent` utility can be used to normalize Fortran indentation

### Include/Use Ordering

**Fortran `use` statements** — place at the top of the program unit, before `implicit none`:
1. Standard/external library modules (e.g., `use mpi`)
2. QUICK utility modules (e.g., `use allMod`, `use quick_constants_module`)
3. Functional QUICK modules (e.g., `use quick_method_module, only: quick_method`)

**Preprocessor include:** Every Fortran source file must include the project-wide utility
header near the very top (before the `module`/`subroutine` statement):
```fortran
#include "util.fh"
```
This header (`src/util/util.fh`) provides timing macros (`RECORD_TIME`, `START_TIME`,
`STOP_TIME`) and the `OUTFILEHANDLE` constant.

**C/CUDA headers:** Standard library headers first, then local project headers:
```c
#include <stdio.h>
#include <string>
#include "gpu.h"
#include "../gpu_common.h"
```

### Error Handling

- **Fortran:** Use an integer error flag `ierr` declared `intent(inout)` and passed through
  the call chain. Non-zero values indicate errors. Check and raise via:
  ```fortran
  call RaiseException(ierr)   ! from quick_exception_module
  ```
  This calls `quick_exit(OUTFILEHANDLE, 1)` on failure. Error codes are defined in
  `src/modules/quick_exception_module.f90` with `select case(ierr)` messages.
- **MPI code:** Wrap MPI-only logic in `#ifdef MPIV` / `#endif` preprocessor guards.
  Only the master process should do I/O:
  ```fortran
  if (master) write(OUTFILEHANDLE, ...) ...
  ```
- **CUDA:** Check CUDA API return codes explicitly within GPU utility functions.

### Comment Style

**Fortran:**
```fortran
! Single-line comment

!-----------------------------------------------------------------------!
! Section header banner (width ~72 chars)                               !
!_______________________________________________________________________!

subroutine foo(x, y)  ! inline comment after code
```

**C/CUDA:**
```c
// Single-line comment
/* Block comment for longer explanations */
```

Every source file should carry a copyright/license header (MPLv2):
```fortran
!
! (C) Copyright <year> QUICK contributors
! All rights reserved.
! ...
! This Source Code Form is subject to the terms of the Mozilla Public
! License, v. 2.0.
!
```

### MPI/GPU Portability

- CPU-only, MPI, CUDA, and HIP builds share the same source tree. Use preprocessor guards:
  - `#ifdef MPIV` for MPI-specific code
  - `#ifdef CUDA_MPIV` for CUDA+MPI code
- GPU-callable device functions are annotated with `__device__` / `__global__` (CUDA) or
  the HIP equivalents.
- Fortran-to-GPU interoperability is done via `iso_c_binding` and `extern "C"` interfaces.

---

## Repository Layout (Key Paths)

```
src/modules/         Fortran modules (quick_*_module.f90) — all shared state/types
src/subs/            Fortran utility subroutines and C++ utility files
src/gpu/cuda/        NVIDIA CUDA kernels and GPU driver (gpu.cu)
src/gpu/hip/         AMD HIP kernels
src/octree/          C++ octree for DFT quadrature grids
src/libxc/           DFT XC functional library (libxc)
src/util/util.fh     Project-wide Fortran preprocessor header (MUST include)
quick-cmake/         Amber build-system glue and GPU helpers
test/                Regression test inputs (*.in), GPU lists, references (saved/*.out)
test/testlist_full.txt      Full CPU test list (~186 tests)
test/testlist_short.txt     Short CPU test list (31 tests)
test/testlist_full_gpu.txt  Full GPU test list (~187 tests)
test/testlist_short_gpu.txt Short GPU test list
tools/runtest        Test runner script
basis/               Basis set data files
.github/workflows/   CI definitions (serial + MPI)
unit-tests/          Unit/prototype harnesses
CMake-Options.md     Reference for all CMake build options
```

---

## CI

GitHub Actions workflows are in `.github/workflows/`:
- `build_test_serial.yml` — Serial builds (legacy + CMake) on Ubuntu 22.04/24.04 (x86/ARM) and macOS 14/15 across GNU 10–15, Clang 17/18/20, IntelLLVM 2024/2025, and NVHPC 25.1.
- `build_test_mpi.yml` — MPI builds (legacy + CMake) on the same OS spread using OpenMPI, MPICH, and Intel-MPI toolchains.

Each workflow installs HDF5 when needed, builds QUICK, runs `./runtest --serial --full` or `./runtest --mpi --full`, and archives `runtest.log`, `runtest-verbose.log`, and `test/runs/*` outputs.
