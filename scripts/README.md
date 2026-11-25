# Collect CuTe DSL compiler artifacts

Run from the repository root with a CuTe DSL build that includes [CUTLASS PR #3702](https://github.com/NVIDIA/cutlass/pull/3702) and a CUDA `nvdisasm` executable:

```sh
CUDA_VISIBLE_DEVICES='' python scripts/collect_artifacts.py src/sample/sample.py \
  --out-dir ./compiled/sample --zip ./compiled/sample.zip --arch sm_80 --nvdisasm nvdisasm
```

The collector executes the **trusted** Python file and expects its `compile_for_explorer()` function to return the result of `cute.compile(...)`. The sample passes three fake FP32 tensor descriptors and `no_jit_engine=True`; it never allocates GPU tensors or runs the compiled function. The sample adds 4,096 contiguous elements using 32 blocks of 128 threads. `--arch` sets the compile target, not the device; `--keep-cubin` copies the binary to the output directory if wanted. Use a new output directory if you want to keep previous artifacts.

The collector sets `CUTE_DSL_KEEP=ir,ir-debug`, `CUTE_DSL_LINEINFO=1`, and the PTX/CUBIN retention flags `CUTE_DSL_KEEP_PTX=1` and `CUTE_DSL_KEEP_CUBIN=1` before importing CUTLASS. It sets the equivalent `CUTE_EXPERIMENTAL_DSL_*` variables for `@cute.experimental.jit` kernels as well. It reads the compiled object's `artifacts` and uses a temporary directory for compiler dumps and disassembly input.

`--zip` packages just the five viewer text artifacts (Python, raw MLIR, clean MLIR, PTX and SASS), not stale files in `--out-dir` or the optional CUBIN. Open that ZIP with **Open files**. For an existing kernel without a `compile_for_explorer()` entry point, use a separate compile adapter and pass `--artifact-source path/to/kernel.py` so the archive contains the original Python file instead of the adapter. See the [async-compute reproduction command](../examples/README.md#regenerate-the-zip).

| Output | Provenance and source mapping |
| --- | --- |
| `sample.py` | The exact source passed to the compiler. |
| `sample.raw.mlir` | `JitFunctionArtifacts.MLIR_RAW`, before the dump passes. MLIR uses `loc(...)` and `#loc` aliases with Python filename, line and column. |
| `sample.mlir` | `JitFunctionArtifacts.MLIR`, the inspection dump after `canonicalize` and `cse` on a clone. PR #3702 preserves source locations in this clone via MLIR bytecode. This is not the full compilation pipeline's final lowered IR. |
| `sample.ptx` | `JitFunctionArtifacts.PTX`. `.file` resolves file IDs; `.loc` provides file ID, line and column. |
| `sample.sass` | `nvdisasm --print-code --print-line-info` on `JitFunctionArtifacts.CUBIN`. `//## File "…", line N` comes from the ELF `.debug_line`; mapping applies to subsequent instructions until the next marker. |
| `sample.cubin` | Optional `JitFunctionArtifacts.CUBIN` bytes. Not in the bundled sample. |

The collector leaves generated text unchanged. Source paths normally reflect the build machine; the checked-in examples replace those paths with repository-relative names for distribution without changing line or column numbers.

## Bundled sample provenance

- Python 3.12.13 and the `nvidia-cutlass-dsl` 4.8.0 wheel. The CUTLASS checkout and installed wheel were not modified.
- The 4.8.0 wheel lacked `MLIR_RAW`. The sample was built with the PR's exact Python changes from commit [`84aa3a3bad3a18b6222fddc93d584189582714c2`](https://github.com/NVIDIA/cutlass/commit/84aa3a3bad3a18b6222fddc93d584189582714c2) applied to a temporary copy of the wheel's `cutlass` package. The patch was byte-identical to the PR patch (SHA-256 `f4d2a307387d0aec530b2aa2d7703db796dad7f418612278b62022c1deda82a2`).
- PTX reports CUDA 13.4, NVVM 23.0.0, compiler build `CL-38657139`, PTX ISA 9.4, target `sm_80`. SASS was generated with `nvdisasm` 12.9.88. It warns about the newer standard ELF format and disables dataflow analysis, but still emits the Python line annotations.

If using an unpatched 4.8.0 CuTeDSL environment with the CUDA 13 runtime, activate it and set up a temporary overlay. The overlay's `sitecustomize.py` takes precedence over the venv's `.pth` entries, which otherwise put the unpatched wheel ahead of `PYTHONPATH`.

```sh
py=$(command -v python)
export NVDISASM=$(command -v nvdisasm)
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
overlay=$(mktemp -d "${TMPDIR:-/tmp}/cute-pr3702.XXXXXX")
wheel=$(env -u PYTHONPATH "$py" -c 'import cutlass; from pathlib import Path; print(Path(cutlass.__file__).parent)')
cp -a "$wheel" "$overlay/"
curl -fsSL https://github.com/NVIDIA/cutlass/commit/84aa3a3bad3a18b6222fddc93d584189582714c2.patch \
  -o "$overlay/pr3702.patch"
git -C "$overlay" apply -p3 --include='cutlass/base_dsl/*' "$overlay/pr3702.patch"
printf '%s\n' 'from pathlib import Path' 'import sys' \
  'sys.path.insert(0, str(Path(__file__).resolve().parent))' > "$overlay/sitecustomize.py"
export PYTHONPATH="$overlay"
export CUTE_DSL_LIBS="$(dirname "$(dirname "$wheel")")/cu13/lib/libcute_dsl_runtime.so"
"$py" -c 'import cutlass; from cutlass.base_dsl.jit_executor import JitFunctionArtifacts as A; print(cutlass.__file__); assert "MLIR_RAW" in A.__dataclass_fields__'
"$py" scripts/collect_artifacts.py src/sample/sample.py --out-dir ./compiled/sample --arch sm_80 --nvdisasm "$NVDISASM"
"$py" -m unittest discover -s scripts -v
```

The tests compile with CUDA devices hidden, check source locations and instructions in the emitted artifacts, reject an empty disassembly, and verify that the ZIP contains the original source and only the current text artifacts.
