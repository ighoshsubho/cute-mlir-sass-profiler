# CuTe Compiler Explorer

A local, static viewer for NVIDIA CuTeDSL compilation artifacts. No compilation, GPU execution or upload to a server happens in the viewer.

## Launch

Requires Node.js 22.13+.

```sh
npm ci --include=dev
npm run dev
```

Open the localhost URL Vite prints (normally http://127.0.0.1:5173). The included vector-add compilation loads on first visit. Click its `out[i] = lhs + rhs` line to follow it through raw/clean MLIR, PTX and SASS. `npm run build` produces the static site in `dist/`; `npm run preview` serves that build for inspection.

## Load artifacts

For a larger example, open [`examples/async-compute-feedback-sm100a.zip`](examples/async-compute-feedback-sm100a.zip). It contains a pipelined Blackwell kernel and its compiler output. See [example notes](examples/README.md) for source lines to inspect.

Open or drop individual text files or one ZIP (nested folders are preserved):

| File | Pane |
| --- | --- |
| `kernel.py` | CuTeDSL Python source |
| `kernel.raw.mlir` or `kernel_debug.mlir` | Raw MLIR |
| `kernel.mlir` | Clean MLIR |
| `kernel.ptx` | PTX |
| `kernel.sass`, `.disasm`, or `.s` | Text SASS |
| `resources.txt` or `compile.log` | Optional diagnostics evidence |

The first import replaces the sample; subsequent imports add or replace matching filenames. If a stage has several files, its pane lets you choose one. Use a `raw` or `debug` suffix immediately before `.mlir` for raw IR; rename other dump names as needed. Load artifacts from the same specialization/compilation. Clear the workspace when switching to an unrelated kernel. UTF-8 text only; individual text files are limited to 8 MiB and the workspace to 32 MiB. The browser does not disassemble binary CUBIN files.

Use the separators to resize (drag or use arrows), expand any pane, and use its search button or Ctrl/⌘+F while focused in an editor. Click code or line numbers to highlight related lines. Raw/Clean switches between supplied files and can preserve the active correlation across both. Theme preference alone is persisted; file contents are held in memory.

## Collect artifacts from a CuTeDSL kernel

Use a CUTLASS/CuTeDSL build that includes [CUTLASS #3702](https://github.com/NVIDIA/cutlass/pull/3702) for `JitFunctionArtifacts.MLIR_RAW` as well as `JitFunctionArtifacts.MLIR` with source locations. The collector sets `CUTE_DSL_KEEP=ir,ir-debug` and `CUTE_DSL_LINEINFO=1` before importing your local module. It reads the compiled function's artifacts for raw/clean MLIR, PTX and CUBIN and runs `nvdisasm --print-line-info` on the CUBIN. See [collector instructions](scripts/README.md) for its CLI and adapter example. Only run the Python collector on source you trust; unlike the viewer, it imports and executes the local adapter function.

## What the evidence means

- Green highlights share matching source file and line metadata. Amber marks an approximate association between matching line numbers when the compiler and uploaded file paths differ but the filename is unambiguous. No metadata means no invented line mapping. A shared source line is not proof that two specific instructions are equivalent.
- MLIR location aliases and nested locations can link multiple source lines to one operation. PTX `.file`/`.loc` directives and `nvdisasm` line markers carry source associations to instructions. Compiler-generated rows and optimized-away operations may have no match. This is not a full MLIR or ISA parser; unusual formatting or absent metadata may reduce coverage.
- The six diagnostic rules flag evidence for scalarization, repeated pure operations, spills, layout, synchronization and conversions. `confirmed` means the cited evidence was observed, not that it is unwanted. `potential` requires developer review. For example, local-memory instructions alone do not prove register spills, and a non-unit stride need not be incorrect. The coverage section explains when the data cannot support a finding.
- Diagnostics run in a browser worker and never estimate performance or run compiler passes. Report-only findings open their cited report line; a report without source locations does not highlight unrelated code.

## Verification

```sh
npm test
npm run build
npx playwright install chromium
npm run test:browser
```
