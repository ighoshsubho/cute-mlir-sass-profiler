# Async compute / feedback example

Open `async-compute-feedback-sm100a.zip` in the viewer to inspect a pipelined Blackwell kernel. The Python pane contains the kernel and its host launcher. The implementation comes from the CuTe workset's `02_async_compute.py`; the original BSD notice is retained. The lesson text and test harness are omitted.

The bundle specializes feedback mode to 129 rows, 9 steps, two input stages, and two output slots, with FP16 inputs and FP32 output. It was compiled for `sm_100a` without launching a kernel. The shipped artifacts use repository-relative source paths instead of build-machine paths; line and column numbers are unchanged.

Try source lines 111–112 (TMA loads), 120 and 128 (Tensor Core MMA), or 153 (register-to-TMEM writeback).

## Regenerate the ZIP

From the explorer directory, activate a CuTeDSL environment with CUTLASS PR #3702 and `nvdisasm` on `PATH` (see [toolchain setup](../scripts/README.md)):

```sh
CUDA_VISIBLE_DEVICES='' python scripts/collect_artifacts.py examples/compile_async_compute.py \
  --artifact-source examples/02_async_compute.py --arch sm_100a \
  --out-dir ./compiled/async-compute \
  --zip ./compiled/async-compute-feedback-sm100a.zip
```

The adapter specializes the host launcher using fake tensors; the ZIP contains the displayed kernel file, not the adapter. To use another copy of the kernel, set `ASYNC_COMPUTE_SOURCE` and pass the same path to `--artifact-source`. Generated text can differ by compiler version and source path.
