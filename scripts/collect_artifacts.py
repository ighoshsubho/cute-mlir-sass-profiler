#!/usr/bin/env python3
"""Collect CuTe DSL compiler output for a trusted Python source file."""

import argparse
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile


def collect(args):
    source = args.source.resolve(strict=True)
    artifact_source = (args.artifact_source or source).resolve(strict=True)
    nvdisasm = shutil.which(args.nvdisasm)
    if nvdisasm is None:
        raise RuntimeError(f"nvdisasm executable not found: {args.nvdisasm}")
    with tempfile.TemporaryDirectory(prefix="cute-explorer-") as tmp:
        # Set these before the first CUTLASS import in the source file.
        settings = dict(KEEP="ir,ir-debug", LINEINFO="1", KEEP_PTX="1",
                        KEEP_CUBIN="1", DUMP_DIR=tmp, ARCH=args.arch)
        for prefix in ("CUTE_DSL", "CUTE_EXPERIMENTAL_DSL"):
            os.environ.update({f"{prefix}_{key}": value for key, value in settings.items()})
        sys.path.insert(0, str(source.parent))
        sys.dont_write_bytecode = True
        from cutlass.base_dsl.jit_executor import JitFunctionArtifacts

        if "MLIR_RAW" not in JitFunctionArtifacts.__dataclass_fields__:
            raise RuntimeError("CUTLASS PR #3702 is required (JitFunctionArtifacts.MLIR_RAW is missing)")
        module = runpy.run_path(str(source))
        compile_fn = module.get("compile_for_explorer")
        if not callable(compile_fn):
            raise RuntimeError(f"{source} must provide a compile_for_explorer() function")
        artifacts = compile_fn().artifacts
        text_artifacts = {
            "raw.mlir": artifacts.MLIR_RAW,
            "mlir": artifacts.MLIR,
            "ptx": artifacts.PTX,
        }
        for suffix, text in text_artifacts.items():
            if not isinstance(text, str) or "\n" not in text:
                raise RuntimeError(f"Compilation did not return {suffix} text")
        cubin = artifacts.CUBIN
        if not isinstance(cubin, bytes) or not cubin.startswith(b"\x7fELF"):
            raise RuntimeError("Compilation did not return ELF CUBIN bytes")
        cubin_path = Path(tmp) / artifact_source.with_suffix(".cubin").name
        cubin_path.write_bytes(cubin)
        text_artifacts["sass"] = subprocess.run(
            [nvdisasm, "--print-code", "--print-line-info", str(cubin_path)],
            check=True, stdout=subprocess.PIPE, text=True,
        ).stdout
        if not text_artifacts["sass"].strip():
            raise RuntimeError("nvdisasm produced no disassembly")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        destination = args.out_dir / artifact_source.name
        if destination.resolve() != artifact_source:
            shutil.copyfile(artifact_source, destination)
        outputs = [destination]
        for suffix, text in text_artifacts.items():
            path = args.out_dir / f"{artifact_source.stem}.{suffix}"
            path.write_text(text, encoding="utf-8")
            outputs.append(path)
        if args.keep_cubin:
            (args.out_dir / cubin_path.name).write_bytes(cubin)
        print(f"Wrote compiler artifacts for {artifact_source.name} ({args.arch}) to {args.out_dir}")
        if args.zip:
            args.zip.parent.mkdir(parents=True, exist_ok=True)
            with ZipFile(args.zip, 'w', compression=ZIP_DEFLATED) as archive:
                for path in outputs:
                    archive.write(path, arcname=path.name)
            print(f"Wrote viewer ZIP to {args.zip}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="trusted .py file with compile_for_explorer()")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--artifact-source", type=Path, help="original .py to package when source is a compile adapter")
    parser.add_argument("--zip", type=Path, help="also write a ZIP of the viewer's five text artifacts")
    parser.add_argument("--arch", default="sm_80", help="explicit CUDA target (default: sm_80)")
    parser.add_argument("--nvdisasm", default="nvdisasm", help="nvdisasm executable or path")
    parser.add_argument("--keep-cubin", action="store_true", help="also retain the binary CUBIN")
    args = parser.parse_args()
    if args.source.suffix != ".py" or (args.artifact_source and args.artifact_source.suffix != ".py"):
        parser.error("source and --artifact-source must be .py files")
    if args.zip and args.zip.suffix.lower() != ".zip":
        parser.error("--zip must end in .zip")
    try:
        collect(args)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
