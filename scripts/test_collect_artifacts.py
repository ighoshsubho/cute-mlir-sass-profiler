"""Compile-only integration test; requires the documented toolchain."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
COLLECT = ROOT / "scripts" / "collect_artifacts.py"
SAMPLE = ROOT / "src" / "sample" / "sample.py"


class CollectArtifactsTest(unittest.TestCase):
    def test_zip_packages_the_original_source_instead_of_the_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            adapter = tmp / 'adapter.py'
            adapter.write_text(
                f'import runpy\ncompile_for_explorer = runpy.run_path({str(SAMPLE)!r})["compile_for_explorer"]\n',
                encoding='utf-8',
            )
            archive = tmp / 'sample.zip'
            out = tmp / 'generated'
            out.mkdir()
            (out / 'stale.ptx').write_text('old')
            result = subprocess.run(
                [sys.executable, str(COLLECT), str(adapter), '--artifact-source', str(SAMPLE),
                 '--out-dir', str(out), '--zip', str(archive), '--keep-cubin',
                 '--nvdisasm', os.environ.get('NVDISASM', 'nvdisasm')],
                cwd=tmp, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertTrue((out / 'sample.cubin').read_bytes().startswith(b'\x7fELF'))
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(set(z.namelist()), {'sample.py', 'sample.raw.mlir', 'sample.mlir', 'sample.ptx', 'sample.sass'})
                self.assertIsNone(z.testzip())
                self.assertEqual(z.read('sample.py'), SAMPLE.read_bytes())
                for member in z.namelist():
                    self.assertEqual(z.read(member), (out / member).read_bytes())

    def test_sample_emits_all_four_source_mapped_compiler_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "generated"
            result = subprocess.run(
                [sys.executable, str(COLLECT), str(SAMPLE), "--out-dir", str(out),
                 "--nvdisasm", os.environ.get("NVDISASM", "nvdisasm")],
                cwd=tmp,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual({p.name for p in out.iterdir()}, {
                "sample.py", "sample.raw.mlir", "sample.mlir", "sample.ptx", "sample.sass",
            })
            self.assertEqual((out / "sample.py").read_bytes(), SAMPLE.read_bytes())
            raw = (out / "sample.raw.mlir").read_text()
            clean = (out / "sample.mlir").read_text()
            ptx = (out / "sample.ptx").read_text()
            sass = (out / "sample.sass").read_text()
            self.assertNotEqual(raw, clean)
            for assembly in (raw, clean):
                self.assertRegex(assembly, r'loc\("[^"\n]*sample\.py":\d+:\d+\)')
                self.assertIn("vector_add_kernel", assembly)
            self.assertRegex(ptx, r'(?m)^\s*\.file\s+\d+\s+"[^"\n]*sample\.py"')
            self.assertRegex(ptx, r"(?m)^\s*\.loc\s+\d+\s+\d+")
            self.assertIn("add.f32", ptx)
            self.assertRegex(sass, r'File "[^"\n]*sample\.py", line \d+')
            self.assertIn("FADD", sass)

    def test_collector_rejects_an_empty_disassembly(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            no_output = tmp / "no-output-nvdisasm"
            no_output.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            no_output.chmod(0o755)
            out = tmp / "generated"
            result = subprocess.run(
                [sys.executable, str(COLLECT), str(SAMPLE), "--out-dir", str(out),
                 "--nvdisasm", str(no_output)],
                cwd=tmp,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("nvdisasm produced no disassembly", result.stderr)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
