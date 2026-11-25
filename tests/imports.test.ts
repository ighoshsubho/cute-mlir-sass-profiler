import { test } from 'node:test';
import assert from 'node:assert/strict';
import { strToU8, zipSync } from 'fflate';
import { classify, readUploads } from '../src/core/imports.ts';
const f = (name: string, text: string) => new File([text], name);
const zip = (members: Record<string, string>) => new File([zipSync(Object.fromEntries(Object.entries(members).map(([k,v]) => [k,strToU8(v)])))], 'input.zip');
test('recognizes separate raw and clean artifacts, plus reports', () => {
  assert.equal(classify('raw.mlir'), 'raw');
  assert.equal(classify('kernel_debug.mlir'), 'raw');
  assert.equal(classify('copy_debug_tile.mlir'), 'clean');
  assert.equal(classify('copy_debug_tile.raw.mlir'), 'raw');
  assert.equal(classify('kernel.mlir'), 'clean');
  assert.equal(classify('compile.log'), 'report');
  assert.equal(classify('sample.cubin'), undefined);
});
test('reads nested archive as text, ignoring cubin and preserving paths', async () => {
  const result = await readUploads([zip({'run/test.py':'print(1)', 'run/test.raw.mlir':'m', 'run/test.ptx':'p', 'run/test.cubin':'binary'})]);
  assert.deepEqual(result.artifacts.map(x => [x.stage,x.name,x.text]), [['python','run/test.py','print(1)'], ['raw','run/test.raw.mlir','m'], ['ptx','run/test.ptx','p']]);
});
test('rejects invalid and oversized archives before expansion', async () => {
  await assert.rejects(readUploads([f('bad.zip','invalid')]), /ZIP/i);
  await assert.rejects(readUploads([zip({'bomb.ptx':'x'.repeat(9 * 1024 * 1024)})]), /limit/i);
  await assert.rejects(readUploads([f('binary.ptx','a\0b')]), /text/i);
});
test('multiple known artifacts are preserved and unsupported input is actionable', async () => {
  assert.equal((await readUploads([f('a.py','a'),f('b.py','b')])).artifacts.length, 2);
  await assert.rejects(readUploads([f('a.cubin','123')]), /No supported/);
});
