import assert from 'node:assert/strict';
import test from 'node:test';
import { buildIndex, type Ref, type Stage } from '../src/core/locations.ts';

const ref = (stage: Stage, line: number): Ref => ({ stage, line });
const metadata = (stage: Stage, line: number) => ({ ref: ref(stage, line), quality: 'metadata' });
const approximate = (stage: Stage, line: number) => ({ ref: ref(stage, line), quality: 'approximate' });

test('indexes explicit MLIR locations against the corresponding Python line', () => {
  const index = buildIndex([
    { stage: 'python', name: '/src/kernel.py', text: 'def kernel():\n    return x\n' },
    { stage: 'raw', name: 'raw.mlir', text: 'module {\n  func.return loc("/src/kernel.py":2:5)\n}' },
    { stage: 'clean', name: 'clean.mlir', text: '  func.return loc(unknown)' },
  ]);
  assert.deepEqual(index.spots(ref('raw', 2)), [{ file: '/src/kernel.py', line: 2, column: 5 }]);
  assert.deepEqual(index.related(ref('raw', 2)), [metadata('raw', 2), metadata('python', 2)]);
  assert.deepEqual(index.related(ref('python', 2)), [metadata('python', 2), metadata('raw', 2)]);
  assert.deepEqual(index.related(ref('clean', 1)), [metadata('clean', 1)]);
  assert.deepEqual(index.spots(ref('clean', 1)), []);
  assert.deepEqual(index.spots(ref('python', 3)), []);
});

test('resolves forward aliases in multiline fused, named, and callsite locations', () => {
  const index = buildIndex([{ stage: 'raw', name: 'raw.mlir', text: [
    '%0 = "cute.op"(%arg0,',
    '    %arg1) : (i32, i32) -> i32',
    '    loc(fused<{note = loc("decoy.py":44:1)}>[',
    '      #loc1, #missing, callsite("callee"("/lib/a.py":3:2) at #loc2), ?',
    '    ])',
    '#loc1 = loc(#loc2)',
    '#loc2 = loc("/with space/a.py":7:4 to 9:6)',
    '%1 = "cute.op"() {msg = "loc(\\22fake.py\\22:3:5)"} : () -> i32 // loc("fake.py":4:5)',
    '%2 = "cute.op"() : () -> i32 loc(#loop)',
    '#loop = loc(#loop)',
    '%3 = "cute.op"() : () -> i32 loc("C:\\\\temp\\5Cwith space\\5Ca.py":8)',
  ].join('\n') }]);
  assert.deepEqual(index.spots(ref('raw', 1)), [
    { file: '/with space/a.py', line: 7, column: 4 },
    { file: '/lib/a.py', line: 3, column: 2 },
  ]);
  for (const row of [2, 3, 4, 5, 6, 7, 8, 9, 10]) {
    assert.deepEqual(index.spots(ref('raw', row)), [], `raw:${row}`);
  }
  assert.deepEqual(index.spots(ref('raw', 11)), [{ file: 'C:\\temp\\with space\\a.py', line: 8 }]);
});

test('PTX locations persist across instructions, not metadata rows or function boundaries', () => {
  const index = buildIndex([{ stage: 'ptx', name: 'kernel.ptx', text: [
    '.visible .entry first()',
    '{',
    '  .reg .b32 %r;',
    '  .loc 1 14 5',
    '  mov.u32 %r, %tid.x;',
    '  .pragma "nothing";',
    'L_0:',
    '  @%p add.u32 %r, %r, 1;',
    '  .loc 2 15 0, function_name info, inlined_at 1 14 5',
    '  // .loc 1 666 1',
    '  mov.u32 %r, 0;',
    '  .loc 0 0 0',
    '  ret;',
    '  .loc 1 16 0',
    '  ret;',
    '}',
    '.visible .func second()',
    '{',
    '  ret;',
    '  .loc 99 22 8',
    '  ret;',
    '}',
    '.file 1 "/path/with space/kernel.py", 12, 42',
    '.file 2 "C:\\\\src\\\\helper.py"',
    '.section .debug_info {',
    '  .b8 0',
    '}',
  ].join('\r\n') }]);
  const caller = { file: '/path/with space/kernel.py', line: 14, column: 5 };
  for (const row of [5, 8]) assert.deepEqual(index.spots(ref('ptx', row)), [caller]);
  assert.deepEqual(index.spots(ref('ptx', 11)), [{ file: 'C:\\src\\helper.py', line: 15 }, caller]);
  assert.deepEqual(index.spots(ref('ptx', 15)), [{ file: caller.file, line: 16 }]);
  for (const row of [1, 2, 3, 4, 6, 7, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]) {
    assert.deepEqual(index.spots(ref('ptx', row)), [], `ptx:${row}`);
  }
});

test('maps tcgen05 MMA with double-colon opcode modifiers, including after an inline label', () => {
  const index = buildIndex([{stage:'ptx', name:'kernel.ptx', text: [
    '.file 1 "kernel.py"',
    '.visible .entry test() {',
    '.loc 1 196 20',
    'tcgen05.mma.cta_group::1.kind::f16.collector::a::discard [%r23], %rd6, %rd7, %r62, {%r61, %r61, %r61, %r61}, %p12;',
    'L_0: @%p0 tcgen05.mma.cta_group::1.kind::f16 [%r23], %rd6, %rd7, %r62, %p12;',
    'L_1:',
    'ret;',
    '}',
  ].join('\n')}]);
  const spot = {file:'kernel.py',line:196,column:20};
  for (const line of [4,5,7]) assert.deepEqual(index.spots(ref('ptx',line)), [spot]);
  assert.deepEqual(index.spots(ref('ptx',6)), []);
});

test('tracks nvdisasm source, inline, and embedded PTX annotations without indexing their rows', () => {
  // Annotation shape observed with CUDA 12.9 nvdisasm -c -gi -gp.
  const index = buildIndex([
    { stage: 'sass', name: 'kernel.sass', text: [
      '.section .text.first,"ax",@progbits',
      'first:',
      '.text.first:',
      '  //## File "/path/with space/kernel.py", line 15 inlined at "/other/caller.py", line 14',
      '  //## File "/other/caller.py", line 14',
      '  //## File ".nv_debug_ptx_txt", line 12',
      '  /*0010*/  S2R R5, SR_TID.X ;',
      '  /* 0x0012160800007319 */',
      '  //## File ".nv_debug_ptx_txt", line 13',
      '  /*0020*/  @!P0 MOV R2, c[0x0][0x160] ;',
      '  //## File "/path/with space/kernel.py", line 17',
      '  /*0030*/  EXIT ;',
      '.section .text.second,"ax",@progbits',
      '  /*0000*/  EXIT ;',
      '.section .debug_line',
      '  .byte 0x00',
    ].join('\n') },
    // An embedded PTX marker is not identity evidence for a separate artifact.
    { stage: 'ptx', name: 'probe.ptx', text: '.entry p() {\n' + '\n'.repeat(10) + 'ret;\n}' },
  ]);
  const callee = { file: '/path/with space/kernel.py', line: 15 };
  const caller = { file: '/other/caller.py', line: 14 };
  const embedded = (line: number) => ({ file: '.nv_debug_ptx_txt', line });
  assert.deepEqual(index.spots(ref('sass', 7)), [callee, caller, embedded(12)]);
  assert.deepEqual(index.spots(ref('sass', 10)), [callee, caller, embedded(13)]);
  assert.deepEqual(index.spots(ref('sass', 12)), [{ file: callee.file, line: 17 }, embedded(13)]);
  assert.deepEqual(index.related(ref('sass', 7)), [metadata('sass', 7), metadata('sass', 10)]);
  for (const row of [1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 14, 15, 16]) {
    assert.deepEqual(index.spots(ref('sass', row)), [], `sass:${row}`);
  }
});

test('related is a deduplicated union of the selected row’s source lines, not transitive closure', () => {
  const index = buildIndex([
    { stage: 'raw', name: 'r.mlir', text: 'op.a loc("/src/a.py":7:2)\nop.b loc("/src/a.py":7:99)' },
    { stage: 'clean', name: 'c.mlir', text: 'op.fused loc(fused["/src/a.py":7:8, "/src/a.py":7:11, "/src/b.py":9])\nop.b loc("/src/b.py":9:1)' },
  ]);
  assert.deepEqual(index.related(ref('raw', 1)), [metadata('raw', 1), metadata('raw', 2), metadata('clean', 1)]);
  assert.deepEqual(index.related(ref('clean', 1)), [metadata('clean', 1), metadata('raw', 1), metadata('raw', 2), metadata('clean', 2)]);
  assert.deepEqual(index.related(ref('report', 500)), [metadata('report', 500)]);
});

test('marks unique basename and matching line as approximate, preferring direct evidence', () => {
  const index = buildIndex([
    { stage: 'python', name: 'kernel.py', text: 'a\nb\nc\n' },
    { stage: 'raw', name: 'r.mlir', text: 'op.a loc("/src/kernel.py":2:2)\nop.b loc("/src/kernel.py":3:3)' },
    { stage: 'clean', name: 'c.mlir', text: 'op.fused loc(fused["kernel.py":2, "/src/kernel.py":2:1])' },
  ]);
  assert.deepEqual(index.related(ref('raw', 1)), [metadata('raw', 1), approximate('python', 2), metadata('clean', 1)]);
  assert.deepEqual(index.related(ref('python', 2)), [metadata('python', 2), approximate('raw', 1), metadata('clean', 1)]);
  assert.deepEqual(index.related(ref('python', 1)), [metadata('python', 1)]);
});

test('links a unique nested Python member to a compiler path only as approximate', () => {
  const index = buildIndex([
    { stage: 'python', name: 'bundle/kernel.py', text: 'a\nb' },
    { stage: 'raw', name: 'r.mlir', text: 'op.a loc("/original/kernel.py":2:2)' },
  ]);
  assert.deepEqual(index.related(ref('raw', 1)), [metadata('raw', 1), approximate('python', 2)]);
  assert.deepEqual(index.related(ref('python', 2)), [metadata('python', 2), approximate('raw', 1)]);
  assert.deepEqual(index.spots(ref('raw', 1)), [{ file: '/original/kernel.py', line: 2, column: 2 }]);
});

test('withholds nested-member fallback if an unselected source has the same basename', () => {
  const index = buildIndex([
    { stage: 'python', name: 'kernel.py', text: 'a\nb' },
    { stage: 'raw', name: 'r.mlir', text: 'op.a loc("/original/kernel.py":2:2)' },
  ], ['kernel.py', 'bundle/kernel.py']);
  assert.deepEqual(index.related(ref('raw', 1)), [metadata('raw', 1)]);
  assert.deepEqual(index.related(ref('python', 2)), [metadata('python', 2)]);
});

test('does not resolve an unqualified compiler filename to one of several loaded Python identities', () => {
  const index = buildIndex([
    { stage: 'python', name: 'kernel.py', text: 'a\nb' },
    { stage: 'raw', name: 'r.mlir', text: 'op.a loc("kernel.py":2:2)' },
  ], ['kernel.py', 'bundle/kernel.py']);
  assert.deepEqual(index.related(ref('raw', 1)), [metadata('raw', 1)]);
  assert.deepEqual(index.related(ref('python', 2)), [metadata('python', 2)]);
});

test('does not equate distinct qualified compiler paths matching one Python member', () => {
  const index = buildIndex([
    { stage: 'python', name: 'bundle/kernel.py', text: 'a\nb' },
    { stage: 'raw', name: 'r.mlir', text: 'op.a loc("/original/kernel.py":2:2)' },
    { stage: 'clean', name: 'c.mlir', text: 'op.b loc("bundle/kernel.py":2:1)\nop.c loc("/elsewhere/kernel.py":2:1)' },
  ]);
  assert.deepEqual(index.related(ref('raw', 1)), [metadata('raw', 1)]);
  assert.deepEqual(index.related(ref('clean', 1)), [metadata('clean', 1), metadata('python', 2)]);
  assert.deepEqual(index.related(ref('python', 2)), [metadata('python', 2), metadata('clean', 1)]);
});

test('withholds basename links if another source or generated artifact can own the name', () => {
  const index = buildIndex([
    { stage: 'python', name: 'same.py', text: 'a\nb\nc' },
    { stage: 'raw', name: '/out/generated.mlir', text: [
      'op.a loc("/src/same.py":2:2)',
      'op.b loc("/other/same.py":77:2)',
      'op.c loc("same.py":2:3)',
      'op.d loc("generated.mlir":9:3)',
    ].join('\n') },
    { stage: 'clean', name: '/out/clean.mlir', text: [
      'op.a loc("/src/same.py":2:1)',
      'op.b loc("/src/generated.mlir":9:3)',
      'op.c loc("same.py":2:3)',
    ].join('\n') },
  ]);
  assert.deepEqual(index.related(ref('python', 2)), [metadata('python', 2)]);
  assert.deepEqual(index.related(ref('raw', 1)), [metadata('raw', 1), metadata('clean', 1)]);
  assert.deepEqual(index.related(ref('raw', 3)), [metadata('raw', 3)]);
  assert.deepEqual(index.related(ref('raw', 4)), [metadata('raw', 4)]);
});

test('unused file declarations can make a basename ambiguous without creating selectable rows', () => {
  const index = buildIndex([
    { stage: 'python', name: 'kernel.py', text: 'contents' },
    { stage: 'raw', name: 'raw.mlir', text: 'op.a loc("/src/kernel.py":1:2)' },
    { stage: 'ptx', name: 'kernel.ptx', text: '.file 1 "/unused/kernel.py"' },
  ]);
  assert.deepEqual(index.related(ref('python', 1)), [metadata('python', 1)]);
  assert.deepEqual(index.spots(ref('ptx', 1)), []);
});

test('accepts PTX instruction excerpts with explicit line metadata', () => {
  const index = buildIndex([{ stage: 'ptx', name: 'snippet.ptx', text: [
    '.file 1 "kernel.py"',
    '.loc 1 5 3',
    'L_0: mov.u32 %r, %tid.x;',
    '.section .debug_line {',
    '  .b8 0',
    '}',
  ].join('\n') }]);
  assert.deepEqual(index.spots(ref('ptx', 3)), [{ file: 'kernel.py', line: 5, column: 3 }]);
  assert.deepEqual(index.spots(ref('ptx', 5)), []);
});

test('attaches a trailing location to an operation whose operands close on their own line', () => {
  const index = buildIndex([{ stage: 'raw', name: 'raw.mlir', text: [
    '  %0, %1 = "cute.op"(',
    '    %arg0,',
    '    %arg1',
    '  ) : (i32, i32) -> (i32, i32) loc("kernel.py":7:5)',
    '  cute.consume %0, %1 loc("kernel.py":8:5)',
  ].join('\n') }]);
  assert.deepEqual(index.spots(ref('raw', 1)), [{ file: 'kernel.py', line: 7, column: 5 }]);
  assert.deepEqual(index.spots(ref('raw', 4)), []);
  assert.deepEqual(index.spots(ref('raw', 5)), [{ file: 'kernel.py', line: 8, column: 5 }]);
});

test('attaches locations after nested MLIR regions and else branches to their opening operations', () => {
  const index = buildIndex([{ stage: 'raw', name: 'raw.mlir', text: [
    'func.func @f(%cond: i1) attributes {test.flag} {',
    '  scf.if %cond {',
    '    scf.if %cond {',
    '      scf.yield',
    '    } loc(#inner)',
    '    scf.yield loc("kernel.py":9:1)',
    '  } else {',
    '    scf.yield loc("kernel.py":10:1)',
    '  } loc("kernel.py":7:3)',
    '  return loc("kernel.py":11:1)',
    '} loc("kernel.py":6:1)',
    '#inner = loc("kernel.py":8:5)',
  ].join('\n') }]);
  const spot = (line: number, column: number) => [{ file: 'kernel.py', line, column }];
  assert.deepEqual(index.spots(ref('raw', 1)), spot(6, 1));
  assert.deepEqual(index.spots(ref('raw', 2)), spot(7, 3));
  assert.deepEqual(index.spots(ref('raw', 3)), spot(8, 5));
  assert.deepEqual(index.spots(ref('raw', 6)), spot(9, 1));
  assert.deepEqual(index.spots(ref('raw', 8)), spot(10, 1));
  assert.deepEqual(index.spots(ref('raw', 10)), spot(11, 1));
  for (const row of [4, 5, 7, 9, 11, 12]) assert.deepEqual(index.spots(ref('raw', row)), [], `raw:${row}`);
});

test('does not attach a location on an unmatched region close to a previous operation', () => {
  const index = buildIndex([{ stage: 'raw', name: 'raw.mlir', text: [
    '  scf.yield loc("kernel.py":9:1)',
    '} loc("kernel.py":8:5)',
  ].join('\n') }]);
  assert.deepEqual(index.spots(ref('raw', 1)), [{ file: 'kernel.py', line: 9, column: 1 }]);
  assert.deepEqual(index.spots(ref('raw', 2)), []);
});
