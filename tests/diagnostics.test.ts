import assert from 'node:assert/strict';
import test from 'node:test';
import { analyze } from '../src/core/diagnostics.ts';
import { buildIndex } from '../src/core/locations.ts';
import type { Artifact, LocationIndex, Ref, Stage } from '../src/core/locations.ts';

const artifact = (stage: Stage, ...lines: string[]): Artifact => ({stage, name: stage, text: lines.join('\n')});
const ref = (stage: Stage, line: number): Ref => ({stage, line});
const index = (groups: readonly (readonly Ref[])[] = [], quality: 'metadata' | 'approximate' = 'metadata'): LocationIndex => ({
  spots: () => [],
  related: (source: Ref) => [source, ...(groups.find((group) => group.some((r) => r.stage === source.stage && r.line === source.line)) ?? [])]
    .map((r) => ({ref: r, quality})),
});

const vectorMLIR = artifact('clean', 'gpu.func @k() kernel {', '  %x = vector.load %a[%i] : memref<128xf32>, vector<4xf32>', '}');
const loads = artifact('ptx', '.visible .entry k() {', '  ld.global.f32 %f1, [%rd1];', '  ld.global.f32 %f2, [%rd1+4];', '}');
const vectorLink = [ref('clean', 2), ref('ptx', 2), ref('ptx', 3)];

test('scalarization needs a vector operation and multiple metadata-linked scalar instructions', () => {
  const result = analyze([vectorMLIR, loads], index([vectorLink]));
  const finding = result.findings.find(({kind}) => kind === 'scalarization');
  assert.ok(finding);
  assert.equal(finding.status, 'potential');
  assert.deepEqual(finding.refs, vectorLink);
});

test('scalarization abstains with approximate-only links, one instruction, another entry, or a linked vector instruction', () => {
  const crossEntry = artifact('ptx', '.entry first() {', '  ld.global.f32 %f1, [%rd1];', '}', '.entry second() {', '  ld.global.f32 %f2, [%rd1+4];', '}');
  const repeatedSymbol = {...crossEntry, text: crossEntry.text.replace(/first|second/g, 'k')};
  const mixedWidth = artifact('ptx', ...loads.text.split('\n').slice(0, 3), '  ld.global.v4.f32 {%f3,%f4,%f5,%f6}, [%rd2];', '}');
  for (const [ptx, links, quality] of [
    [loads, [], 'metadata'],
    [loads, [vectorLink], 'approximate'],
    [loads, [vectorLink.slice(0, 2)], 'metadata'],
    [crossEntry, [[ref('clean', 2), ref('ptx', 2), ref('ptx', 5)]], 'metadata'],
    [repeatedSymbol, [[ref('clean', 2), ref('ptx', 2), ref('ptx', 5)]], 'metadata'],
    [mixedWidth, [[...vectorLink, ref('ptx', 4)]], 'metadata'],
  ] as const) {
    const result = analyze([vectorMLIR, ptx], index(links, quality));
    assert.ok(!result.findings.some(({kind}) => kind === 'scalarization'));
    assert.ok(result.coverage.some(({kind}) => kind === 'scalarization'));
  }
});

const rawDuplicates = artifact('raw', 'gpu.func @k(%i: i32, %j: i32) kernel {', '  %a = arith.addi %i, %j : i32', '  %b = arith.addi %i, %j : i32', '}');
const cleanSingle = artifact('clean', 'gpu.func @k(%i: i32, %j: i32) kernel {', '  %a = arith.addi %i, %j : i32', '}');

test('reports repeated pure operations with an identical retained expression in clean MLIR', () => {
  const {findings} = analyze([rawDuplicates, cleanSingle], index());
  const finding = findings.find(({kind}) => kind === 'redundancy');
  assert.ok(finding);
  assert.equal(finding.status, 'confirmed');
  assert.deepEqual(finding.refs, [ref('raw', 2), ref('raw', 3), ref('clean', 2)]);
  assert.match(finding.explanation, /2 .*1 /);
});

test('does not match across functions, sibling regions, different operands, or effectful operations', () => {
  const rawSiblings = artifact('raw', 'gpu.func @k(%i: i32, %j: i32) kernel {', '  scf.if %pred {', '    %a = arith.addi %i, %j : i32', '  } else {', '    %b = arith.addi %i, %j : i32', '  }', '}');
  for (const artifacts of [
    [rawDuplicates],
    [rawDuplicates, {...cleanSingle, text: cleanSingle.text.replace('@k(', '@other(')}],
    [rawDuplicates, {...cleanSingle, text: cleanSingle.text.replace('%i, %j', '%i, %i')}],
    [rawDuplicates, {...cleanSingle, text: cleanSingle.text.replace(' : i32\n', ' overflow<nsw> : i32\n')}],
    [rawSiblings, cleanSingle],
    [rawDuplicates, {...rawDuplicates, stage: 'clean' as const}],
    [rawDuplicates, cleanSingle].map((a) => ({...a, text: a.text.replaceAll('arith.addi', 'memref.load')})),
  ]) {
    const {findings, coverage} = analyze(artifacts, index());
    assert.ok(!findings.some(({kind}) => kind === 'redundancy'));
    assert.ok(coverage.some(({kind}) => kind === 'redundancy'));
  }
});

test('nonzero spill storage traffic is confirmed by the report itself', () => {
  const report = artifact('report', "ptxas info : Function properties for k", '    0 bytes stack frame, 8 bytes spill stores, 0 bytes spill loads');
  const sass = artifact('sass', '.section .text.other', '/*0000*/ LDL R3, [R1];');
  const findings = analyze([report, sass], index()).findings.filter(({kind}) => kind === 'spills');
  assert.equal(findings.length, 1);
  assert.equal(findings[0].status, 'confirmed');
  assert.deepEqual(findings[0].refs, [ref('report', 2)]);
  assert.match(findings[0].explanation, /8 bytes of spill stores/);
});

test('local instructions alone indicate potential spills, not a stack frame or zero spill counters', () => {
  const report = artifact('report', '32 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads');
  const ptx = artifact('ptx', '.entry k() {', '  .local .align 4 .b8 arr[32];', '  @%p1 st.local.u32 [%rd1], %r1;', '}');
  const withoutOps = analyze([report], index());
  assert.ok(!withoutOps.findings.some(({kind}) => kind === 'spills'));
  assert.ok(withoutOps.coverage.some(({kind, note}) => kind === 'spills' && /zero/i.test(note)));
  const finding = analyze([report, ptx], index()).findings.find(({kind}) => kind === 'spills');
  assert.ok(finding);
  assert.equal(finding.status, 'potential');
  assert.deepEqual(finding.refs, [ref('ptx', 3)]);
  assert.match(finding.explanation, /stack|private/i);
});

test('non-unit declared stride is cited on its access, without claiming incorrect layout or coalescing', () => {
  const clean = artifact('clean', '%x = vector.load %a[%i, %j] : memref<64x64xf32, strided<[1, 64], offset: ?>>, vector<4xf32>');
  const finding = analyze([clean], index()).findings.find(({kind}) => kind === 'layout');
  assert.ok(finding);
  assert.equal(finding.status, 'potential');
  assert.deepEqual(finding.refs, [ref('clean', 1)]);
  assert.match(finding.explanation, /stride.*64/);
  assert.match(finding.impact, /coalesc/i);
});

test('layout abstains for declarations, unit-stride vector dimensions, dynamic strides and instruction-width-only clues', () => {
  const raw = artifact('raw', 'func.func @k(%a: memref<64x64xf32, strided<[64, 1]>>) {',
    '%b = vector.load %a[%i, %j] : memref<64x64xf32, strided<[64, 1]>>, vector<4xf32>',
    '%c = vector.load %d[%i, %j] : memref<64x64xf32, strided<[64, ?]>>, vector<4xf32>',
    '%e = memref.alloc() : memref<64x64xf32, strided<[1, 64]>>', '}');
  const result = analyze([raw, loads], index());
  assert.ok(!result.findings.some(({kind}) => kind === 'layout'));
  assert.ok(result.coverage.some(({kind}) => kind === 'layout'));
});

test('barriers, waits, and fences are observable synchronization instructions, not diagnosed bugs', () => {
  const ptx = artifact('ptx', '.entry k() {', '  @%p2 bar.sync 0;', '  cp.async.wait_group 0;', '  fence.acq_rel.cta;',
    '  // bar.warp.sync 0xffffffff;', '  cp.async.commit_group;', '  mbarrier.init.shared.b64 [%rd1], 32;', '}');
  const sass = artifact('sass', 'Function : k', '/*0040*/ @!P0 BAR.SYNC 0x0;', '/*0050*/ DEPBAR.LE SB0, 0x0;');
  const findings = analyze([ptx, sass], index()).findings.filter(({kind}) => kind === 'synchronization');
  assert.equal(findings.length, 2);
  assert.ok(findings.every(({status}) => status === 'confirmed'));
  assert.deepEqual(findings.map(({refs}) => refs), [[ref('ptx', 2), ref('ptx', 3), ref('ptx', 4)], [ref('sass', 2), ref('sass', 3)]]);
  assert.match(findings[0].explanation, /not .*unnecessary|does not .*unnecessary/);
  assert.match(findings[0].impact, /wait|order/);
});

test('numeric conversions exclude address conversions, moves and bitcasts', () => {
  const ptx = artifact('ptx', '.entry k() {', '  cvt.rn.f32.s32 %f1, %r1;', '  cvta.to.global.u64 %rd1, %rd2;',
    '  mov.b32 %f1, %r1;', '  // cvt.rn.f16.f32 %h1, %f1;', '  cvt.pack.sat.s8.s32.b32 %r4, %r1, %r2, %r3;', '}');
  const sass = artifact('sass', '.section .text.k', '/*0040*/ F2FP.PACK_AB R0, R1, R2;', '/*0050*/ I2FP.F32.S32 R1, R2;', '/*0060*/ MOV R1, R2;');
  const findings = analyze([ptx, sass], index()).findings.filter(({kind}) => kind === 'conversion');
  assert.deepEqual(findings.map(({refs}) => refs), [[ref('ptx', 2), ref('ptx', 6)], [ref('sass', 2), ref('sass', 3)]]);
  assert.deepEqual(findings.map(({title}) => title), ['Numeric conversion instructions in PTX', 'Numeric conversion instructions in SASS']);
  assert.ok(findings.every(({status}) => status === 'confirmed'));
  assert.match(findings[0].impact, /round|precision/);
});

test('numeric conversion findings cite recognized casts in raw and clean MLIR', () => {
  const raw = artifact('raw', 'func.func @k(%arg: f16) {',
    '  %wide = arith.extf %arg : f16 to f32',
    '  %narrow = "arith.truncf"(%wide) : (f32) -> f16',
    '  %i = arith.fptosi %wide : f32 to i32',
    '  %j = arith.index_cast %i : i32 to index',
    '  %f = arith.bitcast %i : i32 to f32',
    '  // %x = arith.sitofp %i : i32 to f32', '}');
  const clean = artifact('clean', 'llvm.func @k(%arg: i16) {',
    '  %wide = llvm.sext %arg : i16 to i32',
    '  %float = llvm.uitofp %wide : i32 to f32',
    '  %bits = llvm.bitcast %float : f32 to i32', '}');
  const {findings, coverage} = analyze([raw, clean], index());
  const conversions = findings.filter(({kind}) => kind === 'conversion');
  assert.deepEqual(conversions.map(({title}) => title), ['Numeric conversion operations in raw MLIR', 'Numeric conversion operations in clean MLIR']);
  assert.deepEqual(conversions.map(({refs}) => refs), [[ref('raw', 2), ref('raw', 3), ref('raw', 4)], [ref('clean', 2), ref('clean', 3)]]);
  assert.ok(conversions.every(({status}) => status === 'confirmed'));
  assert.match(conversions[0].explanation, /arith.extf, arith.truncf, arith.fptosi/);
  assert.ok(!coverage.some(({kind}) => kind === 'conversion'));
});

test('empty inputs produce six distinct coverage notes and no findings', () => {
  const {findings, coverage} = analyze([], index());
  assert.deepEqual(findings, []);
  assert.equal(coverage.length, 6);
  assert.equal(new Set(coverage.map(({kind}) => kind)).size, 6);
  assert.ok(coverage.every(({note}) => note.length > 0));
});

test('repeated instruction sites keep evidence bounded without treating separate kernels as linked', () => {
  const ptx = artifact('ptx', '.entry k1() {', ...Array(10).fill('bar.sync 0;'), '}', '.entry k2() {', ...Array(10).fill('bar.sync 0;'), '}');
  const {findings} = analyze([ptx], index());
  const sync = findings.filter(({kind}) => kind === 'synchronization');
  assert.equal(sync.length, 1);
  assert.match(sync[0].explanation, /20 synchronization instruction site/);
  assert.equal(sync[0].refs.length, 5);
  assert.equal(new Set(sync[0].refs.map(({line}) => line)).size, 5);
});

test('shared source metadata does not link an MLIR kernel to a differently named PTX kernel', () => {
  const mlir = artifact('clean', 'cuda.kernel @k() {', '  %x = vector.load %a[%i] : memref<128xf32>, vector<4xf32> loc("k.py":2:5)', '}');
  const ptx = artifact('ptx', '.file 1 "k.py"', '.entry k_other() {', '  .loc 1 2 5', '  ld.global.f32 %f1, [%rd1];', '  ld.global.f32 %f2, [%rd1+4];', '}');
  const mismatched = [mlir, ptx];
  const matching = [mlir, {...ptx, text: ptx.text.replace('k_other', 'k')}];
  assert.equal(analyze(mismatched, buildIndex(mismatched)).findings.some(({kind}) => kind === 'scalarization'), false);
  const finding = analyze(matching, buildIndex(matching)).findings.find(({kind}) => kind === 'scalarization');
  assert.ok(finding);
  assert.deepEqual(finding.refs, [ref('clean', 2), ref('ptx', 4), ref('ptx', 5)]);
});

test('scalarization reuses metadata evidence across thousands of vector operations at the same source line', () => {
  const rows = 2000;
  const mlir = artifact('clean', 'cuda.kernel @k() {', ...Array.from({length: rows}, (_, i) =>
    `  %x${i} = vector.load %a[%i] : memref<128xf32>, vector<4xf32> loc("k.py":2:5)`), '}');
  const ptx = artifact('ptx', '.file 1 "k.py"', '.entry k() {', '  .loc 1 2 5', ...Array.from({length: rows}, (_, i) =>
    `  ld.global.f32 %f${i}, [%rd1+${i * 4}];`), '}');
  const artifacts = [mlir, ptx];
  const locations = buildIndex(artifacts);
  let queries = 0;
  const result = analyze(artifacts, {...locations, related(source) {
    queries++;
    return locations.related(source);
  }});
  const findings = result.findings.filter(({kind}) => kind === 'scalarization');
  assert.equal(findings.length, 1);
  assert.match(findings[0].explanation, /with 2000 scalar PTX instructions/);
  assert.deepEqual(findings[0].refs, [ref('clean', 2), ref('ptx', 4), ref('ptx', 5), ref('ptx', 6), ref('ptx', 7)]);
  assert.ok(queries <= 1, `expected at most one shared metadata query; got ${queries}`);
});

test('scalarization preserves width, direction, and evidence deduplication across fused locations', () => {
  const mlir = artifact('clean', 'cuda.kernel @k() {',
    '  %narrow = vector.load %a[%i] : memref<128xf16>, vector<4xf16> loc(fused["k.py":2:10, "caller.py":7:1])',
    '  vector.store %v, %a[%i] : memref<128xf32>, vector<4xf32> loc(fused["caller.py":7:2, "k.py":2:5])',
    '  %wide = vector.load %a[%i] : memref<128xf32>, vector<4xf32> loc(fused["k.py":2:5, "caller.py":7:3])',
    '  %again = vector.load %a[%i] : memref<128xf32>, vector<4xf32> loc("k.py":2:5)', '}');
  const ptx = artifact('ptx', '.file 1 "k.py"', '.entry k() {', '  .loc 1 2 5',
    '  ld.global.f32 %f1, [%rd1];', '  ld.global.f32 %f2, [%rd1+4];',
    '  st.global.f32 [%rd1], %f1;', '  st.global.f32 [%rd1+4], %f2;', '}');
  const artifacts = [mlir, ptx];
  const locations = buildIndex(artifacts);
  let queries = 0;
  const {findings} = analyze(artifacts, {...locations, related(source) {
    queries++;
    return locations.related(source);
  }});
  assert.deepEqual(findings.filter(({kind}) => kind === 'scalarization').map(({refs}) => refs), [
    [ref('clean', 3), ref('ptx', 6), ref('ptx', 7)],
    [ref('clean', 4), ref('ptx', 4), ref('ptx', 5)],
  ]);
  assert.ok(queries <= 2, `expected at most one metadata query per location set; got ${queries}`);
});

test('redundancy does not merge identical function headers in different modules', () => {
  const raw = artifact('raw', 'gpu.module @left {', cleanSingle.text, '}', 'gpu.module @right {', cleanSingle.text, '}');
  const clean = artifact('clean', 'gpu.module @left {', cleanSingle.text, '}');
  assert.equal(analyze([raw, clean], index()).findings.some(({kind}) => kind === 'redundancy'), false);
});

test('rank-one CuTe memref layout aliases provide explicit stride evidence only at a matching access', () => {
  const type = '!gmem = !cute.memref<f32, gmem, align<16>, "(4096):(2)">';
  const access = '%4 = cute.memref.load(%arg0, %coord) : (!gmem, !cute.coord<"?">) -> f32 loc(#loc16)';
  const finding = analyze([artifact('clean', type, access)], index()).findings.find(({kind}) => kind === 'layout');
  assert.ok(finding);
  assert.deepEqual(finding.refs, [ref('clean', 2), ref('clean', 1)]);
  assert.match(finding.explanation, /stride 2/);
  assert.match(finding.impact, /lane mapping/);
});

test('block comments and invalid signed counters never produce findings', () => {
  const ptx = artifact('ptx', '/*', 'bar.sync 0;', 'cvt.f32.s32 %f1, %r1;', '*/');
  const sass = artifact('sass', '/* a block', 'LDL R4, [R1];', '*/');
  const report = artifact('report', '-8 bytes spill loads');
  assert.deepEqual(analyze([ptx, sass, report], index()).findings, []);
});
