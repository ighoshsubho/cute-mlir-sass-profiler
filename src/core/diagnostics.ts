import type { Artifact, LocationIndex, Ref, Stage } from './locations.ts';

export type DiagnosticKind = 'scalarization' | 'redundancy' | 'spills' | 'layout' | 'synchronization' | 'conversion';
export type Finding = {
  kind: DiagnosticKind;
  title: string;
  status: 'confirmed' | 'potential';
  explanation: string;
  impact: string;
  refs: Ref[];
};
export type Coverage = { kind: DiagnosticKind; note: string };
type Emit = (finding: Finding) => void;

type Line = { ref: Ref; text: string };
type Assembly = Line & { op: string; symbol?: string; body: number };
type MLIR = Line & { scope?: {symbol: string; identity: string}; entry: boolean };
type Memory = { direction: string; space: string; bits: number; scalar: boolean };
const MAX_PER_RULE = 4;
const MAX_REFS = 5;
const key = ({stage, line}: Ref) => `${stage}:${line}`;

function toLines({stage, text}: Artifact): Line[] {
  // Keep quoted strings and preserve original line numbers. SASS address
  // annotations are short block comments; removing them leaves the opcode.
  const tokens = stage === 'ptx' || stage === 'sass'
    ? /"(?:\\.|[^"\\\r\n])*"|\/\/[^\r\n]*|\/\*[\s\S]*?(?:\*\/|$)/g
    : /"(?:\\.|[^"\\\r\n])*"|\/\/[^\r\n]*/g;
  const code = text.replace(tokens, (token) => token.startsWith('"') ? token : ` ${token.replace(/[^\r\n]/g, '')}`);
  return code.split(/\r?\n/).map((text, n) => ({ref: {stage, line: n + 1}, text: text.trim()}));
}

function asm(lines: Line[]): Assembly[] {
  const result: Assembly[] = [];
  let symbol: string | undefined;
  let ordinal = 0;
  for (const line of lines) {
    const name = line.ref.stage === 'ptx'
      ? line.text.match(/(?:^|\s)\.(?:entry|func)\s+(?:\([^)]*\)\s*)?([\w$%.]+)/)?.[1]
      : line.text.match(/^(?:\.section\s+\.text\.|Function\s*:\s*)([\w$%.]+)/)?.[1];
    if (name) { symbol = name; ordinal++; }
    if (line.ref.stage === 'ptx' && line.text === '}') symbol = undefined;
    const text = line.text.replace(/^[\w$.-]+:(?!:)\s*/, '');
    const op = text.match(/^(?:@!?%?[\w.]+\s+)?([a-z][\w.:]*)(?=[ \t;])(?=[^;]*;)/i)?.[1];
    if (op) result.push({...line, op, symbol, body: ordinal});
  }
  return result;
}

function memory(op: string, stage: Stage): Memory | undefined {
  if (stage === 'ptx') {
    const m = op.match(/^(ld|st)\.(global|shared|local)(?:::[\w]+)?((?:\.[\w:]+)*?)\.[bfsu](8|16|32|64|128)$/);
    if (m) return {direction: m[1], space: m[2], bits: Number(m[4]), scalar: !/\.v\d+(?:\.|$)/.test(m[3]) && m[4] !== '128'};
  } else {
    const m = op.match(/^(LD|ST)(G|S|L)(?:\.|$)/);
    if (m) {
      const bits = Number(op.match(/\.(?:[US])?(8|16|32|64|128)(?:\.|$)/)?.[1] ?? 32);
      return {direction: m[1].toLowerCase(), space: {G: 'global', S: 'shared', L: 'local'}[m[2]]!, bits, scalar: bits <= 32};
    }
  }
}

type MemorySite = {instruction: Assembly; mem: Memory};
type ScalarGroup = {symbol?: string; bits: number; count: number; refs: Ref[]};

function scalarization(mlir: MLIR[], lowered: Assembly[], index: LocationIndex, add: Emit): string | undefined {
  const byRef = new Map(lowered.map((instruction) => [key(instruction.ref), instruction]));
  const byLocations = new Map<string, Map<string, ScalarGroup>>();
  const byEvidence = new Map<string, ScalarGroup>();
  const stages = ['ptx', 'sass'] as const;

  function groupsFor(source: Ref): Map<string, ScalarGroup> {
    const spots = index.spots(source);
    // Exact metadata links depend on the set of file/line pairs, not on columns or
    // provenance order. Keep related() authoritative for ambiguous source names.
    const locationKey = spots.length ? JSON.stringify([...new Set(spots.map(({file, line}) => `${line}:${file}`))].sort()) : undefined;
    const cached = locationKey === undefined ? undefined : byLocations.get(locationKey);
    if (cached) return cached;

    const buckets = new Map<string, MemorySite[]>();
    const visited = new Set<string>();
    for (const link of index.related(source)) {
      if (link.quality !== 'metadata') continue;
      const id = key(link.ref);
      const instruction = byRef.get(id);
      if (!instruction || visited.has(id)) continue;
      visited.add(id);
      const mem = memory(instruction.op, instruction.ref.stage);
      if (!mem) continue;
      const bucket = `${instruction.ref.stage}:${mem.direction}`;
      const sites = buckets.get(bucket) ?? [];
      sites.push({instruction, mem});
      buckets.set(bucket, sites);
    }

    const groups = new Map<string, ScalarGroup>();
    for (const [bucket, sites] of buckets) {
      const {instruction: first, mem: {bits, space}} = sites[0];
      // Abstain on mixed symbols/bodies, even if a subset would qualify on its own.
      if (sites.length < 2 || sites.some(({instruction, mem}) => !mem.scalar || mem.bits !== bits
        || mem.space !== space || instruction.body !== first.body || instruction.symbol !== first.symbol)) continue;
      const refs = sites.map(({instruction}) => instruction.ref).sort((a, b) => a.line - b.line);
      const signature = refs.map(key).join(',');
      // The same lowered group can be reached through different source locations.
      const group = byEvidence.get(signature) ?? {symbol: first.symbol, bits, count: refs.length, refs: refs.slice(0, MAX_REFS - 1)};
      byEvidence.set(signature, group);
      groups.set(bucket, group);
    }
    // An index with no spots for this row can still supply links; do not cache it under a shared empty key.
    if (locationKey !== undefined) byLocations.set(locationKey, groups);
    return groups;
  }

  let vectors = false;
  let found = false;
  const seen = new Set<ScalarGroup>();
  for (const line of mlir) {
    const text = line.text.replace(/\s+loc\(.*\)\s*$/, '');
    const direction = text.match(/^(?:%[\w.$-]+\s*=\s*)?(?:vector\.(?:transfer_)?(load|read|store|write)|(?:cute\.memref|memref|llvm)\.(load|store))(?=[\s(])/)?.slice(1).find(Boolean);
    const shape = text.match(/\bvector<((?:[1-9]\d*x)+)(?:[bfiu]|bf)(8|16|32|64)>/);
    if (!direction || !shape || !shape[1].split('x').some((size) => Number(size) > 1)) continue;
    vectors = true;
    const groups = groupsFor(line.ref);
    const loweredDirection = /load|read/.test(direction) ? 'ld' : 'st';
    const bits = Number(shape[2]);
    for (const stage of stages) {
      const group = groups.get(`${stage}:${loweredDirection}`);
      if (!group || group.symbol !== line.scope?.symbol || group.bits !== bits || seen.has(group)) continue;
      seen.add(group);
      found = true;
      add({kind: 'scalarization', title: 'Possible scalarized vector memory operation', status: 'potential',
        explanation: `A vector MLIR memory operation shares source metadata with ${group.count} scalar ${stage.toUpperCase()} instructions of matching element width. The source location can also cover unrelated accesses or tail handling.`,
        impact: 'Scalar lowering can increase the static instruction count for an access; dynamic cost and coalescing are not established.', refs: [line.ref, ...group.refs]});
      break;
    }
  }
  if (!found) return vectors
    ? 'No vector MLIR access links by metadata to multiple compatible scalar instructions within one lowered symbol.'
    : 'A supported vector MLIR memory operation and correlated PTX or SASS instructions are needed.';
}

const pureArith = /^(arith\.(?:addi|subi|muli|andi|ori|xori|shli|shrsi|shrui|addf|subf|mulf|negf|cmpi|cmpf|select))\s/;

function scopedMLIR(lines: Line[]): MLIR[] {
  const modules: {name: string; depth: number}[] = [];
  let depth = 0;
  let unnamed = 0;
  let scope: (NonNullable<MLIR['scope']> & {depth: number; entry: boolean}) | undefined;
  return lines.map((line) => {
    const {text} = line;
    if (text.endsWith('{')) {
      const module = text.match(/^(?:gpu\.|builtin\.)?module(?:\s+@([^\s{]+))?(?=\s|\{)/);
      if (module) modules.push({name: module[1] ?? `#${++unnamed}`, depth});
      const symbol = text.match(/^(?:(?:gpu|func|llvm)\.func|cuda\.kernel|func)\s+[^@]*@([^\s(]+)\s*\(/)?.[1];
      if (symbol) scope = {symbol, identity: `${modules.map(({name}) => name).join('::')}\n${text.replace(/\s+/g, ' ')}`, depth, entry: true};
    }
    if (scope && depth === scope.depth + 1 && text.startsWith('^')) scope.entry = false;
    const scoped = {...line, scope, entry: !!scope && depth === scope.depth + 1 && scope.entry};
    const braces = text.replace(/"(?:\\.|[^"\\])*"/g, '').match(/[{}]/g) ?? [];
    for (const brace of braces) depth += brace === '{' ? 1 : -1;
    if (scope && depth <= scope.depth) scope = undefined;
    while (modules.length && depth <= modules[modules.length - 1].depth) modules.pop();
    return scoped;
  });
}

function pureExpressions(lines: MLIR[]): Map<string, {op: string; refs: Ref[]}> {
  const result = new Map<string, {op: string; refs: Ref[]}>();
  for (const {text, ref, scope, entry} of lines) {
    // Do not compare nested regions, block-local names, or text with unknown effects.
    if (scope && entry) {
      const rhs = text.match(/^%[\w.$-]+\s*=\s*(.+)$/)?.[1].replace(/\s+loc\(.*\)\s*$/, '');
      const op = rhs?.match(pureArith)?.[1];
      if (rhs && op && !rhs.includes('"') && !/[{}]/.test(rhs) && /%[\w.$-]+/.test(rhs) && rhs.includes(':')) {
        const signature = `${scope.identity}\n${rhs.replace(/\s+/g, ' ').replace(/\s*([,:])\s*/g, '$1')}`;
        const entry = result.get(signature) ?? {op, refs: []};
        entry.refs.push(ref);
        result.set(signature, entry);
      }
    }
  }
  return result;
}

function redundancy(raw: MLIR[], clean: MLIR[], add: Emit): string | undefined {
  if (!raw.length || !clean.length) return 'Both raw and clean MLIR are needed to compare repeated pure operations.';
  const cleaned = pureExpressions(clean);
  let found = false;
  for (const [signature, before] of pureExpressions(raw)) {
    const after = cleaned.get(signature);
    if (!after || before.refs.length < 2 || before.refs.length <= after.refs.length) continue;
    found = true;
    add({kind: 'redundancy', title: 'Fewer duplicate pure operations in clean MLIR', status: 'confirmed',
      explanation: `${before.op} with identical printed operands, type, and flags occurs ${before.refs.length} times in the raw entry block and ${after.refs.length} in the matching clean entry block. This is a textual reduction, not proof of which pass caused it.`,
      impact: 'There are fewer copies of this expression in the clean IR; effects on machine instructions or runtime are not established.',
      refs: [...before.refs.slice(0, MAX_REFS - 1), after.refs[0]]});
  }
  if (!found) return 'No reduced count of identical supported pure operations was found in matching raw and clean function entry blocks.';
}

function spills(report: Line[], lowered: Assembly[], add: Emit): string | undefined {
  let confirmed = false;
  let zero = false;
  for (const {ref, text} of report) {
    const counters = [
      ...[...text.matchAll(/(?<![\w+.-])(\d+)\s+bytes?\s+(?:of\s+)?spill\s+(stores?|loads?)\b/gi)].map((m) => [m[1], m[2]]),
      ...[...text.matchAll(/\bspill\s+(stores?|loads?)\s*[:=]\s*(\d+)\s+bytes?\b/gi)].map((m) => [m[2], m[1]]),
    ];
    const nonzero = counters.filter(([n]) => /[1-9]/.test(n));
    zero ||= counters.length > 0 && nonzero.length === 0;
    if (!nonzero.length) continue;
    confirmed = true;
    add({kind: 'spills', title: 'Compiler report records spill traffic', status: 'confirmed',
      explanation: `The compiler report lists ${nonzero.map(([n, op]) => `${n} bytes of spill ${op.toLowerCase()}`).join(' and ')}. These are compile-time counters, not measured runtime traffic.`,
      impact: 'Spills can introduce per-thread local-memory loads or stores; execution frequency and latency are not established.', refs: [ref]});
  }
  if (confirmed) return;
  const local = lowered.filter(({op, ref}) => memory(op, ref.stage)?.space === 'local');
  if (local.length) {
    add({kind: 'spills', title: 'Local-memory accesses may be spills', status: 'potential',
      explanation: 'Local-memory load/store instructions are present, but they may address stack storage, private arrays, or spills. No nonzero spill counter was found.',
      impact: 'These instructions may generate local-memory traffic if executed. Their purpose and dynamic cost are unknown.', refs: local.slice(0, MAX_REFS).map(({ref}) => ref)});
    return;
  }
  return zero ? 'Recognized spill counters are zero; no recognized local-memory instructions provide other evidence.'
    : 'No explicit spill byte counters or recognized local-memory load/store instructions were found.';
}

function layout(mlir: Line[][], add: Emit): string | undefined {
  let found = false;
  const seen = new Set<string>();
  for (const lines of mlir) {
    const cuteLayout = /!cute\.memref<[^"\n]+"\(\s*[1-9]\d*\s*\):\(\s*(-?\d+)\s*\)"\s*>/;
    const aliases = new Map<string, {stride: string; ref: Ref}>();
    for (const {ref, text} of lines) {
      const name = text.match(/^(![\w.$-]+)\s*=\s*!cute\.memref</)?.[1];
      const stride = text.match(cuteLayout)?.[1];
      if (name && stride) aliases.set(name, {stride, ref});
    }
    for (const {ref, text} of lines) {
      const op = text.match(/^(?:%[\w.$-]+\s*=\s*)?((?:vector|cute\.memref)\.(?:load|store))(?=[\s(])/)?.[1];
      if (!op) continue;
      let explicit: string | undefined;
      let typeRef: Ref | undefined;
      if (op.startsWith('vector.')) {
        // Plain vector.load/store advance in the trailing memref dimension.
        // A transfer map, gather or thread-to-index mapping needs more context.
        const strides = text.match(/:\s*memref<[^,<>]+,\s*strided<\s*\[([\d?,\s-]+)\]/)?.[1];
        const vector = text.match(/,\s*vector<(\d+)x(?:[bfius]|bf)\d+>/)?.[1];
        if (vector && Number(vector) > 1) explicit = strides?.split(',').at(-1)?.trim();
      } else {
        const types = text.split(/\)\s*:\s*\(/, 2)[1];
        const alias = types?.match(/^(![\w.$-]+)(?=\s*[,)]|$)/)?.[1];
        const entry = alias ? aliases.get(alias) : undefined;
        explicit = (types?.startsWith('!cute.memref<') ? types.match(cuteLayout)?.[1] : undefined) ?? entry?.stride;
        typeRef = entry?.ref;
      }
      if (!explicit || !/^-?\d+$/.test(explicit)) continue;
      const stride = Number(explicit);
      const signature = `${stride}:${text.replace(/\s+loc\(.*\)\s*$/, '')}`;
      if (!Number.isSafeInteger(stride) || stride === 1 || seen.has(signature)) continue;
      seen.add(signature);
      found = true;
      add({kind: 'layout', title: 'Memory access has a declared non-unit stride', status: 'potential',
        explanation: `${op} uses a memref with declared stride ${stride} elements in the accessed dimension. This layout may be intentional; the declaration alone does not establish a layout bug.`,
        impact: 'A non-unit stride can constrain contiguous vector accesses. Alignment, lane mapping, and coalescing are not established.', refs: typeRef ? [ref, typeRef] : [ref]});
    }
  }
  if (!found) return 'No supported access with an explicit non-unit stride in its accessed dimension was found. Instruction width alone does not establish layout or coalescing.';
}

const ptxSync = /^(?:bar(?:rier)?\.(?:(?:warp\.)?sync|arrive|red)|mbarrier\.(?:arrive(?:_drop)?|(?:try_|test_)?wait)|fence|membar|cp\.async(?:\.bulk)?\.wait_(?:all|group)|wgmma\.(?:wait_group|fence)|tcgen05\.(?:wait|fence))(?:\.|$)/;
const sassSync = /^(?:BAR|WARPSYNC|DEPBAR|MEMBAR|ERRBAR|BSYNC|MBARRIER|SYNCS\.(?:ARRIVE|PHASECHK|CCTL))(?:\.|$)/;
const ptxConvert = /^cvt\./;
const sassConvert = /^(?:F2F|F2FP|F2I|I2F|I2FP|I2I)(?:\.|$)/;
const mlirConvert = /^(?:arith\.(?:extf|truncf|extsi|extui|trunci)|llvm\.(?:fpext|fptrunc|sext|zext|trunc)|(?:arith|llvm)\.(?:fptosi|fptoui|sitofp|uitofp))$/;

function mlirConversions(raw: Line[], clean: Line[], add: Emit): boolean {
  let found = false;
  for (const [stage, lines] of [['raw', raw], ['clean', clean]] as const) {
    const ops = new Set<string>();
    const refs: Ref[] = [];
    let count = 0;
    for (const {ref, text} of lines) {
      const op = text.match(/^%[\w.$-]+\s*=\s*(?:"([\w.]+)"(?=\s*\()|([\w.]+)(?=\s))/)?.slice(1).find(Boolean);
      // Only count numeric casts; bitcasts, index casts and pointer casts are not evidence.
      if (!op || !mlirConvert.test(op)) continue;
      ops.add(op);
      count++;
      if (refs.length < MAX_REFS) refs.push(ref);
    }
    if (!count) continue;
    found = true;
    add({kind: 'conversion', title: `Numeric conversion operations in ${stage} MLIR`, status: 'confirmed',
      explanation: `Found ${count} numeric conversion operation site(s), including ${[...ops].slice(0, MAX_REFS).join(', ')}. Conversion may be intentional; operation presence does not establish an unexpected cast.`,
      impact: 'Conversion may round, change precision, or change range. Its effect on machine instructions and runtime is not established.', refs});
  }
  return found;
}

function instructionPresence(kind: 'synchronization' | 'conversion', lowered: Assembly[], add: Emit): string | undefined {
  const byStage = new Map<Stage, Assembly[]>();
  const sync = kind === 'synchronization';
  for (const instruction of lowered) {
    const ptx = instruction.ref.stage === 'ptx';
    const pattern = sync ? (ptx ? ptxSync : sassSync) : (ptx ? ptxConvert : sassConvert);
    if (!pattern.test(instruction.op)) continue;
    const group = byStage.get(instruction.ref.stage) ?? [];
    group.push(instruction);
    byStage.set(instruction.ref.stage, group);
  }
  for (const [stage, group] of byStage) {
    const opcodes = [...new Set(group.map(({op}) => op))].slice(0, MAX_REFS).join(', ');
    add({kind, title: `${sync ? 'Synchronization' : 'Numeric conversion'} instructions in ${stage.toUpperCase()}`, status: 'confirmed',
      explanation: `Found ${group.length} ${sync ? 'synchronization' : 'numeric conversion'} instruction site(s), including ${opcodes}. ${sync
        ? 'Instruction presence does not establish that synchronization is unnecessary or incorrect.'
        : 'Conversion may be intentional; instruction presence does not establish an unexpected cast.'}`,
      impact: sync ? 'Barriers and waits may pause progress; fences impose ordering. Their runtime effect depends on execution and protocol.'
        : 'Conversions may round, change precision, or change range depending on the opcode and operands. Runtime cost is not established.',
      refs: group.slice(0, MAX_REFS).map(({ref}) => ref)});
  }
  if (!byStage.size) return `No recognized ${sync ? 'synchronization' : 'numeric conversion'} instructions were found in the supplied PTX or SASS.`;
}

export function analyze(artifacts: Artifact[], index: LocationIndex): {findings: Finding[]; coverage: Coverage[]} {
  const findings: Finding[] = [];
  const coverage: Coverage[] = [];
  const byStage = new Map(artifacts.map((artifact) => [artifact.stage, toLines(artifact)]));
  const lines = (stage: Stage): Line[] => byStage.get(stage) ?? [];
  const lowered = [...asm(lines('ptx')), ...asm(lines('sass'))];
  const clean = scopedMLIR(lines('clean'));
  const raw = scopedMLIR(lines('raw'));
  const counts = new Map<DiagnosticKind, number>();
  const signatures = new Set<string>();
  const add = (finding: Finding) => {
    const count = counts.get(finding.kind) ?? 0;
    if (count >= MAX_PER_RULE) return;
    const refs = [...new Map(finding.refs.map((ref) => [key(ref), ref])).values()].slice(0, MAX_REFS);
    const signature = `${finding.kind}:${refs.map(key).join(',')}`;
    if (signatures.has(signature)) return;
    signatures.add(signature);
    counts.set(finding.kind, count + 1);
    findings.push({...finding, refs});
  };
  const note = (kind: DiagnosticKind, note: string | undefined) => { if (note) coverage.push({kind, note}); };
  note('scalarization', scalarization([...clean, ...raw], lowered, index, add));
  note('redundancy', redundancy(raw, clean, add));
  note('spills', spills(lines('report'), lowered, add));
  note('layout', layout([clean, raw], add));
  note('synchronization', instructionPresence('synchronization', lowered, add));
  const noMachineConversions = instructionPresence('conversion', lowered, add);
  if (!mlirConversions(raw, clean, add) && noMachineConversions) {
    note('conversion', 'No recognized numeric conversion operations were found in the supplied MLIR, PTX or SASS.');
  }
  return {findings, coverage};
}
