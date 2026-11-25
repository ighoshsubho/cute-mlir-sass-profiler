export type Stage = 'python' | 'raw' | 'clean' | 'ptx' | 'sass' | 'report';
export type Artifact = { stage: Stage; name: string; text: string };
export type Ref = { stage: Stage; line: number };
export type Spot = { file: string; line: number; column?: number };
type Related = { ref: Ref; quality: 'metadata' | 'approximate' };
export type LocationIndex = {
  spots(ref: Ref): Spot[];
  related(ref: Ref): Related[];
};

type Rows = Map<number, Spot[]>;
type RecordFile = (file: string) => void;
const stages: Stage[] = ['python', 'raw', 'clean', 'ptx', 'sass', 'report'];
const refKey = ({ stage, line }: Ref) => `${stage}:${line}`;
const lineKey = (file: string, line: number) => `${line}:${file}`;
const textLines = (text: string) => text.replace(/\r?\n$/, '').split(/\r?\n/);

function makeSpot(file: string, line: number, column = 0): Spot | undefined {
  if (!file || !Number.isSafeInteger(line) || line <= 0) return;
  return Number.isSafeInteger(column) && column > 0 ? { file, line, column } : { file, line };
}

const spotKey = ({ file, line, column }: Spot) => `${column ?? 0}:${line}:${file}`;
const uniqueSpots = (spots: Spot[]) => [...new Map(spots.map(spot => [spotKey(spot), spot])).values()];
const alias = /^#[\w$-]+$/;
const integer = /^\d+$/;
const quoted = String.raw`"(?:[^"\\\r\n]|\\.)*"`;

type Token = { value: string; line: number; indent: number };
type LocationPart = Spot | string;

function unquote(value: string, dialect: 'mlir' | 'ptx' | 'sass'): string {
  const inner = value.slice(1, -1);
  // MLIR prints byte escapes (including non-ASCII UTF-8) as \\XX.
  if (dialect === 'mlir') {
    return inner.replace(/(?:\\[0-9a-f]{2})+|\\["\\nrt]/gi, escape => {
      if (escape.length > 2) {
        const bytes = escape.match(/[0-9a-f]{2}/gi) ?? [];
        return new TextDecoder().decode(Uint8Array.from(bytes, x => parseInt(x, 16)));
      }
      return ({ n: '\n', r: '\r', t: '\t' } as Record<string, string>)[escape[1]] ?? escape[1];
    });
  }
  return dialect === 'ptx' ? inner.replace(/\\(["\\])/g, '$1') : inner;
}

function mlirTokens(text: string): Token[] {
  const pattern = new RegExp(String.raw`${quoted}|//.*|->|[#%@!]?[\w$.-]+|[^\s]`, 'g');
  return textLines(text).flatMap((line, i) => {
    const indent = line.search(/\S/);
    return [...line.matchAll(pattern)].filter(m => !m[0].startsWith('//'))
      .map(m => ({ value: m[0], line: i + 1, indent }));
  });
}

// Parse only the location grammar, not arbitrary nested attribute values.
function mlirLocation(tokens: Token[], start: number): { end: number; parts: LocationPart[] } | undefined {
  const value = () => tokens[pos]?.value ?? '';
  const eat = (token: string) => {
    if (value() !== token) return false;
    pos++;
    return true;
  };
  const parts: LocationPart[] = [];
  let pos = start + 2; // loc (
  function parse(depth: number): boolean {
    if (depth > 128) return false;
    if (eat('unknown') || eat('?')) return true;
    if (eat('loc')) return eat('(') && parse(depth + 1) && eat(')');
    if (eat('callsite')) return eat('(') && parse(depth + 1) && eat('at') && parse(depth + 1) && eat(')');
    if (eat('fused')) {
      if (eat('<')) {
        let level = 1;
        while (value() && level > 0) {
          const token = tokens[pos++].value;
          if (token === '<') level++;
          if (token === '>') level--;
        }
        if (level) return false;
      }
      if (!eat('[')) return false;
      if (eat(']')) return true;
      do {
        if (!parse(depth + 1)) return false;
      } while (eat(','));
      return eat(']');
    }
    const token = value();
    if (alias.test(token)) {
      parts.push(token);
      pos++;
      return true;
    }
    if (!token.startsWith('"') || token.length < 2) return false;
    pos++;
    if (!eat(':')) return !eat('(') || (parse(depth + 1) && eat(')'));
    if (!integer.test(value())) return false;
    const line = Number(tokens[pos++].value);
    let column = 0;
    if (eat(':')) {
      if (!integer.test(value())) return false;
      column = Number(tokens[pos++].value);
      if (eat('to')) {
        if (integer.test(value())) pos++;
        if (!eat(':') || !integer.test(value())) return false;
        pos++;
      }
    }
    const spot = makeSpot(unquote(token, 'mlir'), line, column);
    if (spot) parts.push(spot);
    return true;
  }
  return parse(0) && eat(')') ? { end: pos, parts } : undefined;
}

function mlirRows(text: string, recordFile: RecordFile): Rows {
  const aliases = new Map<string, LocationPart[]>();
  const pending = new Map<number, LocationPart[]>();
  const tokens = mlirTokens(text);
  let row = 0;
  let owner: Token | undefined;
  // Both regions and attribute dictionaries use braces. Balance all of them so
  // closing one restores the operation at its opener, not the last nested row.
  const braceOwners: (Token | undefined)[] = [];
  let declaration: string | undefined;
  for (let i = 0; i < tokens.length; ) {
    const token = tokens[i];
    const next = tokens[i + 1]?.value;
    if (token.line > row) {
      row = token.line;
      declaration = alias.test(token.value) && next === '=' ? token.value : undefined;
      let j = i;
      // SSA results can be a list (%a, %b) or a result group (%a:2).
      while (tokens[j]?.line === row && tokens[j]?.value.startsWith('%')) {
        j++;
        if (tokens[j]?.value === ':' && integer.test(tokens[j + 1]?.value ?? '')) j += 2;
        if (tokens[j]?.value !== ',') break;
        j++;
      }
      const hasResult = j > i && tokens[j]?.line === row && tokens[j]?.value === '=';
      const looksLikeOp = /^[\w][\w.-]*$/.test(token.value) || token.value.startsWith('"');
      // A wrapped operand/type list can close at the operation's own indentation.
      const isTrailing = ['loc', '{', '}', ')', ']', ':', '->', 'else'].includes(token.value);
      if (declaration || token.value.startsWith('^')) owner = undefined;
      else if (!isTrailing && (hasResult || looksLikeOp || !owner || token.indent <= owner.indent)) {
        owner = token;
      }
    }
    if (token.value === '{') braceOwners.push(owner);
    // In `} else {`, the next opener inherits this same owner.
    else if (token.value === '}') owner = braceOwners.pop();
    if (token.value === 'loc' && next === '(') {
      const parsed = mlirLocation(tokens, i);
      if (parsed) {
        for (const part of parsed.parts) if (typeof part !== 'string') recordFile(part.file);
        const lastLine = tokens[parsed.end - 1].line;
        if (declaration) aliases.set(declaration, parsed.parts);
        else if (owner && tokens[i - 1]?.value !== '=') {
          const line = owner.line;
          pending.set(line, [...(pending.get(line) ?? []), ...parsed.parts]);
        }
        i = parsed.end;
        row = lastLine;
        continue;
      }
    }
    i++;
  }
  const rows: Rows = new Map();
  for (const [line, parts] of pending) {
    const spots: Spot[] = [];
    const seen = new Set<string>();
    const todo = [...parts].reverse();
    while (todo.length) {
      const part = todo.pop()!;
      if (typeof part !== 'string') spots.push(part);
      else if (!seen.has(part)) {
        seen.add(part);
        const targets = aliases.get(part) ?? [];
        for (let i = targets.length - 1; i >= 0; i--) todo.push(targets[i]);
      }
    }
    if (spots.length) rows.set(line, uniqueSpots(spots));
  }
  return rows;
}

function ptxRows(text: string, recordFile: RecordFile): Rows {
  // Keep strings intact while stripping comments, including multiline block comments.
  const clean = text.replace(new RegExp(String.raw`${quoted}|//[^\r\n]*|/\*[\s\S]*?\*/`, 'g'), token =>
    token.startsWith('/') ? token.replace(/[^\r\n]/g, ' ') : token);
  const lines = textLines(clean);
  const fileDirective = new RegExp(String.raw`^\s*\.file\s+(\d+)\s+(${quoted})`);
  const files = new Map<number, string | undefined>();
  for (const line of lines) {
    const match = fileDirective.exec(line);
    if (!match) continue;
    const id = Number(match[1]);
    const file = unquote(match[2], 'ptx');
    recordFile(file);
    // Reused indices with conflicting identities are not resolvable.
    files.set(id, !files.has(id) || files.get(id) === file ? file : undefined);
  }
  const source = (id: string, line: string, column?: string) =>
    makeSpot(files.get(Number(id)) ?? '', Number(line), Number(column ?? 0));
  const rows: Rows = new Map();
  let active: Spot[] = [];
  let header = false;
  let depth = 0;
  let continuation = false;
  const functionHeader = /^(?:\.[\w]+\s+)*\.(?:entry|func)\b/;
  let excerpt = !lines.some(line => functionHeader.test(line.trimStart()));
  const strings = new RegExp(quoted, 'g');
  for (const [i, raw] of lines.entries()) {
    let line = raw.trim();
    if (functionHeader.test(line)) {
      active = [];
      header = true;
      depth = 0;
      continuation = false;
    }
    if (header) {
      const opening = line.indexOf('{');
      if (opening < 0) {
        if (line.endsWith(';')) header = false; // Prototype, not a body.
        continue;
      }
      depth = 1;
      header = false;
      line = line.slice(opening + 1).trimStart();
    }
    if (!depth && /^\.section\b/.test(line)) excerpt = false;
    if (!depth && !excerpt) continue;
    if (/^\.loc\b/.test(line)) {
      const loc = /^\.loc\s+(\d+)\s+(\d+)(?:\s+(\d+))?(?=$|[\s,;])/.exec(line);
      active = [];
      const primary = loc && source(loc[1], loc[2], loc[3]);
      if (primary) {
        active.push(primary);
        for (const match of line.matchAll(/\binlined_at\s+(\d+)\s+(\d+)\s+(\d+)(?=$|[\s,;])/g)) {
          const caller = source(match[1], match[2], match[3]);
          if (caller) active.push(caller);
        }
        active = uniqueSpots(active);
      }
      continue;
    }
    const code = line.replace(strings, '""');
    // A label ends in a single colon. PTX opcode qualifiers use double
    // colons (e.g. tcgen05.mma.cta_group::1 and shared::cta).
    const instruction = code.replace(/^(?:[{}]\s*|[\w$.-]+:(?!:)\s*)+/, '');
    if (!continuation && /^(?:@!?[%\w$]+\s+)?[a-z][a-z\d._]*(?:::[a-z\d._]+)*(?=\s|;|$)/i.test(instruction)) {
      if (active.length) rows.set(i + 1, active);
      continuation = !code.includes(';');
    } else if (continuation && code.includes(';')) {
      continuation = false;
    }
    for (const token of code.matchAll(/[{}]/g)) depth += token[0] === '{' ? 1 : -1;
    if (depth <= 0 && !excerpt) {
      depth = 0;
      active = [];
      continuation = false;
    }
  }
  return rows;
}

const embeddedPtx = '.nv_debug_ptx_txt';

function sassRows(text: string, recordFile: RecordFile): Rows {
  const rows: Rows = new Map();
  const location = String.raw`(${quoted}),\s*line\s+(\d+)(?:,\s*column\s+(\d+))?`;
  const annotation = new RegExp(String.raw`^\s*//##\s*File\s+${location}`);
  const inlinedAt = new RegExp(String.raw`\binlined at\s+${location}`, 'g');
  const instruction = /^\s*\/\*\s*[\da-f]+\s*\*\/\s*\{?\s*(?:@!?[\w]+\s+)?[A-Z][A-Z\d._]*(?=\s|;)/;
  let inCode = true; // Also allow snippets without section headers.
  let source: Spot[] = [];
  let pendingSource: Spot[] | undefined;
  let ptx: Spot | undefined;
  for (const [i, line] of textLines(text).entries()) {
    const section = /^\s*\.section\s+([^\s,]+)/.exec(line);
    if (section) {
      inCode = /^\.text(?:\.|$)/.test(section[1]);
      source = [];
      pendingSource = undefined;
      ptx = undefined;
      continue;
    }
    if (!inCode) continue;
    const match = annotation.exec(line);
    if (match) {
      const decode = (tokens: RegExpMatchArray) => {
        const file = unquote(tokens[1], 'sass');
        recordFile(file);
        return makeSpot(file, Number(tokens[2]), Number(tokens[3] ?? 0));
      };
      const spot = decode(match);
      if (unquote(match[1], 'sass') === embeddedPtx) {
        ptx = spot;
      } else {
        pendingSource ??= [];
        if (spot) pendingSource.push(spot);
        for (const frame of line.slice(match[0].length).matchAll(inlinedAt)) {
          const caller = decode(frame);
          if (caller) pendingSource.push(caller);
        }
      }
      continue;
    }
    if (!instruction.test(line)) continue;
    if (pendingSource) {
      source = uniqueSpots(pendingSource);
      pendingSource = undefined;
    }
    const spots = ptx ? [...source, ptx] : source;
    if (spots.length) rows.set(i + 1, spots);
  }
  return rows;
}

function artifactRows({ stage, name, text }: Artifact, recordFile: RecordFile): Rows {
  if (stage === 'raw' || stage === 'clean') return mlirRows(text, recordFile);
  if (stage === 'ptx') return ptxRows(text, recordFile);
  if (stage === 'sass') return sassRows(text, recordFile);
  const rows: Rows = new Map();
  if (stage === 'python' && name && text) {
    textLines(text).forEach((_, i) => rows.set(i + 1, [{ file: name, line: i + 1 }]));
  }
  return rows;
}

const basename = (file: string) => file.split(/[/\\]/).at(-1) ?? '';

/** Pass all loaded Python names when indexing only the currently visible artifacts. */
export function buildIndex(artifacts: Artifact[], loadedPythonNames: readonly string[] = []): LocationIndex {
  const byRef = new Map<string, Spot[]>();
  const bySourceLine = new Map<string, Map<string, Ref>>();
  const loadedByBasename = new Map<string, Set<string>>();
  const qualifiedByBasename = new Map<string, Set<string>>();
  const generatedNames = new Set(artifacts.filter(artifact => artifact.stage !== 'python').map(artifact => basename(artifact.name)));
  generatedNames.add(embeddedPtx);
  const addIdentity = (files: Map<string, Set<string>>, name: string, file: string) => {
    const known = files.get(name) ?? new Set<string>();
    known.add(file);
    files.set(name, known);
  };
  const recordFile = (file: string) => {
    const name = basename(file);
    if (file !== name) addIdentity(qualifiedByBasename, name, file);
  };
  for (const file of loadedPythonNames) addIdentity(loadedByBasename, basename(file), file);
  for (const artifact of artifacts) {
    if (artifact.stage === 'python') addIdentity(loadedByBasename, basename(artifact.name), artifact.name);
    else recordFile(artifact.name);
    for (const [line, spots] of artifactRows(artifact, recordFile)) {
      const ref = { stage: artifact.stage, line };
      const rowKey = refKey(ref);
      byRef.set(rowKey, spots);
      for (const spot of spots) {
        const key = lineKey(spot.file, spot.line);
        const refs = bySourceLine.get(key) ?? new Map<string, Ref>();
        refs.set(rowKey, ref);
        bySourceLine.set(key, refs);
      }
    }
  }
  return {
    spots(ref) {
      return (byRef.get(refKey(ref)) ?? []).map(spot => ({ ...spot }));
    },
    related(ref) {
      const origin = refKey(ref);
      const matches = new Map<string, Related>();
      const collect = (file: string, line: number, quality: Related['quality'], scope?: 'python' | 'compiler') => {
        for (const [key, candidate] of bySourceLine.get(lineKey(file, line)) ?? []) {
          if (scope && (candidate.stage === 'python') !== (scope === 'python')) continue;
          if (key !== origin && (quality === 'metadata' || !matches.has(key))) {
            matches.set(key, { ref: { ...candidate }, quality });
          }
        }
      };
      // Match directly against this row's provenance, never a matched row's provenance.
      for (const { file, line } of byRef.get(origin) ?? []) {
        const name = basename(file);
        const qualified = qualifiedByBasename.get(name);
        const loaded = loadedByBasename.get(name);
        const isBare = file === name;
        const ambiguousCompiler = (qualified?.size ?? 0) > 1;
        const ambiguousLoaded = (loaded?.size ?? 0) > 1;
        if (!isBare || (!ambiguousCompiler && !ambiguousLoaded)) collect(file, line, 'metadata');
        if (!name || generatedNames.has(name) || ambiguousCompiler || ambiguousLoaded) continue;
        const compilerPath = qualified?.values().next().value;
        const loadedPath = loaded?.values().next().value;
        // A ZIP member is a loaded source identity, not another compiler path.
        if (ref.stage === 'python') {
          if (!isBare) collect(name, line, 'approximate', 'compiler');
          if (compilerPath && compilerPath !== file) collect(compilerPath, line, 'approximate', 'compiler');
        } else {
          if (loadedPath && loadedPath !== file) collect(loadedPath, line, 'approximate', 'python');
          const counterpart = isBare ? compilerPath : name;
          if (counterpart) collect(counterpart, line, 'approximate', 'compiler');
        }
      }
      const sorted = [...matches.values()].sort(({ ref: a }, { ref: b }) =>
        stages.indexOf(a.stage) - stages.indexOf(b.stage) || a.line - b.line);
      // A row is always its own origin; this is not evidence about any other row.
      return [{ ref: { ...ref }, quality: 'metadata' }, ...sorted];
    },
  };
}
