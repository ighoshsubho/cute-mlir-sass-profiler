import { unzipSync } from 'fflate';
import type { Artifact, Stage } from './locations.ts';

const FILE_LIMIT = 8 * 1024 * 1024;
const TOTAL_LIMIT = 32 * 1024 * 1024;
const COUNT_LIMIT = 256;

export function classify(name: string): Stage | undefined {
  const lower = name.toLowerCase();
  const base = lower.split(/[\\/]/).at(-1) ?? lower;
  if (base.endsWith('.py')) return 'python';
  if (base.endsWith('.mlir')) return /(?:^|[._-])(?:raw|debug)\.mlir$/.test(base) ? 'raw' : 'clean';
  if (base.endsWith('.ptx')) return 'ptx';
  if (/\.(?:sass|disasm|s)$/.test(base)) return 'sass';
  if (/\.(?:txt|log)$/.test(base)) return 'report';
  return undefined;
}

export async function readUploads(files: readonly File[]): Promise<{ artifacts: Artifact[]; skipped: number }> {
  if (files.length > COUNT_LIMIT) throw new Error('File count limit exceeded.');
  let bytes = 0, skipped = 0;
  const artifacts: Artifact[] = [];
  const decode = new TextDecoder('utf-8', {fatal: true});
  function reserve(name: string, size: number) {
    bytes += size;
    if (size > FILE_LIMIT || bytes > TOTAL_LIMIT) throw new Error(`Text size limit exceeded: ${name} (8 MiB per file, 32 MiB total).`);
  }
  function append(stage: Stage, name: string, data: Uint8Array) {
    let text: string;
    try { text = decode.decode(data); }
    catch { throw new Error(`${name} is not UTF-8 text.`); }
    if (text.includes('\0')) throw new Error(`${name} is not a plain-text artifact.`);
    if (text.trim()) artifacts.push({stage, name, text});
    else skipped++;
    if (artifacts.length > COUNT_LIMIT) throw new Error('File count limit exceeded.');
  }
  for (const file of files) {
    if (file.name.toLowerCase().endsWith('.zip')) {
      if (file.size > TOTAL_LIMIT || file.size < 22) throw new Error(`Invalid ZIP or ZIP size limit exceeded: ${file.name}.`);
      const data = new Uint8Array(await file.arrayBuffer());
      const seen = new Set<string>();
      let entries = 0;
      let unpacked: Record<string, Uint8Array>;
      try {
        unpacked = unzipSync(data, {filter: ({name, originalSize}) => {
          if (++entries > COUNT_LIMIT) throw new Error('ZIP entry count limit exceeded.');
          if (!classify(name) || /(?:^|\/)__MACOSX(?:\/|$)/.test(name) || /(?:^|\/)\._/.test(name)) { skipped++; return false; }
          if (seen.has(name)) throw new Error('ZIP contains duplicate artifact paths.');
          seen.add(name);
          reserve(name, originalSize);
          return true;
        }});
      } catch (error) {
        if (error instanceof Error && /limit exceeded|duplicate artifact/.test(error.message)) throw error;
        throw new Error(`Could not read ZIP: ${file.name}.`);
      }
      for (const [name, content] of Object.entries(unpacked)) {
        if (content.length > FILE_LIMIT) throw new Error(`ZIP entry size limit exceeded: ${name}.`);
        append(classify(name)!, name, content);
      }
    } else {
      const stage = classify(file.name);
      if (!stage) { skipped++; continue; }
      reserve(file.name, file.size);
      append(stage, file.name, new Uint8Array(await file.arrayBuffer()));
    }
  }
  if (!artifacts.length) throw new Error('No supported nonempty text artifacts found. Upload Python, MLIR, PTX, SASS, a resource report, or a ZIP.');
  return {artifacts, skipped};
}
