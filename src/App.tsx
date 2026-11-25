import { useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties, DragEvent, PointerEvent, ReactNode } from 'react';
import { CodePane } from './CodePane';
import type { Mark } from './CodePane';
import { buildIndex } from './core/locations';
import type { Artifact, Ref, Stage } from './core/locations';
import type { Coverage, Finding } from './core/diagnostics';
import { classify, readUploads } from './core/imports';

const sampleFiles = import.meta.glob<string>('./sample/*', { eager: true, query: '?raw', import: 'default' });
const sample = Object.entries(sampleFiles).flatMap(([path, text]): Artifact[] => {
  const name = path.split('/').at(-1)!;
  const stage = classify(name);
  return stage ? [{stage, name, text}] : [];
});
const title: Record<Stage, string> = {python:'CuTeDSL', raw:'MLIR raw', clean:'MLIR clean', ptx:'PTX', sass:'SASS', report:'Report'};
type Pane = 'python' | 'mlir' | 'ptx' | 'sass';
type Analysis = {findings: Finding[]; coverage: Coverage[]};
const panes: {id: Pane; title: string; hint: string}[] = [
  {id:'python',title:'CuTeDSL',hint:'Python source (.py)'},
  {id:'mlir',title:'MLIR',hint:'Raw and clean IR (.mlir)'},
  {id:'ptx',title:'PTX',hint:'Virtual ISA (.ptx)'},
  {id:'sass',title:'SASS',hint:'Disassembly (.sass, .disasm)'}
];
const short = (name: string) => name.split(/[\\/]/).at(-1) ?? name;
const refKey = (ref: Ref) => `${ref.stage}:${ref.line}`;

// Used only to choose where to scroll among existing source links, not to
// establish new links or claim instruction-to-instruction identity.
const navigationPatterns = {
  mma: {
    python: /\b(?:dot|gemm)\s*\(/,
    mlir: /^\s*(?:%[\w.$-]+\s*=\s*)?(?:lir\.dot|cute\.gemm|cute_nvgpu\.mma)(?=\s|\()/,
    ptx: /^(?:tcgen05\.mma|mma\.sync|wgmma\.mma_async)\./,
    sass: /^(?:UTCHMMA|HMMA|IMMA|BMMA|DMMA)(?=\s|\.)/,
  },
  tmaLoad: {
    python: /\btma_load\s*\(/,
    mlir: /^\s*(?:%[\w.$-]+\s*=\s*)?lir\.tma_load(?=\s|\()/,
    ptx: /^cp\.async\.bulk\.tensor\.[\w.:]*\.shared::(?:cta|cluster)\.global(?=\s|\.)/,
    sass: /^UTMALDG(?=\s|\.)/,
  },
};

function operationFamily(stage: Stage, text: string) {
  if (stage === 'report') return;
  const dialect = stage === 'clean' || stage === 'raw' ? 'mlir' : stage;
  const code = stage === 'python' ? text.split('#',1)[0] : stage === 'ptx'
    ? text.replace(/^\s*(?:[\w$.-]+:(?!:)\s*)?(?:@!?%[\w$]+\s+)?/, '') : stage === 'sass'
    ? text.replace(/^\s*\/\*\s*[\da-f]+\s*\*\/\s*(?:@!?\w+\s+)?/, '') : text;
  return (Object.keys(navigationPatterns) as (keyof typeof navigationPatterns)[])
    .find(family => navigationPatterns[family][dialect].test(code));
}

function Icon({name}: {name: 'upload'|'search'|'expand'|'restore'|'theme'|'code'}) {
  const paths = {
    upload:'M12 16V4m-4 4 4-4 4 4M5 15v5h14v-5', search:'M15.5 15.5 21 21M17 10.5a6.5 6.5 0 1 1-13 0 6.5 6.5 0 0 1 13 0Z',
    expand:'M9 4H4v5m11-5h5v5M4 15v5h5m11-5v5h-5',restore:'M4 9h5V4m6 0v5h5M4 15h5v5m6 0v-5h5',
    theme:'M12 4a8 8 0 1 0 8 8 6.5 6.5 0 0 1-8-8Z',code:'m8 6-5 6 5 6m8-12 5 6-5 6m-5 3 2-6'
  };
  return <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}

function Panel({pane, stage, artifact, choices, pick, expanded, expand, controls, marks, target, select}: {
  pane: typeof panes[number]; stage: Stage; artifact?: Artifact; choices: Artifact[]; pick: (name:string)=>void;
  expanded: boolean; expand: ()=>void; controls?: ReactNode; marks:Mark[]; target?:number; select:(ref:Ref)=>void;
}) {
  const search = useRef<(() => void) | null>(null);
  return <section className="pane" data-pane={pane.id} data-stage={stage} aria-label={`${pane.title} code pane`}>
    <header className="pane-header">
      <span className={`stage-dot ${pane.id}`} aria-hidden="true" />
      <h2>{pane.title}</h2>
      {choices.length > 1 ? <select className="artifact-name" aria-label={`${pane.title} artifact`} value={artifact?.name} onChange={e=>pick(e.target.value)}>
        {choices.map(a=><option key={a.name} value={a.name}>{a.name}</option>)}
      </select> : <span className="artifact-name" title={artifact?.name}>{artifact ? short(artifact.name) : 'Not loaded'}</span>}
      <div className="pane-actions">
        {controls}
        <button className="icon" aria-label={`Search ${pane.title}`} title={`Search ${pane.title} (Ctrl/⌘+F)`} disabled={!artifact} onClick={()=>search.current?.()}><Icon name="search" /></button>
        <button className="icon" aria-label={expanded ? 'Restore panes' : `Expand ${pane.title}`} title={expanded ? 'Restore panes' : `Expand ${pane.title}`} onClick={expand}><Icon name={expanded?'restore':'expand'} /></button>
      </div>
    </header>
    {artifact ? <CodePane stage={stage} text={artifact.text} marks={marks} targetLine={target} onLine={line=>select({stage,line})} searchHandle={search} /> :
      <div className="empty"><span>No {pane.title} artifact</span><small>Load {pane.hint}</small></div>}
  </section>;
}

const getTheme = () => { try { return localStorage.getItem('cute-explorer-theme') === 'light' ? 'light' : 'dark'; } catch { return 'dark'; } };
export default function App() {
  const [artifacts, setArtifacts] = useState<Artifact[]>(sample);
  const [sampleMode, setSampleMode] = useState(sample.length > 0);
  const [chosen, setChosen] = useState<Partial<Record<Stage, string>>>({});
  const [mlir, setMlir] = useState<'raw'|'clean'>('clean');
  const [selection, setSelection] = useState<{refs: Ref[]; finding?: string; message?: string} | null>(null);
  const [expanded, setExpanded] = useState<Pane | null>(null);
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(true);
  const [theme, setTheme] = useState(getTheme);
  const [drag, setDrag] = useState(0);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const [split, setSplit] = useState({col:50,row:50});
  const [diagnostics, setDiagnostics] = useState<{input: Artifact[]; result?: Analysis; error?: string} | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const workspace = useRef<HTMLDivElement>(null);
  const reportEvidence = useRef<HTMLElement>(null);
  const importing = useRef(false);
  const choices = useMemo(() => groupArtifacts(artifacts), [artifacts]);
  const active = useMemo(() => {
    const byStage: Partial<Record<Stage,Artifact>> = {};
    for (const stage of Object.keys(choices) as Stage[]) {
      const list = choices[stage]!;
      byStage[stage] = list.find(a=>a.name===chosen[stage]) ?? list[0];
    }
    return byStage;
  }, [choices, chosen]);
  const visibleMlir = active[mlir] ? mlir : active.raw ? 'raw' : 'clean';
  const indexed = useMemo(() => Object.values(active), [active]);
  const pythonNames = useMemo(() => (choices.python ?? []).map(a=>a.name), [choices]);
  const index = useMemo(() => buildIndex(indexed, pythonNames), [indexed,pythonNames]);
  const diagnosticLines = useMemo(() => new Map(indexed.map(a=>[a.stage, a.text.split(/\r?\n/)])), [indexed]);
  const analysis = diagnostics?.input === indexed ? diagnostics.result : undefined;
  const analysisError = diagnostics?.input === indexed ? diagnostics.error : undefined;
  useEffect(() => {
    if (!indexed.length) return;
    const fail = () => setDiagnostics({input:indexed, error:'Analysis could not finish. The code panes and source links are still available.'});
    let worker: Worker | undefined;
    try {
      worker = new Worker(new URL('./analysis.worker.ts', import.meta.url), {type:'module'});
      worker.onmessage = (event: MessageEvent<Analysis | null>) => {
        if (event.data) setDiagnostics({input:indexed,result:event.data});
        else fail();
        worker?.terminate();
      };
      worker.onerror = () => {fail(); worker?.terminate();};
      worker.postMessage({artifacts:indexed, pythonNames});
    } catch { fail(); worker?.terminate(); }
    return () => worker?.terminate();
  }, [indexed,pythonNames]);
  useEffect(() => {
    if (selection?.refs.some(ref=>ref.stage==='report')) reportEvidence.current?.focus();
  }, [selection]);
  const marked = useMemo(() => {
    const result: Partial<Record<Stage, Mark[]>> = {};
    const destination: Partial<Record<Stage, number>> = {};
    const preferred: Partial<Record<Stage, number>> = {};
    const related = new Map<string, {ref:Ref; quality:Mark['quality']} >();
    const rowText = (ref: Ref) => diagnosticLines.get(ref.stage)?.[ref.line-1] ?? '';
    const origin = selection?.refs[0];
    const focusOperation = !selection?.finding && origin && operationFamily(origin.stage,rowText(origin));
    for (const ref of selection?.refs ?? []) {
      for (const item of index.related(ref)) {
        const key = refKey(item.ref);
        if (!related.has(key) || related.get(key)!.quality === 'approximate') related.set(key,item);
      }
    }
    for (const ref of selection?.refs ?? []) {
      related.set(refKey(ref), {ref, quality:'selected'});
      destination[ref.stage] ??= ref.line;
    }
    let metadata = 0, approximate = 0;
    for (const {ref,quality} of related.values()) {
      (result[ref.stage] ??= []).push({line:ref.line,quality});
      destination[ref.stage] ??= ref.line;
      if (focusOperation && quality !== 'selected' && operationFamily(ref.stage,rowText(ref)) === focusOperation) preferred[ref.stage] ??= ref.line;
      if (quality === 'metadata') metadata++;
      if (quality === 'approximate') approximate++;
    }
    return {result,destination:{...destination,...preferred},metadata,approximate};
  }, [index,selection,diagnosticLines]);
  const primary = selection?.refs[0];
  const reportRows = selection?.refs.filter(ref=>ref.stage==='report') ?? [];
  const status = selection?.message ?? (primary ?
    !index.spots(primary).length ? 'No source metadata for the selected line.' :
    marked.metadata + marked.approximate === 0 ? 'Source location found; no related lines in the loaded artifacts.' :
    `${marked.metadata} metadata-linked · ${marked.approximate} approximate` : 'Click a line or line number to trace its source location.');

  function reset(next: boolean) {
    setArtifacts(next ? sample : []); setSampleMode(next); setChosen({}); setMlir('clean'); setSelection(null); setError(''); setNotice('');
  }
  async function load(files: File[]) {
    if (!files.length || importing.current) return;
    importing.current = true;
    setBusy(true); setError(''); setNotice('');
    try {
      const {artifacts: incoming, skipped} = await readUploads(files);
      const merged = new Map((sampleMode ? [] : artifacts).map(a=>[`${a.stage}:${a.name}`,a]));
      for (const a of incoming) merged.set(`${a.stage}:${a.name}`,a);
      const next = [...merged.values()];
      const encoder = new TextEncoder();
      if (next.length > 256 || next.reduce((sum,a)=>sum+encoder.encode(a.text).length,0) > 32 * 1024 * 1024) throw new Error('Workspace limit exceeded. Clear the workspace before loading a new compilation.');
      setArtifacts(next); setSampleMode(false); setSelection(null);
      setChosen(previous=>({...(!sampleMode ? previous : {}), ...Object.fromEntries(incoming.slice().reverse().map(a=>[a.stage,a.name]))}));
      if (incoming.some(a=>a.stage==='clean')) setMlir('clean');
      else if (incoming.some(a=>a.stage==='raw')) setMlir('raw');
      setNotice(`Loaded ${incoming.length} artifact${incoming.length===1?'':'s'}${skipped ? `; ${skipped} unsupported or empty entries skipped.` : '.'}`);
    } catch (e) { setError(e instanceof Error ? e.message : 'Unable to read artifacts.'); }
    finally { importing.current=false; setBusy(false); if(input.current) input.current.value=''; }
  }
  function drop(e: DragEvent) { if (!e.dataTransfer.files.length) return; e.preventDefault(); setDrag(0); void load(Array.from(e.dataTransfer.files)); }
  function resize(axis: 'col'|'row', e: PointerEvent<HTMLDivElement>) {
    e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId);
  }
  function move(axis: 'col'|'row', e: PointerEvent<HTMLDivElement>) {
    if(!e.currentTarget.hasPointerCapture(e.pointerId)) return;
    const box = workspace.current!.getBoundingClientRect();
    const value = axis==='col' ? (e.clientX-box.left)/box.width : (e.clientY-box.top)/box.height;
    setSplit(d=>({...d,[axis]:clamp(value*100)}));
  }
  function chooseFinding(finding: Finding, id: string) {
    if (expanded) setExpanded(null);
    if (finding.refs.some(ref=>ref.stage==='raw') && !finding.refs.some(ref=>ref.stage==='clean')) setMlir('raw');
    if (finding.refs.some(ref=>ref.stage==='clean') && !finding.refs.some(ref=>ref.stage==='raw')) setMlir('clean');
    setSelection({refs:finding.refs,finding:id,message: `${finding.status === 'confirmed' ? 'Confirmed evidence' : 'Potential issue'}: ${finding.title}`});
  }
  function toggleTheme() {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next); try { localStorage.setItem('cute-explorer-theme',next); } catch { /* optional preference */ }
  }
  return <main data-theme={theme} onDrop={drop} onDragEnter={e=>{if(e.dataTransfer.types.includes('Files')) {e.preventDefault();setDrag(n=>n+1);}}}
    onDragLeave={()=>setDrag(n=>Math.max(0,n-1))} onDragOver={e=>{if(e.dataTransfer.types.includes('Files')) {e.preventDefault();e.dataTransfer.dropEffect='copy';}}}>
    <header className="topbar">
      <div className="brand"><span className="brand-icon"><Icon name="code" /></span><div><h1>CuTe Compiler Explorer</h1><span>Local artifact inspector</span></div></div>
      <div className="toolbar">
        <button className="primary" onClick={()=>input.current?.click()} disabled={busy}><Icon name="upload" />{busy?'Reading…':'Open files'}</button>
        <input ref={input} type="file" multiple hidden accept=".py,.mlir,.ptx,.sass,.disasm,.s,.txt,.log,.zip" onChange={e=>void load(Array.from(e.target.files ?? []))} />
        <button onClick={()=>reset(true)} disabled={busy || !sample.length}>Load sample</button>
        <button onClick={()=>reset(false)} disabled={busy || !artifacts.length}>Clear workspace</button>
        <button aria-expanded={diagnosticsOpen} onClick={()=>setDiagnosticsOpen(v=>!v)} aria-label={diagnosticsOpen?'Hide diagnostics':'Show diagnostics'}>Diagnostics</button>
        <button className="icon theme" onClick={toggleTheme} title="Toggle light/dark theme" aria-label="Toggle theme"><Icon name="theme" /></button>
      </div>
    </header>
    <div className="context">
      <span className="workspace-name">{sampleMode?'Sample kernel':artifacts.length?'Local workspace':'Empty workspace'}</span>
      <span className="context-separator" aria-hidden="true" />
      <span>Drop a ZIP or individual files</span>
      <span className="context-note">{notice || 'Artifacts stay in this browser'}</span>
    </div>
    {error && <div className="error" role="alert">{error}<button onClick={()=>setError('')} aria-label="Dismiss error">Dismiss</button></div>}
    <div className="body-grid" data-diagnostics={diagnosticsOpen}>
      <div className="workspace" ref={workspace} data-expanded={expanded ?? ''} style={{'--col':split.col/100,'--row':split.row/100} as CSSProperties}>
        {panes.map(pane=>{
          const stage = pane.id==='mlir' ? visibleMlir : pane.id;
          return <Panel key={pane.id} pane={pane} stage={stage} artifact={active[stage]} choices={choices[stage] ?? []}
            pick={name=>{setChosen(v=>({...v,[stage]:name}));setSelection(null);}} expanded={expanded===pane.id}
            expand={()=>setExpanded(v=>v===pane.id?null:pane.id)} marks={marked.result[stage] ?? []} target={marked.destination[stage]}
            select={ref=>setSelection({refs:[ref]})} controls={pane.id==='mlir' && <div className="ir-toggle" aria-label="MLIR version">
              {(['raw','clean'] as const).map(kind=><button key={kind} disabled={!active[kind]} title={!active[kind]?'Artifact not supplied':undefined} aria-pressed={kind===visibleMlir && Boolean(active[kind])} onClick={()=>setMlir(kind)}>{kind==='raw'?'Raw IR':'Clean IR'}</button>)}
            </div>} />;
        })}
        {(['col','row'] as const).map(axis=><div key={axis} className={`split ${axis}`} role="separator" tabIndex={0}
          aria-label={axis==='col'?'Resize columns':'Resize rows'} aria-orientation={axis==='col'?'vertical':'horizontal'}
          aria-valuemin={20} aria-valuemax={80} aria-valuenow={Math.round(split[axis])}
          onPointerDown={e=>resize(axis,e)} onPointerMove={e=>move(axis,e)}
          onKeyDown={e=>{const delta=axis==='col' ? (e.key==='ArrowRight'?3:e.key==='ArrowLeft'?-3:0) : (e.key==='ArrowDown'?3:e.key==='ArrowUp'?-3:0); if(delta){e.preventDefault();setSplit(p=>({...p,[axis]:clamp(p[axis]+delta)}));}}} />)}
      </div>
      <aside className="diagnostics" aria-label="Kernel diagnostics" hidden={!diagnosticsOpen}>
        <header><h2>Kernel diagnostics</h2><span className="count">{analysis?.findings.length ?? '–'}</span></header>
        <p className="aside-note">Rule-based observations. Confirmed means the evidence is present, not that it is a bug.</p>
        {choices.report && <label className="report-picker">Resource report
          <select className="artifact-name" value={active.report?.name} onChange={e=>{setChosen(v=>({...v,report:e.target.value}));setSelection(null);}}>
            {choices.report.map(a=><option key={a.name} value={a.name}>{a.name}</option>)}
          </select>
        </label>}
        {reportRows.length>0 && active.report && <section ref={reportEvidence} tabIndex={-1} className="report-evidence" aria-label="Resource report evidence">
          <h3>{short(active.report.name)}</h3>
          {reportRows.map(ref=><pre key={refKey(ref)}><mark>{ref.line}: {diagnosticLines.get('report')?.[ref.line-1]}</mark></pre>)}
        </section>}
        {analysisError && <p className="no-findings">{analysisError}</p>}
        {!analysis && !analysisError && <p className="no-findings">{indexed.length ? 'Analyzing artifacts…' : 'Load artifacts to run diagnostics.'}</p>}
        {analysis?.findings.length===0 && <p className="no-findings">No findings from the current artifacts.</p>}
        <div className="finding-list">
          {analysis?.findings.map((f,i)=>{
            const id = `${f.kind}:${i}`;
            const locations = [...new Map(f.refs.flatMap(ref=>index.spots(ref)).map(loc=>[`${loc.file}:${loc.line}`,loc])).values()];
            return <button className="finding" key={id} aria-pressed={selection?.finding===id} onClick={()=>chooseFinding(f,id)}>
              <span className="finding-top"><span className="finding-title">{f.title}</span><span className={`badge ${f.status}`}>{f.status}</span></span>
              <span className="finding-body">{f.explanation}</span>
              <span className="finding-impact">{f.impact}</span>
              <span className="evidence">{locations.length ? `Source: ${locations.slice(0,3).map(loc=>`${short(loc.file)}:${loc.line}`).join(', ')}${locations.length>3?' …':''}` : 'No source location in supplied evidence.'}</span>
              {f.refs.slice(0,4).map(ref=>{
                const lines = diagnosticLines.get(ref.stage);
                return <span className="evidence" key={refKey(ref)}>{title[ref.stage]}:{ref.line} {lines?.[ref.line-1]?.trim().slice(0,160)}</span>;
              })}
            </button>;
          })}
        </div>
        {analysis && <details><summary>Analysis coverage</summary><ul className="coverage">{analysis.coverage.map(c=><li key={c.kind}>{c.note}</li>)}</ul></details>}
      </aside>
    </div>
    <footer><span className="selection-status" role="status">{status}</span><div className="legend"><span><i className="direct" /> Metadata</span><span><i className="approx" /> Approximate</span></div></footer>
    {drag>0 && <div className="drop-overlay"><span>Drop compilation artifacts</span><small>Python · MLIR · PTX · SASS · ZIP</small></div>}
  </main>;
}

function groupArtifacts(artifacts: Artifact[]) {
  return artifacts.reduce<Partial<Record<Stage,Artifact[]>>>((groups,artifact)=>{
    (groups[artifact.stage] ??= []).push(artifact); return groups;
  }, {});
}
function clamp(value:number) {return Math.min(80,Math.max(20,value));}
