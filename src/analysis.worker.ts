import { analyze } from './core/diagnostics';
import { buildIndex } from './core/locations';
import type { Artifact } from './core/locations';

self.onmessage = (event: MessageEvent<{artifacts: Artifact[]; pythonNames: string[]}>) => {
  const {artifacts, pythonNames} = event.data;
  try { self.postMessage(analyze(artifacts, buildIndex(artifacts, pythonNames))); }
  catch { self.postMessage(null); }
};
