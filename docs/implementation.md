# CuTe Compiler Explorer MVP

Goal: Inspect uploaded Python, raw/clean MLIR, PTX and SASS locally with metadata-driven line correlations.

- Create React/Vite shell, test runner and four CodeMirror viewers. Browser tests cover loading, mapping, controls and import. Use static build with no API.
- Implement typed parsers and location index; exercise actual compiler output and fixtures for aliases, multi-source lines, path ambiguity and missing markers. Use source evidence grades instead of matching opcodes for correlation.
- Implement separately tested rules and searchable diagnostic cards; each finding links to evidence rows and reports uncertainty.
- Implement bounded ZIP/text imports and explicit source identity. Real CuTe sample and optional Python collector stay independent of the browser. Exercise collector with installed toolchain and preserve missing-artifact honesty.
- Verify production build and browser workflows at desktop and mobile sizes. No repository commits or pushes.
