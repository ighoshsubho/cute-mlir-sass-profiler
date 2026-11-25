import { defineConfig } from 'vite';
export default defineConfig({
  base: './',
  server: { host: '127.0.0.1' },
  build: { rollupOptions: { output: { manualChunks(id) {
    if (/node_modules\/(?:@codemirror|@lezer|@marijn)/.test(id)) return 'editor';
    if (/node_modules\/(?:react|react-dom|scheduler)\//.test(id)) return 'react';
  } } } },
});
