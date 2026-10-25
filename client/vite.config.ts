import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  // host: true so the dev server is reachable from other machines on the LAN.
  // fs.allow: the client imports ../proto/gvs.proto.
  server: { host: true, port: 5173, fs: { allow: ['..'] } },
  build: { outDir: 'dist' },
});
