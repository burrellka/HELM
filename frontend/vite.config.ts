import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Local dev: Vite serves the SPA and proxies /api to the backend, so the
// browser sees ONE origin — same as production, where nginx does it. Keeping
// dev same-origin is what lets us ship with no CORS middleware at all.
//
// Run the backend with HELM_DEV_MODE=true so the session cookie isn't marked
// Secure (it would be dropped over http://localhost otherwise).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: false,
        // The log tail must not be buffered by the dev proxy either, or
        // you'll "fix" a streaming bug that only exists in dev.
        ws: false,
      },
    },
  },
  build: { sourcemap: true },
});
