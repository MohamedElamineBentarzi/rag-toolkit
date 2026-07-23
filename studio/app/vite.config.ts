import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// A plain static SPA: no backend, no proxy. `blocks.json` is served from
// public/ and everything else compiles to static assets under dist/.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
