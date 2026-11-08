import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

import courseCatalog from './plugins/course-catalog.js'

export default defineConfig({
  base: '/llm-engineering-lab/',
  plugins: [vue(), courseCatalog(), {
    name: 'watch-course-python',
    configureServer(server) {
      // glob 的 Python 文件在 web 根目录外；显式监听目录才能发现新增文件。
      for (const pkg of ['llm_agent', 'llm_finetune', 'llm_models', 'llm_train', 'llm_infer']) {
        server.watcher.add(fileURLToPath(new URL(`../${pkg}/`, import.meta.url)))
      }
    },
  }],
  build: { manifest: true },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 5173,
    host: '0.0.0.0',
    fs: { allow: ['..'] }, // SourceSnippet 要读仓库根下的 llm_*/**/*.py
  },
})
