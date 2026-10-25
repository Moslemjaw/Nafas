import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { VitePWA } from 'vite-plugin-pwa'

// Relative base so the same build works on localhost and on GitHub Pages (/Nafas/).
export default defineConfig({
  base: './',
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.svg', 'capture-worklet.js', 'model/cough*', 'model/lung_config.json'],
      manifest: {
        name: 'Nafas Cough Detector',
        short_name: 'Nafas',
        description: 'Detects coughs from the phone microphone, on the device.',
        start_url: '.',
        display: 'standalone',
        background_color: '#f6f5f8',
        theme_color: '#2f6fde',
        icons: [
          { src: 'icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        // cough model + onnxruntime wasm (~14 MB) are precached so the app works offline
        globPatterns: ['**/*.{js,mjs,css,html,svg,png,wasm,onnx,json}'],
        // the 45 MB lung model downloads only when Lung check is first opened, then is cached
        globIgnores: ['**/lung.onnx'],
        maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
        runtimeCaching: [
          {
            urlPattern: /\/model\/lung\.onnx$/,
            handler: 'CacheFirst',
            options: { cacheName: 'lung-model', expiration: { maxEntries: 1 } },
          },
        ],
      },
    }),
  ],
})
