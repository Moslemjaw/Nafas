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
      includeAssets: ['favicon.svg', 'capture-worklet.js', 'model/*'],
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
        // model + onnxruntime wasm (~14 MB) are precached so the app works offline
        globPatterns: ['**/*.{js,mjs,css,html,svg,png,wasm,onnx,json}'],
        maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
      },
    }),
  ],
})
