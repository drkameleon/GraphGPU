import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
    resolve: {
        alias: {
            'graphgpu': resolve(__dirname, '../src'),
        },
    },
});
