// ============================================================
// graphGPU - Fake WebGPU + animation frames (vitest helpers)
// ============================================================

import { vi } from 'vitest';
import {
    fakeDevice, resetGPU,
    WEBGPU_GLOBALS, fakeNavigatorGPU, fakeWebGPUContext,
} from './fake-device';

export { gpu, fakeDevice, releaseMaps, FakeBuffer } from './fake-device';

/**
 * Install WebGPU globals, a fake `navigator.gpu` and a fake
 * 'webgpu' canvas context, so Renderer.init() succeeds.
 */
export function installFakeWebGPU(canvas?: HTMLCanvasElement): void {
    resetGPU();
    for (const [name, value] of Object.entries(WEBGPU_GLOBALS)) vi.stubGlobal(name, value);

    if (canvas) {
        vi.stubGlobal('navigator', { ...globalThis.navigator, gpu: fakeNavigatorGPU(fakeDevice()) });
        const context = fakeWebGPUContext();
        const original = canvas.getContext.bind(canvas);
        vi.spyOn(canvas, 'getContext').mockImplementation(((type: string) =>
            type === 'webgpu' ? context : original(type as '2d')) as typeof canvas.getContext);
    }
}

// -----------------------------------------------------------
// Animation frames
// -----------------------------------------------------------

let frames = new Map<number, FrameRequestCallback>();
let nextFrameId = 1;
let now = 0;

/** Replace requestAnimationFrame with a manually stepped one */
export function installFakeFrames(): void {
    frames = new Map();
    now = 0;
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
        const id = nextFrameId++;
        frames.set(id, cb);
        return id;
    });
    vi.stubGlobal('cancelAnimationFrame', (id: number) => { frames.delete(id); });
}

export function pendingFrames(): number {
    return frames.size;
}

/** Run `count` frames (each ~16ms apart), letting promises settle in between */
export async function flushFrames(count: number): Promise<void> {
    for (let i = 0; i < count && frames.size > 0; i++) {
        const pending = [...frames.values()];
        frames = new Map();
        now += 16;
        for (const cb of pending) cb(now);
        await new Promise<void>(r => setTimeout(r, 0));
    }
}
