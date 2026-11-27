// ============================================================
// graphGPU - Headless browser environment for benchmarks
// ============================================================
// happy-dom for the DOM, the fake WebGPU device from the tests,
// and a no-op 2D context for the label overlay (so its JS work
// is measured, minus actual text rasterization).
// ============================================================

import { Window } from 'happy-dom';
import {
    fakeDevice, WEBGPU_GLOBALS,
    fakeNavigatorGPU, fakeWebGPUContext,
} from '../tests/fake-device';

const WIDTH = 1280;
const HEIGHT = 800;

function fake2DContext() {
    const noop = () => {};
    return {
        font: '', fillStyle: '', textAlign: 'center', textBaseline: 'middle',
        save: noop, restore: noop, scale: noop, translate: noop, rotate: noop,
        clearRect: noop, fillText: noop,
        measureText: (text: string) => ({ width: text.length * 7 }),
    };
}

let installed = false;

/** Install the globals the renderer needs (idempotent) */
export function installBrowserEnv(): void {
    if (installed) return;
    installed = true;

    const window = new Window({ width: WIDTH, height: HEIGHT });
    const g = globalThis as Record<string, unknown>;
    g.window = window;
    g.document = window.document;
    g.getComputedStyle = window.getComputedStyle.bind(window);
    for (const [name, value] of Object.entries(WEBGPU_GLOBALS)) g[name] = value;
    Object.defineProperty(globalThis, 'navigator', {
        value: { gpu: fakeNavigatorGPU(fakeDevice()) },
        configurable: true,
    });

    const proto = window.HTMLCanvasElement.prototype as unknown as Record<string, unknown>;
    proto.getContext = (type: string) => type === 'webgpu' ? fakeWebGPUContext() : fake2DContext();
    proto.getBoundingClientRect = () => ({
        left: 0, top: 0, right: WIDTH, bottom: HEIGHT,
        width: WIDTH, height: HEIGHT, x: 0, y: 0,
    });
    Object.defineProperty(proto, 'clientWidth', { get: () => WIDTH, configurable: true });
    Object.defineProperty(proto, 'clientHeight', { get: () => HEIGHT, configurable: true });
}

/** A canvas attached to the document */
export function createCanvas(): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
    return canvas;
}
