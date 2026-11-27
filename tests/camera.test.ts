// ============================================================
// graphGPU - Camera tests
// ============================================================

import { describe, it, expect } from 'vitest';
import { Camera } from '../src/core/Camera';

const W = 800;
const H = 600;

describe('Camera', () => {
    it('maps the canvas center to the world origin', () => {
        const cam = new Camera(W, H);
        const [wx, wy] = cam.screenToWorld(W / 2, H / 2);
        expect(wx).toBeCloseTo(0);
        expect(wy).toBeCloseTo(0);
    });

    it('round-trips screen ↔ world', () => {
        const cam = new Camera(W, H);
        cam.setPosition(0.3, -0.2);
        cam.setZoom(2.5);
        const [wx, wy] = cam.screenToWorld(123, 456);
        const [sx, sy] = cam.worldToScreen(wx, wy);
        expect(sx).toBeCloseTo(123);
        expect(sy).toBeCloseTo(456);
    });

    it('flips the y axis (screen down = world down)', () => {
        const cam = new Camera(W, H);
        const [, top] = cam.screenToWorld(W / 2, 0);
        const [, bottom] = cam.screenToWorld(W / 2, H);
        expect(top).toBeGreaterThan(bottom);
    });

    it('pans vertically so the content follows the cursor', () => {
        const cam = new Camera(W, H);
        const before = cam.screenToWorld(400, 300);
        cam.pan(0, 20);
        const [sx, sy] = cam.worldToScreen(before[0], before[1]);
        expect(sx).toBeCloseTo(400);
        expect(sy).toBeCloseTo(320);
    });

    it('pans horizontally so the content follows the cursor', () => {
        for (const [w, h] of [[800, 600], [600, 800], [1920, 400]]) {
            const cam = new Camera(w, h);
            cam.setZoom(1.7);
            cam.update();
            const before = cam.screenToWorld(w / 2, h / 2);
            cam.pan(50, 0);
            const [sx, sy] = cam.worldToScreen(before[0], before[1]);
            expect(sx).toBeCloseTo(w / 2 + 50);
            expect(sy).toBeCloseTo(h / 2);
        }
    });

    it('pans diagonally so the content follows the cursor', () => {
        const cam = new Camera(W, H);
        const before = cam.screenToWorld(100, 100);
        cam.pan(-30, 45);
        const [sx, sy] = cam.worldToScreen(before[0], before[1]);
        expect(sx).toBeCloseTo(70);
        expect(sy).toBeCloseTo(145);
    });

    it('zooms toward the cursor, keeping the point under it fixed', () => {
        const cam = new Camera(W, H);
        const before = cam.screenToWorld(100, 500);
        cam.zoom(1, 100, 500);
        const after = cam.screenToWorld(100, 500);
        expect(after[0]).toBeCloseTo(before[0]);
        expect(after[1]).toBeCloseTo(before[1]);
        expect(cam.getZoom()).toBeGreaterThan(1);
    });

    it('zooms out with negative delta', () => {
        const cam = new Camera(W, H);
        cam.zoom(-1);
        expect(cam.getZoom()).toBeLessThan(1);
    });

    it('zooms by an arbitrary factor, keeping the cursor point fixed', () => {
        const cam = new Camera(W, H);
        const before = cam.screenToWorld(600, 150);
        cam.zoomBy(3.5, 600, 150);
        expect(cam.getZoom()).toBeCloseTo(3.5);
        const after = cam.screenToWorld(600, 150);
        expect(after[0]).toBeCloseTo(before[0]);
        expect(after[1]).toBeCloseTo(before[1]);
    });

    it('ignores a non-finite cursor position', () => {
        const cam = new Camera(W, H);
        cam.zoomBy(2, NaN, 100);
        expect(cam.getZoom()).toBe(2);
        expect(cam.getState().x).toBe(0);
        expect(cam.getState().y).toBe(0);
    });

    it('ignores invalid zoom factors', () => {
        const cam = new Camera(W, H);
        for (const f of [0, -2, NaN, Infinity]) cam.zoomBy(f);
        expect(cam.getZoom()).toBe(1);
    });

    it('clamps zoom to the configured range', () => {
        const cam = new Camera(W, H, [0.5, 2]);
        cam.setZoom(100);
        expect(cam.getZoom()).toBe(2);
        cam.setZoom(0.001);
        expect(cam.getZoom()).toBe(0.5);
        for (let i = 0; i < 50; i++) cam.zoom(1);
        expect(cam.getZoom()).toBe(2);
    });

    it('fits bounds inside the viewport', () => {
        const cam = new Camera(W, H);
        cam.fitBounds(-3, -1, 5, 2, 0.1);
        const state = cam.getState();
        expect(state.x).toBeCloseTo(1);
        expect(state.y).toBeCloseTo(0.5);

        for (const [x, y] of [[-3, -1], [5, 2], [-3, 2], [5, -1]]) {
            const [sx, sy] = cam.worldToScreen(x, y);
            expect(sx).toBeGreaterThanOrEqual(0);
            expect(sx).toBeLessThanOrEqual(W);
            expect(sy).toBeGreaterThanOrEqual(0);
            expect(sy).toBeLessThanOrEqual(H);
        }
    });

    it('centers on a single point, keeping the zoom', () => {
        const cam = new Camera(W, H);
        cam.setPosition(4, 4);
        cam.setZoom(3);
        cam.fitBounds(1, 2, 1, 2);
        expect(cam.getState()).toEqual({ x: 1, y: 2, zoom: 3, rotation: 0 });
    });

    it('builds the projection matrix lazily', () => {
        const cam = new Camera(W, H);
        cam.setPosition(0.5, 0.25);
        cam.setZoom(2);
        expect(cam.dirty).toBe(true);
        cam.update();
        expect(cam.dirty).toBe(false);

        const m = cam.matrix;
        const sx = 2 / (W / H);
        expect(m[0]).toBeCloseTo(sx);
        expect(m[5]).toBeCloseTo(2);
        expect(m[8]).toBeCloseTo(-0.5 * sx);
        expect(m[9]).toBeCloseTo(-0.5);
        expect(m[10]).toBe(1);
    });

    it('keeps world coords consistent after resize', () => {
        const cam = new Camera(W, H);
        cam.resize(1000, 500);
        const [wx, wy] = cam.screenToWorld(500, 250);
        expect(wx).toBeCloseTo(0);
        expect(wy).toBeCloseTo(0);
        const [sx, sy] = cam.worldToScreen(0.2, 0.2);
        const back = cam.screenToWorld(sx, sy);
        expect(back[0]).toBeCloseTo(0.2);
        expect(back[1]).toBeCloseTo(0.2);
    });
});
