// ============================================================
// graphGPU - Camera
// ============================================================
// 2D camera with pan, zoom, and coordinate transforms.
// Outputs a projection matrix for the vertex shader.
// ============================================================

import type { CameraState } from '../types';

export class Camera {
    private x: number = 0;
    private y: number = 0;
    private zoomLevel: number = 1;
    private rotation: number = 0;

    private canvasWidth: number;
    private canvasHeight: number;
    private aspectRatio: number;

    private zoomMin: number;
    private zoomMax: number;

    /** 3x3 projection matrix (column-major, padded to std140) */
    readonly matrix: Float32Array = new Float32Array(12); // 3 vec4s for std140

    dirty: boolean = true;

    constructor(
        width: number,
        height: number,
        zoomRange: [number, number] = [0.01, 100],
    ) {
        this.canvasWidth = width;
        this.canvasHeight = height;
        this.aspectRatio = width / height;
        this.zoomMin = zoomRange[0];
        this.zoomMax = zoomRange[1];
        this.update();
    }

    /** Resize the viewport */
    resize(width: number, height: number): void {
        this.canvasWidth = width;
        this.canvasHeight = height;
        this.aspectRatio = width / height;
        this.dirty = true;
    }

    /** Pan by delta in screen pixels */
    pan(dx: number, dy: number): void {
        this.x -= dx / (this.zoomLevel * this.canvasWidth * 0.5);
        this.y += dy / (this.zoomLevel * this.canvasHeight * 0.5);
        this.dirty = true;
    }

    /** Zoom by delta at a screen point */
    zoom(delta: number, screenX?: number, screenY?: number): void {
        const oldZoom = this.zoomLevel;
        const factor = delta > 0 ? 1.1 : 1 / 1.1;
        this.zoomLevel = Math.max(
            this.zoomMin,
            Math.min(this.zoomMax, this.zoomLevel * factor),
        );

        // Zoom toward cursor position
        if (screenX !== undefined && screenY !== undefined) {
            const worldBefore = this.screenToWorld(screenX, screenY, oldZoom);
            const worldAfter = this.screenToWorld(screenX, screenY, this.zoomLevel);
            this.x += worldBefore[0] - worldAfter[0];
            this.y += worldBefore[1] - worldAfter[1];
        }

        this.dirty = true;
    }

    /** Set zoom level directly */
    setZoom(level: number): void {
        this.zoomLevel = Math.max(this.zoomMin, Math.min(this.zoomMax, level));
        this.dirty = true;
    }

    /** Set position directly */
    setPosition(x: number, y: number): void {
        this.x = x;
        this.y = y;
        this.dirty = true;
    }

    /** Reset to default view */
    reset(): void {
        this.x = 0;
        this.y = 0;
        this.zoomLevel = 1;
        this.rotation = 0;
        this.dirty = true;
    }

    /** Fit view to show all nodes within given bounds */
    fitBounds(
        minX: number, minY: number,
        maxX: number, maxY: number,
        padding: number = 0.1,
    ): void {
        const w = maxX - minX;
        const h = maxY - minY;
        if (w === 0 && h === 0) {
            this.reset();
            return;
        }

        this.x = (minX + maxX) / 2;
        this.y = (minY + maxY) / 2;

        const scaleX = (2 * (1 - padding)) / (w * this.aspectRatio);
        const scaleY = (2 * (1 - padding)) / h;
        this.zoomLevel = Math.min(scaleX, scaleY);
        this.zoomLevel = Math.max(this.zoomMin, Math.min(this.zoomMax, this.zoomLevel));

        this.dirty = true;
    }

    /** Convert screen coords to world coords */
    screenToWorld(
        screenX: number,
        screenY: number,
        overrideZoom?: number,
    ): [number, number] {
        const z = overrideZoom ?? this.zoomLevel;
        const ndcX = (screenX / this.canvasWidth) * 2 - 1;
        const ndcY = -((screenY / this.canvasHeight) * 2 - 1);
        const worldX = ndcX / (z / this.aspectRatio) + this.x;
        const worldY = ndcY / z + this.y;
        return [worldX, worldY];
    }

    /** Convert world coords to screen coords */
    worldToScreen(worldX: number, worldY: number): [number, number] {
        const ndcX = (worldX - this.x) * (this.zoomLevel / this.aspectRatio);
        const ndcY = (worldY - this.y) * this.zoomLevel;
        const screenX = (ndcX + 1) / 2 * this.canvasWidth;
        const screenY = (-ndcY + 1) / 2 * this.canvasHeight;
        return [screenX, screenY];
    }

    /** Recompute the projection matrix (call when dirty) */
    update(): void {
        if (!this.dirty) return;

        const sx = this.zoomLevel / this.aspectRatio;
        const sy = this.zoomLevel;
        const tx = -this.x * sx;
        const ty = -this.y * sy;

        // Column-major 3x3 padded to std140 (3 × vec4)
        // col0
        this.matrix[0] = sx;
        this.matrix[1] = 0;
        this.matrix[2] = 0;
        this.matrix[3] = 0; // pad
        // col1
        this.matrix[4] = 0;
        this.matrix[5] = sy;
        this.matrix[6] = 0;
        this.matrix[7] = 0; // pad
        // col2
        this.matrix[8] = tx;
        this.matrix[9] = ty;
        this.matrix[10] = 1;
        this.matrix[11] = 0; // pad

        this.dirty = false;
    }

    /** Get current state */
    getState(): CameraState {
        return {
            x: this.x,
            y: this.y,
            zoom: this.zoomLevel,
            rotation: this.rotation,
        };
    }

    /** Get current zoom level */
    getZoom(): number {
        return this.zoomLevel;
    }
}
