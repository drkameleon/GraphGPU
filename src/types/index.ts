// ============================================================
// graphGPU - Type Definitions
// ============================================================
// Compile-time only. Zero runtime cost.
// All data lives in flat typed arrays (SoA layout).
// ============================================================

/** Opaque numeric handle into the node buffer */
export type NodeId = number;

/** Opaque numeric handle into the edge buffer */
export type EdgeId = number;

// -----------------------------------------------------------
// Node
// -----------------------------------------------------------

/** Logical view of a node (returned by accessors, never stored) */
export interface Node {
    readonly id: NodeId;
    readonly tag: string;
    readonly x: number;
    readonly y: number;
    readonly size: number;
    readonly r: number;
    readonly g: number;
    readonly b: number;
    readonly a: number;
    readonly properties: Record<string, unknown>;
}

/** Minimal input to create a node */
export interface NodeInput {
    tag: string;
    properties?: Record<string, unknown>;
    x?: number;
    y?: number;
    size?: number;
    color?: ColorInput;
}

// -----------------------------------------------------------
// Edge
// -----------------------------------------------------------

/** Logical view of an edge */
export interface Edge {
    readonly id: EdgeId;
    readonly tag: string;
    readonly source: NodeId;
    readonly target: NodeId;
    readonly weight: number;
    readonly properties: Record<string, unknown>;
}

/** Minimal input to create an edge */
export interface EdgeInput {
    tag: string;
    source: NodeId;
    target: NodeId;
    weight?: number;
    properties?: Record<string, unknown>;
}

// -----------------------------------------------------------
// Color
// -----------------------------------------------------------

export type ColorInput =
    | string                          // "#ff6050" or "rgb(255,96,80)"
    | [number, number, number]        // [r, g, b] 0-1
    | [number, number, number, number] // [r, g, b, a] 0-1
    | { r: number; g: number; b: number; a?: number };

// -----------------------------------------------------------
// Palette (inspired by graphGPU's .art palette system)
// -----------------------------------------------------------

export interface Palette {
    name: string;
    colors: string[];  // hex strings
}

// -----------------------------------------------------------
// Graph Configuration
// -----------------------------------------------------------

export interface GraphGPUOptions {
    /** Target canvas element or selector */
    canvas: HTMLCanvasElement | string;

    /** Pre-allocate for this many nodes (auto-grows) */
    maxNodes?: number;

    /** Pre-allocate for this many edges (auto-grows) */
    maxEdges?: number;

    /** Default node radius in pixels */
    nodeSize?: number;

    /** Edge base opacity 0-1 */
    edgeOpacity?: number;

    /** Edge base width in pixels */
    edgeWidth?: number;

    /** Active color palette */
    palette?: Palette | string;

    /** Enable antialiasing (MSAA) */
    antialias?: boolean;

    /** Background color */
    background?: ColorInput;

    /** Pixel ratio (defaults to devicePixelRatio) */
    pixelRatio?: number;

    /** Layout options */
    layout?: LayoutOptions;

    /** Interaction options */
    interaction?: InteractionOptions;

    /** Label rendering options */
    labels?: LabelOptions;

    /** Called when WebGPU is not available */
    onWebGPUUnavailable?: () => void;
}

// -----------------------------------------------------------
// Layout
// -----------------------------------------------------------

export type LayoutType = 'force' | 'force-gpu' | 'circular' | 'grid' | 'none';

export interface LayoutOptions {
    type: LayoutType;

    /** Force-directed: repulsion strength */
    repulsion?: number;

    /** Force-directed: attraction strength */
    attraction?: number;

    /** Force-directed: gravity toward center */
    gravity?: number;

    /** Force-directed: velocity damping 0-1 */
    damping?: number;

    /** Force-directed: Barnes-Hut theta */
    barnesHutTheta?: number;

    /** Max iterations before auto-stop */
    maxIterations?: number;

    /** Run layout in Web Worker */
    useWorker?: boolean;
}

// -----------------------------------------------------------
// Interaction
// -----------------------------------------------------------

export interface InteractionOptions {
    /** Enable pan */
    pan?: boolean;

    /** Enable zoom (scroll/pinch) */
    zoom?: boolean;

    /** Zoom range [min, max] */
    zoomRange?: [number, number];

    /** Enable node dragging */
    dragNodes?: boolean;

    /** Enable hover detection */
    hover?: boolean;

    /** Enable node/edge selection */
    selection?: boolean;

    /** Enable multi-select (shift+click) */
    multiSelect?: boolean;

    /** Enable lasso selection */
    lasso?: boolean;
}

// -----------------------------------------------------------
// Labels
// -----------------------------------------------------------

export interface LabelOptions {
    /** Show node labels */
    enabled?: boolean;

    /** CSS font string */
    font?: string;

    /** Label color (auto-contrast if omitted) */
    color?: ColorInput;

    /** Min zoom level to show labels */
    visibleAtZoom?: number;

    /** Which property to use as label */
    property?: string;

    /** Max characters before truncation */
    maxLength?: number;
}

// -----------------------------------------------------------
// Events
// -----------------------------------------------------------

export type GraphGPUEventType =
    | 'node:click'
    | 'node:dblclick'
    | 'node:hover'
    | 'node:hoverout'
    | 'node:dragstart'
    | 'node:drag'
    | 'node:dragend'
    | 'node:select'
    | 'node:deselect'
    | 'edge:click'
    | 'edge:hover'
    | 'edge:hoverout'
    | 'edge:select'
    | 'edge:deselect'
    | 'canvas:click'
    | 'canvas:pan'
    | 'canvas:zoom'
    | 'layout:start'
    | 'layout:tick'
    | 'layout:stop'
    | 'render:frame';

export interface GraphGPUEvent<T = unknown> {
    type: GraphGPUEventType;
    target: T;
    originalEvent?: Event;
    x?: number;
    y?: number;
}

export type GraphGPUEventHandler<T = unknown> = (event: GraphGPUEvent<T>) => void;

// -----------------------------------------------------------
// Render stats (exposed per-frame)
// -----------------------------------------------------------

export interface RenderStats {
    fps: number;
    frameTime: number;
    nodeCount: number;
    edgeCount: number;
    visibleNodes: number;
    visibleEdges: number;
    gpuTime?: number;
}

// -----------------------------------------------------------
// Camera state
// -----------------------------------------------------------

export interface CameraState {
    x: number;
    y: number;
    zoom: number;
    rotation: number;
}
