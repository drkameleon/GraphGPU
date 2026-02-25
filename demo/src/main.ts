// ============================================================
// graphGPU Demo — Vue 3 + TypeScript
// ============================================================

import { GraphGPU } from 'graphgpu';
import type { Node, NodeId } from 'graphgpu';

const { createApp, ref, reactive, computed, onMounted } = Vue;

// ── Types ──

interface LegendItem {
    tag: string;
    color: string;
}

interface EditModalState {
    open: boolean;
    nodeId: NodeId | null;
    properties: Record<string, unknown>;
}

interface DeleteModalState {
    open: boolean;
    nodeId: NodeId | null;
    label: string;
}

interface SettingsModalState {
    open: boolean;
    nodeSize: number;
    edgeOpacity: number;
    showLabels: boolean;
    showFps: boolean;
    gravitationalConstant: number;
    springLength: number;
    springConstant: number;
    centralGravity: number;
    damping: number;
    barnesHutTheta: number;
}

// ── Constants ──

const NODE_TYPE_TAGS = ['person', 'movie', 'country', 'book'] as const;
const PALETTE_NAMES = ['default', 'vibrant', 'pastel', 'earthy', 'inferno', 'playful', 'viridis', 'rainbow'] as const;

const LAYOUT_OPTS = {
    maxIterations: 1000,
} as const;

const DEFAULT_SETTINGS: Omit<SettingsModalState, 'open'> = {
    nodeSize: 8,
    edgeOpacity: 0.8,
    showLabels: true,
    showFps: false,
    gravitationalConstant: -0.25,
    springLength: 0.2,
    springConstant: 0.06,
    centralGravity: 0.012,
    damping: 0.18,
    barnesHutTheta: 0.3,
};

const LIGHT_BG: [number, number, number, number] = [0.96, 0.96, 0.965, 1];
const DARK_BG: [number, number, number, number] = [0.118, 0.122, 0.149, 1];

// ── Helpers ──

function rgbaToHex(bg: readonly number[]): string {
    return `rgb(${Math.round(bg[0] * 255)},${Math.round(bg[1] * 255)},${Math.round(bg[2] * 255)})`;
}

function getNodeLabel(node: Node): string {
    return (node.properties.name ?? node.properties.title ?? node.tag) as string;
}

// ── App ──

createApp({
    setup() {
        let g: GraphGPU | null = null;

        // State
        const darkMode = ref(false);
        const layoutRunning = ref(true);
        const animatedEnabled = ref(false);
        const activePalette = ref('vibrant');
        const nodeCount = ref(0);
        const edgeCount = ref(0);

        // Selection / hover
        const selectedNode = ref<Node | null>(null);
        const selectedNodeId = ref<NodeId | null>(null);
        const selectedNodeColor = ref('#888');
        const hoveredNode = ref<Node | null>(null);
        const hoveredNodeColor = ref('#888');

        // Tooltip
        const tooltipVisible = ref(false);
        const tooltipStyle = ref<Record<string, string>>({});
        const tooltipTag = ref('');
        const tooltipName = ref('');
        const tooltipProps = ref('');
        const tooltipColor = ref('#888');

        // Modals
        const editModal = reactive<EditModalState>({ open: false, nodeId: null, properties: {} });
        const deleteModal = reactive<DeleteModalState>({ open: false, nodeId: null, label: '' });
        const settingsModal = reactive<SettingsModalState>({
            open: false,
            ...DEFAULT_SETTINGS,
        });

        // Legend
        const legendItems = ref<LegendItem[]>([]);
        const tagColorCache: Record<string, string> = {};

        // FPS display
        const fpsDisplay = reactive({ fps: 0, nodes: 0, edges: 0 });
        let fpsInterval: number | null = null;

        // Computed
        const hasSelection = computed(() => selectedNode.value !== null);
        const paletteNames = PALETTE_NAMES;

        // ── Palette / Legend ──

        function refreshLegend(): void {
            if (!g) return;
            const items: LegendItem[] = [];
            for (const [tag, { bg }] of g.getTagColors()) {
                const hex = rgbaToHex(bg);
                tagColorCache[tag] = hex;
                if ((NODE_TYPE_TAGS as readonly string[]).includes(tag)) {
                    items.push({ tag, color: hex });
                }
            }
            legendItems.value = items;
        }

        function updateCounts(): void {
            if (!g) return;
            nodeCount.value = g.nodeCount;
            edgeCount.value = g.edgeCount;
        }

        function getPalettePreview(name: string): string {
            return (GraphGPU as any).palettes[name]?.colors?.[0] ?? '#666';
        }

        function switchPalette(name: string): void {
            activePalette.value = name;
            g?.setPalette(name);
            refreshLegend();
        }

        // ── Layout / View ──

        function toggleLayout(): void {
            if (!g) return;
            if (layoutRunning.value) {
                g.stopLayout();
                layoutRunning.value = false;
            } else {
                g.startLayout(LAYOUT_OPTS);
                layoutRunning.value = true;
            }
        }

        function fitView(): void {
            g?.fitView(0.15);
        }

        function resetGraph(): void {
            if (!g) return;
            if (layoutRunning.value) g.stopLayout();
            g.resetPositions();
            // Pre-stabilize offscreen
            for (let i = 0; i < 300; i++) {
                g.stepLayout(3);
            }
            g.fitView(0.15);
            g.startLayout(LAYOUT_OPTS);
            layoutRunning.value = true;
            setTimeout(() => g!.fitView(0.15), 800);
        }

        function toggleAnimated(): void {
            animatedEnabled.value = !animatedEnabled.value;
            g?.setAnimated(animatedEnabled.value);
        }

        // ── Theme ──

        function toggleDarkMode(): void {
            darkMode.value = !darkMode.value;
            document.body.classList.toggle('dark', darkMode.value);
            if (g) {
                g.setBackground(darkMode.value ? DARK_BG : LIGHT_BG);
            }
        }

        // ── Node editing ──

        function showEditModal(): void {
            if (!g || selectedNodeId.value === null) return;
            const node = g.getGraph().getNode(selectedNodeId.value);
            if (!node) return;
            editModal.nodeId = selectedNodeId.value;
            editModal.properties = { ...node.properties };
            editModal.open = true;
        }

        function saveEdit(): void {
            if (!g || editModal.nodeId === null) return;
            const graph = g.getGraph();
            for (const [key, val] of Object.entries(editModal.properties)) {
                graph.setNodeProperty(editModal.nodeId, key, val);
            }
            const updated = graph.getNode(editModal.nodeId);
            if (updated) selectedNode.value = { ...updated };
            editModal.open = false;
            updateCounts();
        }

        // ── Node deletion ──

        function deleteSelected(): void {
            if (!g || selectedNodeId.value === null) return;
            const node = g.getGraph().getNode(selectedNodeId.value);
            if (!node) return;
            deleteModal.nodeId = selectedNodeId.value;
            deleteModal.label = getNodeLabel(node);
            deleteModal.open = true;
        }

        function confirmDelete(): void {
            if (!g || deleteModal.nodeId === null) return;
            g.unput(deleteModal.nodeId);
            selectedNode.value = null;
            selectedNodeId.value = null;
            deleteModal.open = false;
            updateCounts();
        }

        // ── Settings ──

        function getPhysicsOpts() {
            return {
                gravitationalConstant: settingsModal.gravitationalConstant,
                springLength: settingsModal.springLength,
                springConstant: settingsModal.springConstant,
                centralGravity: settingsModal.centralGravity,
                damping: settingsModal.damping,
                barnesHutTheta: settingsModal.barnesHutTheta,
                maxIterations: 1000,
            };
        }

        function onSettingChange(key: string, event: Event): void {
            const val = parseFloat((event.target as HTMLInputElement).value);
            (settingsModal as any)[key] = val;

            if (!g) return;

            // Appearance — apply immediately
            if (key === 'nodeSize') {
                g.setNodeSize(val);
            } else if (key === 'edgeOpacity') {
                g.setEdgeOpacity(val);
            }

            // Physics — restart layout with new params
            if (['gravitationalConstant', 'springLength', 'springConstant',
                 'centralGravity', 'damping', 'barnesHutTheta'].includes(key)) {
                if (layoutRunning.value) {
                    g.stopLayout();
                    g.startLayout(getPhysicsOpts());
                }
            }
        }

        function onToggleSetting(key: string): void {
            if (key === 'showLabels') {
                settingsModal.showLabels = !settingsModal.showLabels;
                g?.setLabelsVisible(settingsModal.showLabels);
            } else if (key === 'showFps') {
                settingsModal.showFps = !settingsModal.showFps;
                if (settingsModal.showFps) {
                    fpsInterval = window.setInterval(() => {
                        if (!g) return;
                        fpsDisplay.fps = g.getFps();
                        fpsDisplay.nodes = g.nodeCount;
                        fpsDisplay.edges = g.edgeCount;
                    }, 250);
                } else if (fpsInterval !== null) {
                    clearInterval(fpsInterval);
                    fpsInterval = null;
                }
            }
        }

        function resetSettings(): void {
            Object.assign(settingsModal, DEFAULT_SETTINGS);
            if (!g) return;
            g.setNodeSize(DEFAULT_SETTINGS.nodeSize);
            g.setEdgeOpacity(DEFAULT_SETTINGS.edgeOpacity);
            g.setLabelsVisible(DEFAULT_SETTINGS.showLabels);
            if (fpsInterval !== null) {
                clearInterval(fpsInterval);
                fpsInterval = null;
            }
            if (layoutRunning.value) {
                g.stopLayout();
                g.startLayout(getPhysicsOpts());
            }
        }

        // ── Stress Test ──

        function stressTest(): void {
            if (!g) return;

            // Stop current layout
            if (layoutRunning.value) g.stopLayout();

            // Clear existing graph
            const graph = g.getGraph();
            for (const id of [...graph.activeNodeIds()]) {
                g.unput(id);
            }

            // Use smaller nodes for stress test
            g.setNodeSize(4);
            settingsModal.nodeSize = 4;

            // Generate random graph: 500 nodes, ~1500 edges
            const N = 500;
            const tags = ['alpha', 'beta', 'gamma', 'delta', 'epsilon'];
            const edgeTags = ['connects', 'links', 'references', 'depends'];
            const nodeIds: number[] = [];

            for (let i = 0; i < N; i++) {
                const tag = tags[Math.floor(Math.random() * tags.length)];
                const id = g.put(tag, { name: `${tag}-${i}` });
                nodeIds.push(id);
            }

            // Each node gets ~3 random edges on average
            const numEdges = Math.floor(N * 3);
            for (let i = 0; i < numEdges; i++) {
                const src = nodeIds[Math.floor(Math.random() * N)];
                let tgt = nodeIds[Math.floor(Math.random() * N)];
                if (src === tgt) tgt = nodeIds[(nodeIds.indexOf(src) + 1) % N];
                const tag = edgeTags[Math.floor(Math.random() * edgeTags.length)];
                g.link(src, tag, tgt);
            }

            // Scatter positions randomly before layout
            g.resetPositions();

            // Light pre-stabilization (don't block for too long)
            for (let i = 0; i < 50; i++) {
                g.stepLayout(3);
            }
            g.fitView(0.15);

            // Start layout and let it settle live
            g.startLayout(getPhysicsOpts());
            layoutRunning.value = true;
            setTimeout(() => g!.fitView(0.15), 500);
            setTimeout(() => g!.fitView(0.15), 2000);

            updateCounts();
            refreshLegend();

            // Auto-enable FPS display
            if (!settingsModal.showFps) {
                settingsModal.showFps = true;
                fpsInterval = window.setInterval(() => {
                    if (!g) return;
                    fpsDisplay.fps = g.getFps();
                    fpsDisplay.nodes = g.nodeCount;
                    fpsDisplay.edges = g.edgeCount;
                }, 250);
            }
        }

        // ── Init ──

        onMounted(async () => {
            const canvas = document.getElementById('graph-canvas') as HTMLCanvasElement;
            const fallback = document.getElementById('fallback')!;

            g = new GraphGPU({
                canvas,
                palette: 'vibrant',
                nodeSize: 8,
                edgeOpacity: 0.8,
                antialias: true,
                background: LIGHT_BG,
                interaction: {
                    pan: true, zoom: true, dragNodes: true,
                    hover: true, selection: true, multiSelect: false,
                },
            });

            const ok = await g.init();
            if (!ok) { fallback.classList.add('visible'); return; }

            // ── Tag colors ──
            const graph = g.getGraph();
            graph.tagColors.setColor('person', '#ff6b6b');
            graph.tagColors.setColor('movie', '#ffd93d');
            graph.tagColors.setColor('country', '#6bcb77');
            graph.tagColors.setColor('book', '#4d96ff');

            // ── Populate ──
            populateGraph(g);

            // ── Pre-stabilize layout (run offscreen before showing) ──
            // Run many iterations synchronously so the graph is settled
            // before the user sees anything.
            for (let i = 0; i < 300; i++) {
                g.stepLayout(3);
            }
            g.fitView(0.15);

            // Now start the live layout for final settling
            g.startLayout(LAYOUT_OPTS);
            // Fit again shortly after for any remaining drift
            setTimeout(() => g!.fitView(0.15), 800);
            updateCounts();
            refreshLegend();

            // ── Events ──
            g.on('node:hover', (e: { target: Node }) => {
                const node = e.target;
                if (!node) return;
                hoveredNode.value = { ...node };
                hoveredNodeColor.value = tagColorCache[node.tag] ?? '#888';
                tooltipTag.value = node.tag;
                tooltipName.value = getNodeLabel(node);
                tooltipColor.value = tagColorCache[node.tag] ?? '#888';
                tooltipProps.value = Object.entries(node.properties)
                    .filter(([k]) => k !== 'name' && k !== 'title')
                    .map(([k, v]) => `${k}: ${v}`)
                    .join('\n');
                tooltipVisible.value = true;
            });

            g.on('node:hoverout', () => {
                hoveredNode.value = null;
                tooltipVisible.value = false;
            });

            g.on('node:select', (e: { target: Node }) => {
                const node = e.target;
                if (!node) return;
                selectedNode.value = { ...node };
                selectedNodeId.value = node.id;
                selectedNodeColor.value = tagColorCache[node.tag] ?? '#888';
            });

            g.on('node:deselect', () => {
                selectedNode.value = null;
                selectedNodeId.value = null;
            });

            canvas.addEventListener('mousemove', (e: MouseEvent) => {
                tooltipStyle.value = {
                    left: `${e.clientX + 14}px`,
                    top: `${e.clientY - 8}px`,
                };
            });

            window.addEventListener('resize', () => g!.resize());

            // Console access
            (window as any).g = g;
        });

        return {
            darkMode, layoutRunning, animatedEnabled, activePalette,
            nodeCount, edgeCount, fpsDisplay,
            selectedNode, selectedNodeColor, hoveredNode, hoveredNodeColor,
            hasSelection,
            tooltipVisible, tooltipStyle, tooltipTag, tooltipName, tooltipProps, tooltipColor,
            legendItems, paletteNames,
            editModal, deleteModal, settingsModal,
            toggleLayout, fitView, resetGraph, toggleAnimated, toggleDarkMode,
            switchPalette, getPalettePreview,
            showEditModal, saveEdit, deleteSelected, confirmDelete,
            onSettingChange, onToggleSetting, resetSettings, stressTest,
        };
    },
}).mount('#app');

// ============================================================
// Graph data (cinema sample)
// ============================================================

function populateGraph(g: GraphGPU): void {
    // Countries
    const uk = g.put('country', { name: 'United Kingdom' });
    const au = g.put('country', { name: 'Australia' });
    const us = g.put('country', { name: 'United States' });
    const ca = g.put('country', { name: 'Canada' });
    const fr = g.put('country', { name: 'France' });
    const de = g.put('country', { name: 'Germany' });
    const se = g.put('country', { name: 'Sweden' });
    const pl = g.put('country', { name: 'Poland' });

    // People
    const nolan = g.put('person', { name: 'Christopher Nolan', birthday: 1970 });
    const pearce = g.put('person', { name: 'Guy Pearce', birthday: 1967 });
    const hanson = g.put('person', { name: 'Curtis Hanson', birthday: 1945 });
    const spacey = g.put('person', { name: 'Kevin Spacey', birthday: 1959 });
    const dicaprio = g.put('person', { name: 'Leonardo DiCaprio', birthday: 1974 });
    const hardy = g.put('person', { name: 'Tom Hardy', birthday: 1977 });
    const cotillard = g.put('person', { name: 'Marion Cotillard', birthday: 1975 });
    const moss = g.put('person', { name: 'Carrie-Ann Moss', birthday: 1967 });
    const kidman = g.put('person', { name: 'Nicole Kidman', birthday: 1967 });
    const cruise = g.put('person', { name: 'Tom Cruise', birthday: 1962 });
    const kubrick = g.put('person', { name: 'Stanley Kubrick', birthday: 1928, died: 1999 });
    const burton = g.put('person', { name: 'Tim Burton', birthday: 1958 });
    const depp = g.put('person', { name: 'Johnny Depp', birthday: 1965 });
    const hallstrom = g.put('person', { name: 'Lasse Hallström', birthday: 1946 });
    const scorsese = g.put('person', { name: 'Martin Scorsese', birthday: 1942 });
    const sydow = g.put('person', { name: 'Max von Sydow', birthday: 1929, died: 2020 });
    const binoche = g.put('person', { name: 'Juliette Binoche', birthday: 1964 });
    const dench = g.put('person', { name: 'Judi Dench', birthday: 1934 });
    const eastwood = g.put('person', { name: 'Clint Eastwood', birthday: 1930 });
    const polanski = g.put('person', { name: 'Roman Polanski', birthday: 1933 });
    const olin = g.put('person', { name: 'Lena Olin', birthday: 1955 });
    const zimmer = g.put('person', { name: 'Hans Zimmer', birthday: 1957 });
    const pook = g.put('person', { name: 'Jocelyn Pook', birthday: 1960 });
    const lehane = g.put('person', { name: 'Dennis Lehane', birthday: 1965 });
    const penn = g.put('person', { name: 'Sean Penn', birthday: 1960 });
    const malick = g.put('person', { name: 'Terrence Malick', birthday: 1943 });
    const brody = g.put('person', { name: 'Adrien Brody', birthday: 1973 });
    const wach1 = g.put('person', { name: 'Lana Wachowski', birthday: 1965 });
    const wach2 = g.put('person', { name: 'Lilly Wachowski', birthday: 1967 });

    // Movies
    const memento = g.put('movie', { title: 'Memento', year: 2000 });
    const inception = g.put('movie', { title: 'Inception', year: 2010 });
    const laconfidential = g.put('movie', { title: 'L.A. Confidential', year: 1997 });
    const matrix = g.put('movie', { title: 'The Matrix', year: 1999 });
    const eyes = g.put('movie', { title: 'Eyes Wide Shut', year: 1999 });
    const bigfish = g.put('movie', { title: 'Big Fish', year: 2003 });
    const sleepyhollow = g.put('movie', { title: 'Sleepy Hollow', year: 1999 });
    const chocolat = g.put('movie', { title: 'Chocolat', year: 2000 });
    const jedgar = g.put('movie', { title: 'J. Edgar', year: 2011 });
    const ninthgate = g.put('movie', { title: 'The Ninth Gate', year: 1999 });
    const shutter = g.put('movie', { title: 'Shutter Island', year: 2010 });
    const mystic = g.put('movie', { title: 'Mystic River', year: 2003 });
    const redline = g.put('movie', { title: 'Thin Red Line', year: 1998 });
    const pianist = g.put('movie', { title: 'The Pianist', year: 2002 });

    // Books
    const mysticB = g.put('book', { title: 'Mystic River', year: 2001, language: 'en' });

    // ── Relationships ──
    g.link([nolan, hardy, dench, pook], 'isFrom', uk);
    g.link([pearce, kidman], 'isFrom', au);
    g.link([malick, brody, hanson, spacey, dicaprio, wach1, wach2, cruise, kubrick, burton, depp, eastwood, scorsese, lehane, penn], 'isFrom', us);
    g.link(moss, 'isFrom', ca);
    g.link([cotillard, binoche], 'isFrom', fr);
    g.link(polanski, 'isFrom', [fr, pl]);
    g.link([hallstrom, olin, sydow], 'isFrom', se);
    g.link(zimmer, 'isFrom', de);

    g.link(nolan, 'directed', [memento, inception]);
    g.link(hanson, 'directed', laconfidential);
    g.link([wach1, wach2], 'directed', matrix);
    g.link(kubrick, 'directed', eyes);
    g.link(burton, 'directed', [bigfish, sleepyhollow]);
    g.link(hallstrom, 'directed', chocolat);
    g.link(eastwood, 'directed', [jedgar, mystic]);
    g.link(polanski, 'directed', [pianist, ninthgate]);
    g.link(scorsese, 'directed', shutter);
    g.link(malick, 'directed', [pianist, redline]);

    g.link(pearce, 'actedIn', [memento, laconfidential]);
    g.link(spacey, 'actedIn', laconfidential);
    g.link([dicaprio, hardy, cotillard], 'actedIn', inception);
    g.link([dicaprio, sydow], 'actedIn', shutter);
    g.link(cotillard, 'actedIn', bigfish);
    g.link(moss, 'actedIn', [memento, matrix, chocolat]);
    g.link([cruise, kidman], 'actedIn', eyes);
    g.link(depp, 'actedIn', [chocolat, sleepyhollow]);
    g.link([binoche, dench, olin], 'actedIn', chocolat);
    g.link([dicaprio, dench], 'actedIn', jedgar);
    g.link([depp, olin], 'actedIn', ninthgate);
    g.link(penn, 'actedIn', [mystic, redline]);
    g.link(brody, 'actedIn', [redline, pianist]);

    g.link(zimmer, 'composed', inception);
    g.link(pook, 'composed', eyes);
    g.link(nolan, 'written', inception);
    g.link(lehane, 'written', mysticB);
    g.link(mystic, 'basedOn', mysticB);
    g.link(wach1, 'sibling', wach2);
    g.link(cruise, 'married', kidman);
}