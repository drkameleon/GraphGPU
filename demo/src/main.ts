// ============================================================
// graphGPU Demo - Cinema Graph (Sample 11)
// ============================================================
// Recreates the iconic graphGPU sample: directors, actors,
// movies, countries, and books - all connected.
// ============================================================

import { GraphGPU } from 'graphgpu';
import type { Node, NodeId } from 'graphgpu';

// -----------------------------------------------------------
// Color mappings for legend
// -----------------------------------------------------------
const TAG_COLORS: Record<string, string> = {
    person: '#ff6b6b',
    movie: '#ffd93d',
    country: '#6bcb77',
    book: '#4d96ff',
};

// -----------------------------------------------------------
// Initialize graphGPU
// -----------------------------------------------------------
async function main() {
    const canvas = document.getElementById('graph-canvas') as HTMLCanvasElement;
    const fallback = document.getElementById('fallback')!;

    const g = new GraphGPU({
        canvas,
        palette: 'vibrant',
        nodeSize: 5,
        edgeOpacity: 0.7,
        antialias: true,
        background: [0.03, 0.03, 0.05, 1],
        interaction: {
            pan: true,
            zoom: true,
            dragNodes: true,
            hover: true,
            selection: true,
            multiSelect: true,
        },
    });

    const ok = await g.init();
    if (!ok) {
        fallback.classList.add('visible');
        return;
    }

    // -----------------------------------------------------------
    // Populate the graph (mirrors graphGPU's sample11.art)
    // -----------------------------------------------------------

    // Assign colors by tag
    const graph = g.getGraph();
    graph.tagColors.setColor('person', '#ff6b6b');
    graph.tagColors.setColor('movie', '#ffd93d');
    graph.tagColors.setColor('country', '#6bcb77');
    graph.tagColors.setColor('book', '#4d96ff');

    // --- Countries ---
    const uk = g.put('country', { name: 'United Kingdom' });
    const au = g.put('country', { name: 'Australia' });
    const us = g.put('country', { name: 'United States' });
    const ca = g.put('country', { name: 'Canada' });
    const fr = g.put('country', { name: 'France' });
    const de = g.put('country', { name: 'Germany' });
    const se = g.put('country', { name: 'Sweden' });
    const es = g.put('country', { name: 'Spain' });
    const pl = g.put('country', { name: 'Poland' });

    // --- People ---
    const nolan    = g.put('person', { name: 'Christopher Nolan', birthday: 1970 });
    const pearce   = g.put('person', { name: 'Guy Pearce', birthday: 1967 });
    const hanson   = g.put('person', { name: 'Curtis Hanson', birthday: 1945 });
    const spacey   = g.put('person', { name: 'Kevin Spacey', birthday: 1959 });
    const dicaprio = g.put('person', { name: 'Leonardo DiCaprio', birthday: 1974 });
    const hardy    = g.put('person', { name: 'Tom Hardy', birthday: 1977 });
    const cotillard = g.put('person', { name: 'Marion Cotillard', birthday: 1975 });
    const moss     = g.put('person', { name: 'Carrie-Ann Moss', birthday: 1967 });
    const kidman   = g.put('person', { name: 'Nicole Kidman', birthday: 1967 });
    const cruise   = g.put('person', { name: 'Tom Cruise', birthday: 1962 });
    const kubrick  = g.put('person', { name: 'Stanley Kubrick', birthday: 1928, died: 1999 });
    const burton   = g.put('person', { name: 'Tim Burton', birthday: 1958 });
    const depp     = g.put('person', { name: 'Johnny Depp', birthday: 1965 });
    const hallstrom = g.put('person', { name: 'Lasse Hallström', birthday: 1946 });
    const scorsese = g.put('person', { name: 'Martin Scorsese', birthday: 1942 });
    const sydow    = g.put('person', { name: 'Max von Sydow', birthday: 1929, died: 2020 });
    const binoche  = g.put('person', { name: 'Juliette Binoche', birthday: 1964 });
    const dench    = g.put('person', { name: 'Judi Dench', birthday: 1934 });
    const eastwood = g.put('person', { name: 'Clint Eastwood', birthday: 1930 });
    const polanski = g.put('person', { name: 'Roman Polanski', birthday: 1933 });
    const olin     = g.put('person', { name: 'Lena Olin', birthday: 1955 });
    const zimmer   = g.put('person', { name: 'Hans Zimmer', birthday: 1957 });
    const pook     = g.put('person', { name: 'Jocelyn Pook', birthday: 1960 });
    const lehane   = g.put('person', { name: 'Dennis Lehane', birthday: 1965 });
    const penn     = g.put('person', { name: 'Sean Penn', birthday: 1960 });
    const malick   = g.put('person', { name: 'Terrence Malick', birthday: 1943 });
    const brody    = g.put('person', { name: 'Adrien Brody', birthday: 1973 });
    const wach1    = g.put('person', { name: 'Lana Wachowski', birthday: 1965 });
    const wach2    = g.put('person', { name: 'Lilly Wachowski', birthday: 1967 });

    // --- Movies ---
    const memento        = g.put('movie', { title: 'Memento', year: 2000 });
    const inception      = g.put('movie', { title: 'Inception', year: 2010 });
    const laconfidential = g.put('movie', { title: 'L.A. Confidential', year: 1997 });
    const matrix         = g.put('movie', { title: 'The Matrix', year: 1999 });
    const eyes           = g.put('movie', { title: 'Eyes Wide Shut', year: 1999 });
    const bigfish        = g.put('movie', { title: 'Big Fish', year: 2003 });
    const sleepyhollow   = g.put('movie', { title: 'Sleepy Hollow', year: 1999 });
    const chocolat       = g.put('movie', { title: 'Chocolat', year: 2000 });
    const jedgar         = g.put('movie', { title: 'J. Edgar', year: 2011 });
    const ninthgate      = g.put('movie', { title: 'The Ninth Gate', year: 1999 });
    const shutter        = g.put('movie', { title: 'Shutter Island', year: 2010 });
    const mystic         = g.put('movie', { title: 'Mystic River', year: 2003 });
    const redline        = g.put('movie', { title: 'Thin Red Line', year: 1998 });
    const pianist        = g.put('movie', { title: 'The Pianist', year: 2002 });

    // --- Books ---
    const mysticB = g.put('book', { title: 'Mystic River', year: 2001, language: 'en' });

    // -----------------------------------------------------------
    // Relationships (mirrors graphGPU's ~> operator)
    // -----------------------------------------------------------

    // isFrom
    g.link([nolan, hardy, dench, pook], 'isFrom', uk);
    g.link([pearce, kidman], 'isFrom', au);
    g.link([malick, brody, hanson, spacey, dicaprio, wach1, wach2, cruise, kubrick, burton, depp, eastwood, scorsese, lehane, penn], 'isFrom', us);
    g.link(moss, 'isFrom', ca);
    g.link([cotillard, binoche], 'isFrom', fr);
    g.link(polanski, 'isFrom', [fr, pl]);
    g.link([hallstrom, olin, sydow], 'isFrom', se);
    g.link(zimmer, 'isFrom', de);

    // directed
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

    // actedIn
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

    // composed
    g.link(zimmer, 'composed', inception);
    g.link(pook, 'composed', eyes);

    // written
    g.link(nolan, 'written', inception);
    g.link(lehane, 'written', mysticB);

    // basedOn
    g.link(mystic, 'basedOn', mysticB);

    // sibling / married
    g.link(wach1, 'sibling', wach2);
    g.link(cruise, 'married', kidman);

    // -----------------------------------------------------------
    // Layout
    // -----------------------------------------------------------

    g.startLayout({
        repulsion: 1.2,
        attraction: 0.015,
        gravity: 0.12,
        damping: 0.9,
        maxIterations: 400,
    });

    // Auto-fit after layout settles
    // Single fitView after layout has had a moment to settle
    setTimeout(() => g.fitView(0.15), 1500);

    // -----------------------------------------------------------
    // UI: Stats
    // -----------------------------------------------------------
    const statsEl = document.getElementById('stats')!;

    function updateStats() {
        statsEl.innerHTML = `
            <span class="value">${g.nodeCount}</span> nodes ·
            <span class="value">${g.edgeCount}</span> edges<br>
            <span class="badge">WebGPU</span>
        `;
    }
    updateStats();

    // -----------------------------------------------------------
    // UI: Legend (refreshable)
    // -----------------------------------------------------------
    const legendEl = document.getElementById('legend')!;

    function refreshLegend() {
        legendEl.innerHTML = '';
        const tagColors = g.getTagColors();
        for (const [tag, { bg }] of tagColors) {
            const hex = `rgb(${Math.round(bg[0]*255)},${Math.round(bg[1]*255)},${Math.round(bg[2]*255)})`;
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `<div class="legend-dot" style="background:${hex}"></div>${tag}`;
            legendEl.appendChild(item);
        }
    }
    refreshLegend();

    // -----------------------------------------------------------
    // UI: Tooltip on hover
    // -----------------------------------------------------------
    const tooltip = document.getElementById('tooltip')!;
    const tooltipTag = document.getElementById('tooltip-tag')!;
    const tooltipName = document.getElementById('tooltip-name')!;
    const tooltipProps = document.getElementById('tooltip-props')!;

    g.on('node:hover', (e) => {
        const node = e.target as Node;
        if (!node) return;

        const color = TAG_COLORS[node.tag] ?? '#888';
        tooltipTag.textContent = node.tag;
        tooltipTag.style.color = color;
        tooltipName.textContent =
            (node.properties.name ?? node.properties.title ?? `#${node.id}`) as string;

        const propsText = Object.entries(node.properties)
            .filter(([k]) => k !== 'name' && k !== 'title')
            .map(([k, v]) => `${k}: ${v}`)
            .join('\n');
        tooltipProps.textContent = propsText;

        tooltip.classList.add('visible');
    });

    g.on('node:hoverout', () => {
        tooltip.classList.remove('visible');
    });

    // Track mouse for tooltip positioning
    canvas.addEventListener('mousemove', (e: MouseEvent) => {
        tooltip.style.left = `${e.clientX + 16}px`;
        tooltip.style.top = `${e.clientY - 10}px`;
    });

    // -----------------------------------------------------------
    // UI: Buttons
    // -----------------------------------------------------------
    const btnLayout = document.getElementById('btn-layout')!;
    const btnFit = document.getElementById('btn-fit')!;
    const btnReset = document.getElementById('btn-reset')!;
    const btnBg = document.getElementById('btn-bg')!;
    const btnSelect = document.getElementById('btn-select')!;
    const btnMulti = document.getElementById('btn-multi')!;
    const btnGravity = document.getElementById('btn-gravity')!;

    let layoutRunning = true;
    btnLayout.classList.add('active');

    btnLayout.addEventListener('click', () => {
        if (layoutRunning) {
            g.stopLayout();
            layoutRunning = false;
            btnLayout.classList.remove('active');
            btnLayout.textContent = '▶ layout';
        } else {
            g.startLayout({ repulsion: 1.2, attraction: 0.015, gravity: 0.12 });
            layoutRunning = true;
            btnLayout.classList.add('active');
            btnLayout.textContent = '⟳ layout';
        }
    });

    btnFit.addEventListener('click', () => g.fitView(0.15));
    btnReset.addEventListener('click', () => {
        g.resetView();
        g.fitView(0.15);
    });

    // Background toggle: dark ↔ light
    let darkBg = true;
    btnBg.addEventListener('click', () => {
        darkBg = !darkBg;
        if (darkBg) {
            g.setBackground([0.03, 0.03, 0.05, 1]);
            document.body.style.background = '#08080c';
        } else {
            g.setBackground([0.96, 0.96, 0.97, 1]);
            document.body.style.background = '#f5f5f7';
        }
        btnBg.classList.toggle('active', !darkBg);
    });

    // Selection toggle
    let selectEnabled = true;
    btnSelect.addEventListener('click', () => {
        selectEnabled = !selectEnabled;
        g.setSelectionEnabled(selectEnabled);
        btnSelect.classList.toggle('active', selectEnabled);
    });

    // Multi-select toggle
    let multiEnabled = true;
    btnMulti.addEventListener('click', () => {
        multiEnabled = !multiEnabled;
        g.setMultiSelectEnabled(multiEnabled);
        btnMulti.classList.toggle('active', multiEnabled);
    });

    // Gravity pull toggle
    let gravityEnabled = false;
    btnGravity.addEventListener('click', () => {
        gravityEnabled = !gravityEnabled;
        g.setGravityPull(gravityEnabled, 0.15);
        btnGravity.classList.toggle('active', gravityEnabled);
    });

    // -----------------------------------------------------------
    // UI: Palette swatches
    // -----------------------------------------------------------
    const paletteBar = document.getElementById('palette-bar')!;
    const paletteNames = ['default', 'vibrant', 'pastel', 'earthy', 'inferno', 'playful', 'viridis', 'rainbow'];
    let activePalette = 'vibrant';

    for (const name of paletteNames) {
        const palettes = GraphGPU.palettes;
        const colors = palettes[name]?.colors ?? [];
        const swatch = document.createElement('div');
        swatch.className = `palette-swatch ${name === activePalette ? 'active' : ''}`;
        swatch.title = name;

        // Use first color as swatch
        swatch.style.background = colors[0] ?? '#666';

        swatch.addEventListener('click', () => {
            g.setPalette(name);
            activePalette = name;
            paletteBar.querySelectorAll('.palette-swatch').forEach(s => s.classList.remove('active'));
            swatch.classList.add('active');
            refreshLegend();
        });

        paletteBar.appendChild(swatch);
    }

    window.addEventListener('resize', () => {
        g.getGraph(); // force re-render
    });

    // -----------------------------------------------------------
    // Resize handler
    // -----------------------------------------------------------
    console.log(`
    ┌─────────────────────────────────────────┐
    │  graphGPU - Cinema Graph Demo         │
    │  ${g.nodeCount} nodes · ${g.edgeCount} edges              │
    │  WebGPU renderer active                 │
    │                                         │
    │  Try:                                   │
    │    g.fetch('person', {name: 'Nolan'})   │
    │    g.related(0)                         │
    │    g.setPalette('inferno')              │
    └─────────────────────────────────────────┘
    `);

    // Expose globally for console experimentation
    (window as any).g = g;
    (window as any).graphgpu = g;
}

main().catch(console.error);
