<div align="center">

<img align="center" width="500" src="logo.png"/>

<p align="center">
     <i>A WebGPU-accelerated, force-directed<br>graph visualization library for the browser</i> 
     <br><br>
     <img src="https://img.shields.io/github/license/drkameleon/GraphGPU?style=for-the-badge">
     <img src="https://img.shields.io/badge/language-TypeScript-3178C6.svg?style=for-the-badge" alt="TypeScript"/>
     <img src="https://img.shields.io/github/actions/workflow/status/drkameleon/GraphGPU/ci.yml?branch=main&style=for-the-badge" alt="Build status"/>
</p>
</div>

---

<!--ts-->

- [What does this do?](#what-does-this-package-do)
- [How do I use it?](#how-do-i-use-it)
- [Features](#features)
- [API Reference](#api-reference)
- [Demo](#demo)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

<!--te-->

---

### What does this do?

**GraphGPU** is a hardware-accelerated graph visualization library. Nodes, edges, labels, force-directed layout and interactive controls all run on the GPU via WebGPU.

### How do I use it?

```typescript
import { GraphGPU } from 'graphgpu';

const g = new GraphGPU({
    canvas: '#my-canvas',
    nodeSize: 6,
    edgeOpacity: 0.7,
    palette: 'vibrant',
});

await g.init();

const alice = g.put('person', 'Alice', { role: 'engineer' });
const bob   = g.put('person', 'Bob', { role: 'designer' });
const proj  = g.put('project', 'graphGPU');

g.link(alice, 'worksOn', proj);
g.link(bob, 'worksOn', proj);
g.link(alice, 'knows', bob);

g.startLayout({ repulsion: 1.2, gravity: 0.12 });

setTimeout(() => g.fitView(), 2000);
```

> [!TIP]
> Don't want to use a bundler? Grab `graphgpu.standalone.js` or `graphgpu.standalone.min.js` straight from the [latest release](https://github.com/drkameleon/GraphGPU/releases/latest) and drop it into a script tag:
> ```html
> <script src="graphgpu.standalone.min.js"></script>
> <script>
>     const g = new GraphGPU.GraphGPU({ canvas: '#my-canvas' });
> </script>
> ```

### Features

- WebGPU-accelerated rendering via instanced draw calls
- Force-directed layout (CPU and GPU compute engines)
- 8 built-in color palettes (`default`, `vibrant`, `pastel`, `earthy`, `inferno`, `playful`, `viridis`, `rainbow`)
- Pan, zoom, drag, select, multi-select, hover
- Gravity-pull mode: drag a node and the whole graph follows with spring physics
- Canvas2D label overlay with auto-sizing and truncation
- Tag-based coloring, palette-switchable at runtime
- SoA (Structure-of-Arrays) data layout for large graphs

### API Reference

#### Constructor

```typescript
new GraphGPU({
    canvas: string | HTMLCanvasElement,
    nodeSize?: number,          // default: 6
    edgeOpacity?: number,       // default: 0.12
    palette?: string | Palette, // default: 'default'
    background?: string,        // default: '#08080c'
    antialias?: boolean,        // default: true
    pixelRatio?: number,        // default: devicePixelRatio
    maxNodes?: number,          // default: 10000
    maxEdges?: number,          // default: 50000
})
```

#### Graph operations

| Method | Description |
|--------|-------------|
| `put(tag, label, properties?)` | Add a node with a label string |
| `put(tag, properties?)` | Add a node with properties dict |
| `link(source, tag, target)` | Add an edge (supports arrays for batch) |
| `unput(nodeId)` | Remove a node |
| `unlink(edgeId)` | Remove an edge |

#### Layout

| Method | Description |
|--------|-------------|
| `startLayout(opts?)` | Start CPU force-directed layout |
| `stopLayout()` | Stop the layout |
| `startGPULayout(opts?)` | Start GPU compute layout |
| `fitView(padding?)` | Fit camera to show all nodes |
| `resize()` | Handle container/window resize (updates canvas, MSAA, camera) |

#### Appearance

| Method | Description |
|--------|-------------|
| `setPalette(name)` | Switch color palette |
| `setBackground(rgba)` | Change background color |
| `getTagColors()` | Get current tag-to-color assignments |

#### Interaction

| Method | Description |
|--------|-------------|
| `setSelectionEnabled(bool)` | Toggle node selection |
| `setMultiSelectEnabled(bool)` | Toggle multi-select mode |
| `setGravityPull(bool, strength?)` | Toggle physics-based drag pull |
| `on(event, handler)` | Listen for events |
| `off(event, handler)` | Remove listener |

#### Events

`node:click`, `node:select`, `node:deselect`, `node:hover`, `node:hoverout`, `node:drag`, `node:dragstart`, `node:dragend`, `node:dblclick`, `canvas:click`, `canvas:pan`, `canvas:zoom`

### Demo

The `demo/` folder contains a cinema graph example (52 nodes, 80+ edges):

```bash
cd demo
npm install
npm run dev
```

### Architecture

```
src/
    index.ts              # Public API (GraphGPU class)
    core/
        Graph.ts          # SoA graph data structure
        Camera.ts         # 2D camera with pan/zoom
        Renderer.ts       # WebGPU render pipeline
    shaders/
        index.ts          # WGSL vertex/fragment/compute shaders
    interaction/
        Controls.ts       # Mouse/touch input handling
    layout/
        ForceLayout.ts    # CPU force-directed layout
        GPUForceLayout.ts # GPU compute force layout
    utils/
        color.ts          # Color parsing utilities
        palette.ts        # 9 built-in color palettes
    types/
        index.ts          # TypeScript type definitions
```

### Contributing

You are 100% welcome! Just make a PR. 🚀

<hr/>

### License

MIT License

Copyright (c) 2026 Yanis Zafirópulos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
