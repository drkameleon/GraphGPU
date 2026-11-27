// ============================================================
// graphGPU - Graph tests
// ============================================================

import { describe, it, expect } from 'vitest';
import { Graph } from '../src/core/Graph';
import { parseColor } from '../src/utils/color';

describe('Graph: nodes', () => {
    it('assigns sequential ids', () => {
        const g = new Graph();
        expect(g.addNode({ tag: 'a' })).toBe(0);
        expect(g.addNode({ tag: 'a' })).toBe(1);
        expect(g.addNode({ tag: 'b' })).toBe(2);
        expect(g.numNodes).toBe(3);
    });

    it('stores tag, properties, position, size and color', () => {
        const g = new Graph();
        const id = g.addNode({
            tag: 'person',
            properties: { name: 'Alice', age: 30 },
            x: 0.25, y: -0.5,
            size: 12,
            color: '#ff0000',
        });

        const node = g.getNode(id)!;
        expect(node.tag).toBe('person');
        expect(node.properties).toEqual({ name: 'Alice', age: 30 });
        expect(node.x).toBeCloseTo(0.25);
        expect(node.y).toBeCloseTo(-0.5);
        expect(node.size).toBe(12);
        expect([node.r, node.g, node.b, node.a]).toEqual([1, 0, 0, 1]);
    });

    it('uses the configured default node size', () => {
        const g = new Graph({ nodeSize: 9 });
        const id = g.addNode({ tag: 'a' });
        expect(g.sizes[id]).toBe(9);
    });

    it('places nodes without explicit position inside [-1, 1]', () => {
        const g = new Graph();
        for (let i = 0; i < 100; i++) g.addNode({ tag: 'a' });
        for (const v of g.positions.subarray(0, 200)) {
            expect(v).toBeGreaterThanOrEqual(-1);
            expect(v).toBeLessThanOrEqual(1);
        }
    });

    it('auto-colors nodes by tag', () => {
        const g = new Graph();
        const a1 = g.getNode(g.addNode({ tag: 'a' }))!;
        const a2 = g.getNode(g.addNode({ tag: 'a' }))!;
        const b = g.getNode(g.addNode({ tag: 'b' }))!;

        expect([a1.r, a1.g, a1.b]).toEqual([a2.r, a2.g, a2.b]);
        expect([a1.r, a1.g, a1.b]).not.toEqual([b.r, b.g, b.b]);
    });

    it('grows buffers past initial capacity, preserving data', () => {
        const g = new Graph({ maxNodes: 2 });
        for (let i = 0; i < 50; i++) {
            g.addNode({ tag: 'a', x: i / 100, y: -i / 100, size: i + 1 });
        }
        expect(g.numNodes).toBe(50);
        for (let i = 0; i < 50; i++) {
            const n = g.getNode(i)!;
            expect(n.x).toBeCloseTo(i / 100);
            expect(n.y).toBeCloseTo(-i / 100);
            expect(n.size).toBe(i + 1);
        }
    });

    it('updates position, color, size and properties', () => {
        const g = new Graph();
        const id = g.addNode({ tag: 'a', properties: { k: 1 } });

        g.dirtyNodes = false;
        g.dirtyEdges = false;
        g.setNodePosition(id, 0.1, 0.2);
        expect(g.dirtyNodes).toBe(true);
        expect(g.dirtyEdges).toBe(true);
        expect(g.positions[id * 2]).toBeCloseTo(0.1);
        expect(g.positions[id * 2 + 1]).toBeCloseTo(0.2);

        g.setNodeColor(id, parseColor('#00ff00'));
        expect(Array.from(g.colors.subarray(id * 4, id * 4 + 4))).toEqual([0, 1, 0, 1]);

        g.setNodeSize(id, 3);
        expect(g.sizes[id]).toBe(3);

        g.setNodeProperty(id, 'k', 2);
        expect(g.getNode(id)!.properties.k).toBe(2);
    });

    it('removes nodes and reuses their slots', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'a' });
        const b = g.addNode({ tag: 'a' });
        g.addNode({ tag: 'a' });

        g.removeNode(b);
        expect(g.isNodeActive(b)).toBe(false);
        expect(g.getNode(b)).toBeNull();
        expect(g.sizes[b]).toBe(0);
        expect(g.isNodeActive(a)).toBe(true);

        expect(g.addNode({ tag: 'a' })).toBe(b);
        expect(g.isNodeActive(b)).toBe(true);
    });

    it('ignores removal of invalid or already removed nodes', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'a' });
        g.addNode({ tag: 'a' });
        g.removeNode(a);
        expect(() => g.removeNode(a)).not.toThrow();
        expect(() => g.removeNode(-1)).not.toThrow();
        expect(() => g.removeNode(999)).not.toThrow();
    });

    it('returns null for out-of-range node ids', () => {
        const g = new Graph();
        expect(g.getNode(0)).toBeNull();
        expect(g.getNode(-1)).toBeNull();
        expect(g.isNodeActive(0)).toBe(false);
    });
});

describe('Graph: edges', () => {
    it('stores source, target, tag, weight and properties', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n' });
        const b = g.addNode({ tag: 'n' });
        const e = g.addEdge({ tag: 'knows', source: a, target: b, properties: { since: 2020 } });

        expect(g.numEdges).toBe(1);
        expect(g.getEdge(e)).toEqual({
            id: e,
            tag: 'knows',
            source: a,
            target: b,
            weight: 1,
            properties: { since: 2020 },
        });
    });

    it('honors explicit weight', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n' });
        const b = g.addNode({ tag: 'n' });
        const e = g.addEdge({ tag: 'w', source: a, target: b, weight: 2.5 });
        expect(g.getEdge(e)!.weight).toBe(2.5);
    });

    it('colors edges by tag', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n' });
        const b = g.addNode({ tag: 'n' });
        const e = g.addEdge({ tag: 'knows', source: a, target: b });
        const expected = g.tagColors.getColor('knows').bg;
        expect(g.edgeColors[e * 3]).toBeCloseTo(expected[0]);
        expect(g.edgeColors[e * 3 + 1]).toBeCloseTo(expected[1]);
        expect(g.edgeColors[e * 3 + 2]).toBeCloseTo(expected[2]);
    });

    it('grows edge buffers past initial capacity', () => {
        const g = new Graph({ maxEdges: 2 });
        const a = g.addNode({ tag: 'n' });
        const b = g.addNode({ tag: 'n' });
        for (let i = 0; i < 40; i++) {
            g.addEdge({ tag: 'e', source: a, target: b, weight: i });
        }
        expect(g.numEdges).toBe(40);
        expect(g.getEdge(39)!.weight).toBe(39);
    });

    it('removes edges and reuses their slots', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n' });
        const b = g.addNode({ tag: 'n' });
        const e0 = g.addEdge({ tag: 'e', source: a, target: b });
        g.addEdge({ tag: 'e', source: b, target: a });

        g.removeEdge(e0);
        expect(g.isEdgeActive(e0)).toBe(false);
        expect(g.getEdge(e0)).toBeNull();
        expect(g.addEdge({ tag: 'e', source: a, target: b })).toBe(e0);
    });

    it('removing a node removes all its edges', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n' });
        const b = g.addNode({ tag: 'n' });
        const c = g.addNode({ tag: 'n' });
        const ab = g.addEdge({ tag: 'e', source: a, target: b });
        const bc = g.addEdge({ tag: 'e', source: b, target: c });
        const ca = g.addEdge({ tag: 'e', source: c, target: a });

        g.removeNode(b);
        expect(g.isEdgeActive(ab)).toBe(false);
        expect(g.isEdgeActive(bc)).toBe(false);
        expect(g.isEdgeActive(ca)).toBe(true);
    });
});

describe('Graph: empty tags', () => {
    it('treats a node with an empty tag as a normal node', () => {
        const g = new Graph();
        const a = g.addNode({ tag: '', properties: { name: 'x' } });
        expect(g.isNodeActive(a)).toBe(true);
        expect(g.getNode(a)).not.toBeNull();
        expect(g.getNode(a)!.tag).toBe('');
        expect(g.fetchByTag('')).toHaveLength(1);
        expect(g.getAllTags()).toEqual(['']);
        expect(g.getRelatedNodes(a)).toEqual([]);
    });

    it('treats an edge with an empty tag as a normal edge', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n' });
        const b = g.addNode({ tag: 'n' });
        const e = g.addEdge({ tag: '', source: a, target: b });
        expect(g.isEdgeActive(e)).toBe(true);
        expect(g.getEdge(e)).not.toBeNull();
        expect(g.getNodeEdges(a)).toHaveLength(1);
        expect(g.getRelatedNodes(a).map(n => n.id)).toEqual([b]);
    });
});

describe('Graph: counts', () => {
    it('counts only live nodes and edges', () => {
        const g = new Graph();
        const ids = [0, 1, 2, 3].map(() => g.addNode({ tag: 'n' }));
        const e01 = g.addEdge({ tag: 'e', source: ids[0], target: ids[1] });
        g.addEdge({ tag: 'e', source: ids[2], target: ids[3] });
        g.addEdge({ tag: 'e', source: ids[1], target: ids[2] });

        g.removeNode(ids[1]); // middle node, takes 2 edges with it
        expect(g.numNodes).toBe(3);
        expect(g.numEdges).toBe(1);
        expect(g.nodeSlots).toBe(4);

        g.removeEdge(e01); // already gone with its node
        expect(g.numEdges).toBe(1);
    });

    it('does not double-count repeated removals', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n' });
        g.addNode({ tag: 'n' });
        g.removeNode(a);
        g.removeNode(a);
        g.removeNode(-1);
        expect(g.numNodes).toBe(1);
    });

    it('counts reused slots once', () => {
        const g = new Graph();
        g.addNode({ tag: 'n' });
        const b = g.addNode({ tag: 'n' });
        g.addNode({ tag: 'n' });
        g.removeNode(b);
        g.addNode({ tag: 'n' });
        expect(g.numNodes).toBe(3);
        expect(g.nodeSlots).toBe(3);
    });

    it('shrinks slots when the tail is removed', () => {
        const g = new Graph();
        g.addNode({ tag: 'n' });
        const b = g.addNode({ tag: 'n' });
        g.removeNode(b);
        expect(g.numNodes).toBe(1);
        expect(g.nodeSlots).toBe(1);
    });

    it('resets counts on clear', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n' });
        const b = g.addNode({ tag: 'n' });
        g.addEdge({ tag: 'e', source: a, target: b });
        g.clear();
        expect(g.numNodes).toBe(0);
        expect(g.numEdges).toBe(0);
        expect(g.nodeSlots).toBe(0);
        expect(g.edgeSlots).toBe(0);
    });
});

describe('Graph: queries', () => {
    function sample() {
        const g = new Graph();
        const john = g.addNode({ tag: 'person', properties: { name: 'John', surname: 'Doe' } });
        const mary = g.addNode({ tag: 'person', properties: { name: 'Mary', surname: 'Doe' } });
        const tom = g.addNode({ tag: 'person', properties: { name: 'Tom', surname: 'Smith' } });
        const acme = g.addNode({ tag: 'company', properties: { name: 'Acme' } });
        g.addEdge({ tag: 'marriedTo', source: john, target: mary });
        g.addEdge({ tag: 'worksAt', source: john, target: acme });
        g.addEdge({ tag: 'worksAt', source: tom, target: acme });
        return { g, john, mary, tom, acme };
    }

    it('fetches nodes by tag', () => {
        const { g } = sample();
        expect(g.fetchByTag('person').map(n => n.properties.name)).toEqual(['John', 'Mary', 'Tom']);
        expect(g.fetchByTag('company')).toHaveLength(1);
        expect(g.fetchByTag('missing')).toEqual([]);
    });

    it('fetches nodes by tag and property filter', () => {
        const { g } = sample();
        const does = g.fetchByTag('person', { surname: 'Doe' });
        expect(does.map(n => n.properties.name)).toEqual(['John', 'Mary']);
        expect(g.fetchByTag('person', { surname: 'Doe', name: 'Mary' })).toHaveLength(1);
    });

    it('returns the edges of a node', () => {
        const { g, john, acme } = sample();
        expect(g.getNodeEdges(john).map(e => e.tag).sort()).toEqual(['marriedTo', 'worksAt']);
        expect(g.getNodeEdges(acme)).toHaveLength(2);
    });

    it('returns related nodes in both directions', () => {
        const { g, john, mary, acme, tom } = sample();
        expect(g.getRelatedNodes(john).map(n => n.id).sort()).toEqual([mary, acme].sort());
        expect(g.getRelatedNodes(acme).map(n => n.id).sort()).toEqual([john, tom].sort());
    });

    it('lists unique tags', () => {
        const { g } = sample();
        expect(g.getAllTags().sort()).toEqual(['company', 'person']);
    });

    it('iterates only active ids', () => {
        const { g, mary } = sample();
        g.removeNode(mary);
        expect([...g.activeNodeIds()]).not.toContain(mary);
        expect([...g.activeNodeIds()]).toHaveLength(3);
        expect([...g.activeEdgeIds()]).toHaveLength(2);
    });
});

describe('Graph: lifecycle', () => {
    it('clears everything', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n' });
        const b = g.addNode({ tag: 'n' });
        g.addEdge({ tag: 'e', source: a, target: b });

        g.clear();
        expect(g.numNodes).toBe(0);
        expect(g.numEdges).toBe(0);
        expect(g.getNode(a)).toBeNull();
        expect(g.getEdge(0)).toBeNull();
        expect([...g.activeNodeIds()]).toEqual([]);
        expect(g.tagColors.getAll().size).toBe(0);
    });

    it('forgets old edges after clear', () => {
        const g = new Graph();
        const a = g.addNode({ tag: 'n' });
        const b = g.addNode({ tag: 'n' });
        g.addEdge({ tag: 'old', source: a, target: b });
        g.clear();

        // Same ids get reused: node 0 must not still "own" edge 0
        const c = g.addNode({ tag: 'n' });
        const d = g.addNode({ tag: 'n' });
        const e = g.addNode({ tag: 'n' });
        const de = g.addEdge({ tag: 'new', source: d, target: e });

        g.removeNode(c);
        expect(g.isEdgeActive(de)).toBe(true);
        expect(g.getEdge(de)!.tag).toBe('new');
        expect(g.getNodeEdges(d)).toHaveLength(1);
    });

    it('starts fresh ids after clear', () => {
        const g = new Graph();
        g.addNode({ tag: 'n' });
        g.addNode({ tag: 'n' });
        g.clear();
        expect(g.addNode({ tag: 'n' })).toBe(0);
    });

    it('re-randomizes positions inside [-1, 1]', () => {
        const g = new Graph();
        for (let i = 0; i < 50; i++) g.addNode({ tag: 'n', x: 5, y: 5 });
        g.resetPositions();
        for (const v of g.positions.subarray(0, 100)) {
            expect(Math.abs(v)).toBeLessThanOrEqual(1);
        }
        expect(g.dirtyNodes).toBe(true);
        expect(g.dirtyEdges).toBe(true);
    });
});
