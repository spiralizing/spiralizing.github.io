@def title = "The Spiral Array"
@def noindex = true
@def sitemap_exclude = true

<!--
Unlisted page: not linked from _layout/sidebar.html, excluded from sitemap.xml
and feed.xml, and served with a robots noindex/nofollow meta tag. Reachable
only at /spiral_array/. This is obscurity, not access control.

The 3D model below is ported from ../MusicScoresTrajectories/index.html
(landing page): same traces, labels, colors, camera and interaction. Only the
canvas sizing differs (fluid instead of a fixed 1000x1000 px square).
-->

## Interactive 3D Spiral Array

Explore the three-dimensional structure of musical harmony using the spiral array representation. Drag to orbit, scroll to zoom, and click legend entries to toggle pitches, chords and keys.

~~~
<script src="https://cdn.plot.ly/plotly-2.33.0.min.js"></script>

<div id="spiral-plot"></div>

<style>
    #spiral-plot {
        width: 100%;
        max-width: 1000px;
        aspect-ratio: 1 / 1;
        margin: 0 auto;
        display: block;
        background: #1a1a1a;
        border: 1px solid #6D6B66;
        overflow: visible;
    }
</style>

<script>
    // Musical constants
    const r = 1;
    const h = Math.sqrt(2 / 15);
    const weights = [0.536, 0.274, 0.19];
    const alpha = 0.75;
    const beta = 0.75;
    const noteNames = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F'];

    // Initial visibility of each layer
    let spiralVisibility = {
        pitches: true,
        chords: true,
        keys: true,
        spirals: true
    };

    // Calculate positions
    function getPitchPosition(k) {
        return {
            x: r * Math.sin(k * Math.PI / 2),
            y: r * Math.cos(k * Math.PI / 2),
            z: k * h
        };
    }

    function getChordPosition(k, isMinor = false) {
        const p1 = getPitchPosition(k);
        const p2 = getPitchPosition(k + 1);
        const p3 = getPitchPosition(isMinor ? k - 3 : k + 4);

        return {
            x: weights[0] * p1.x + weights[1] * p2.x + weights[2] * p3.x,
            y: weights[0] * p1.y + weights[1] * p2.y + weights[2] * p3.y,
            z: weights[0] * p1.z + weights[1] * p2.z + weights[2] * p3.z
        };
    }

    function getKeyPosition(k, isMinor = false) {
        if (isMinor) {
            const tonic = getChordPosition(k, true);
            const dominant_maj = getChordPosition(k + 1, false);
            const dominant_min = getChordPosition(k + 1, true);
            const subdominant_maj = getChordPosition(k - 1, false);
            const subdominant_min = getChordPosition(k - 1, true);

            return {
                x: weights[0] * tonic.x + weights[1] * (alpha * dominant_maj.x + (1 - alpha) * dominant_min.x) + weights[2] * (beta * subdominant_min.x + (1 - beta) * subdominant_maj.x),
                y: weights[0] * tonic.y + weights[1] * (alpha * dominant_maj.y + (1 - alpha) * dominant_min.y) + weights[2] * (beta * subdominant_min.y + (1 - beta) * subdominant_maj.y),
                z: weights[0] * tonic.z + weights[1] * (alpha * dominant_maj.z + (1 - alpha) * dominant_min.z) + weights[2] * (beta * subdominant_min.z + (1 - beta) * subdominant_maj.z)
            };
        } else {
            const tonic = getChordPosition(k, false);
            const dominant = getChordPosition(k + 1, false);
            const subdominant = getChordPosition(k - 1, false);

            return {
                x: weights[0] * tonic.x + weights[1] * dominant.x + weights[2] * subdominant.x,
                y: weights[0] * tonic.y + weights[1] * dominant.y + weights[2] * subdominant.y,
                z: weights[0] * tonic.z + weights[1] * dominant.z + weights[2] * subdominant.z
            };
        }
    }

    // Generate plot data
    function generateSpiralData() {
        const traces = [];

        // Pitches
        const pitchData = { x: [], y: [], z: [], text: [] };
        for (let k = 0; k < 12; k++) {
            const pos = getPitchPosition(k);
            pitchData.x.push(pos.x);
            pitchData.y.push(pos.y);
            pitchData.z.push(pos.z);
            pitchData.text.push(noteNames[k]);
        }
        traces.push({
            x: pitchData.x, y: pitchData.y, z: pitchData.z,
            text: pitchData.text,
            mode: 'markers+text',
            type: 'scatter3d',
            name: 'Pitches',
            marker: { color: '#006400', size: 8 }, // Dark green
            textposition: 'top center',
            textfont: { size: 12 },
            visible: spiralVisibility.pitches
        });

        // Major Chords
        const majorChordData = { x: [], y: [], z: [], text: [] };
        for (let k = 0; k < 12; k++) {
            const pos = getChordPosition(k, false);
            majorChordData.x.push(pos.x);
            majorChordData.y.push(pos.y);
            majorChordData.z.push(pos.z);
            majorChordData.text.push(noteNames[k] + ' Maj');
        }
        traces.push({
            x: majorChordData.x, y: majorChordData.y, z: majorChordData.z,
            text: majorChordData.text,
            mode: 'markers+text',
            type: 'scatter3d',
            name: 'Major Chords',
            marker: { color: '#ff0000', size: 10, symbol: 'diamond' }, // Red
            textposition: 'top center',
            textfont: { size: 10 },
            visible: spiralVisibility.chords
        });

        // Minor Chords
        const minorChordData = { x: [], y: [], z: [], text: [] };
        for (let k = 0; k < 12; k++) {
            const pos = getChordPosition(k, true);
            minorChordData.x.push(pos.x);
            minorChordData.y.push(pos.y);
            minorChordData.z.push(pos.z);
            minorChordData.text.push(noteNames[k] + ' min');
        }
        traces.push({
            x: minorChordData.x, y: minorChordData.y, z: minorChordData.z,
            text: minorChordData.text,
            mode: 'markers+text',
            type: 'scatter3d',
            name: 'Minor Chords',
            marker: { color: '#000080', size: 10, symbol: 'diamond' }, // Dark blue
            textposition: 'top center',
            textfont: { size: 10 },
            visible: spiralVisibility.chords
        });

        // Major Keys
        const majorKeyData = { x: [], y: [], z: [], text: [] };
        for (let k = 0; k < 12; k++) {
            const pos = getKeyPosition(k, false);
            majorKeyData.x.push(pos.x);
            majorKeyData.y.push(pos.y);
            majorKeyData.z.push(pos.z);
            majorKeyData.text.push(noteNames[k] + ' Major');
        }
        traces.push({
            x: majorKeyData.x, y: majorKeyData.y, z: majorKeyData.z,
            text: majorKeyData.text,
            mode: 'markers+text',
            type: 'scatter3d',
            name: 'Major Keys',
            marker: { color: 'gold', size: 12, symbol: 'square' },
            textposition: 'top center',
            textfont: { size: 10 },
            visible: spiralVisibility.keys
        });

        // Minor Keys
        const minorKeyData = { x: [], y: [], z: [], text: [] };
        for (let k = 0; k < 12; k++) {
            const pos = getKeyPosition(k, true);
            minorKeyData.x.push(pos.x);
            minorKeyData.y.push(pos.y);
            minorKeyData.z.push(pos.z);
            minorKeyData.text.push(noteNames[k] + ' minor');
        }
        traces.push({
            x: minorKeyData.x, y: minorKeyData.y, z: minorKeyData.z,
            text: minorKeyData.text,
            mode: 'markers+text',
            type: 'scatter3d',
            name: 'Minor Keys',
            marker: { color: 'brown', size: 12, symbol: 'square' },
            textposition: 'top center',
            textfont: { size: 10 },
            visible: spiralVisibility.keys
        });

        // Pitch spiral
        const spiralData = { x: [], y: [], z: [] };
        for (let k = 0; k <= 12; k += 0.05) {
            const pos = getPitchPosition(k);
            spiralData.x.push(pos.x);
            spiralData.y.push(pos.y);
            spiralData.z.push(pos.z);
        }
        traces.push({
            x: spiralData.x, y: spiralData.y, z: spiralData.z,
            mode: 'lines',
            type: 'scatter3d',
            name: 'Pitch Spiral',
            line: { color: 'gray', width: 2 },
            visible: spiralVisibility.spirals,
            showlegend: false
        });

        return traces;
    }

    // Layout configuration (no fixed width/height: the div sizes the plot)
    const spiralLayout = {
        scene: {
            camera: {
                up: { y: 0, z: 1, x: 0 },
                center: { y: 0, z: 0, x: 0 },
                eye: { y: 1.5, z: 1.5, x: 1.5 }
            },
            xaxis: { title: 'X' },
            yaxis: { title: 'Y' },
            zaxis: { title: 'Z' },
            aspectmode: 'cube'
        },
        margin: { l: 0, b: 0, r: 0, t: 0 },
        paper_bgcolor: '#1a1a1a',
        plot_bgcolor: '#1a1a1a',
        font: { color: 'white' }
    };

    // Initialize plot
    (function () {
        function drawSpiral() {
            if (typeof Plotly === 'undefined') return;
            Plotly.newPlot('spiral-plot', generateSpiralData(), spiralLayout, {
                editable: false,
                responsive: true,
                staticPlot: false,
                scrollZoom: true
            });
        }
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', drawSpiral);
        } else {
            drawSpiral();
        }
    })();
</script>
~~~
