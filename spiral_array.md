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
    /* Definite height (not aspect-ratio): Plotly measures the container at
       newPlot time and collapses the scene if the height is indeterminate. */
    #spiral-plot {
        width: 100%;
        max-width: 1000px;
        height: min(1000px, 85vh);
        min-height: 420px;
        margin: 0 auto;
        display: block;
        background: #1a1a1a;
        border: 1px solid #6D6B66;
        overflow: visible;
    }

    #spiral-fallback {
        color: #e9edec;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85em;
        padding: 1.2em;
        text-align: center;
    }
</style>

<script src="/assets/js/spiral_array.js"></script>
~~~
