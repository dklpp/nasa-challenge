# Transitarium (NASA Eyes–style Exoplanet Viewer)

A lightweight 3D exoplanet system viewer for hackathon demos. Elliptical, variable-speed orbits (Kepler's 2nd law), system list, search, and a details panel. Includes a tiny test harness for the Kepler solver.

## Files

- `index.html` — markup, import map, and script bootstrap
- `styles.css` — UI styles
- `data.js` — demo systems (TRAPPIST‑1, Kepler‑90, Kepler‑10, WASP‑121)
- `utils.js` — `teffColor`, `scaleAU`, `trueAnomaly`, and test harness (`test`, `approx`, `reportTests`)
- `tests.js` — unit tests (circular orbit properties, periapsis/apoapsis, high‑e stability)
- `main.js` — Three.js scene, OrbitControls, animation loop, UI wiring

## Run

Because modules are used, serve via a simple local server:

```bash
# Python 3
cd transitarium
python -m http.server 8000
# open http://localhost:8000 in a modern browser (Chrome, Edge, Firefox)
```

Or use any static server. Import maps require modern browsers.

## Notes

- Sizes can be exaggerated (toggle). Orbits can be shown on a log scale for compact systems.
- Click a planet to populate details in the right panel.
- HUD badge shows unit test results for the orbital solver.

## Next

- Replace `data.js` with a live fetch from NASA Exoplanet Archive.
- Wire your AI candidate evidence (phase‑folded LC, BLS plots) into the right panel.
