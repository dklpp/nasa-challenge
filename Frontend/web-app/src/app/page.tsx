"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { systems as data, type Planet as PlanetT, type System as SystemT } from "./data";
import { teffColor, scaleAU, trueAnomaly } from "./utils";
import SearchBar from "../components/SearchBar";
import FiltersPanel from "../components/FiltersPanel";
import SystemsList from "../components/SystemsList";
import HUD from "../components/HUD";
import TitleBar from "../components/TitleBar";
import Legend from "../components/Legend";
import DetailsPanel from "../components/DetailsPanel";
import Toast from "../components/Toast";

type Planet = (PlanetT & Partial<{ radius_rj: number; radius_re: number }>) & {
  aU?: number;
  bU?: number;
  cU?: number;
  M?: number;
};
type Sys = SystemT;

export default function Home() {
  const [q, setQ] = useState("");
  const [exSizes, setExSizes] = useState(true);
  const [logScale, setLogScale] = useState(true);
  const [showHZ, setShowHZ] = useState(false);
  const [dimOrbits, setDimOrbits] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [toast, setToast] = useState("");
  const [active, setActive] = useState<string | null>(null);
  const [titleSub, setTitleSub] = useState("Click a system → click a planet");

  const centerRef = useRef<HTMLDivElement | null>(null);
  const detailsRef = useRef<HTMLDivElement | null>(null);
  const tourRef = useRef<{ timer: any; idx: number } | null>(null);
  const sceneRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    starField: THREE.Points | null;
    ray: THREE.Raycaster;
    current: { group: THREE.Group | null; star: THREE.Object3D | null; planets: THREE.Mesh[]; orbits: THREE.Line[]; hz: THREE.Mesh | null };
    clock: THREE.Clock;
    raf?: number;
  } | null>(null);

  function showToast(msg: string, t = 1800) {
    setToast(msg);
    setTimeout(() => setToast(""), t);
  }

  const filtered = useMemo(() => {
    const qq = q.trim().toLowerCase();
    return data.filter(
      (s: Sys) =>
        !qq ||
        s.name.toLowerCase().includes(qq) ||
        s.planets.some((p: Planet) => `${s.name} ${p.name}`.toLowerCase().includes(qq))
    );
  }, [q]);

  useEffect(() => {
    // Register tests and report after mount
    (async () => {
      try {
        await import("./tests");
        const utils = await import("./utils");
        utils.reportTests?.(document.getElementById("testBadge"));
      } catch (e) {
        console.warn("Could not run tests", e);
      }
    })();
  }, []);

  // Initialize Three scene once
  useEffect(() => {
    if (!centerRef.current) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    centerRef.current.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 5000);
    camera.position.set(0, 120, 260);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 30;
    controls.maxDistance = 1500;

    scene.add(new THREE.AmbientLight(0x506080, 0.4));
    const starField = new THREE.Points(
      new THREE.BufferGeometry(),
      new THREE.PointsMaterial({ size: 0.8, color: 0x9fb7ff, transparent: true, opacity: 0.5 })
    );
    // Generate background stars
    {
      const N = 2000,
        R = 1600;
      const pos = new Float32Array(N * 3);
      for (let i = 0; i < N; i++) {
        const r = R * Math.cbrt(Math.random());
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        pos[3 * i] = r * Math.sin(ph) * Math.cos(th);
        pos[3 * i + 1] = r * Math.cos(ph);
        pos[3 * i + 2] = r * Math.sin(ph) * Math.sin(th);
      }
      starField.geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      scene.add(starField);
    }

    const ray = new THREE.Raycaster();
    const current = { group: null as any, star: null as any, planets: [] as THREE.Mesh[], orbits: [] as THREE.Line[], hz: null as any };
    const clock = new THREE.Clock();

    function onResize() {
      if (!centerRef.current) return;
      const w = centerRef.current.clientWidth,
        h = centerRef.current.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    window.addEventListener("resize", onResize);
    onResize();

    function clearSystem() {
      if (current.group) {
        scene.remove(current.group);
        current.group.traverse((obj: any) => {
          obj.geometry?.dispose?.();
          obj.material?.dispose?.();
        });
      }
      current.group = null as any;
      current.star = null as any;
      current.planets = [];
      current.orbits = [];
      current.hz = null as any;
    }

    function buildSystem(sys: Sys) {
      clearSystem();
      const g = new THREE.Group();
      scene.add(g);
      current.group = g;

      const starCol = teffColor(sys.star.teff, THREE);
      const starGeom = new THREE.SphereGeometry(exSizes ? 10 : Math.max(2, (sys.star as any).radius_rs * 2.5), 32, 16);
      const starMat = new THREE.MeshStandardMaterial({ emissive: starCol, emissiveIntensity: 1.5, color: 0x222233, roughness: 0.4, metalness: 0.1 });
      const starMesh = new THREE.Mesh(starGeom, starMat);
      g.add(starMesh);
      current.star = starMesh;
      const lamp = new THREE.PointLight(starCol.getHex(), 2.2, 0, 2);
      lamp.position.set(0, 0, 0);
      g.add(lamp);

      if (showHZ) {
        const hz = new THREE.RingGeometry(scaleAU(0.75, logScale), scaleAU(1.77, logScale), 128);
        const hzmat = new THREE.MeshBasicMaterial({ color: 0x66ffcc, transparent: true, opacity: 0.08, side: THREE.DoubleSide });
        const hzmesh = new THREE.Mesh(hz, hzmat);
        hzmesh.rotation.x = -Math.PI / 2;
        g.add(hzmesh);
        current.hz = hzmesh as any;
      }

      sys.planets.forEach((p0) => {
        const p = { ...p0 } as Planet;
        const a = Math.max(0.004, p.a_au);
        const e = Math.min(0.95, Math.max(0, p.e || 0));
        const aU = scaleAU(a, logScale);
        const bU = aU * Math.sqrt(1 - e * e);
        const cU = Math.sqrt(Math.max(0, aU * aU - bU * bU));
        const inc = THREE.MathUtils.degToRad(p.incl || 0);

        const N = 256;
        const pts: THREE.Vector3[] = [];
        for (let i = 0; i <= N; i++) {
          const th = (i / N) * 2 * Math.PI;
          const x = aU * Math.cos(th) - cU;
          const y = bU * Math.sin(th);
          pts.push(new THREE.Vector3(x, 0, y));
        }
        const orbGeom = new THREE.BufferGeometry().setFromPoints(pts);
        const orbMat = new THREE.LineBasicMaterial({ color: dimOrbits ? 0x28406f : 0x4066aa, transparent: true, opacity: dimOrbits ? 0.35 : 0.7 });
        const orbit = new THREE.LineLoop(orbGeom, orbMat);
        orbit.rotation.x = -Math.PI / 2;
        orbit.rotation.z = inc;
        g.add(orbit);
        current.orbits.push(orbit);

        const rScene = exSizes ? Math.max(1.8, (p.radius_rj ? p.radius_rj * 6 : (p.radius_re || 0) * 0.8)) : Math.max(0.4, (p.radius_rj ? p.radius_rj * 1.2 : (p.radius_re || 0) * 0.2));
        const geom = new THREE.SphereGeometry(rScene, 24, 16);
        const color = teffColor(sys.star.teff, THREE).clone().offsetHSL(0, 0, -0.15);
        const mat = new THREE.MeshStandardMaterial({ color, metalness: 0.2, roughness: 0.4 });
        const m = new THREE.Mesh(geom, mat);
        (m as any).userData = { ...p, aU, bU, cU, inc, M: Math.random() * Math.PI * 2, name: `${sys.name} ${p.name}` };
        g.add(m);
        current.planets.push(m);
      });

      setActive(sys.name);
      setTitleSub(`${sys.planets.length} planet(s) • star T_eff ${sys.star.teff}K`);
    }

    function selectPlanet(m: THREE.Mesh) {
      const p: any = (m as any).userData;
      current.orbits.forEach((o) => ((o.material as THREE.LineBasicMaterial).opacity = dimOrbits ? 0.5 : 0.9));
      controls.target.lerp(new THREE.Vector3(0, 0, 0), 0.2);
      camera.position.lerp(new THREE.Vector3(0, 40 + p.aU * 0.08, 90 + p.aU * 0.2), 0.2);
      const rows: [string, string][] = [
        ["Planet", p.name],
        ["Orbital period", p.period_days.toFixed(3) + " days"],
        ["Semi‑major (a)", p.a_au.toFixed(4) + " AU"],
        ["Eccentricity", (p.e || 0).toFixed(3)],
        ["Inclination", (p.incl || 0).toFixed(1) + "°"],
        ["Radius", p.radius_rj ? p.radius_rj.toFixed(2) + " Rj" : (p.radius_re || 0).toFixed(2) + " R⊕"],
      ];
      if (detailsRef.current) {
        detailsRef.current.innerHTML =
          rows.map(([k, v]) => `<div class="row"><div>${k}</div><div>${v}</div></div>`).join("") +
          `<div style="margin-top:8px"><span class="pill">Evidence</span> <span class="pill">MAST link</span> <span class="pill">Archive row</span></div>`;
      }
    }

    function tick() {
      sceneRef.current!.raf = requestAnimationFrame(tick);
      const dt = clock.getDelta();
      controls.update();
      const timeFactor = speed || 1;
      for (const m of current.planets) {
        const p: any = (m as any).userData;
        const e = Math.min(0.95, Math.max(0, p.e || 0));
        p.M += timeFactor * dt * (2 * Math.PI) / p.period_days; // mean anomaly advance
        const v = trueAnomaly(p.M, e);
        const a = (p.aU = scaleAU(p.a_au, logScale));
        const b = (p.bU = a * Math.sqrt(1 - e * e));
        const c = (p.cU = Math.sqrt(Math.max(0, a * a - b * b)));
        const r = (a * (1 - e * e)) / (1 + e * Math.cos(v));
        const x = r * Math.cos(v);
        const y = r * Math.sin(v);
        m.position
          .set(x, 0, y)
          .applyAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(0))
          .applyAxisAngle(new THREE.Vector3(1, 0, 0), THREE.MathUtils.degToRad((p.incl as number) || 0));
      }
      renderer.render(scene, camera);
    }

    // Pointer picking
  renderer.domElement.addEventListener("pointerdown", (e: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      const mouse = new THREE.Vector2();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      ray.setFromCamera(mouse, camera);
      const hits = ray.intersectObjects(current.planets, false);
      if (hits.length) selectPlanet(hits[0].object as THREE.Mesh);
    });

    // Save in ref for external UI handlers
    sceneRef.current = { renderer, scene, camera, controls, starField, ray, current, clock } as any;

    // Build initial system and start anim
    buildSystem(data[0]);
    tick();

    return () => {
      cancelAnimationFrame(sceneRef.current?.raf || 0);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      starField.geometry.dispose();
      (starField.material as THREE.Material).dispose();
      clearSystem();
      sceneRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Rebuild when toggles change
  useEffect(() => {
    const s = sceneRef.current;
    if (!s?.current.group) return;
    const sys = data.find((x) => x.name === active) || data[0];
    // Rebuild to apply exSizes/logScale/showHZ changes
    // Clear details on rebuild
    if (detailsRef.current) detailsRef.current.textContent = "Select a planet to see parameters and evidence.";
    // Use the internal builder by re-calling the effect's function via simple local reimplementation
    // We can't access buildSystem here, so mimic by triggering a full rebuild via resetting state:
    // Remove existing and rebuild
    const { scene, current } = s;
    if (current.group) {
      scene.remove(current.group);
      current.group.traverse((obj: any) => {
        obj.geometry?.dispose?.();
        obj.material?.dispose?.();
      });
      current.group = null as any;
      current.star = null as any;
      current.planets = [];
      current.orbits = [];
      current.hz = null as any;
    }
    // Reuse the helper by temporarily instantiating a local function
    // We duplicate minimal logic to avoid lifting builder into ref
    const buildAgain = (sys: Sys) => {
      const g = new THREE.Group();
      scene.add(g);
      s.current.group = g as any;
      const starCol = teffColor(sys.star.teff, THREE);
      const starGeom = new THREE.SphereGeometry(exSizes ? 10 : Math.max(2, (sys.star as any).radius_rs * 2.5), 32, 16);
      const starMat = new THREE.MeshStandardMaterial({ emissive: starCol, emissiveIntensity: 1.5, color: 0x222233, roughness: 0.4, metalness: 0.1 });
      const starMesh = new THREE.Mesh(starGeom, starMat);
      g.add(starMesh);
      s.current.star = starMesh as any;
      const lamp = new THREE.PointLight(starCol.getHex(), 2.2, 0, 2);
      lamp.position.set(0, 0, 0);
      g.add(lamp);
      if (showHZ) {
        const hz = new THREE.RingGeometry(scaleAU(0.75, logScale), scaleAU(1.77, logScale), 128);
        const hzmat = new THREE.MeshBasicMaterial({ color: 0x66ffcc, transparent: true, opacity: 0.08, side: THREE.DoubleSide });
        const hzmesh = new THREE.Mesh(hz, hzmat);
        hzmesh.rotation.x = -Math.PI / 2;
        g.add(hzmesh);
        s.current.hz = hzmesh as any;
      }
      sys.planets.forEach((p0) => {
        const p = { ...p0 } as Planet;
        const a = Math.max(0.004, p.a_au);
        const e = Math.min(0.95, Math.max(0, p.e || 0));
        const aU = scaleAU(a, logScale);
        const bU = aU * Math.sqrt(1 - e * e);
        const cU = Math.sqrt(Math.max(0, aU * aU - bU * bU));
        const inc = THREE.MathUtils.degToRad(p.incl || 0);
        const N = 256;
        const pts: THREE.Vector3[] = [];
        for (let i = 0; i <= N; i++) {
          const th = (i / N) * 2 * Math.PI;
          const x = aU * Math.cos(th) - cU;
          const y = bU * Math.sin(th);
          pts.push(new THREE.Vector3(x, 0, y));
        }
        const orbGeom = new THREE.BufferGeometry().setFromPoints(pts);
        const orbMat = new THREE.LineBasicMaterial({ color: dimOrbits ? 0x28406f : 0x4066aa, transparent: true, opacity: dimOrbits ? 0.35 : 0.7 });
        const orbit = new THREE.LineLoop(orbGeom, orbMat);
        orbit.rotation.x = -Math.PI / 2;
        orbit.rotation.z = inc;
        g.add(orbit);
        s.current.orbits.push(orbit as any);
        const rScene = exSizes ? Math.max(1.8, (p.radius_rj ? p.radius_rj * 6 : (p.radius_re || 0) * 0.8)) : Math.max(0.4, (p.radius_rj ? p.radius_rj * 1.2 : (p.radius_re || 0) * 0.2));
        const geom = new THREE.SphereGeometry(rScene, 24, 16);
        const color = teffColor(sys.star.teff, THREE).clone().offsetHSL(0, 0, -0.15);
        const mat = new THREE.MeshStandardMaterial({ color, metalness: 0.2, roughness: 0.4 });
        const m = new THREE.Mesh(geom, mat);
        (m as any).userData = { ...p, aU, bU, cU, inc, M: Math.random() * Math.PI * 2, name: `${sys.name} ${p.name}` };
        g.add(m);
        s.current.planets.push(m as any);
      });
      setActive(sys.name);
      setTitleSub(`${sys.planets.length} planet(s) • star T_eff ${sys.star.teff}K`);
    };
    buildAgain(sys);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exSizes, logScale, showHZ]);

  // Dim orbit material opacity live toggle
  useEffect(() => {
    const s = sceneRef.current;
    if (!s) return;
    s.current.orbits.forEach((o) => {
      (o.material as THREE.LineBasicMaterial).opacity = dimOrbits ? 0.35 : 0.7;
    });
  }, [dimOrbits]);

  // Handlers
  function handleBuild(sys: Sys) {
    const s = sceneRef.current;
    if (!s) return;
    // Trigger rebuild by toggling a dependency-less rebuild path: reuse the toggle effect
    setActive(sys.name);
    // Quick rebuild using the same path as options change: temporarily nudge state to re-run effect
    // But more directly: call a local builder mirroring initial effect
    const { scene, current } = s;
    if (current.group) {
      scene.remove(current.group);
      current.group.traverse((obj: any) => {
        obj.geometry?.dispose?.();
        obj.material?.dispose?.();
      });
      current.group = null as any;
      current.star = null as any;
      current.planets = [];
      current.orbits = [];
      current.hz = null as any;
    }

    const g = new THREE.Group();
    scene.add(g);
    s.current.group = g as any;
    const starCol = teffColor(sys.star.teff, THREE);
    const starGeom = new THREE.SphereGeometry(exSizes ? 10 : Math.max(2, (sys.star as any).radius_rs * 2.5), 32, 16);
    const starMat = new THREE.MeshStandardMaterial({ emissive: starCol, emissiveIntensity: 1.5, color: 0x222233, roughness: 0.4, metalness: 0.1 });
    const starMesh = new THREE.Mesh(starGeom, starMat);
    g.add(starMesh);
    s.current.star = starMesh as any;
    const lamp = new THREE.PointLight(starCol.getHex(), 2.2, 0, 2);
    lamp.position.set(0, 0, 0);
    g.add(lamp);
    if (showHZ) {
      const hz = new THREE.RingGeometry(scaleAU(0.75, logScale), scaleAU(1.77, logScale), 128);
      const hzmat = new THREE.MeshBasicMaterial({ color: 0x66ffcc, transparent: true, opacity: 0.08, side: THREE.DoubleSide });
      const hzmesh = new THREE.Mesh(hz, hzmat);
      hzmesh.rotation.x = -Math.PI / 2;
      g.add(hzmesh);
      s.current.hz = hzmesh as any;
    }
    sys.planets.forEach((p0) => {
      const p = { ...p0 } as Planet;
      const a = Math.max(0.004, p.a_au);
      const e = Math.min(0.95, Math.max(0, p.e || 0));
      const aU = scaleAU(a, logScale);
      const bU = aU * Math.sqrt(1 - e * e);
      const cU = Math.sqrt(Math.max(0, aU * aU - bU * bU));
      const inc = THREE.MathUtils.degToRad(p.incl || 0);
      const N = 256;
      const pts: THREE.Vector3[] = [];
      for (let i = 0; i <= N; i++) {
        const th = (i / N) * 2 * Math.PI;
        const x = aU * Math.cos(th) - cU;
        const y = bU * Math.sin(th);
        pts.push(new THREE.Vector3(x, 0, y));
      }
      const orbGeom = new THREE.BufferGeometry().setFromPoints(pts);
      const orbMat = new THREE.LineBasicMaterial({ color: dimOrbits ? 0x28406f : 0x4066aa, transparent: true, opacity: dimOrbits ? 0.35 : 0.7 });
      const orbit = new THREE.LineLoop(orbGeom, orbMat);
      orbit.rotation.x = -Math.PI / 2;
      orbit.rotation.z = inc;
      g.add(orbit);
      s.current.orbits.push(orbit as any);

      const rScene = exSizes ? Math.max(1.8, (p.radius_rj ? p.radius_rj * 6 : (p.radius_re || 0) * 0.8)) : Math.max(0.4, (p.radius_rj ? p.radius_rj * 1.2 : (p.radius_re || 0) * 0.2));
      const geom = new THREE.SphereGeometry(rScene, 24, 16);
      const color = teffColor(sys.star.teff, THREE).clone().offsetHSL(0, 0, -0.15);
      const mat = new THREE.MeshStandardMaterial({ color, metalness: 0.2, roughness: 0.4 });
      const m = new THREE.Mesh(geom, mat);
      (m as any).userData = { ...p, aU, bU, cU, inc, M: Math.random() * Math.PI * 2, name: `${sys.name} ${p.name}` };
      g.add(m);
      s.current.planets.push(m as any);
    });
    setTitleSub(`${sys.planets.length} planet(s) • star T_eff ${sys.star.teff}K`);
    showToast(`Loaded ${sys.name}`);
  }

  function handleTour(toggle?: boolean) {
    if (!tourRef.current) tourRef.current = { timer: null as any, idx: 0 } as any;
    const tr = tourRef.current!;
    if (tr && tr.timer) {
      clearInterval(tr.timer as any);
      tr.timer = null as any;
      showToast("Tour stopped");
      return;
    }
    tr.idx = 0;
    tr.timer = setInterval(() => {
      const s = data[tr.idx % data.length];
      handleBuild(s);
      tr.idx++;
    }, 4000);
    showToast("Tour started");
  }

  function resetCam() {
    const s = sceneRef.current;
    if (!s) return;
    s.camera.position.set(0, 120, 260);
    s.controls.target.set(0, 0, 0);
    s.controls.update();
  }

  const speedVal = useMemo(() => speed.toFixed(2) + "×", [speed]);

  return (
    <div id="app">
      <aside id="left">
        <h1>Systems</h1>
        <SearchBar q={q} onChange={setQ} onTour={() => handleTour(true)} />
        <FiltersPanel
          exSizes={exSizes}
          logScale={logScale}
          showHZ={showHZ}
          dimOrbits={dimOrbits}
          onChange={(patch: Partial<{ exSizes: boolean; logScale: boolean; showHZ: boolean; dimOrbits: boolean }>) => {
            if (patch.exSizes !== undefined) setExSizes(patch.exSizes);
            if (patch.logScale !== undefined) setLogScale(patch.logScale);
            if (patch.showHZ !== undefined) setShowHZ(patch.showHZ);
            if (patch.dimOrbits !== undefined) setDimOrbits(patch.dimOrbits);
          }}
        />
        <SystemsList systems={filtered} onSelect={handleBuild} />
      </aside>

      <main id="center" ref={centerRef}>
        <HUD speed={speed} onSpeed={setSpeed} speedVal={speedVal} onReset={resetCam} />
        <TitleBar system={active} sub={titleSub} />
        <Legend />
        <Toast message={toast} />
      </main>

      <aside id="right">
        <DetailsPanel detailsRef={detailsRef} />
      </aside>
    </div>
  );
}
