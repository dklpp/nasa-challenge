"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { type Planet as PlanetT, type System as SystemT } from "./data";
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
  const [data, setData] = useState<SystemT[]>([]);
  const [loading, setLoading] = useState(true);
  const [exSizes, setExSizes] = useState(true);
  const [logScale, setLogScale] = useState(true);
  const [showHZ, setShowHZ] = useState(false);
  const [dimOrbits, setDimOrbits] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [toast, setToast] = useState("");
  const [active, setActive] = useState<string | null>(null);
  const [titleSub, setTitleSub] = useState("Click a system → click a planet");
  const [sizeScale, setSizeScale] = useState(0.12);
  const [mode, setMode] = useState<"explore" | "system">("explore");
  const [showDetails, setShowDetails] = useState(true);

  const centerRef = useRef<HTMLDivElement | null>(null);
  const detailsRef = useRef<HTMLDivElement | null>(null);
  const tourRef = useRef<{ timer: any; idx: number } | null>(null);
  const sceneRef = useRef<any | null>(null);

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
        s.planets.some((p: Planet) =>
          `${s.name} ${p.name}`.toLowerCase().includes(qq)
        )
    );
  }, [q, data]);

  // Fetch systems on mount
  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const res = await fetch("/api/systems");
        const js = await res.json();
        setData(js || []);
      } catch (e) {
        console.warn("Failed to fetch systems", e);
        setData([]);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // When layout width changes (e.g., toggling details panel), update renderer size
  useEffect(() => {
    const s = sceneRef.current;
    if (!s || !centerRef.current) return;
    const w = centerRef.current.clientWidth;
    const h = centerRef.current.clientHeight;
    s.renderer.setSize(w, h, false);
    s.camera.aspect = w / h;
    s.camera.updateProjectionMatrix();
  }, [showDetails]);

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

  scene.add(new THREE.AmbientLight(0x506080, 0.5));
    // Global light to avoid creating hundreds of per-star lights
    const globalLight = new THREE.DirectionalLight(0xffffff, 1.0);
    globalLight.position.set(300, 450, 350);
    globalLight.target.position.set(0, 0, 0);
    scene.add(globalLight);
    scene.add(globalLight.target);
    // Layered starry background for richer depth
    // Use a tiny circular sprite so stars render as round points (not squares)
  let discTex: any = null;
    function getDiscTexture() {
      if (discTex) return discTex;
      const c = document.createElement("canvas");
      c.width = c.height = 64;
      const ctx = c.getContext("2d")!;
      ctx.clearRect(0, 0, 64, 64);
      ctx.beginPath();
      ctx.arc(32, 32, 30, 0, Math.PI * 2);
      const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 30);
      grad.addColorStop(0, "rgba(255,255,255,1)");
      grad.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = grad;
      ctx.fill();
      discTex = new THREE.CanvasTexture(c);
      discTex.needsUpdate = true;
      return discTex;
    }
    const starField = new THREE.Group();
    function makeStars(
      N: number,
      R: number,
      opts: { size: number; opacity: number; warmChance?: number }
    ) {
      const geom = new THREE.BufferGeometry();
      const pos = new Float32Array(N * 3);
      const cols = new Float32Array(N * 3);
      for (let i = 0; i < N; i++) {
        const r = R * Math.cbrt(Math.random());
        const th = Math.random() * Math.PI * 2;
        const ph = Math.acos(2 * Math.random() - 1);
        const x = r * Math.sin(ph) * Math.cos(th);
        const y = r * Math.cos(ph);
        const z = r * Math.sin(ph) * Math.sin(th);
        pos[3 * i] = x;
        pos[3 * i + 1] = y;
        pos[3 * i + 2] = z;
        // Mostly blue-white with some warmer stars
        const isWarm = Math.random() < (opts.warmChance ?? 0.18);
        const h = isWarm
          ? 0.06 + Math.random() * 0.04
          : 0.58 + Math.random() * 0.12;
        const s = 0.58 + Math.random() * 0.34;
        const l = 0.6 + Math.random() * 0.32; // slightly brighter for visibility
        const c = new THREE.Color().setHSL(h, s, l);
        cols[3 * i] = c.r;
        cols[3 * i + 1] = c.g;
        cols[3 * i + 2] = c.b;
      }
      geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geom.setAttribute("color", new THREE.BufferAttribute(cols, 3));
      const mat = new THREE.PointsMaterial({
        size: opts.size,
        transparent: true,
        opacity: opts.opacity,
        vertexColors: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true,
        map: getDiscTexture(),
        alphaTest: 0.5,
      });
      return new THREE.Points(geom, mat);
    }
    starField.add(
      makeStars(5500, 2200, { size: 0.75, opacity: 0.36, warmChance: 0.14 })
    );
    starField.add(
      makeStars(2500, 1800, { size: 1.2, opacity: 0.5, warmChance: 0.2 })
    );
    starField.add(
      makeStars(700, 1400, { size: 1.6, opacity: 0.7, warmChance: 0.22 })
    );
    // Sparse bright stars for emphasis
    starField.add(
      makeStars(220, 1200, { size: 2.0, opacity: 0.85, warmChance: 0.25 })
    );
    scene.add(starField);

    const ray = new THREE.Raycaster();
    const current = {
      group: null as any,
      star: null as any,
      planets: [] as any[],
      orbits: [] as any[],
      hz: null as any,
    };
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
      // Reset camera to a good vantage point on each system build
      controls.target.set(0, 0, 0);
      camera.position.set(0, 120, 260);
      controls.update();

      const starCol = teffColor(sys.star.teff, THREE);
      const starGeom = new THREE.SphereGeometry(
        exSizes ? 10 : Math.max(2, ((sys.star as any).radius_rs ?? 1) * 2.5),
        32,
        16
      );
      const starMat = new THREE.MeshStandardMaterial({
        emissive: starCol,
        emissiveIntensity: 1.5,
        color: 0x222233,
        roughness: 0.4,
        metalness: 0.1,
      });
      const starMesh = new THREE.Mesh(starGeom, starMat);
      g.add(starMesh);
      current.star = starMesh;
      const lamp = new THREE.PointLight(starCol.getHex(), 2.2, 0, 2);
      lamp.position.set(0, 0, 0);
      g.add(lamp);

      if (showHZ) {
        const hz = new THREE.RingGeometry(
          scaleAU(0.75, logScale),
          scaleAU(1.77, logScale),
          128
        );
        const hzmat = new THREE.MeshBasicMaterial({
          color: 0x66ffcc,
          transparent: true,
          opacity: 0.08,
          side: THREE.DoubleSide,
        });
        const hzmesh = new THREE.Mesh(hz, hzmat);
        hzmesh.rotation.x = -Math.PI / 2;
        g.add(hzmesh);
        current.hz = hzmesh as any;
      }

      let maxAU = 0;
      sys.planets.forEach((p0) => {
        const p = { ...p0 } as Planet;
        const a = Math.max(0.004, p.a_au);
        const e = Math.min(0.95, Math.max(0, p.e || 0));
        const aU = scaleAU(a, logScale);
        const bU = aU * Math.sqrt(1 - e * e);
        const cU = Math.sqrt(Math.max(0, aU * aU - bU * bU));
        const inc = THREE.MathUtils.degToRad(p.incl || 0);
        maxAU = Math.max(maxAU, aU * (1 + e));

        const N = 256;
        const pts: any[] = [];
        for (let i = 0; i <= N; i++) {
          const th = (i / N) * 2 * Math.PI;
          const x = aU * Math.cos(th) - cU;
          const y = bU * Math.sin(th);
          pts.push(new THREE.Vector3(x, 0, y));
        }
        const orbGeom = new THREE.BufferGeometry().setFromPoints(pts);
        const orbMat = new THREE.LineBasicMaterial({
          color: dimOrbits ? 0x28406f : 0x4066aa,
          transparent: true,
          opacity: dimOrbits ? 0.35 : 0.7,
        });
        const orbit = new THREE.LineLoop(orbGeom, orbMat);
        // Keep orbit geometry in XZ plane and tilt around X by inclination to match planet motion
        orbit.rotation.set(inc, 0, 0);
        g.add(orbit);
        current.orbits.push(orbit);

        const rScene = exSizes
          ? Math.max(
              1.8,
              p.radius_rj ? p.radius_rj * 6 : (p.radius_re || 0) * 0.8
            )
          : Math.max(
              0.4,
              p.radius_rj ? p.radius_rj * 1.2 : (p.radius_re || 0) * 0.2
            );
        const geom = new THREE.SphereGeometry(rScene, 24, 16);
        const color = teffColor(sys.star.teff, THREE)
          .clone()
          .offsetHSL(0, 0, -0.15);
        const mat = new THREE.MeshStandardMaterial({
          color,
          metalness: 0.2,
          roughness: 0.4,
        });
        const m = new THREE.Mesh(geom, mat);
        (m as any).userData = {
          ...p,
          aU,
          bU,
          cU,
          inc,
          M: Math.random() * Math.PI * 2,
          name: `${sys.name} ${p.name}`,
        };
        // Set an initial position so the planet is visible immediately
        const v0 = trueAnomaly((m as any).userData.M, e);
        const r0 = (aU * (1 - e * e)) / (1 + e * Math.cos(v0));
        const x0 = r0 * Math.cos(v0);
        const y0 = r0 * Math.sin(v0);
        m.position
          .set(x0, 0, y0)
          .applyAxisAngle(new THREE.Vector3(1, 0, 0), inc);
        g.add(m);
        current.planets.push(m);
      });

      setActive(sys.name);
      setTitleSub(
        `${sys.planets.length} planet(s) • star T_eff ${sys.star.teff}K`
      );
      // Frame the system based on its size
      const dist = Math.min(1500, Math.max(120, 90 + maxAU * 0.6));
      camera.position.set(0, Math.min(400, 40 + maxAU * 0.25), dist);
      controls.target.set(0, 0, 0);
      controls.update();
    }

    function buildExplore(systems: Sys[]) {
      clearSystem();
      const g = new THREE.Group();
      scene.add(g);
      current.group = g;

      // Earth at the center
      const earthGeom = new THREE.SphereGeometry(6, 32, 16);
      const earthMat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0x2f71ff),
        emissive: new THREE.Color(0x1f3fb8),
        emissiveIntensity: 0.5,
        metalness: 0.1,
        roughness: 0.6,
      });
      const earth = new THREE.Mesh(earthGeom, earthMat);
      g.add(earth);
      const earthGlow = new THREE.PointLight(0x3a7cff, 0.6, 150, 2);
      earthGlow.position.set(0, 0, 0);
      g.add(earthGlow);

      // Distribute systems around Earth using a Fibonacci sphere
      const Nsys = systems.length;
      const R = 950; // radius of distribution sphere
      const golden = Math.PI * (3 - Math.sqrt(5));
      let totalPlanets = 0;

      systems.forEach((sys, i) => {
        const y = 1 - (i / Math.max(1, Nsys - 1)) * 2; // 1..-1
        const r = Math.sqrt(Math.max(0, 1 - y * y));
        const theta = golden * i;
        const x = Math.cos(theta) * r;
        const z = Math.sin(theta) * r;
        const pos = new THREE.Vector3(x, y, z).multiplyScalar(R);

        const sg = new THREE.Group();
        sg.position.copy(pos);
        g.add(sg);

        // Star
        const starCol = teffColor(sys.star.teff, THREE);
        const starGeom = new THREE.SphereGeometry(
          exSizes ? 7 : Math.max(2, ((sys.star as any).radius_rs ?? 1) * 2.0),
          24,
          12
        );
        const starMat = new THREE.MeshStandardMaterial({
          emissive: starCol,
          emissiveIntensity: 1.2,
          color: 0x222233,
          roughness: 0.45,
          metalness: 0.1,
        });
  const starMesh = new THREE.Mesh(starGeom, starMat);
  sg.add(starMesh);

        // Planets and orbits (lighter geometry for perf)
        sys.planets.forEach((p0) => {
          const p = { ...p0 } as Planet;
          const a = Math.max(0.004, p.a_au);
          const e = Math.min(0.95, Math.max(0, p.e || 0));
          const aU = scaleAU(a, logScale);
          const bU = aU * Math.sqrt(1 - e * e);
          const cU = Math.sqrt(Math.max(0, aU * aU - bU * bU));
          const inc = THREE.MathUtils.degToRad(p.incl || 0);

          // Orbit (reduced resolution for performance)
          const N = 72;
          const pts: any[] = [];
          for (let j = 0; j <= N; j++) {
            const th = (j / N) * 2 * Math.PI;
            const ox = aU * Math.cos(th) - cU;
            const oy = bU * Math.sin(th);
            pts.push(new THREE.Vector3(ox, 0, oy));
          }
          const orbGeom = new THREE.BufferGeometry().setFromPoints(pts);
          const orbMat = new THREE.LineBasicMaterial({
            color: dimOrbits ? 0x28406f : 0x4066aa,
            transparent: true,
            opacity: dimOrbits ? 0.28 : 0.55,
          });
          const orbit = new THREE.LineLoop(orbGeom, orbMat);
          orbit.rotation.set(inc, 0, 0);
          sg.add(orbit);
          current.orbits.push(orbit as any);

          // Planet mesh
          const rScene = exSizes
            ? Math.max(1.4, p.radius_rj ? p.radius_rj * 5 : (p.radius_re || 0) * 0.7)
            : Math.max(0.35, p.radius_rj ? p.radius_rj * 1.0 : (p.radius_re || 0) * 0.18);
          const geom = new THREE.SphereGeometry(rScene, 20, 12);
          const color = teffColor(sys.star.teff, THREE).clone().offsetHSL(0, 0, -0.15);
          const mat = new THREE.MeshStandardMaterial({ color, metalness: 0.2, roughness: 0.4 });
          const m = new THREE.Mesh(geom, mat);
          (m as any).userData = {
            ...p,
            aU,
            bU,
            cU,
            inc,
            M: Math.random() * Math.PI * 2,
            name: `${sys.name} ${p.name}`,
          };
          const v0 = trueAnomaly((m as any).userData.M, e);
          const r0 = (aU * (1 - e * e)) / (1 + e * Math.cos(v0));
          const x0 = r0 * Math.cos(v0);
          const y0 = r0 * Math.sin(v0);
          m.position.set(x0, 0, y0).applyAxisAngle(new THREE.Vector3(1, 0, 0), inc);
          sg.add(m);
          current.planets.push(m as any);
        });
      });

      setActive("Explore");
      setTitleSub(`${systems.length} systems • click any planet`);
      // Frame the whole distribution sphere
      controls.target.set(0, 0, 0);
      camera.position.set(0, 600, 1200);
      controls.maxDistance = 2500;
      controls.update();
    }

    function selectPlanet(m: any) {
      const p: any = (m as any).userData;
      current.orbits.forEach(
        (o: any) => ((o.material as any).opacity = dimOrbits ? 0.5 : 0.9)
      );
      const wp = new THREE.Vector3();
      (m as any).getWorldPosition(wp);
      const camTarget = wp.clone();
      const camPos = wp.clone().add(new THREE.Vector3(0, 40 + p.aU * 0.08, 90 + p.aU * 0.2));
      controls.target.lerp(camTarget, 0.25);
      camera.position.lerp(camPos, 0.25);
      const rows: [string, string][] = [
        ["Planet", p.name],
        ["Orbital period", p.period_days.toFixed(3) + " days"],
        ["Semi‑major (a)", p.a_au.toFixed(4) + " AU"],
        ["Eccentricity", (p.e || 0).toFixed(3)],
        ["Inclination", (p.incl || 0).toFixed(1) + "°"],
        [
          "Radius",
          p.radius_rj
            ? p.radius_rj.toFixed(2) + " Rj"
            : (p.radius_re || 0).toFixed(2) + " R⊕",
        ],
      ];
      if (detailsRef.current) {
        detailsRef.current.innerHTML =
          rows
            .map(
              ([k, v]) =>
                `<div class="row"><div>${k}</div><div>${v}</div></div>`
            )
            .join("") +
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
        p.M += (timeFactor * dt * (2 * Math.PI)) / p.period_days; // mean anomaly advance
        const v = trueAnomaly(p.M, e);
        const a = (p.aU = scaleAU(p.a_au, logScale));
        const b = (p.bU = a * Math.sqrt(1 - e * e));
        const c = (p.cU = Math.sqrt(Math.max(0, a * a - b * b)));
        const r = (a * (1 - e * e)) / (1 + e * Math.cos(v));
        const x = r * Math.cos(v);
        const y = r * Math.sin(v);
        m.position
          .set(x, 0, y)
          .applyAxisAngle(
            new THREE.Vector3(0, 1, 0),
            THREE.MathUtils.degToRad(0)
          )
          .applyAxisAngle(
            new THREE.Vector3(1, 0, 0),
            THREE.MathUtils.degToRad((p.incl as number) || 0)
          );
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
      if (hits.length) selectPlanet(hits[0].object as any);
    });

    // Save in ref for external UI handlers
    sceneRef.current = {
      renderer,
      scene,
      camera,
      controls,
      starField,
      ray,
      current,
      clock,
    } as any;

  // Build initial scene and start anim
  if (data.length) {
    if (mode === "explore") buildExplore(data);
    else buildSystem(data[0]);
  }
    tick();

    return () => {
      cancelAnimationFrame(sceneRef.current?.raf || 0);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      // Remove global light and its target to avoid leaks
      scene.remove(globalLight);
      scene.remove(globalLight.target);
      starField.children.forEach((child: any) => {
        child.geometry?.dispose?.();
        child.material?.dispose?.();
      });
      clearSystem();
      sceneRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.length]);

  // Rebuild when toggles change
  useEffect(() => {
    const s = sceneRef.current;
    if (!s?.current.group) return;
    if (mode === "explore") {
      // Rebuild explore scene using current options
      // Clear details on rebuild
      if (detailsRef.current)
        detailsRef.current.textContent =
          "Select a planet to see parameters and evidence.";
      const { scene, current } = s;
      if (current.group) {
        scene.remove(current.group);
        current.group.traverse((obj: any) => {
          obj.geometry?.dispose?.();
          obj.material?.dispose?.();
        });
        current.group = null as any;
        current.star = null as any;
        current.planets = [] as any[];
        current.orbits = [] as any[];
        current.hz = null as any;
      }
      // Use the initial helper created in init effect via re-running that logic here minimally
      // Since buildExplore is in closure, replicate minimal rebuild here
      // We'll reconstruct explore similar to the first build
      // Build Earth + distributed systems
      // Inline small builder to avoid refactoring
  const sceneLocal = s.scene as any;
  const controls = s.controls as OrbitControls;
      const g = new THREE.Group();
      sceneLocal.add(g);
      s.current.group = g as any;
      const earthGeom = new THREE.SphereGeometry(6, 32, 16);
      const earthMat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0x2f71ff),
        emissive: new THREE.Color(0x1f3fb8),
        emissiveIntensity: 0.5,
        metalness: 0.1,
        roughness: 0.6,
      });
      const earth = new THREE.Mesh(earthGeom, earthMat);
      g.add(earth);
      const earthGlow = new THREE.PointLight(0x3a7cff, 0.6, 150, 2);
      earthGlow.position.set(0, 0, 0);
      g.add(earthGlow);
      const Nsys = data.length;
      const R = 950;
      const golden = Math.PI * (3 - Math.sqrt(5));
      data.forEach((sys, i) => {
        const y = 1 - (i / Math.max(1, Nsys - 1)) * 2;
        const r = Math.sqrt(Math.max(0, 1 - y * y));
        const theta = golden * i;
        const x = Math.cos(theta) * r;
        const z = Math.sin(theta) * r;
        const pos = new THREE.Vector3(x, y, z).multiplyScalar(R);
        const sg = new THREE.Group();
        sg.position.copy(pos);
        g.add(sg);
        const starCol = teffColor(sys.star.teff, THREE);
        const starGeom = new THREE.SphereGeometry(
          exSizes ? 7 : Math.max(2, ((sys.star as any).radius_rs ?? 1) * 2.0),
          24,
          12
        );
        const starMat = new THREE.MeshStandardMaterial({
          emissive: starCol,
          emissiveIntensity: 1.2,
          color: 0x222233,
          roughness: 0.45,
          metalness: 0.1,
        });
  const starMesh = new THREE.Mesh(starGeom, starMat);
  sg.add(starMesh);
        sys.planets.forEach((p0) => {
          const p = { ...p0 } as any as Planet;
          const a = Math.max(0.004, p.a_au);
          const e = Math.min(0.95, Math.max(0, p.e || 0));
          const aU = scaleAU(a, logScale);
          const bU = aU * Math.sqrt(1 - e * e);
          const cU = Math.sqrt(Math.max(0, aU * aU - bU * bU));
          const inc = THREE.MathUtils.degToRad(p.incl || 0);
          const N = 72;
          const pts: any[] = [];
          for (let j = 0; j <= N; j++) {
            const th = (j / N) * 2 * Math.PI;
            const ox = aU * Math.cos(th) - cU;
            const oy = bU * Math.sin(th);
            pts.push(new THREE.Vector3(ox, 0, oy));
          }
          const orbGeom = new THREE.BufferGeometry().setFromPoints(pts);
          const orbMat = new THREE.LineBasicMaterial({
            color: dimOrbits ? 0x28406f : 0x4066aa,
            transparent: true,
            opacity: dimOrbits ? 0.28 : 0.55,
          });
          const orbit = new THREE.LineLoop(orbGeom, orbMat);
          orbit.rotation.set(inc, 0, 0);
          sg.add(orbit);
          s.current.orbits.push(orbit as any);
          const rScene = exSizes
            ? Math.max(1.4, p.radius_rj ? p.radius_rj * 5 : (p.radius_re || 0) * 0.7)
            : Math.max(0.35, p.radius_rj ? p.radius_rj * 1.0 : (p.radius_re || 0) * 0.18);
          const geom = new THREE.SphereGeometry(rScene, 20, 12);
          const color = teffColor(sys.star.teff, THREE).clone().offsetHSL(0, 0, -0.15);
          const mat = new THREE.MeshStandardMaterial({ color, metalness: 0.2, roughness: 0.4 });
          const m = new THREE.Mesh(geom, mat);
          (m as any).userData = { ...p, aU, bU, cU, inc, M: Math.random() * Math.PI * 2, name: `${sys.name} ${p.name}` };
          const v0 = trueAnomaly((m as any).userData.M, e);
          const r0 = (aU * (1 - e * e)) / (1 + e * Math.cos(v0));
          const x0 = r0 * Math.cos(v0);
          const y0 = r0 * Math.sin(v0);
          m.position.set(x0, 0, y0).applyAxisAngle(new THREE.Vector3(1, 0, 0), inc);
          sg.add(m);
          s.current.planets.push(m as any);
        });
      });
      setActive("Explore");
      setTitleSub(`${data.length} systems • click any planet`);
      controls.target.set(0, 0, 0);
      s.camera.position.set(0, 600, 1200);
      controls.maxDistance = 2500;
      controls.update();
      return;
    }
    const sys = data.find((x) => x.name === active) || data[0];
    // Rebuild to apply exSizes/logScale/showHZ changes
    // Clear details on rebuild
    if (detailsRef.current)
      detailsRef.current.textContent =
        "Select a planet to see parameters and evidence.";
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
      const starGeom = new THREE.SphereGeometry(
        exSizes ? 10 : Math.max(2, ((sys.star as any).radius_rs ?? 1) * 2.5),
        32,
        16
      );
      const starMat = new THREE.MeshStandardMaterial({
        emissive: starCol,
        emissiveIntensity: 1.5,
        color: 0x222233,
        roughness: 0.4,
        metalness: 0.1,
      });
      const starMesh = new THREE.Mesh(starGeom, starMat);
      g.add(starMesh);
      s.current.star = starMesh as any;
      const lamp = new THREE.PointLight(starCol.getHex(), 2.2, 0, 2);
      lamp.position.set(0, 0, 0);
      g.add(lamp);
      if (showHZ) {
        const hz = new THREE.RingGeometry(
          scaleAU(0.75, logScale),
          scaleAU(1.77, logScale),
          128
        );
        const hzmat = new THREE.MeshBasicMaterial({
          color: 0x66ffcc,
          transparent: true,
          opacity: 0.08,
          side: THREE.DoubleSide,
        });
        const hzmesh = new THREE.Mesh(hz, hzmat);
        hzmesh.rotation.x = -Math.PI / 2;
        g.add(hzmesh);
        s.current.hz = hzmesh as any;
      }
      let maxAU = 0;
      sys.planets.forEach((p0) => {
        const p = { ...p0 } as Planet;
        const a = Math.max(0.004, p.a_au);
        const e = Math.min(0.95, Math.max(0, p.e || 0));
        const aU = scaleAU(a, logScale);
        const bU = aU * Math.sqrt(1 - e * e);
        const cU = Math.sqrt(Math.max(0, aU * aU - bU * bU));
        const inc = THREE.MathUtils.degToRad(p.incl || 0);
        maxAU = Math.max(maxAU, aU * (1 + e));
        const N = 256;
        const pts: any[] = [];
        for (let i = 0; i <= N; i++) {
          const th = (i / N) * 2 * Math.PI;
          const x = aU * Math.cos(th) - cU;
          const y = bU * Math.sin(th);
          pts.push(new THREE.Vector3(x, 0, y));
        }
        const orbGeom = new THREE.BufferGeometry().setFromPoints(pts);
        const orbMat = new THREE.LineBasicMaterial({
          color: dimOrbits ? 0x28406f : 0x4066aa,
          transparent: true,
          opacity: dimOrbits ? 0.35 : 0.7,
        });
        const orbit = new THREE.LineLoop(orbGeom, orbMat);
        // Keep orbit geometry in XZ plane and tilt around X by inclination to match planet motion
        orbit.rotation.set(inc, 0, 0);
        g.add(orbit);
        s.current.orbits.push(orbit as any);
        const rScene = exSizes
          ? Math.max(
              1.8,
              p.radius_rj ? p.radius_rj * 6 : (p.radius_re || 0) * 0.8
            )
          : Math.max(
              0.4,
              p.radius_rj ? p.radius_rj * 1.2 : (p.radius_re || 0) * 0.2
            );
        const geom = new THREE.SphereGeometry(rScene, 24, 16);
        const color = teffColor(sys.star.teff, THREE)
          .clone()
          .offsetHSL(0, 0, -0.15);
        const mat = new THREE.MeshStandardMaterial({
          color,
          metalness: 0.2,
          roughness: 0.4,
        });
        const m = new THREE.Mesh(geom, mat);
        (m as any).userData = {
          ...p,
          aU,
          bU,
          cU,
          inc,
          M: Math.random() * Math.PI * 2,
          name: `${sys.name} ${p.name}`,
        };
        const v0 = trueAnomaly((m as any).userData.M, e);
        const r0 = (aU * (1 - e * e)) / (1 + e * Math.cos(v0));
        const x0 = r0 * Math.cos(v0);
        const y0 = r0 * Math.sin(v0);
        m.position
          .set(x0, 0, y0)
          .applyAxisAngle(new THREE.Vector3(1, 0, 0), inc);
        g.add(m);
        s.current.planets.push(m as any);
      });
      setActive(sys.name);
      setTitleSub(
        `${sys.planets.length} planet(s) • star T_eff ${sys.star.teff}K`
      );
      const dist = Math.min(1500, Math.max(120, 90 + maxAU * 0.6));
      s.camera.position.set(0, Math.min(400, 40 + maxAU * 0.25), dist);
      s.controls.target.set(0, 0, 0);
      s.controls.update();
    };
    buildAgain(sys);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exSizes, logScale, showHZ, mode, dimOrbits]);

  // Dim orbit material opacity live toggle
  useEffect(() => {
    const s = sceneRef.current;
    if (!s) return;
    s.current.orbits.forEach((o: any) => {
      (o.material as any).opacity = dimOrbits ? 0.35 : 0.7;
    });
  }, [dimOrbits]);

  // Handlers
  function handleBuild(sys: Sys) {
    const s = sceneRef.current;
    if (!s) return;
    // switch to system mode on manual selection
    if (mode !== "system") setMode("system");
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
    const starGeom = new THREE.SphereGeometry(
      exSizes ? 10 : Math.max(2, ((sys.star as any).radius_rs ?? 1) * 2.5),
      32,
      16
    );
    const starMat = new THREE.MeshStandardMaterial({
      emissive: starCol,
      emissiveIntensity: 1.5,
      color: 0x222233,
      roughness: 0.4,
      metalness: 0.1,
    });
    const starMesh = new THREE.Mesh(starGeom, starMat);
    g.add(starMesh);
    s.current.star = starMesh as any;
    const lamp = new THREE.PointLight(starCol.getHex(), 2.2, 0, 2);
    lamp.position.set(0, 0, 0);
    g.add(lamp);
    if (showHZ) {
      const hz = new THREE.RingGeometry(
        scaleAU(0.75, logScale),
        scaleAU(1.77, logScale),
        128
      );
      const hzmat = new THREE.MeshBasicMaterial({
        color: 0x66ffcc,
        transparent: true,
        opacity: 0.08,
        side: THREE.DoubleSide,
      });
      const hzmesh = new THREE.Mesh(hz, hzmat);
      hzmesh.rotation.x = -Math.PI / 2;
      g.add(hzmesh);
      s.current.hz = hzmesh as any;
    }
    let maxAU = 0;
    sys.planets.forEach((p0) => {
      const p = { ...p0 } as Planet;
      const a = Math.max(0.004, p.a_au);
      const e = Math.min(0.95, Math.max(0, p.e || 0));
      const aU = scaleAU(a, logScale);
      const bU = aU * Math.sqrt(1 - e * e);
      const cU = Math.sqrt(Math.max(0, aU * aU - bU * bU));
      const inc = THREE.MathUtils.degToRad(p.incl || 0);
      maxAU = Math.max(maxAU, aU * (1 + e));
      const N = 256;
      const pts: any[] = [];
      for (let i = 0; i <= N; i++) {
        const th = (i / N) * 2 * Math.PI;
        const x = aU * Math.cos(th) - cU;
        const y = bU * Math.sin(th);
        pts.push(new THREE.Vector3(x, 0, y));
      }
      const orbGeom = new THREE.BufferGeometry().setFromPoints(pts);
      const orbMat = new THREE.LineBasicMaterial({
        color: dimOrbits ? 0x28406f : 0x4066aa,
        transparent: true,
        opacity: dimOrbits ? 0.35 : 0.7,
      });
      const orbit = new THREE.LineLoop(orbGeom, orbMat);
      // Keep orbit geometry in XZ plane and tilt around X by inclination to match planet motion
      orbit.rotation.set(inc, 0, 0);
      g.add(orbit);
      s.current.orbits.push(orbit as any);

      const rScene = exSizes
        ? Math.max(
            1.8,
            p.radius_rj ? p.radius_rj * 6 : (p.radius_re || 0) * 0.8
          )
        : Math.max(
            0.4,
            p.radius_rj ? p.radius_rj * 1.2 : (p.radius_re || 0) * 0.2
          );
      const geom = new THREE.SphereGeometry(rScene, 24, 16);
      const color = teffColor(sys.star.teff, THREE)
        .clone()
        .offsetHSL(0, 0, -0.15);
      const mat = new THREE.MeshStandardMaterial({
        color,
        metalness: 0.2,
        roughness: 0.4,
      });
      const m = new THREE.Mesh(geom, mat);
      (m as any).userData = {
        ...p,
        aU,
        bU,
        cU,
        inc,
        M: Math.random() * Math.PI * 2,
        name: `${sys.name} ${p.name}`,
      };
      const v0 = trueAnomaly((m as any).userData.M, e);
      const r0 = (aU * (1 - e * e)) / (1 + e * Math.cos(v0));
      const x0 = r0 * Math.cos(v0);
      const y0 = r0 * Math.sin(v0);
      m.position
        .set(x0, 0, y0)
        .applyAxisAngle(new THREE.Vector3(1, 0, 0), inc);
      g.add(m);
      s.current.planets.push(m as any);
    });
    setTitleSub(
      `${sys.planets.length} planet(s) • star T_eff ${sys.star.teff}K`
    );
      s.controls.target.set(0, 0, 0);
      s.camera.position.set(0, 120, 260);
      s.controls.update();
    const dist = Math.min(1500, Math.max(120, 90 + maxAU * 0.6));
    s.camera.position.set(0, Math.min(400, 40 + maxAU * 0.25), dist);
    s.controls.target.set(0, 0, 0);
    s.controls.update();
    showToast(`Loaded ${sys.name}`);
  }

  function handleTour(toggle?: boolean) {
    if (!data.length) {
      showToast("No systems loaded");
      return;
    }
    if (!tourRef.current)
      tourRef.current = { timer: null as any, idx: 0 } as any;
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
      if (s) handleBuild(s);
      tr.idx++;
    }, 4000);
    showToast("Tour started");
  }

  function resetCam() {
    const s = sceneRef.current;
    if (!s) return;
    if (mode === "explore") {
      s.camera.position.set(0, 600, 1200);
      s.controls.target.set(0, 0, 0);
      s.controls.maxDistance = 2500;
      s.controls.update();
    } else {
      s.camera.position.set(0, 120, 260);
      s.controls.target.set(0, 0, 0);
      s.controls.update();
    }
  }

  const speedVal = useMemo(() => speed.toFixed(2) + "×", [speed]);
  const sizeVal = useMemo(() => sizeScale.toFixed(2) + "×", [sizeScale]);
  const modeLabel = mode === "explore" ? "Explore" : "System";

  // Fullscreen toggle
  const [isFS, setIsFS] = useState(false);
  useEffect(() => {
    const onFS = () => setIsFS(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", onFS);
    return () => document.removeEventListener("fullscreenchange", onFS);
  }, []);
  function toggleFullscreen() {
    const el: any = document.documentElement as any;
    if (!document.fullscreenElement) {
      (el.requestFullscreen || el.webkitRequestFullscreen || el.msRequestFullscreen)?.call(el);
    } else {
      (document.exitFullscreen || (document as any).webkitExitFullscreen || (document as any).msExitFullscreen)?.call(document);
    }
  }

  function handleExplore() {
    setMode((m) => (m === "explore" ? "system" : "explore"));
    // A toast to indicate mode; actual rebuild handled by options effect and initial init
    showToast("Toggled view mode");
  }

  return (
    <div id="app" className={showDetails ? "" : "noRight"}>
      <aside id="left">
        <div className="leftSticky">
          <h1>Systems</h1>
          <SearchBar q={q} onChange={setQ} onTour={() => handleTour(true)} />
          <FiltersPanel
            exSizes={exSizes}
            logScale={logScale}
            showHZ={showHZ}
            dimOrbits={dimOrbits}
            onChange={(
              patch: Partial<{
                exSizes: boolean;
                logScale: boolean;
                showHZ: boolean;
                dimOrbits: boolean;
              }>
            ) => {
              if (patch.exSizes !== undefined) setExSizes(patch.exSizes);
              if (patch.logScale !== undefined) setLogScale(patch.logScale);
              if (patch.showHZ !== undefined) setShowHZ(patch.showHZ);
              if (patch.dimOrbits !== undefined) setDimOrbits(patch.dimOrbits);
            }}
          />
        </div>
        {loading ? (
          <div className="list"><div className="sys">Loading…</div></div>
        ) : filtered.length ? (
          <SystemsList systems={filtered} onSelect={handleBuild} />
        ) : (
          <div className="list"><div className="sys">No systems</div></div>
        )}
      </aside>

      <main id="center" ref={centerRef}>
        <HUD
          speed={speed}
          onSpeed={setSpeed}
          speedVal={speedVal}
          onReset={resetCam}
          sizeScale={sizeScale}
          onSizeScale={setSizeScale}
          sizeVal={sizeVal}
          modeLabel={modeLabel}
          onExplore={handleExplore}
        />
        <TitleBar system={active} sub={titleSub} />
        <button
          id="detailsbtn"
          title={showDetails ? "Hide details" : "Show details"}
          aria-pressed={showDetails}
          onClick={() => setShowDetails((v) => !v)}
        >
          {showDetails ? "Hide details" : "Show details"}
        </button>
        <Legend
          line1={
            mode === "explore" ? (
              <>Earth at center • Stars with planets distributed in 3D</>
            ) : (
              <>
                Size ∝ radius • Color ∝ T<sub>eff</sub>
              </>
            )
          }
          line2={
            mode === "explore" ? (
              <>Color ≈ star Tₑff • Size ≈ planet radius</>
            ) : (
              <>Speed via Kepler's 2nd law</>
            )
          }
        />
        <button
          id="fsbtn"
          title={isFS ? "Exit full screen" : "Full screen"}
          aria-pressed={isFS}
          onClick={toggleFullscreen}
        >
          {isFS ? (
            // minimize icon
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="9 3 9 9 3 9" />
              <polyline points="15 21 15 15 21 15" />
              <line x1="21" y1="3" x2="14" y2="10" />
              <line x1="3" y1="21" x2="10" y2="14" />
            </svg>
          ) : (
            // fullscreen icon
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="3 9 3 3 9 3" />
              <polyline points="21 15 21 21 15 21" />
              <line x1="21" y1="3" x2="14" y2="10" />
              <line x1="3" y1="21" x2="10" y2="14" />
            </svg>
          )}
        </button>
        <Toast message={toast} />
      </main>

      {showDetails && (
        <aside id="right">
          <DetailsPanel detailsRef={detailsRef} />
        </aside>
      )}
    </div>
  );
}
