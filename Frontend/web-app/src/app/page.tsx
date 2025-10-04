"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { type Planet as PlanetT, type System as SystemT } from "./data";
import { teffColor, scaleAU, trueAnomaly, celestialToCartesian } from "./utils";
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
  const exSizes = true; 
  const [logScale, setLogScale] = useState(true);
  const [showHZ, setShowHZ] = useState(false);
  const dimOrbits = false; 
  const [speed, setSpeed] = useState(1);
  const [toast, setToast] = useState("");
  const [active, setActive] = useState<string | null>(null);
  const [titleSub, setTitleSub] = useState("Click a system → click a planet");
  const [sizeScale, setSizeScale] = useState(0.12);
  const [mode, setMode] = useState<"explore" | "system" | "earth">("earth");
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

  scene.add(new THREE.AmbientLight(0x506080, 0.8));
    // Global light to avoid creating hundreds of per-star lights
    const globalLight = new THREE.DirectionalLight(0xffffff, 1.2);
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
      stars: [] as any[],
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
      current.stars = [];
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
      const earthGeom = new THREE.SphereGeometry(6, 64, 32);
      
      // Create a simple procedural Earth texture using a canvas
      const earthCanvas = document.createElement('canvas');
      earthCanvas.width = 512;
      earthCanvas.height = 256;
      const ctx = earthCanvas.getContext('2d')!;
      
      // Base ocean color
      ctx.fillStyle = '#1a4d8f';
      ctx.fillRect(0, 0, 512, 256);
      
      // Add continents using green patches
      ctx.fillStyle = '#2d5a2d';
      for (let i = 0; i < 150; i++) {
        const x = Math.random() * 512;
        const y = Math.random() * 256;
        const size = Math.random() * 60 + 20;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }
      
      // Add lighter green patches for variety
      ctx.fillStyle = '#4a7c4a';
      for (let i = 0; i < 100; i++) {
        const x = Math.random() * 512;
        const y = Math.random() * 256;
        const size = Math.random() * 40 + 10;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }
      
      // Add white clouds
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      for (let i = 0; i < 80; i++) {
        const x = Math.random() * 512;
        const y = Math.random() * 256;
        const size = Math.random() * 30 + 10;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }
      
      const earthTexture = new THREE.CanvasTexture(earthCanvas);
      earthTexture.needsUpdate = true;
      
      const earthMat = new THREE.MeshStandardMaterial({
        map: earthTexture,
        metalness: 0.2,
        roughness: 0.8,
      });
      const earth = new THREE.Mesh(earthGeom, earthMat);
      earth.rotation.x = THREE.MathUtils.degToRad(23.5); // Earth's axial tilt
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
  (starMesh as any).userData = {
    systemName: sys.name,
    type: 'star',
  };
  sg.add(starMesh);
  current.stars.push(starMesh as any);

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

    function buildEarthCentered(systems: Sys[]) {
      // Note: Despite the name, this creates a heliocentric (Sun-centered) view
      // with exoplanet systems positioned by their celestial coordinates
      clearSystem();
      const g = new THREE.Group();
      scene.add(g);
      current.group = g;

      // The Sun at the center with fiery appearance
      const sunGeom = new THREE.SphereGeometry(10, 64, 64);
      
      // Create a fiery sun material
      const sunMat = new THREE.MeshStandardMaterial({
        emissive: new THREE.Color(0xff4400),
        emissiveIntensity: 2.5,
        color: new THREE.Color(0xff6600),
        roughness: 1.0,
        metalness: 0,
      });
      
      const sunMesh = new THREE.Mesh(sunGeom, sunMat);
      g.add(sunMesh);
      current.star = sunMesh; // Store for animation
      
      // Add solar flares/prominences using particle system
      const flareGeom = new THREE.BufferGeometry();
      const flareCount = 2000;
      const flarePositions = new Float32Array(flareCount * 3);
      const flareColors = new Float32Array(flareCount * 3);
      const flareSizes = new Float32Array(flareCount);
      
      for (let i = 0; i < flareCount; i++) {
        // Position particles on and around sun surface
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const radius = 10 + Math.random() * 3; // Extend slightly beyond surface
        
        flarePositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
        flarePositions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
        flarePositions[i * 3 + 2] = radius * Math.cos(phi);
        
        // Color variation from bright yellow to deep orange-red
        const colorMix = Math.random();
        const r = 1.0;
        const g = 0.3 + colorMix * 0.5;
        const b = colorMix * 0.2;
        
        flareColors[i * 3] = r;
        flareColors[i * 3 + 1] = g;
        flareColors[i * 3 + 2] = b;
        
        flareSizes[i] = Math.random() * 1.5 + 0.5;
      }
      
      flareGeom.setAttribute('position', new THREE.BufferAttribute(flarePositions, 3));
      flareGeom.setAttribute('color', new THREE.BufferAttribute(flareColors, 3));
      flareGeom.setAttribute('size', new THREE.BufferAttribute(flareSizes, 1));
      
      const flareMat = new THREE.PointsMaterial({
        size: 0.8,
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      
      const flares = new THREE.Points(flareGeom, flareMat);
      g.add(flares);
      (current as any).sunFlares = flares; // Store for animation
      
      // Inner fire layer
      const fireGeom1 = new THREE.SphereGeometry(11.5, 32, 32);
      const fireMat1 = new THREE.MeshBasicMaterial({
        color: 0xff6600,
        transparent: true,
        opacity: 0.4,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
      });
      const fire1 = new THREE.Mesh(fireGeom1, fireMat1);
      g.add(fire1);
      
      // Middle flame layer
      const fireGeom2 = new THREE.SphereGeometry(13.5, 32, 32);
      const fireMat2 = new THREE.MeshBasicMaterial({
        color: 0xff8800,
        transparent: true,
        opacity: 0.3,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
      });
      const fire2 = new THREE.Mesh(fireGeom2, fireMat2);
      g.add(fire2);
      
      // Outer heat haze
      const fireGeom3 = new THREE.SphereGeometry(16, 32, 32);
      const fireMat3 = new THREE.MeshBasicMaterial({
        color: 0xffaa00,
        transparent: true,
        opacity: 0.15,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
      });
      const fire3 = new THREE.Mesh(fireGeom3, fireMat3);
      g.add(fire3);
      
      // Store fire layers for animation
      (current as any).fireLayers = [fire1, fire2, fire3];
      
      // Intense Sun light with orange tint
      const sunLight = new THREE.PointLight(0xff7733, 5.0, 0, 1.2);
      sunLight.position.set(0, 0, 0);
      g.add(sunLight);

      // Earth orbiting the Sun at ~1 AU
      const earthDistance = 150; // 1 AU in scene units
      const earthGeom = new THREE.SphereGeometry(6.4, 64, 32);
      
      // Create a simple procedural Earth texture using a canvas
      const earthCanvas = document.createElement('canvas');
      earthCanvas.width = 512;
      earthCanvas.height = 256;
      const ctx = earthCanvas.getContext('2d')!;
      
      // Base ocean color
      ctx.fillStyle = '#1a4d8f';
      ctx.fillRect(0, 0, 512, 256);
      
      // Add continents using green patches
      ctx.fillStyle = '#2d5a2d';
      for (let i = 0; i < 150; i++) {
        const x = Math.random() * 512;
        const y = Math.random() * 256;
        const size = Math.random() * 60 + 20;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }
      
      // Add lighter green patches for variety
      ctx.fillStyle = '#4a7c4a';
      for (let i = 0; i < 100; i++) {
        const x = Math.random() * 512;
        const y = Math.random() * 256;
        const size = Math.random() * 40 + 10;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }
      
      // Add white clouds
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      for (let i = 0; i < 80; i++) {
        const x = Math.random() * 512;
        const y = Math.random() * 256;
        const size = Math.random() * 30 + 10;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }
      
      const earthTexture = new THREE.CanvasTexture(earthCanvas);
      earthTexture.needsUpdate = true;
      
      const earthMat = new THREE.MeshStandardMaterial({
        map: earthTexture,
        metalness: 0.2,
        roughness: 0.8,
      });
      const earth = new THREE.Mesh(earthGeom, earthMat);
      earth.position.set(earthDistance, 0, 0);
      earth.rotation.x = THREE.MathUtils.degToRad(23.5); // Earth's axial tilt
      g.add(earth);
      
      // Earth glow
      const earthGlow = new THREE.PointLight(0x3a7cff, 0.8, 200, 2);
      earthGlow.position.copy(earth.position);
      g.add(earthGlow);
      
      // Earth orbit path
      const earthOrbitGeom = new THREE.BufferGeometry();
      const earthOrbitPoints: any[] = [];
      for (let i = 0; i <= 128; i++) {
        const angle = (i / 128) * Math.PI * 2;
        earthOrbitPoints.push(new THREE.Vector3(
          earthDistance * Math.cos(angle),
          0,
          earthDistance * Math.sin(angle)
        ));
      }
      earthOrbitGeom.setFromPoints(earthOrbitPoints);
      const earthOrbitMat = new THREE.LineBasicMaterial({
        color: 0x4488ff,
        transparent: true,
        opacity: 0.3,
      });
      const earthOrbit = new THREE.LineLoop(earthOrbitGeom, earthOrbitMat);
      g.add(earthOrbit);

      // Add exoplanet systems using RA/Dec coordinates
      const systemsWithCoords = systems.filter(s => s.ra !== undefined && s.dec !== undefined);
      
      systemsWithCoords.forEach((sys) => {
        if (sys.ra === undefined || sys.dec === undefined) return;
        
        // Use distance if available, otherwise default to 100 parsecs
        const distance = sys.distance_pc || 100;
        const pos = celestialToCartesian(sys.ra, sys.dec, distance, 10);
        
        const sg = new THREE.Group();
        sg.position.set(pos.x, pos.y, pos.z);
        g.add(sg);

        // Star
        const starCol = teffColor(sys.star.teff, THREE);
        const starSize = exSizes ? 4 : Math.max(1.5, ((sys.star as any).radius_rs ?? 1) * 1.5);
        const starGeom = new THREE.SphereGeometry(starSize, 16, 12);
        const starMat = new THREE.MeshStandardMaterial({
          emissive: starCol,
          emissiveIntensity: 1.5,
          color: 0x222233,
          roughness: 0.4,
          metalness: 0.1,
        });
        const starMesh = new THREE.Mesh(starGeom, starMat);
        sg.add(starMesh);
        
        // Don't add individual lights for distant stars to avoid shader uniform limits
        // The global ambient and directional light will illuminate them

        // Add planets with smaller orbits (scaled down for visibility)
        sys.planets.forEach((p0) => {
          const p = { ...p0 } as Planet;
          const a = Math.max(0.004, p.a_au);
          const e = Math.min(0.95, Math.max(0, p.e || 0));
          // Smaller scale for planets around distant stars
          const localScale = 15;
          const aU = a * localScale;
          const bU = aU * Math.sqrt(1 - e * e);
          const cU = Math.sqrt(Math.max(0, aU * aU - bU * bU));
          const inc = THREE.MathUtils.degToRad(p.incl || 0);

          // Orbit path
          const N = 48;
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
            opacity: dimOrbits ? 0.2 : 0.4,
          });
          const orbit = new THREE.LineLoop(orbGeom, orbMat);
          orbit.rotation.set(inc, 0, 0);
          sg.add(orbit);
          current.orbits.push(orbit as any);

          // Planet mesh
          const rScene = exSizes
            ? Math.max(0.8, p.radius_rj ? p.radius_rj * 3 : (p.radius_re || 0) * 0.5)
            : Math.max(0.3, p.radius_rj ? p.radius_rj * 0.8 : (p.radius_re || 0) * 0.15);
          const geom = new THREE.SphereGeometry(rScene, 12, 8);
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
            systemRA: sys.ra,
            systemDec: sys.dec,
            systemDistance: distance,
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

      setActive("Solar System");
      setTitleSub(`Sun + Earth + ${systemsWithCoords.length} exoplanet systems`);
      
      // Position camera to center Earth in the view
      controls.target.set(earthDistance, 0, 0); // Target Earth's position
      camera.position.set(earthDistance, 100, 300); // Position camera to look at Earth
      controls.maxDistance = 15000;
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
      
      // Animate smooth zoom
      const startPos = camera.position.clone();
      const startTarget = new THREE.Vector3(
        controls.target.x,
        controls.target.y,
        controls.target.z
      );
      
      let progress = 0;
      const zoomDuration = 0.8; // 0.8 second animation
      
      const animateZoom = () => {
        progress += clock.getDelta() / zoomDuration;
        progress = Math.min(progress, 1);
        
        // Smooth easing function (ease-in-out)
        const eased = progress < 0.5 
          ? 2 * progress * progress 
          : 1 - Math.pow(-2 * progress + 2, 2) / 2;
        
        // Interpolate camera position
        camera.position.lerpVectors(startPos, camPos, eased);
        
        // Interpolate target
        const newTarget = new THREE.Vector3().lerpVectors(startTarget, camTarget, eased);
        controls.target.set(newTarget.x, newTarget.y, newTarget.z);
        
        controls.update();
        
        if (progress < 1) {
          requestAnimationFrame(animateZoom);
        }
      };
      
      animateZoom();
      
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

    function selectStar(starMesh: any) {
      const systemName = starMesh.userData.systemName;
      // Get world position of the star
      const wp = new THREE.Vector3();
      starMesh.getWorldPosition(wp);
      
      // Calculate zoom position - closer to the star
      const camTarget = wp.clone();
      const camPos = wp.clone().add(new THREE.Vector3(0, 30, 80));
      
      // Animate smooth zoom using lerp in animation loop
      const startPos = camera.position.clone();
      const startTarget = new THREE.Vector3(
        controls.target.x,
        controls.target.y,
        controls.target.z
      );
      
      let progress = 0;
      const zoomDuration = 1.0; // 1 second animation
      
      const animateZoom = () => {
        progress += clock.getDelta() / zoomDuration;
        progress = Math.min(progress, 1);
        
        // Smooth easing function (ease-in-out)
        const eased = progress < 0.5 
          ? 2 * progress * progress 
          : 1 - Math.pow(-2 * progress + 2, 2) / 2;
        
        // Interpolate camera position
        camera.position.lerpVectors(startPos, camPos, eased);
        
        // Interpolate target
        const newTarget = new THREE.Vector3().lerpVectors(startTarget, camTarget, eased);
        controls.target.set(newTarget.x, newTarget.y, newTarget.z);
        
        controls.update();
        
        if (progress < 1) {
          requestAnimationFrame(animateZoom);
        }
      };
      
      animateZoom();
      
      // Display star name in details panel
      if (detailsRef.current) {
        detailsRef.current.innerHTML = `
          <div class="row"><div style="font-size: 1.3em; font-weight: bold; text-align: center; margin-bottom: 12px;">${systemName}</div></div>
          <div style="text-align: center; margin-top: 8px; color: #888;">Click on a planet to see its details</div>
        `;
      }
      
      // Show toast notification
      showToast(`Selected ${systemName}`);
    }

    function tick() {
      sceneRef.current!.raf = requestAnimationFrame(tick);
      const dt = clock.getDelta();
      controls.update();
      const timeFactor = speed || 1;
      
      // Animate sun fire effects
      if ((current as any).sunFlares) {
        const flares = (current as any).sunFlares;
        flares.rotation.y += dt * 0.05;
        flares.rotation.x += dt * 0.02;
        
        // Pulsate flare opacity for flickering effect
        const pulsate = Math.sin(clock.getElapsedTime() * 2) * 0.2 + 0.8;
        flares.material.opacity = pulsate;
      }
      
      // Animate fire layers for dynamic flame effect
      if ((current as any).fireLayers) {
        const layers = (current as any).fireLayers;
        const time = clock.getElapsedTime();
        
        layers.forEach((layer: any, i: number) => {
          // Different rotation speeds for each layer
          layer.rotation.y += dt * (0.1 + i * 0.05);
          layer.rotation.x += dt * (0.05 - i * 0.01);
          
          // Pulsating scale for breathing effect
          const scale = 1 + Math.sin(time * (1 + i * 0.5)) * 0.05;
          layer.scale.set(scale, scale, scale);
          
          // Opacity flickering
          const baseOpacity = [0.4, 0.3, 0.15][i];
          const flicker = Math.sin(time * (3 + i)) * 0.1;
          layer.material.opacity = baseOpacity + flicker;
        });
      }
      
      // Animate sun surface with emissive intensity pulsing
      if (current.star && mode === "earth") {
        const intensity = 2.5 + Math.sin(clock.getElapsedTime() * 1.5) * 0.3;
        (current.star.material as any).emissiveIntensity = intensity;
      }
      
      for (const m of current.planets) {
        const p: any = (m as any).userData;
        const e = Math.min(0.95, Math.max(0, p.e || 0));
        // Slow down planet orbital motion by factor of 10
        const orbitalSlowdown = 10;
        p.M += (timeFactor * dt * (2 * Math.PI)) / (p.period_days * orbitalSlowdown); // mean anomaly advance
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
      
      // In explore mode, check for star clicks first
      if (mode === "explore" && current.stars.length > 0) {
        const starHits = ray.intersectObjects(current.stars, false);
        if (starHits.length) {
          selectStar(starHits[0].object);
          return;
        }
      }
      
      // Check for planet clicks
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
      buildSystem,
      buildExplore,
      buildEarthCentered,
      selectPlanet,
      selectStar,
    } as any;

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
  }, []);

  // Build initial scene when data is loaded
  useEffect(() => {
    const s = sceneRef.current;
    if (!s || !data.length) return;
    if (mode === "explore") {
      s.buildExplore(data);
    } else if (mode === "earth") {
      s.buildEarthCentered(data);
    } else {
      // Start with first system in system mode
      s.buildSystem(data[0]);
    }
  }, [data.length, data, mode]);

  // Rebuild when toggles change
  useEffect(() => {
    const s = sceneRef.current;
    if (!s?.current.group || !data.length) return;
    if (mode === "explore") {
      // Rebuild explore scene using current options
      // Clear details on rebuild
      if (detailsRef.current)
        detailsRef.current.textContent =
          "Select a planet to see parameters and evidence.";
      s.buildExplore(data);
      return;
    }
    if (mode === "earth") {
      // Rebuild Earth-centered scene
      if (detailsRef.current)
        detailsRef.current.textContent =
          "Select a planet to see parameters and evidence.";
      s.buildEarthCentered(data);
      return;
    }
    const sys = data.find((x) => x.name === active) || data[0];
    // Rebuild to apply exSizes/logScale/showHZ changes
    // Clear details on rebuild
    if (detailsRef.current)
      detailsRef.current.textContent =
        "Select a planet to see parameters and evidence.";
    s.buildSystem(sys);
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
    setActive(sys.name);
    s.buildSystem(sys);
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
    } else if (mode === "earth") {
      const earthDistance = 150; // Same as in buildEarthCentered
      s.camera.position.set(earthDistance, 100, 300);
      s.controls.target.set(earthDistance, 0, 0); // Target Earth
      s.controls.maxDistance = 15000;
      s.controls.update();
    } else {
      s.camera.position.set(0, 120, 260);
      s.controls.target.set(0, 0, 0);
      s.controls.update();
    }
  }

  const speedVal = useMemo(() => speed.toFixed(2) + "×", [speed]);
  const sizeVal = useMemo(() => sizeScale.toFixed(2) + "×", [sizeScale]);
  const modeLabel = mode === "explore" ? "Explore" : mode === "earth" ? "Solar System" : "System";

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
    setMode((m) => {
      if (m === "earth") return "explore";
      if (m === "explore") return "system";
      return "earth";
    });
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
            logScale={logScale}
            showHZ={showHZ}
            onChange={(
              patch: Partial<{
                logScale: boolean;
                showHZ: boolean;
              }>
            ) => {
              if (patch.logScale !== undefined) setLogScale(patch.logScale);
              if (patch.showHZ !== undefined) setShowHZ(patch.showHZ);
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
            mode === "earth" ? (
              <>Sun at center • Earth orbits • Stars positioned by RA/Dec</>
            ) : mode === "explore" ? (
              <>Earth at center • Stars with planets distributed in 3D</>
            ) : (
              <>
                Size ∝ radius • Color ∝ T<sub>eff</sub>
              </>
            )
          }
          line2={
            mode === "earth" ? (
              <>Heliocentric view • Exoplanet systems at celestial coordinates</>
            ) : mode === "explore" ? (
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
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="9 3 9 9 3 9" />
              <polyline points="15 21 15 15 21 15" />
              <line x1="21" y1="3" x2="14" y2="10" />
              <line x1="3" y1="21" x2="10" y2="14" />
            </svg>
          ) : (
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
