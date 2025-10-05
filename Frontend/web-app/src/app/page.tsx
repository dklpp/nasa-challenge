"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { type Planet as PlanetT, type System as SystemT } from "./data";
import {
  teffColor,
  planetTempColor,
  scaleAU,
  trueAnomaly,
  celestialToCartesian,
} from "./utils";
import SearchBar from "../components/SearchBar";
import SystemsList from "../components/SystemsList";
import HUD from "../components/HUD";
import TitleBar from "../components/TitleBar";
import Legend from "../components/Legend";
import DetailsPanel from "../components/DetailsPanel";
import Toast from "../components/Toast";
import UploadModal from "../components/UploadModal";
import ModelPredictionModal from "../components/ModelPredictionModal";

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
  const dimOrbits = false;
  const [speed, setSpeed] = useState(1);
  const [toast, setToast] = useState("");
  const [active, setActive] = useState<string | null>(null);
  const [titleSub, setTitleSub] = useState("Click a system → click a planet");
  const [mode, setMode] = useState<"explore" | "system" | "earth">("earth");
  const [showDetails, setShowDetails] = useState(false);
  const [viewingFromSidebar, setViewingFromSidebar] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showModelPrediction, setShowModelPrediction] = useState(false);
  const [pendingSystem, setPendingSystem] = useState<SystemT | null>(null);
  const [customSystemNames, setCustomSystemNames] = useState<Set<string>>(
    new Set()
  );

  const centerRef = useRef<HTMLDivElement | null>(null);
  const detailsRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<any | null>(null);
  const speedRef = useRef(speed);
  const dataRef = useRef<SystemT[]>([]);

  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  useEffect(() => {
    dataRef.current = data;
  }, [data]);

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

  useEffect(() => {
    const s = sceneRef.current;
    if (!s || !centerRef.current) return;
    const w = centerRef.current.clientWidth;
    const h = centerRef.current.clientHeight;
    s.renderer.setSize(w, h, false);
    s.camera.aspect = w / h;
    s.camera.updateProjectionMatrix();
  }, [showDetails]);
  useEffect(() => {
    if (!centerRef.current) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    centerRef.current.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000508);
    const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 5000);
    camera.position.set(0, 120, 260);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 30;
    controls.maxDistance = 1500;

    let previousControlTarget = { x: controls.target.x, y: controls.target.y, z: controls.target.z };
    let userInteractionDetector = setInterval(() => {
      const dx = Math.abs(controls.target.x - previousControlTarget.x);
      const dy = Math.abs(controls.target.y - previousControlTarget.y);
      const dz = Math.abs(controls.target.z - previousControlTarget.z);
      
      if (dx > 0.5 || dy > 0.5 || dz > 0.5) {
        if ((current as any).zoomAnimation) {
          cancelAnimationFrame((current as any).zoomAnimation);
          (current as any).zoomAnimation = null;
        }
        previousControlTarget = { x: controls.target.x, y: controls.target.y, z: controls.target.z };
      }
    }, 100);

    const keys = {
      w: false,
      a: false,
      s: false,
      d: false,
      q: false,
      e: false,
      arrowup: false,
      arrowdown: false,
      arrowleft: false,
      arrowright: false,
      shift: false,
    };

    const onKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (key in keys) {
        (keys as any)[key] = true;
      }
      if (e.key === "Shift") {
        keys.shift = true;
      }
      if (e.key === "Escape") {
        if ((current as any).zoomAnimation) {
          cancelAnimationFrame((current as any).zoomAnimation);
          (current as any).zoomAnimation = null;
        }
        controls.target.set(0, 0, 0);
        controls.update();
        setShowDetails(false);
        showToast("Camera unlocked - free to move", 1200);
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (key in keys) {
        (keys as any)[key] = false;
      }
      if (e.key === "Shift") {
        keys.shift = false;
      }
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);

    scene.add(new THREE.AmbientLight(0x506080, 0.8));
    const globalLight = new THREE.DirectionalLight(0xffffff, 1.2);
    globalLight.position.set(300, 450, 350);
    globalLight.target.position.set(0, 0, 0);
    scene.add(globalLight);
    scene.add(globalLight.target);
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

    function createPlanetTexture(baseColor: any) {
      const canvas = document.createElement("canvas");
      canvas.width = 1024;
      canvas.height = 512;
      const ctx = canvas.getContext("2d")!;

      const gradient = ctx.createLinearGradient(0, 0, 0, 512);

      const h = { h: 0, s: 0, l: 0 };
      baseColor.getHSL(h);
      const numBands = 20;
      for (let i = 0; i < numBands; i++) {
        const t = i / numBands;

        const lightnessVar = 0.1 + Math.sin(t * Math.PI * 4) * 0.15;
        const saturationVar = Math.cos(t * Math.PI * 3) * 0.1;

        const color = new THREE.Color().setHSL(
          h.h + Math.sin(t * Math.PI * 2) * 0.05,
          Math.max(0, Math.min(1, h.s + saturationVar)),
          Math.max(0.2, Math.min(0.8, h.l + lightnessVar))
        );

        gradient.addColorStop(
          t,
          `rgb(${Math.floor(color.r * 255)}, ${Math.floor(
            color.g * 255
          )}, ${Math.floor(color.b * 255)})`
        );
      }

      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 1024, 512);

      ctx.globalCompositeOperation = "overlay";
      for (let band = 0; band < 12; band++) {
        const yBase = (band / 12) * 512;
        const amplitude = 20 + Math.random() * 30;
        const frequency = 0.005 + Math.random() * 0.005;
        const phase = Math.random() * Math.PI * 2;

        ctx.beginPath();
        ctx.moveTo(0, yBase);

        for (let x = 0; x <= 1024; x += 2) {
          const y = yBase + Math.sin(x * frequency + phase) * amplitude;
          ctx.lineTo(x, y);
        }

        ctx.lineTo(1024, yBase + 60);
        ctx.lineTo(0, yBase + 60);
        ctx.closePath();

        const bandColor = new THREE.Color().setHSL(
          h.h + Math.random() * 0.1 - 0.05,
          h.s * 0.6,
          0.7 + Math.random() * 0.2
        );
        ctx.fillStyle = `rgba(${Math.floor(bandColor.r * 255)}, ${Math.floor(
          bandColor.g * 255
        )}, ${Math.floor(bandColor.b * 255)}, 0.3)`;
        ctx.fill();
      }


      ctx.globalCompositeOperation = "multiply";
      for (let i = 0; i < 8; i++) {
        const x = Math.random() * 1024;
        const y = Math.random() * 512;
        const width = 100 + Math.random() * 150;
        const height = 40 + Math.random() * 60;

        const darkColor = new THREE.Color().setHSL(h.h, h.s, 0.3);
        const grd = ctx.createRadialGradient(x, y, 0, x, y, width);
        grd.addColorStop(
          0,
          `rgba(${Math.floor(darkColor.r * 255)}, ${Math.floor(
            darkColor.g * 255
          )}, ${Math.floor(darkColor.b * 255)}, 0.4)`
        );
        grd.addColorStop(1, "rgba(0, 0, 0, 0)");

        ctx.fillStyle = grd;
        ctx.fillRect(x - width, y - height, width * 2, height * 2);
      }

      const texture = new THREE.CanvasTexture(canvas);
      texture.needsUpdate = true;
      return texture;
    }


    function createStarTexture(baseColor: any) {
      const canvas = document.createElement("canvas");
      canvas.width = 1024;
      canvas.height = 512;
      const ctx = canvas.getContext("2d")!;

      const h = { h: 0, s: 0, l: 0 };
      baseColor.getHSL(h);



      const centerGrad = ctx.createRadialGradient(512, 256, 0, 512, 256, 512);
      

      const centerColor = new THREE.Color().setHSL(h.h, Math.max(0.3, h.s * 0.6), Math.min(0.95, h.l * 1.5));

      const midColor = new THREE.Color().setHSL(h.h, Math.max(0.5, h.s * 0.85), Math.min(0.8, h.l * 1.2));

      const edgeColor = new THREE.Color().setHSL(h.h + 0.02, Math.max(0.6, h.s * 1.0), Math.max(0.4, h.l * 0.7));
      
      centerGrad.addColorStop(0, `rgb(${Math.floor(centerColor.r * 255)}, ${Math.floor(centerColor.g * 255)}, ${Math.floor(centerColor.b * 255)})`);
      centerGrad.addColorStop(0.3, `rgb(${Math.floor(centerColor.r * 255)}, ${Math.floor(centerColor.g * 255)}, ${Math.floor(centerColor.b * 255)})`);
      centerGrad.addColorStop(0.7, `rgb(${Math.floor(midColor.r * 255)}, ${Math.floor(midColor.g * 255)}, ${Math.floor(midColor.b * 255)})`);
      centerGrad.addColorStop(1, `rgb(${Math.floor(edgeColor.r * 255)}, ${Math.floor(edgeColor.g * 255)}, ${Math.floor(edgeColor.b * 255)})`);
      ctx.fillStyle = centerGrad;
      ctx.fillRect(0, 0, 1024, 512);


      ctx.globalCompositeOperation = "overlay";
      for (let i = 0; i < 300; i++) {
        const x = Math.random() * 1024;
        const y = Math.random() * 512;
        const size = 8 + Math.random() * 20;
        const brightness = 0.5 + Math.random() * 0.5;
        
        const cellColor = new THREE.Color().setHSL(h.h, h.s * 0.7, brightness);
        const grd = ctx.createRadialGradient(x, y, 0, x, y, size);
        grd.addColorStop(0, `rgba(${Math.floor(cellColor.r * 255)}, ${Math.floor(cellColor.g * 255)}, ${Math.floor(cellColor.b * 255)}, 0.6)`);
        grd.addColorStop(1, "rgba(0, 0, 0, 0)");
        
        ctx.fillStyle = grd;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }


      ctx.globalCompositeOperation = "screen";
      for (let i = 0; i < 25; i++) {
        const x = Math.random() * 1024;
        const y = Math.random() * 512;
        const size = 30 + Math.random() * 60;
        
        const hotColor = new THREE.Color().setHSL(h.h + (Math.random() - 0.5) * 0.05, Math.max(0.6, h.s), Math.min(0.95, h.l * 1.3));
        const grd = ctx.createRadialGradient(x, y, 0, x, y, size);
        grd.addColorStop(0, `rgba(${Math.floor(hotColor.r * 255)}, ${Math.floor(hotColor.g * 255)}, ${Math.floor(hotColor.b * 255)}, 0.5)`);
        grd.addColorStop(0.5, `rgba(${Math.floor(hotColor.r * 255)}, ${Math.floor(hotColor.g * 255)}, ${Math.floor(hotColor.b * 255)}, 0.2)`);
        grd.addColorStop(1, "rgba(0, 0, 0, 0)");
        
        ctx.fillStyle = grd;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }


      ctx.globalCompositeOperation = "multiply";
      for (let i = 0; i < 15; i++) {
        const x = Math.random() * 1024;
        const y = Math.random() * 512;
        const size = 15 + Math.random() * 40;
        
        const spotColor = new THREE.Color().setHSL(h.h, h.s * 0.5, h.l * 0.4);
        const grd = ctx.createRadialGradient(x, y, 0, x, y, size);
        grd.addColorStop(0, `rgba(${Math.floor(spotColor.r * 255)}, ${Math.floor(spotColor.g * 255)}, ${Math.floor(spotColor.b * 255)}, 0.7)`);
        grd.addColorStop(0.6, `rgba(${Math.floor(spotColor.r * 255)}, ${Math.floor(spotColor.g * 255)}, ${Math.floor(spotColor.b * 255)}, 0.3)`);
        grd.addColorStop(1, "rgba(0, 0, 0, 0)");
        
        ctx.fillStyle = grd;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }

      const texture = new THREE.CanvasTexture(canvas);
      texture.needsUpdate = true;
      return texture;
    }
    const starField = new THREE.Group();

    function makeOrbitMesh(points: THREE.Vector3[], colorHex: number, thickness: number) {
      const curve = new THREE.CatmullRomCurve3(points, true);
      const geom  = new THREE.TubeGeometry(curve, Math.max(128, points.length), thickness, 8, true);
      const mat   = new THREE.MeshBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: 0.7,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(geom, mat);

      (mesh as any).raycast = () => {};
      return mesh;
    }


    function makeOrbitLine(points: THREE.Vector3[], colorHex: number, opacity = 0.85) {
      const geom = new THREE.BufferGeometry().setFromPoints(points);
      const mat = new THREE.LineBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity,
        linewidth: 1,
      });
      const line = new THREE.LineLoop(geom, mat);
      (line as any).raycast = () => {};
      return line;
    }

function computeOrbitScales(
  baseAU: number[],
  radii: number[],
  starRadius: number,
  gap = 16,
  firstOffset = 10
) {
  const items = baseAU.map((a, i) => ({ i, a, r: radii[i] || 0 })).sort((x, y) => x.a - y.a);
  const scales = baseAU.map(() => 1);
  let prev = Math.max(starRadius + firstOffset, starRadius * 1.15);
  for (const it of items) {
    const minForThis = Math.max(prev + gap + it.r * 1.6, starRadius + firstOffset);
    const target = Math.max(it.a, minForThis);
    scales[it.i] = target / Math.max(1e-6, it.a);
    prev = target;
  }
  return scales;
}


    function randFromString(s: string) {
      let h = 2166136261 >>> 0;
      for (let i = 0; i < s.length; i++) {
        h ^= s.charCodeAt(i);
        h = Math.imul(h, 16777619);
      }

      h += h << 13; h ^= h >>> 7; h += h << 3; h ^= h >>> 17; h += h << 5;
      return (h >>> 0) / 4294967295;
    }
    function randAngleDeg(name: string, salt: string, rangeDeg: number) {
      const r = randFromString(name + '|' + salt);
      return THREE.MathUtils.degToRad((r - 0.5) * 2 * rangeDeg);
    }
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
        const isWarm = Math.random() < (opts.warmChance ?? 0.18);
        const h = isWarm
          ? 0.06 + Math.random() * 0.04
          : 0.58 + Math.random() * 0.12;
        const s = 0.58 + Math.random() * 0.34;
        const l = 0.6 + Math.random() * 0.32;
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
        depthTest: true,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true,
        map: getDiscTexture(),
        alphaTest: 0.5,
      });
      const points = new THREE.Points(geom, mat);

      points.renderOrder = -1;
      return points;
    }

    starField.add(
      makeStars(80000, 2800, { size: 0.18, opacity: 0.1, warmChance: 0.1 })
    );
    starField.add(
      makeStars(60000, 2600, { size: 0.3, opacity: 0.18, warmChance: 0.11 })
    );
    starField.add(
      makeStars(50000, 2400, { size: 0.45, opacity: 0.25, warmChance: 0.13 })
    );
    starField.add(
      makeStars(35000, 2100, { size: 0.6, opacity: 0.32, warmChance: 0.15 })
    );
    starField.add(
      makeStars(25000, 1850, { size: 0.75, opacity: 0.4, warmChance: 0.17 })
    );
    starField.add(
      makeStars(18000, 1600, { size: 0.9, opacity: 0.48, warmChance: 0.19 })
    );
    starField.add(
      makeStars(12000, 1400, { size: 1.05, opacity: 0.54, warmChance: 0.21 })
    );

    starField.add(
      makeStars(8000, 1200, { size: 1.2, opacity: 0.62, warmChance: 0.23 })
    );
    starField.add(
      makeStars(5000, 1000, { size: 1.35, opacity: 0.68, warmChance: 0.25 })
    );
    starField.add(
      makeStars(3000, 850, { size: 1.5, opacity: 0.75, warmChance: 0.27 })
    );
    starField.add(
      makeStars(1500, 700, { size: 1.7, opacity: 0.82, warmChance: 0.29 })
    );

    starField.add(
      makeStars(800, 600, { size: 2.0, opacity: 0.88, warmChance: 0.31 })
    );
    starField.add(
      makeStars(400, 500, { size: 2.4, opacity: 0.94, warmChance: 0.33 })
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


function smoothCamTo(camPos: THREE.Vector3, targetPos: THREE.Vector3, duration = 1.2) {
  const startPos = camera.position.clone();
  const startTarget = new THREE.Vector3(controls.target.x, controls.target.y, controls.target.z);
  let progress = 0;

  if ((current as any).zoomAnimation) {
    cancelAnimationFrame((current as any).zoomAnimation);
    (current as any).zoomAnimation = null;
  }

  const animate = () => {
    const dtLocal = clock.getDelta();
    progress = Math.min(1, progress + dtLocal / duration);
    const eased = progress < 0.5 ? 2 * progress * progress : 1 - Math.pow(-2 * progress + 2, 2) / 2;

    camera.position.lerpVectors(startPos, camPos, eased);
    const newTarget = new THREE.Vector3().lerpVectors(startTarget, targetPos, eased);
    controls.target.copy(newTarget);
    controls.update();

    if (progress < 1) (current as any).zoomAnimation = requestAnimationFrame(animate);
    else (current as any).zoomAnimation = null;
  };

  (current as any).zoomAnimation = requestAnimationFrame(animate);
}

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

      g.rotation.set(
        randAngleDeg(sys.name, 'sys-x', 18),
        randAngleDeg(sys.name, 'sys-y', 18),
        randAngleDeg(sys.name, 'sys-z', 18)
      );





      const starCol = teffColor(sys.star.teff, THREE);


      const stellarRadius = sys.star.radius_rs ?? 1;

      const starSize = exSizes
        ? Math.max(4, Math.min(stellarRadius * 8, 50 + Math.log(stellarRadius) * 10))
        : Math.max(2, Math.min(stellarRadius * 2.5, 15 + Math.log(stellarRadius) * 4));
      const starGeom = new THREE.SphereGeometry(starSize, 64, 64);
      

      const starTexture = createStarTexture(starCol);
      
      const starMat = new THREE.MeshStandardMaterial({
        map: starTexture,
        emissive: starCol,
        emissiveIntensity: 0.8,
        color: starCol,
        roughness: 1.0,
        metalness: 0.0,
      });
      const starMesh = new THREE.Mesh(starGeom, starMat);
      starMesh.userData = { systemName: sys.name, type: "star" };
      g.add(starMesh);
      current.star = starMesh;
      current.stars.push(starMesh);
      

      const glowGeom1 = new THREE.SphereGeometry(starSize * 1.15, 32, 32);
      const glowMat1 = new THREE.MeshBasicMaterial({
        color: starCol,
        transparent: true,
        opacity: 0.3,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
      });
      const glow1 = new THREE.Mesh(glowGeom1, glowMat1);
      g.add(glow1);
      
      const glowGeom2 = new THREE.SphereGeometry(starSize * 1.35, 32, 32);
      const glowMat2 = new THREE.MeshBasicMaterial({
        color: starCol,
        transparent: true,
        opacity: 0.15,
        side: THREE.BackSide,
        blending: THREE.AdditiveBlending,
      });
      const glow2 = new THREE.Mesh(glowGeom2, glowMat2);
      g.add(glow2);
      
      const lamp = new THREE.PointLight(starCol.getHex(), 2.2, 0, 2);
      lamp.position.set(0, 0, 0);
      g.add(lamp);

      let maxAU = 0;

      const baseAU_list: number[] = [];
      const radii_list: number[] = [];
      sys.planets.forEach((p0) => {
        const p = { ...p0 } as Planet;
        const aU0 = scaleAU(Math.max(0.004, p.a_au), logScale);
        const rScene = exSizes
          ? Math.max(1.8, p.radius_rj ? p.radius_rj * 6 : (p.radius_re || 0) * 0.8)
          : Math.max(0.4, p.radius_rj ? p.radius_rj * 1.2 : (p.radius_re || 0) * 0.2);
        baseAU_list.push(aU0);
        radii_list.push(rScene);
      });
      const orbitScales_sys = computeOrbitScales(
        baseAU_list,
        radii_list,
        starSize,
        Math.max(16, Math.floor(starSize * 0.5)),
        Math.max(10, Math.floor(starSize * 0.6))
      );
      sys.planets.forEach((p0, idx) => {
        const p = { ...p0 } as Planet;
        const a = Math.max(0.004, p.a_au);
        const e = Math.min(0.95, Math.max(0, p.e || 0));
        const aU0 = scaleAU(a, logScale);
        const mult = orbitScales_sys[idx] ?? 1;
        const aU = aU0 * mult;
        const bU = aU * Math.sqrt(1 - e * e);
        const cU = Math.sqrt(Math.max(0, aU * aU - bU * bU));
        const inc = THREE.MathUtils.degToRad(p.incl || 0);
        maxAU = Math.max(maxAU, aU * (1 + e));

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
        const color = planetTempColor(p.pl_eqt, THREE);


        const planetTexture = createPlanetTexture(color);

        const mat = new THREE.MeshStandardMaterial({
          map: planetTexture,
          metalness: 0.2,
          roughness: 0.6,
        });
        const m = new THREE.Mesh(geom, mat);
        (m as any).userData = {
          ...p,
          aU,
          bU,
          cU,
          inc,
          orbitScale: mult,
          M: 0,
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
        current.planets.push(m);


        {
          const N = 128;
          const pts: THREE.Vector3[] = [];
          for (let i = 0; i <= N; i++) {
            const th = (i / N) * 2 * Math.PI;
            const x = aU * Math.cos(th) - cU;
            const y = bU * Math.sin(th);
            pts.push(new THREE.Vector3(x, 0, y));
          }
          const orbit = makeOrbitLine(pts, 0x5588dd, 0.85);
          orbit.rotation.set(inc, 0, 0);
          orbit.position.y = 0.001;
          (orbit as any).userData.planet = m;
          g.add(orbit);
          current.orbits.push(orbit);
        }
      });

      setActive(sys.name);
      setTitleSub(
        `${sys.planets.length} planet(s) • star T_eff ${sys.star.teff}K`
      );

      const dist = Math.min(1500, Math.max(120, 90 + maxAU * 0.6));
smoothCamTo(
  new THREE.Vector3(0, Math.min(400, 40 + maxAU * 0.25), dist),
  new THREE.Vector3(0, 0, 0),
  1.2
);
    }

    function buildEarthCentered(systems: Sys[], skipCameraAnimation = false) {


      clearSystem();
      const g = new THREE.Group();
      scene.add(g);
      current.group = g;


      const sunGeom = new THREE.SphereGeometry(7, 64, 64);


      const sunColor = new THREE.Color(0xffa500);
      const sunTexture = createStarTexture(sunColor);

      const sunMat = new THREE.MeshStandardMaterial({
        map: sunTexture,
        emissive: new THREE.Color(0xff4400),
        emissiveIntensity: 1.5,
        color: new THREE.Color(0xff6600),
        roughness: 1.0,
        metalness: 0,
      });

      const sunMesh = new THREE.Mesh(sunGeom, sunMat);
      sunMesh.userData = { systemName: "Solar System", type: "star" };
      g.add(sunMesh);
      current.star = sunMesh;
      current.stars.push(sunMesh);


      const flareGeom = new THREE.BufferGeometry();
      const flareCount = 2000;
      const flarePositions = new Float32Array(flareCount * 3);
      const flareColors = new Float32Array(flareCount * 3);
      const flareSizes = new Float32Array(flareCount);

      for (let i = 0; i < flareCount; i++) {

        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const radius = 10 + Math.random() * 3;

        flarePositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
        flarePositions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
        flarePositions[i * 3 + 2] = radius * Math.cos(phi);


        const colorMix = Math.random();
        const r = 1.0;
        const g = 0.3 + colorMix * 0.5;
        const b = colorMix * 0.2;

        flareColors[i * 3] = r;
        flareColors[i * 3 + 1] = g;
        flareColors[i * 3 + 2] = b;

        flareSizes[i] = Math.random() * 1.5 + 0.5;
      }

      flareGeom.setAttribute(
        "position",
        new THREE.BufferAttribute(flarePositions, 3)
      );
      flareGeom.setAttribute(
        "color",
        new THREE.BufferAttribute(flareColors, 3)
      );
      flareGeom.setAttribute("size", new THREE.BufferAttribute(flareSizes, 1));

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
      (current as any).sunFlares = flares;


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


      (current as any).fireLayers = [fire1, fire2, fire3];


      const sunLight = new THREE.PointLight(0xff7733, 5.0, 0, 1.2);
      sunLight.position.set(0, 0, 0);
      g.add(sunLight);


      const earthDistance = 150;
      const earthGeom = new THREE.SphereGeometry(6.4, 64, 32);


      const earthCanvas = document.createElement("canvas");
      earthCanvas.width = 512;
      earthCanvas.height = 256;
      const ctx = earthCanvas.getContext("2d")!;


      ctx.fillStyle = "#1a4d8f";
      ctx.fillRect(0, 0, 512, 256);


      ctx.fillStyle = "#2d5a2d";
      for (let i = 0; i < 150; i++) {
        const x = Math.random() * 512;
        const y = Math.random() * 256;
        const size = Math.random() * 60 + 20;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }


      ctx.fillStyle = "#4a7c4a";
      for (let i = 0; i < 100; i++) {
        const x = Math.random() * 512;
        const y = Math.random() * 256;
        const size = Math.random() * 40 + 10;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }


      ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
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
      earth.rotation.x = THREE.MathUtils.degToRad(23.5);
      g.add(earth);


      const earthGlow = new THREE.PointLight(0x3a7cff, 0.8, 200, 2);
      earthGlow.position.copy(earth.position);
      g.add(earthGlow);


      const earthOrbitGeom = new THREE.BufferGeometry();
      const earthOrbitPoints: any[] = [];
      for (let i = 0; i <= 128; i++) {
        const angle = (i / 128) * Math.PI * 2;
        earthOrbitPoints.push(
          new THREE.Vector3(
            earthDistance * Math.cos(angle),
            0,
            earthDistance * Math.sin(angle)
          )
        );
      }
      earthOrbitGeom.setFromPoints(earthOrbitPoints);
      const earthOrbitMat = new THREE.LineBasicMaterial({
        color: 0x5588dd,
        transparent: true,
        opacity: 0.85,
        linewidth: 2,
      });
      const earthOrbit = new THREE.LineLoop(earthOrbitGeom, earthOrbitMat);
      g.add(earthOrbit);


      const systemsWithCoords = systems.filter(
        (s) => s.ra !== undefined && s.dec !== undefined
      );

      systemsWithCoords.forEach((sys) => {
        if (sys.ra === undefined || sys.dec === undefined) return;


        const isCustomSystem = customSystemNames.has(sys.name);


        const distance = sys.distance_pc || 100;
        const pos = celestialToCartesian(sys.ra, sys.dec, distance, 10);

        const sg = new THREE.Group();
        sg.position.set(pos.x, pos.y, pos.z);

        sg.rotation.set(
          randAngleDeg(sys.name, 'ec-x', 22),
          randAngleDeg(sys.name, 'ec-y', 22),
          randAngleDeg(sys.name, 'ec-z', 22)
        );
        g.add(sg);



        const starCol = teffColor(sys.star.teff, THREE);
        const stellarRadius = sys.star.radius_rs ?? 1;
        


        const starSize = exSizes
          ? Math.max(2, Math.min(stellarRadius * 3.5, 20 + Math.log(stellarRadius) * 7))
          : Math.max(1.5, Math.min(stellarRadius * 1.5, 10 + Math.log(stellarRadius) * 3));
        const starGeom = new THREE.SphereGeometry(starSize, 32, 32);
        

        const starTexture = createStarTexture(starCol);
        
        const starMat = new THREE.MeshStandardMaterial({
          map: starTexture,
          emissive: starCol,
          emissiveIntensity: isCustomSystem ? 1.2 : 0.8,
          color: starCol,
          roughness: 1.0,
          metalness: 0.0,
        });
        const starMesh = new THREE.Mesh(starGeom, starMat);
        starMesh.userData = {
          systemName: sys.name,
          type: "star",
          isCustom: isCustomSystem,
        };
        sg.add(starMesh);
        current.stars.push(starMesh);


        if (isCustomSystem) {

          const glowGeom = new THREE.SphereGeometry(starSize * 2, 16, 12);
          const glowMat = new THREE.MeshBasicMaterial({
            color: 0x00ff88,
            transparent: true,
            opacity: 0.3,
            side: THREE.BackSide,
            blending: THREE.AdditiveBlending,
          });
          const glowMesh = new THREE.Mesh(glowGeom, glowMat);
          sg.add(glowMesh);
          (sg as any).userData.customGlow = glowMesh;


          const ringGeom = new THREE.RingGeometry(
            starSize * 2.5,
            starSize * 3,
            32
          );
          const ringMat = new THREE.MeshBasicMaterial({
            color: 0x00ffaa,
            transparent: true,
            opacity: 0.6,
            side: THREE.DoubleSide,
            blending: THREE.AdditiveBlending,
          });
          const ring = new THREE.Mesh(ringGeom, ringMat);
          ring.rotation.x = Math.PI / 2;
          sg.add(ring);
          (sg as any).userData.customRing = ring; 


          const customLight = new THREE.PointLight(0x00ff88, 2.0, 100, 2);
          customLight.position.set(0, 0, 0);
          sg.add(customLight);
        }

const baseAU_list_ec: number[] = [];
const radii_list_ec: number[] = [];
const localScale = 50;
sys.planets.forEach((p0) => {
  const p = { ...p0 } as Planet;
  const aU0 = Math.max(0.004, p.a_au) * localScale;
  const rSceneTmp = exSizes
    ? Math.max(2.5, p.radius_rj ? p.radius_rj * 8 : (p.radius_re || 0) * 1.5)
    : Math.max(1.0, p.radius_rj ? p.radius_rj * 2.5 : (p.radius_re || 0) * 0.5);
  baseAU_list_ec.push(aU0);
  radii_list_ec.push(rSceneTmp);
});
const orbitScales_ec = computeOrbitScales(
  baseAU_list_ec,
  radii_list_ec,
  starSize,
  Math.max(20, Math.floor(starSize * 0.8)),
  Math.max(12, Math.floor(starSize * 0.7))
);

        sys.planets.forEach((p0, idx) => {
          const p = { ...p0 } as Planet;
          const a = Math.max(0.004, p.a_au);
          const e = Math.min(0.95, Math.max(0, p.e || 0));

          const localScale = 50;
const aU0 = a * localScale;
const mult = orbitScales_ec[idx] ?? 1;
const aU = aU0 * mult;
          const bU = aU * Math.sqrt(1 - e * e);
          const cU = Math.sqrt(Math.max(0, aU * aU - bU * bU));
          const inc = THREE.MathUtils.degToRad(p.incl || 0);


          const rScene = exSizes
            ? Math.max(
                2.5,
                p.radius_rj ? p.radius_rj * 8 : (p.radius_re || 0) * 1.5
              )
            : Math.max(
                1.0,
                p.radius_rj ? p.radius_rj * 2.5 : (p.radius_re || 0) * 0.5
              );
          const geom = new THREE.SphereGeometry(rScene, 16, 12);
          const color = planetTempColor(p.pl_eqt, THREE);


          const planetTexture = createPlanetTexture(color);

          const mat = new THREE.MeshStandardMaterial({
            map: planetTexture,
            metalness: 0.2,
            roughness: 0.6,
          });
          const m = new THREE.Mesh(geom, mat);
          (m as any).userData = {
            ...p,
            aU,
            bU,
            cU,
            inc,
            orbitScale: mult,
            M: 0,
            name: `${sys.name} ${p.name}`,
            systemRA: sys.ra,
            systemDec: sys.dec,
            systemDistance: distance,
            localScale: 50,
          };
          const v0 = trueAnomaly((m as any).userData.M, e);
          const r0 = (aU * (1 - e * e)) / (1 + e * Math.cos(v0));
          const x0 = r0 * Math.cos(v0);
          const y0 = r0 * Math.sin(v0);
          m.position
            .set(x0, 0, y0)
            .applyAxisAngle(new THREE.Vector3(1, 0, 0), inc);
          sg.add(m);
          current.planets.push(m as any);


          {
            const N = 128;
            const pts: THREE.Vector3[] = [];
            for (let j = 0; j <= N; j++) {
              const th = (j / N) * 2 * Math.PI;
              const ox = aU * Math.cos(th) - cU;
              const oy = bU * Math.sin(th);
              pts.push(new THREE.Vector3(ox, 0, oy));
            }
            const orbit = makeOrbitLine(pts, 0x5588dd, 0.85);
            orbit.rotation.set(inc, 0, 0);
            orbit.position.y = 0.001;
            (orbit as any).userData.planet = m;
            sg.add(orbit);
            current.orbits.push(orbit as any);
          }
        });
      });

      setActive("Solar System");
      setTitleSub(
        `Sun + Earth + ${systemsWithCoords.length} exoplanet systems`
      );


      controls.maxDistance = 15000;
      

      if (!skipCameraAnimation) {
        smoothCamTo(new THREE.Vector3(200, 250, 500), new THREE.Vector3(0, 0, 0), 1.2);
      }
    }

    function selectPlanet(m: any) {
      const p: any = (m as any).userData;

      current.orbits.forEach((o: any) => {
        (o.material as any).opacity = 0.85;
        (o.material as any).color.setHex(0x5588dd);
      });
      const wp = new THREE.Vector3();
      (m as any).getWorldPosition(wp);
      const camTarget = wp.clone();
      

      const zoomDistance = Math.max(100, p.aU * 0.4);
      const heightOffset = 50 + p.aU * 0.12;
      

      const directionToCamera = new THREE.Vector3()
        .subVectors(camera.position, wp)
        .normalize();
      

      if (directionToCamera.length() < 0.1) {
        directionToCamera.set(0, 0.3, 1).normalize();
      }
      

      const camPos = wp.clone().add(
        directionToCamera.multiplyScalar(zoomDistance)
      );
      camPos.y += heightOffset;


      const startPos = camera.position.clone();
      const startTarget = new THREE.Vector3(
        controls.target.x,
        controls.target.y,
        controls.target.z
      );

      let progress = 0;
      const zoomDuration = 0.8;
      let animationId: number | null = null;

      const animateZoom = () => {
        progress += clock.getDelta() / zoomDuration;
        progress = Math.min(progress, 1);


        const eased =
          progress < 0.5
            ? 2 * progress * progress
            : 1 - Math.pow(-2 * progress + 2, 2) / 2;


        camera.position.lerpVectors(startPos, camPos, eased);


        const newTarget = new THREE.Vector3().lerpVectors(
          startTarget,
          camTarget,
          eased
        );
        controls.target.set(newTarget.x, newTarget.y, newTarget.z);

        controls.update();

        if (progress < 1) {
          animationId = requestAnimationFrame(animateZoom);
        } else {
          animationId = null;
        }
      };


      if ((current as any).zoomAnimation) {
        cancelAnimationFrame((current as any).zoomAnimation);
      }
      
      animateZoom();
      (current as any).zoomAnimation = animationId;


      setShowDetails(true);
      

      setTimeout(() => {
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
      }, 0);
    }

    function selectStar(starMesh: any) {
      const systemName = starMesh.userData.systemName;

      setActive(systemName);


      const wp = new THREE.Vector3();
      starMesh.getWorldPosition(wp);


      const sys = dataRef.current.find((s) => s.name === systemName);
      

      let maxAU = 0;
      if (sys) {
        sys.planets.forEach((p) => {
          const a = Math.max(0.004, p.a_au);
          const e = Math.min(0.95, Math.max(0, p.e || 0));
          const aU = scaleAU(a, logScale);

          maxAU = Math.max(maxAU, aU * (1 + e));
        });
      }
      

      const camTarget = wp.clone();
      


      const baseDistance = 200;
      const orbitBasedDistance = maxAU > 0 ? Math.max(baseDistance, maxAU * 1.8) : baseDistance;
      const distance = Math.min(1500, orbitBasedDistance);
      

      const heightOffset = maxAU > 0 ? Math.min(400, 80 + maxAU * 0.35) : 80;
      

      const directionToCamera = new THREE.Vector3()
        .subVectors(camera.position, wp)
        .normalize();
      

      if (directionToCamera.length() < 0.1) {
        directionToCamera.set(0, 0.3, 1).normalize();
      }
      

      const camPos = wp.clone().add(
        directionToCamera.multiplyScalar(distance)
      );
      camPos.y += heightOffset;


      const startPos = camera.position.clone();
      const startTarget = new THREE.Vector3(
        controls.target.x,
        controls.target.y,
        controls.target.z
      );

      let progress = 0;
      const zoomDuration = 0.1;
      let animationId: number | null = null;

      const animateZoom = () => {
        progress += clock.getDelta() / zoomDuration;
        progress = Math.min(progress, 1);


        const eased =
          progress < 0.5
            ? 2 * progress * progress
            : 1 - Math.pow(-2 * progress + 2, 2) / 2;


        camera.position.lerpVectors(startPos, camPos, eased);


        const newTarget = new THREE.Vector3().lerpVectors(
          startTarget,
          camTarget,
          eased
        );
        controls.target.set(newTarget.x, newTarget.y, newTarget.z);

        controls.update();

        if (progress < 1) {
          animationId = requestAnimationFrame(animateZoom);
        } else {
          animationId = null;
        }
      };


      if ((current as any).zoomAnimation) {
        cancelAnimationFrame((current as any).zoomAnimation);
      }

      animateZoom();
      (current as any).zoomAnimation = animationId;


      setShowDetails(true);



      setTimeout(() => {
        if (detailsRef.current) {
          const rows: [string, string][] = [["Star", systemName]];

          if (sys) {
            rows.push(["Effective Temperature", `${sys.star.teff} K`]);

            if (sys.star.radius_rs !== undefined) {
              rows.push([
                "Stellar Radius",
                `${sys.star.radius_rs.toFixed(3)} R☉`,
              ]);
            }

            if (sys.star.mass_ms !== undefined) {
              rows.push(["Stellar Mass", `${sys.star.mass_ms.toFixed(3)} M☉`]);
            }

            rows.push(["Number of Planets", `${sys.planets.length}`]);
          }

          detailsRef.current.innerHTML =
            rows
              .map(
                ([k, v]) =>
                  `<div class="row"><div>${k}</div><div>${v}</div></div>`
              )
              .join("") +
            `<div style="margin-top:12px; text-align: center; color: #888;">Click on a planet to see its details</div>`;
        }
      }, 0);


      setTitleSub("Star selected");


      showToast(`Selected ${systemName}`);
    }

    function tick() {
      sceneRef.current!.raf = requestAnimationFrame(tick);
      const dt = clock.getDelta();


      const moveSpeed = keys.shift ? 100 : 50;
      const moveAmount = moveSpeed * dt;


      const forward = new THREE.Vector3();
      camera.getWorldDirection(forward);
      forward.y = 0;
      forward.normalize();

      const right = new THREE.Vector3();
      right.crossVectors(forward, camera.up).normalize();


      const isMoving = keys.w || keys.s || keys.a || keys.d || keys.q || keys.e || 
                       keys.arrowup || keys.arrowdown || keys.arrowleft || keys.arrowright;


      if (keys.w || keys.arrowup) {
        camera.position.addScaledVector(forward, moveAmount);

        if (isMoving) {
          const newTarget = forward.clone().multiplyScalar(moveAmount);
          controls.target.set(
            controls.target.x + newTarget.x,
            controls.target.y + newTarget.y,
            controls.target.z + newTarget.z
          );
        }
      }
      if (keys.s || keys.arrowdown) {
        camera.position.addScaledVector(forward, -moveAmount);
        if (isMoving) {
          const newTarget = forward.clone().multiplyScalar(-moveAmount);
          controls.target.set(
            controls.target.x + newTarget.x,
            controls.target.y + newTarget.y,
            controls.target.z + newTarget.z
          );
        }
      }
      if (keys.a || keys.arrowleft) {
        camera.position.addScaledVector(right, -moveAmount);
        if (isMoving) {
          const newTarget = right.clone().multiplyScalar(-moveAmount);
          controls.target.set(
            controls.target.x + newTarget.x,
            controls.target.y + newTarget.y,
            controls.target.z + newTarget.z
          );
        }
      }
      if (keys.d || keys.arrowright) {
        camera.position.addScaledVector(right, moveAmount);
        if (isMoving) {
          const newTarget = right.clone().multiplyScalar(moveAmount);
          controls.target.set(
            controls.target.x + newTarget.x,
            controls.target.y + newTarget.y,
            controls.target.z + newTarget.z
          );
        }
      }


      if (keys.q) {
        camera.position.y -= moveAmount;
        if (isMoving) {
          controls.target.set(
            controls.target.x,
            controls.target.y - moveAmount,
            controls.target.z
          );
        }
      }
      if (keys.e) {
        camera.position.y += moveAmount;
        if (isMoving) {
          controls.target.set(
            controls.target.x,
            controls.target.y + moveAmount,
            controls.target.z
          );
        }
      }

      controls.update();
      const timeFactor = speedRef.current;


      if ((current as any).sunFlares) {
        const flares = (current as any).sunFlares;
        flares.rotation.y += dt * 0.05;
        flares.rotation.x += dt * 0.02;


        const pulsate = Math.sin(clock.getElapsedTime() * 2) * 0.2 + 0.8;
        flares.material.opacity = pulsate;
      }


      if ((current as any).fireLayers) {
        const layers = (current as any).fireLayers;
        const time = clock.getElapsedTime();

        layers.forEach((layer: any, i: number) => {

          layer.rotation.y += dt * (0.1 + i * 0.05);
          layer.rotation.x += dt * (0.05 - i * 0.01);


          const scale = 1 + Math.sin(time * (1 + i * 0.5)) * 0.05;
          layer.scale.set(scale, scale, scale);


          const baseOpacity = [0.4, 0.3, 0.15][i];
          const flicker = Math.sin(time * (3 + i)) * 0.1;
          layer.material.opacity = baseOpacity + flicker;
        });
      }


      if (current.star && mode === "earth") {
        const intensity = 2.5 + Math.sin(clock.getElapsedTime() * 1.5) * 0.3;
        (current.star.material as any).emissiveIntensity = intensity;
      }


      if (current.group) {
        const time = clock.getElapsedTime();
        current.group.children.forEach((child: any) => {
          if (child.userData.customGlow) {

            const glowPulse = Math.sin(time * 2) * 0.15 + 0.3;
            child.userData.customGlow.material.opacity = glowPulse;


            const scalePulse = 1 + Math.sin(time * 2) * 0.1;
            child.userData.customGlow.scale.set(
              scalePulse,
              scalePulse,
              scalePulse
            );
          }

          if (child.userData.customRing) {

            child.userData.customRing.rotation.z += dt * 0.5;


            const ringPulse = Math.sin(time * 3) * 0.2 + 0.6;
            child.userData.customRing.material.opacity = ringPulse;
          }
        });
      }


      if (current.stars && current.stars.length > 0) {
        const time = clock.getElapsedTime();
        current.stars.forEach((star: any, index: number) => {

          if (star === hoveredStar) return;
          
          const phaseOffset = index * 0.7;
          

          const pulseSpeed = 0.4 + (index % 5) * 0.1;
          const pulse = Math.sin(time * pulseSpeed + phaseOffset);
          

          const baseIntensity = star.userData.type === "star" && mode === "earth" ? 1.5 : 0.8;
          const intensityVariation = pulse * 0.12 + 0.88;
          const newIntensity = baseIntensity * intensityVariation;
          
          if (star.material && star.material.emissiveIntensity !== undefined) {
            star.material.emissiveIntensity = newIntensity;

            if (star.userData.originalIntensity === undefined) {
              star.userData.originalIntensity = baseIntensity;
            }
          }
        });
      }

      for (const m of current.planets) {
  const p: any = (m as any).userData;
  const e = Math.min(0.95, Math.max(0, p.e || 0));
  const orbitalSlowdown = 20;
  p.M += (speedRef.current * dt * (2 * Math.PI)) / (p.period_days * orbitalSlowdown);
  const v = trueAnomaly(p.M, e);



const baseA = p.localScale ? p.a_au * p.localScale : scaleAU(p.a_au, logScale);
const a = (p.aU = baseA * (p.orbitScale ?? 1));
const b = (p.bU = a * Math.sqrt(1 - e * e));
const c = (p.cU = Math.sqrt(Math.max(0, a * a - b * b)));

  const r = (a * (1 - e * e)) / (1 + e * Math.cos(v));
  const x = r * Math.cos(v);
  const y = r * Math.sin(v);

  m.position
    .set(x, 0, y)
    .applyAxisAngle(new THREE.Vector3(1, 0, 0), THREE.MathUtils.degToRad((p.incl as number) || 0));
}
      renderer.render(scene, camera);
    }


    let hoveredStar: any = null;
    let hoveredPlanet: any = null;


    renderer.domElement.addEventListener("pointermove", (e: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      const mouse = new THREE.Vector2();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      ray.setFromCamera(mouse, camera);


      const starHits = ray.intersectObjects(current.stars, false);
      
      if (starHits.length) {
        const newHoveredStar = starHits[0].object as any;
        

        if (newHoveredStar !== hoveredStar) {

          if (hoveredStar && hoveredStar.material) {
            hoveredStar.material.emissiveIntensity = hoveredStar.userData.originalIntensity || 0.8;
          }
          

          if (newHoveredStar.material && newHoveredStar.material.emissiveIntensity !== undefined) {

            if (newHoveredStar.userData.originalIntensity === undefined) {
              newHoveredStar.userData.originalIntensity = newHoveredStar.material.emissiveIntensity;
            }

            newHoveredStar.material.emissiveIntensity = (newHoveredStar.userData.originalIntensity || 0.8) * 2.5;
          }
          
          hoveredStar = newHoveredStar;
          renderer.domElement.style.cursor = "pointer";
        }
        

        if (hoveredPlanet && hoveredPlanet.material) {
          hoveredPlanet.material.emissive.multiplyScalar(1 / 1.5);
          hoveredPlanet = null;
        }
      } else {
        const planetHits = ray.intersectObjects(current.planets, false);
        
        if (planetHits.length) {
          const newHoveredPlanet = planetHits[0].object as any;
          

          if (newHoveredPlanet !== hoveredPlanet) {

            if (hoveredPlanet && hoveredPlanet.material) {
              hoveredPlanet.material.emissive.multiplyScalar(1 / 1.5);
            }
            

            if (newHoveredPlanet.material && newHoveredPlanet.material.emissive) {
              newHoveredPlanet.material.emissive.multiplyScalar(1.5);
            }
            
            hoveredPlanet = newHoveredPlanet;
          }
          
          renderer.domElement.style.cursor = "pointer";
        } else {
          renderer.domElement.style.cursor = "default";
          

          if (hoveredPlanet && hoveredPlanet.material) {
            hoveredPlanet.material.emissive.multiplyScalar(1 / 1.5);
            hoveredPlanet = null;
          }
        }
        

        if (hoveredStar && hoveredStar.material) {
          hoveredStar.material.emissiveIntensity = hoveredStar.userData.originalIntensity || 0.8;
          hoveredStar = null;
        }
      }
    });


    renderer.domElement.addEventListener("pointerdown", (e: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      const mouse = new THREE.Vector2();
      mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      ray.setFromCamera(mouse, camera);


      const starHits = ray.intersectObjects(current.stars, false);
      if (starHits.length) {
        selectStar(starHits[0].object as any);
        return;
      }


      const hits = ray.intersectObjects(current.planets, false);
      if (hits.length) selectPlanet(hits[0].object as any);
    });


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
      buildEarthCentered,
      buildExplore: buildEarthCentered,
      selectPlanet,
      selectStar,
      smoothCamTo, 
    } as any;

    tick();

    return () => {
      cancelAnimationFrame(sceneRef.current?.raf || 0);
      clearInterval(userInteractionDetector);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      renderer.dispose();

      scene.remove(globalLight);
      scene.remove(globalLight.target);
      starField.children.forEach((child: any) => {
        child.geometry?.dispose?.();
        child.material?.dispose?.();
      });
      clearSystem();
      sceneRef.current = null;
    };

  }, []);


  useEffect(() => {
    const s = sceneRef.current;
    if (!s || !data.length) return;
    if (mode === "explore") {
      s.buildExplore(data);
    } else if (mode === "earth") {
      s.buildEarthCentered(data);
    } else {

      s.buildSystem(data[0]);
    }
  }, [data.length, data, mode]);


  useEffect(() => {
    const s = sceneRef.current;
    if (!s?.current.group || !data.length) return;
    if (mode === "earth") {

      if (detailsRef.current)
        detailsRef.current.textContent =
          "Select a planet to see parameters and evidence.";
      s.buildEarthCentered(data);
      return;
    }
    const sys = data.find((x) => x.name === active) || data[0];

    if (detailsRef.current)
      detailsRef.current.textContent =
        "Select a planet to see parameters and evidence.";
    s.buildSystem(sys);

  }, [exSizes, logScale, mode, dimOrbits]);

  useEffect(() => {
    const s = sceneRef.current;
    if (!s) return;
    s.current.orbits.forEach((o: any) => {
      (o.material as any).opacity = 0.85;
      (o.material as any).color.setHex(0x5588dd);
    });
  }, [dimOrbits]);

  function handleBuild(sys: Sys) {
    const s = sceneRef.current;
    if (!s) return;

    if (mode !== "system") setMode("system");
    setActive(sys.name);
    setViewingFromSidebar(true); 
    s.buildSystem(sys);
    
    s.smoothCamTo(new THREE.Vector3(0, 120, 260), new THREE.Vector3(0, 0, 0), 1.2);
    
    showToast(`Loaded ${sys.name}`);
  }

  function resetCam() {
    const s = sceneRef.current;
if (!s) return;

if (mode === "explore") {
  s.smoothCamTo(new THREE.Vector3(0, 600, 1200), new THREE.Vector3(0, 0, 0), 1.0);
  s.controls.maxDistance = 2500;
} else if (mode === "earth") {
  const earthDistance = 150;
  s.smoothCamTo(new THREE.Vector3(earthDistance, 100, 300), new THREE.Vector3(earthDistance, 0, 0), 1.0);
  s.controls.maxDistance = 15000;
} else {
  s.smoothCamTo(new THREE.Vector3(0, 120, 260), new THREE.Vector3(0, 0, 0), 1.0);
}
  }

  function backToMainView() {
    const s = sceneRef.current;
    if (!s) return;

    setViewingFromSidebar(false);
    setActive(null);

    setMode("earth");

    const earthDistance = 150;
    s.camera.position.set(earthDistance, 100, 300);
    s.controls.target.set(earthDistance, 0, 0);
    s.controls.maxDistance = 15000;
    s.controls.update();

    s.buildEarthCentered(data);
  }

  function handleUploadSystem(system: SystemT) {
    setPendingSystem(system);
    setShowUploadModal(false);
    
    setTimeout(() => {
      setShowModelPrediction(true);
    }, 100);
  }

  function handleShowInSpace() {
    if (!pendingSystem) return;
    
    const updatedData = [...data, pendingSystem];
    setData(updatedData);

    setCustomSystemNames((prev) => new Set(prev).add(pendingSystem.name));

    showToast(
      `✨ ${pendingSystem.name} added! Look for the glowing green highlight in the map!`,
      3000
    );

    const s = sceneRef.current;
    if (s && mode === "earth") {
      setTimeout(() => {
        s.buildEarthCentered(updatedData, true); 
      }, 50);
    }

    setPendingSystem(null);
  }

  const speedVal = useMemo(() => speed.toFixed(2) + "×", [speed]);
  const modeLabel = mode === "earth" ? "Solar System" : "System";

  const [isFS, setIsFS] = useState(false);
  useEffect(() => {
    const onFS = () => setIsFS(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", onFS);
    return () => document.removeEventListener("fullscreenchange", onFS);
  }, []);
  function toggleFullscreen() {
    const el: any = document.documentElement as any;
    if (!document.fullscreenElement) {
      (
        el.requestFullscreen ||
        el.webkitRequestFullscreen ||
        el.msRequestFullscreen
      )?.call(el);
    } else {
      (
        document.exitFullscreen ||
        (document as any).webkitExitFullscreen ||
        (document as any).msExitFullscreen
      )?.call(document);
    }
  }

  return (
    <div id="app" className={showDetails ? "" : "noRight"}>
      <aside id="left">
        <div className="leftSticky">
          <h1>Systems</h1>
          <SearchBar q={q} onChange={setQ} />
        </div>
        <div className="list-container">
          {loading ? (
            <div className="list">
              <div className="sys">Loading…</div>
            </div>
          ) : filtered.length ? (
            <SystemsList
              systems={filtered}
              onSelect={handleBuild}
              customSystemNames={customSystemNames}
            />
          ) : (
            <div className="list">
              <div className="sys">No systems</div>
            </div>
          )}
        </div>
        <button
          className="add-system-btn text-md"
          onClick={() => setShowUploadModal(true)}
          title="Add custom system"
        >
          CHECK YOUR EXOPLANET CANDIDATE
        </button>
      </aside>

      <main id="center" ref={centerRef}>
        <HUD
          speed={speed}
          onSpeed={setSpeed}
          speedVal={speedVal}
          onReset={resetCam}
          modeLabel={modeLabel}
          onBackToMain={backToMainView}
          showBackButton={viewingFromSidebar}
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
            ) : (
              <>
                Size ∝ radius • Color ∝ T<sub>eff</sub>
              </>
            )
          }
          line2={
            mode === "earth" ? (
              <>
                Heliocentric view • Press ESC to unlock camera
              </>
            ) : (
              <>Speed via Kepler's 2nd law • Press ESC to unlock camera</>
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
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="9 3 9 9 3 9" />
              <polyline points="15 21 15 15 21 15" />
              <line x1="21" y1="3" x2="14" y2="10" />
              <line x1="3" y1="21" x2="10" y2="14" />
            </svg>
          ) : (
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
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

      <UploadModal
        isOpen={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        onUpload={handleUploadSystem}
      />

      <ModelPredictionModal
        isOpen={showModelPrediction}
        onClose={() => {
          setShowModelPrediction(false);
          setPendingSystem(null);
        }}
        onShowInSpace={handleShowInSpace}
      />
    </div>
  );
}
