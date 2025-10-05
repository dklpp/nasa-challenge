// Scene scale
export const AU_TO_U = 120;

// Map stellar Teff (3000–8000K) to a blue→red-ish color via HSL
export function teffColor(T: number, THREE: typeof import('three')): any{
  const clamped = Math.max(2500, Math.min(10000, T));
  
  let h: number, s: number, l: number;
  
  if (clamped < 3500) {
    // Cool red stars (M-class)
    const t = (clamped - 2500) / 1000;
    h = 0.0 + t * 0.05; // 0.0 to 0.05 (red to orange-red)
    s = 0.9 - t * 0.2;
    l = 0.5;
  } else if (clamped < 5000) {
    // Orange stars (K-class)
    const t = (clamped - 3500) / 1500;
    h = 0.05 + t * 0.08; // 0.05 to 0.13 (orange-red to orange-yellow)
    s = 0.7 - t * 0.1;
    l = 0.55 + t * 0.05;
  } else if (clamped < 6000) {
    // Yellow stars (G-class, like our Sun)
    const t = (clamped - 5000) / 1000;
    h = 0.13 + t * 0.03; // 0.13 to 0.16 (yellow-orange to yellow)
    s = 0.6 - t * 0.3;
    l = 0.60 + t * 0.1;
  } else if (clamped < 7500) {
    // White to blue-white stars (F-class)
    const t = (clamped - 6000) / 1500;
    h = 0.16 + t * 0.37; // 0.16 to 0.53 (yellow to light blue)
    s = 0.3 - t * 0.2;
    l = 0.70 + t * 0.1;
  } else {
    // Blue-white to blue stars (A, B, O-class)
    const t = Math.min((clamped - 7500) / 2500, 1.0);
    h = 0.53 + t * 0.07; // 0.53 to 0.60 (light blue to blue)
    s = 0.1 + t * 0.4;
    l = 0.75 + t * 0.05;
  }
  
  return new (THREE as any).Color().setHSL(h, s, l);
}

// Map planet equilibrium temperature (pl_eqt) to color
// Cold planets (< 200K): blue/purple
// Temperate planets (200-400K): cyan/green/yellow
// Hot planets (> 400K): orange/red
export function planetTempColor(T: number | undefined, THREE: typeof import('three')): any {
  if (T === undefined || T === null || !isFinite(T)) {
    // Default gray color for planets without temperature data
    return new (THREE as any).Color(0x888888);
  }
  
  // Temperature ranges (in Kelvin)
  // Venus: ~735K, Earth: ~255K, Mars: ~210K, Jupiter: ~110K, Neptune: ~55K
  // Hot Jupiters: 1000-2500K
  
  const clamped = Math.max(0, Math.min(2500, T));
  
  let h: number, s: number, l: number;
  
  if (clamped < 100) {
    // Very cold (0-100K): Deep blue to purple
    h = 0.65 - (clamped / 100) * 0.1; // 0.65 to 0.55 (blue to purple-blue)
    s = 0.8;
    l = 0.4;
  } else if (clamped < 200) {
    // Cold (100-200K): Purple-blue to bright blue
    const t = (clamped - 100) / 100;
    h = 0.55 + t * 0.05; // 0.55 to 0.60 (purple-blue to blue)
    s = 0.8;
    l = 0.45;
  } else if (clamped < 300) {
    // Cool temperate (200-300K): Blue to cyan
    const t = (clamped - 200) / 100;
    h = 0.60 - t * 0.10; // 0.60 to 0.50 (blue to cyan)
    s = 0.7;
    l = 0.5;
  } else if (clamped < 400) {
    // Warm temperate (300-400K): Cyan to yellow-green
    const t = (clamped - 300) / 100;
    h = 0.50 - t * 0.25; // 0.50 to 0.25 (cyan to yellow-green)
    s = 0.6;
    l = 0.55;
  } else if (clamped < 600) {
    // Hot (400-600K): Yellow to orange
    const t = (clamped - 400) / 200;
    h = 0.15 - t * 0.05; // 0.15 to 0.10 (yellow to orange)
    s = 0.9;
    l = 0.6;
  } else if (clamped < 1000) {
    // Very hot (600-1000K): Orange to red
    const t = (clamped - 600) / 400;
    h = 0.08 - t * 0.08; // 0.08 to 0.00 (orange to red)
    s = 0.95;
    l = 0.55;
  } else {
    // Extremely hot (1000K+): Deep red to bright red-orange
    const t = Math.min((clamped - 1000) / 1500, 1.0);
    h = 0.02; // red
    s = 1.0;
    l = 0.5 + t * 0.2; // Brighter as it gets hotter
  }
  
  return new (THREE as any).Color().setHSL(h, s, l);
}

// Optional logarithmic scaling for orbit radii
export function scaleAU(a: number, useLog: boolean){
  if (useLog) return Math.log(1 + a * 8) * AU_TO_U; // gentle compression
  return a * AU_TO_U;
}

// Kepler solver: mean anomaly M -> true anomaly v
export function trueAnomaly(M: number, e: number){
  M = M % (2*Math.PI); if (M < 0) M += 2*Math.PI;
  let E = e < 0.8 ? M : Math.PI; // initial guess
  for (let i=0; i<6; i++){
    const f = E - e*Math.sin(E) - M;
    const fp = 1 - e*Math.cos(E);
    E -= f/fp;
  }
  const cosE = Math.cos(E), sinE = Math.sin(E);
  const cosv = (cosE - e) / (1 - e*cosE);
  const sinv = (Math.sqrt(1 - e*e) * sinE) / (1 - e*cosE);
  return Math.atan2(sinv, cosv);
}

// Convert celestial coordinates (RA, Dec) to 3D Cartesian coordinates
// RA and Dec are in degrees, distance in parsecs
// Returns position in scene units where 1 parsec = scale units
export function celestialToCartesian(
  ra: number, 
  dec: number, 
  distance: number = 100, 
  scale: number = 10
): { x: number; y: number; z: number } {
  // Convert degrees to radians
  const raRad = (ra * Math.PI) / 180;
  const decRad = (dec * Math.PI) / 180;
  
  // Convert to Cartesian coordinates
  // In equatorial coordinate system:
  // x points to vernal equinox (RA=0, Dec=0)
  // y points to RA=90°, Dec=0
  // z points to north celestial pole (Dec=90°)
  const d = distance * scale;
  const x = d * Math.cos(decRad) * Math.cos(raRad);
  const y = d * Math.sin(decRad);
  const z = d * Math.cos(decRad) * Math.sin(raRad);
  
  return { x, y, z };
}

// Minimal test harness (ported)
type TestFn = () => void;
const _tests: Array<[string, TestFn]> = [];
export function test(name: string, fn: TestFn){ _tests.push([name, fn]); }
export function approx(a: number, b: number, eps = 1e-6){ if (Math.abs(a-b) > eps) throw new Error(`Expected ${a} ≈ ${b}`); }
export function reportTests(el?: HTMLElement | null){
  let pass = 0; const out: Array<any> = [];
  for (const [name, fn] of _tests){
    try { fn(); pass++; out.push([name, true]); }
    catch(e: any){ console.error('Test failed:', name, e); out.push([name,false,e.message]); }
  }
  if (el) el.textContent = `Tests: ${pass}/${_tests.length} passed`;
  return out;
}
