// Scene scale
export const AU_TO_U = 120;

// Map stellar Teff (3000–8000K) to a blue→red-ish color via HSL
export function teffColor(T: number, THREE: typeof import('three')): any{
  const clamped = Math.max(3000, Math.min(8000, T));
  const h = ((clamped - 3000) / 7000) * 0.7; // 0..0.7 (roughly blue→yellow)
  return new (THREE as any).Color().setHSL(h, 1.0, 0.6);
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
