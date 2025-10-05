
export const AU_TO_U = 120;


export function teffColor(T: number, THREE: typeof import('three')): any{
  const clamped = Math.max(2500, Math.min(10000, T));
  
  let h: number, s: number, l: number;
  
  if (clamped < 3500) {

    const t = (clamped - 2500) / 1000;
    h = 0.0 + t * 0.05;
    s = 0.9 - t * 0.2;
    l = 0.5;
  } else if (clamped < 5000) {

    const t = (clamped - 3500) / 1500;
    h = 0.05 + t * 0.08;
    s = 0.7 - t * 0.1;
    l = 0.55 + t * 0.05;
  } else if (clamped < 6000) {

    const t = (clamped - 5000) / 1000;
    h = 0.13 + t * 0.03;
    s = 0.6 - t * 0.3;
    l = 0.60 + t * 0.1;
  } else if (clamped < 7500) {

    const t = (clamped - 6000) / 1500;
    h = 0.16 + t * 0.37;
    s = 0.3 - t * 0.2;
    l = 0.70 + t * 0.1;
  } else {

    const t = Math.min((clamped - 7500) / 2500, 1.0);
    h = 0.53 + t * 0.07;
    s = 0.1 + t * 0.4;
    l = 0.75 + t * 0.05;
  }
  
  return new (THREE as any).Color().setHSL(h, s, l);
}





export function planetTempColor(T: number | undefined, THREE: typeof import('three')): any {
  if (T === undefined || T === null || !isFinite(T)) {

    return new (THREE as any).Color(0x888888);
  }
  



  
  const clamped = Math.max(0, Math.min(2500, T));
  
  let h: number, s: number, l: number;
  
  if (clamped < 100) {

    h = 0.65 - (clamped / 100) * 0.1;
    s = 0.8;
    l = 0.4;
  } else if (clamped < 200) {

    const t = (clamped - 100) / 100;
    h = 0.55 + t * 0.05;
    s = 0.8;
    l = 0.45;
  } else if (clamped < 300) {

    const t = (clamped - 200) / 100;
    h = 0.60 - t * 0.10;
    s = 0.7;
    l = 0.5;
  } else if (clamped < 400) {

    const t = (clamped - 300) / 100;
    h = 0.50 - t * 0.25;
    s = 0.6;
    l = 0.55;
  } else if (clamped < 600) {

    const t = (clamped - 400) / 200;
    h = 0.15 - t * 0.05;
    s = 0.9;
    l = 0.6;
  } else if (clamped < 1000) {

    const t = (clamped - 600) / 400;
    h = 0.08 - t * 0.08;
    s = 0.95;
    l = 0.55;
  } else {

    const t = Math.min((clamped - 1000) / 1500, 1.0);
    h = 0.02;
    s = 1.0;
    l = 0.5 + t * 0.2;
  }
  
  return new (THREE as any).Color().setHSL(h, s, l);
}


export function scaleAU(a: number, useLog: boolean){
  if (useLog) return Math.log(1 + a * 8) * AU_TO_U;
  return a * AU_TO_U;
}


export function trueAnomaly(M: number, e: number){
  M = M % (2*Math.PI); if (M < 0) M += 2*Math.PI;
  let E = e < 0.8 ? M : Math.PI;
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




export function celestialToCartesian(
  ra: number, 
  dec: number, 
  distance: number = 100, 
  scale: number = 10
): { x: number; y: number; z: number } {

  const raRad = (ra * Math.PI) / 180;
  const decRad = (dec * Math.PI) / 180;
  





  const d = distance * scale;
  const x = d * Math.cos(decRad) * Math.cos(raRad);
  const y = d * Math.sin(decRad);
  const z = d * Math.cos(decRad) * Math.sin(raRad);
  
  return { x, y, z };
}


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
