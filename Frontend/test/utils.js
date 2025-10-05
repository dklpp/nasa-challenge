
export const AU_TO_U = 120;


export const teffColor = (T, THREE) => {
  const clamped = Math.max(3000, Math.min(8000, T));
  const h = ((clamped - 3000) / 7000) * 0.7;
  return new THREE.Color().setHSL(h, 1.0, 0.6);
};


export function scaleAU(a, useLog){
  if (useLog) return Math.log(1 + a * 8) * AU_TO_U;
  return a * AU_TO_U;
}


export function trueAnomaly(M, e){
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


const _tests = [];
export function test(name, fn){ _tests.push([name, fn]); }
export function approx(a, b, eps=1e-6){ if (Math.abs(a-b) > eps) throw new Error(`Expected ${a} ≈ ${b}`); }
export function reportTests(el){
  let pass = 0; const out = [];
  for (const [name, fn] of _tests){
    try { fn(); pass++; out.push([name, true]); }
    catch(e){ console.error('Test failed:', name, e); out.push([name,false,e.message]); }
  }
  if (el) el.textContent = `Tests: ${pass}/${_tests.length} passed`;
  return out;
}
