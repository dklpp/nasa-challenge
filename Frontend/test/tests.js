
import { test, approx, trueAnomaly } from './utils.js';


test('trueAnomaly e=0, M=0', ()=> approx(trueAnomaly(0,0), 0));
test('trueAnomaly e=0, M=π/2', ()=> approx(trueAnomaly(Math.PI/2,0), Math.PI/2));
test('trueAnomaly e=0, M=π', ()=> approx(trueAnomaly(Math.PI,0), Math.PI));


test('Circular orbit radius constant', ()=> { const a=1,e=0; for(let k=0;k<6;k++){ const M=k*0.7; const v=trueAnomaly(M,e); const r=a*(1-e*e)/(1+e*Math.cos(v)); approx(r,a,1e-6);} });


test('Monotonicity small e', ()=> { const e=0.1; const v1=trueAnomaly(0.5,e), v2=trueAnomaly(1.0,e); if(!(v2>v1)) throw new Error('v not increasing'); });


test('Periapsis distance r_p = a(1-e)', ()=> { const a=1, e=0.5; const v=0; const r=a*(1-e*e)/(1+e*Math.cos(v)); approx(r, a*(1-e), 1e-6); });

test('Apoapsis distance r_a = a(1+e)', ()=> { const a=1, e=0.5; const v=Math.PI; const r=a*(1-e*e)/(1+e*Math.cos(v)); approx(r, a*(1+e), 1e-6); });


test('High e solver stable', ()=> { const e=0.9; for(let k=0;k<10;k++){ const M = (k/10)*2*Math.PI; const v = trueAnomaly(M,e); if (!isFinite(v)) throw new Error('v not finite'); } });
