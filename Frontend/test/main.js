import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { systems } from './data.js';
import { AU_TO_U, teffColor, scaleAU, trueAnomaly, reportTests } from './utils.js';
import './tests.js'; // registers tests; we will report after UI mounts

const ui = {
  sysList: document.getElementById('sysList'),
  q: document.getElementById('q'),
  exSizes: document.getElementById('exSizes'),
  logScale: document.getElementById('logScale'),
  showHZ: document.getElementById('showHZ'),
  dimOrbits: document.getElementById('dimOrbits'),
  speed: document.getElementById('speed'),
  speedVal: document.getElementById('speedVal'),
  tourBtn: document.getElementById('tourBtn'),
  resetCam: document.getElementById('resetCam'),
  titleSystem: document.querySelector('#titlebar .system'),
  titleSub: document.querySelector('#titlebar .sub'),
  details: document.getElementById('details'),
  thumbs: document.getElementById('thumbs'),
  toast: document.getElementById('toast'),
  testBadge: document.getElementById('testBadge'),
};

function showToast(msg, t=1800){ ui.toast.textContent = msg; ui.toast.style.display='block'; setTimeout(()=> ui.toast.style.display='none', t); }

// Three.js scene
const center = document.getElementById('center');
const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true });
center.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 5000);
camera.position.set(0, 120, 260);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.05; controls.minDistance = 30; controls.maxDistance = 1500;

const ambient = new THREE.AmbientLight(0x506080, .4); scene.add(ambient);
const starField = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 0.8, color: 0x9fb7ff }));
(function makeStars(){
  const N=2000, R=1600; const pos = new Float32Array(N*3);
  for(let i=0;i<N;i++){ const r = R*Math.cbrt(Math.random()); const th = Math.random()*Math.PI*2; const ph = Math.acos(2*Math.random()-1); pos[3*i]=r*Math.sin(ph)*Math.cos(th); pos[3*i+1]=r*Math.cos(ph); pos[3*i+2]=r*Math.sin(ph)*Math.sin(th); }
  starField.geometry.setAttribute('position', new THREE.BufferAttribute(pos,3)); starField.material.transparent=true; starField.material.opacity=0.5; scene.add(starField);
})();

function onResize(){ const w = center.clientWidth, h = center.clientHeight; renderer.setSize(w,h,false); camera.aspect = w/h; camera.updateProjectionMatrix(); }
window.addEventListener('resize', onResize); onResize();

let current = { group:null, star:null, planets:[], orbits:[], hz:null };

function clearSystem(){ if(current.group){ scene.remove(current.group); current.group.traverse(obj=>{ if(obj.geometry) obj.geometry.dispose(); if(obj.material) obj.material.dispose?.(); }); } current = { group:null, star:null, planets:[], orbits:[], hz:null }; }

function buildSystem(sys){
  clearSystem();
  const g = new THREE.Group(); scene.add(g); current.group = g;

  const starColor = teffColor(sys.star.teff, THREE);
  const starGeom = new THREE.SphereGeometry( ui.exSizes.checked ? 10 : Math.max(2, sys.star.radius_rs*2.5), 32, 16 );
  const starMat = new THREE.MeshStandardMaterial({ emissive: starColor, emissiveIntensity: 1.5, color:0x222233, roughness:0.4, metalness:0.1 });
  const starMesh = new THREE.Mesh(starGeom, starMat); g.add(starMesh); current.star = starMesh;
  const lamp = new THREE.PointLight(starColor.getHex(), 2.2, 0, 2); lamp.position.set(0,0,0); g.add(lamp);

  if(ui.showHZ.checked){
    const hz = new THREE.RingGeometry(scaleAU(0.75, ui.logScale.checked), scaleAU(1.77, ui.logScale.checked), 128);
    const hzmat = new THREE.MeshBasicMaterial({ color:0x66ffcc, transparent:true, opacity:0.08, side:THREE.DoubleSide });
    const hzmesh = new THREE.Mesh(hz, hzmat); hzmesh.rotation.x = -Math.PI/2; g.add(hzmesh); current.hz = hzmesh;
  }

  sys.planets.forEach((p)=>{
    const a = Math.max(0.004, p.a_au); const e = Math.min(0.95, Math.max(0, p.e||0));
    const aU = scaleAU(a, ui.logScale.checked); const bU = aU * Math.sqrt(1-e*e); const cU = Math.sqrt(Math.max(0, aU*aU - bU*bU));
    const inc = THREE.MathUtils.degToRad(p.incl||0);

    const N = 256; const pts = [];
    for(let i=0;i<=N;i++){ const th = i/N*2*Math.PI; const x = aU*Math.cos(th)-cU; const y = bU*Math.sin(th); pts.push(new THREE.Vector3(x,0,y)); }
    const orbGeom = new THREE.BufferGeometry().setFromPoints(pts);
    const orbMat = new THREE.LineBasicMaterial({ color: ui.dimOrbits.checked? 0x28406f : 0x4066aa, transparent:true, opacity: ui.dimOrbits.checked? 0.35 : 0.7 });
    const orbit = new THREE.LineLoop(orbGeom, orbMat); orbit.rotation.x = -Math.PI/2; orbit.rotation.z = inc; g.add(orbit); current.orbits.push(orbit);

    const rScene = ui.exSizes.checked ? Math.max(1.8, (p.radius_rj? p.radius_rj*6 : p.radius_re*0.8)) : Math.max(0.4, (p.radius_rj? p.radius_rj*1.2 : p.radius_re*0.2));
    const geom = new THREE.SphereGeometry(rScene, 24, 16);
    const color = teffColor(sys.star.teff, THREE).clone().offsetHSL(0,0,-0.15);
    const mat = new THREE.MeshStandardMaterial({ color, metalness:0.2, roughness:0.4 });
    const m = new THREE.Mesh(geom, mat);
    m.userData = { ...p, aU, bU, cU, inc, M: Math.random()*Math.PI*2, name: `${sys.name} ${p.name}` };
    g.add(m); current.planets.push(m);
  });

  ui.titleSystem.textContent = sys.name; ui.titleSub.textContent = `${sys.planets.length} planet(s) • star T_eff ${sys.star.teff}K`;
}

function renderList(){
  const q = ui.q.value.trim().toLowerCase();
  ui.sysList.innerHTML = '';
  systems.filter(s => !q || s.name.toLowerCase().includes(q) || s.planets.some(p=>(`${s.name} ${p.name}`).toLowerCase().includes(q)))
    .forEach(s => {
      const div = document.createElement('div'); div.className='sys';
      div.innerHTML = `<div class="name">${s.name}</div><div class="meta">${s.planets.length} planets • T<sub>eff</sub> ${s.star.teff}K</div>`;
      div.onclick = ()=> { buildSystem(s); showToast(`Loaded ${s.name}`); };
      ui.sysList.appendChild(div);
    });
}
ui.q.addEventListener('input', renderList);
['exSizes','logScale','showHZ'].forEach(id => ui[id].addEventListener('change', ()=> current.group && buildSystem(activeSystem())));
ui.dimOrbits.addEventListener('change', ()=> current.orbits.forEach(o => o.material.opacity = ui.dimOrbits.checked? 0.35 : 0.7));
function activeSystem(){ return systems.find(s => s.name === ui.titleSystem.textContent) || systems[0]; }

let tourTimer=null, tourIdx=0;
ui.tourBtn.onclick = ()=>{
  if(tourTimer){ clearInterval(tourTimer); tourTimer=null; ui.tourBtn.textContent='Tour'; showToast('Tour stopped'); return; }
  tourIdx = 0; ui.tourBtn.textContent='Stop';
  tourTimer = setInterval(()=>{ const s = systems[tourIdx % systems.length]; buildSystem(s); tourIdx++; }, 4000);
  showToast('Tour started');
};

ui.resetCam.onclick = ()=> { camera.position.set(0, 120, 260); controls.target.set(0,0,0); controls.update(); };

const ray = new THREE.Raycaster(); const mouse = new THREE.Vector2();
renderer.domElement.addEventListener('pointerdown', (e)=>{
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX-rect.left)/rect.width)*2 - 1;
  mouse.y = -((e.clientY-rect.top)/rect.height)*2 + 1;
  ray.setFromCamera(mouse, camera);
  const hits = ray.intersectObjects(current.planets, false);
  if(hits.length){ selectPlanet(hits[0].object); }
});

function selectPlanet(m){
  const p = m.userData;
  current.orbits.forEach(o=>o.material.opacity = ui.dimOrbits.checked? 0.5 : 0.9);
  controls.target.lerp(new THREE.Vector3(0,0,0), 0.2);
  camera.position.lerp(new THREE.Vector3(0, 40 + p.aU*0.08, 90 + p.aU*0.2), 0.2);
  const rows = [
    ['Planet', p.name],
    ['Orbital period', p.period_days.toFixed(3)+' days'],
    ['Semi‑major (a)', p.a_au.toFixed(4)+' AU'],
    ['Eccentricity', (p.e||0).toFixed(3)],
    ['Inclination', (p.incl||0).toFixed(1)+'°'],
    ['Radius', (p.radius_rj? p.radius_rj.toFixed(2)+' Rj' : p.radius_re.toFixed(2)+' R⊕')]
  ];
  ui.details.innerHTML = rows.map(([k,v])=>`<div class="row"><div>${k}</div><div>${v}</div></div>`).join('') +
    `<div style="margin-top:8px"><span class="pill">Evidence</span> <span class="pill">MAST link</span> <span class="pill">Archive row</span></div>`;
}

const clock = new THREE.Clock();
function tick(){
  requestAnimationFrame(tick);
  const dt = clock.getDelta(); controls.update();
  ui.speedVal.textContent = `${parseFloat(ui.speed.value).toFixed(2)}×`;
  const timeFactor = (parseFloat(ui.speed.value)||1);
  for(const m of current.planets){
    const p = m.userData; const e = Math.min(0.95, Math.max(0, p.e||0));
    p.M += timeFactor * dt * 2*Math.PI / p.period_days; // mean anomaly advance
    const v = trueAnomaly(p.M, e);
    const a = p.aU = scaleAU(p.a_au, ui.logScale.checked); const b = a*Math.sqrt(1-e*e); const c = Math.sqrt(Math.max(0, a*a - b*b));
    const r = a*(1-e*e)/(1+e*Math.cos(v));
    const x = r*Math.cos(v); const y = r*Math.sin(v);
    m.position.set(x, 0, y).applyAxisAngle(new THREE.Vector3(0,1,0), THREE.MathUtils.degToRad(0)).applyAxisAngle(new THREE.Vector3(1,0,0), THREE.MathUtils.degToRad(p.incl||0));
  }
  renderer.render(scene, camera);
}

// Init
renderList(); buildSystem(systems[0]);
// Run tests and update HUD badge
reportTests(ui.testBadge);
// Start animation
tick();
