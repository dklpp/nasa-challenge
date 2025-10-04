import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { systems } from './data.js';
import { teffColor, scaleAU, trueAnomaly, reportTests } from './utils.js';
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
  sizeScale: document.getElementById('sizeScale'),
  sizeVal: document.getElementById('sizeVal'),
  tourBtn: document.getElementById('tourBtn'),
  resetCam: document.getElementById('resetCam'),
  titleSystem: document.querySelector('#titlebar .system'),
  titleSub: document.querySelector('#titlebar .sub'),
  details: document.getElementById('details'),
  thumbs: document.getElementById('thumbs'),
  toast: document.getElementById('toast'),
  testBadge: document.getElementById('testBadge'),
  exploreBtn: document.getElementById('exploreBtn'),
  modeBadge: document.getElementById('modeBadge'),
};

function showToast(msg, t=1800){ ui.toast.textContent = msg; ui.toast.style.display='block'; setTimeout(()=> ui.toast.style.display='none', t); }

// Three.js scene
const center = document.getElementById('center');
const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true });
center.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 10000);
camera.position.set(0, 900, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.05; controls.minDistance = 50; controls.maxDistance = 5000;

const ambient = new THREE.AmbientLight(0x506080, .45); scene.add(ambient);
const starField = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 0.8, color: 0x9fb7ff }));
(function makeStars(){
  const N=1500, R=2500; const pos = new Float32Array(N*3);
  for(let i=0;i<N;i++){ const r = R*Math.cbrt(Math.random()); const th = Math.random()*Math.PI*2; const ph = Math.acos(2*Math.random()-1); pos[3*i]=r*Math.sin(ph)*Math.cos(th); pos[3*i+1]=r*Math.cos(ph); pos[3*i+2]=r*Math.sin(ph)*Math.sin(th); }
  starField.geometry.setAttribute('position', new THREE.BufferAttribute(pos,3)); starField.material.transparent=true; starField.material.opacity=0.35; scene.add(starField);
})();

function onResize(){ const w = center.clientWidth, h = center.clientHeight; renderer.setSize(w,h,false); camera.aspect = w/h; camera.updateProjectionMatrix(); }
window.addEventListener('resize', onResize); onResize();

// -------------------- Explore (All Planets) --------------------
let mode = 'explore'; // 'explore' | 'system'
let explore = { mesh:null, axis:null, idx:[], domain:null };
let current = { group:null, star:null, planets:[], orbits:[], hz:null };

function flattenPlanets(systems){
  const out = [];
  for(let si=0; si<systems.length; si++){
    const s = systems[si]; const starT = s.star?.teff ?? 5500;
    for(let pi=0; pi<s.planets.length; pi++){
      const p = s.planets[pi];
      const P = p.period_days || p.per || null;
      const R = p.radius_re || (p.radius_rj? p.radius_rj*11.21 : null);
      if(!P || !R) continue;
      out.push({ si, pi, sysName: s.name, name: `${s.name} ${p.name}`, P, R, teff: starT, a_au: p.a_au||null, e: p.e||0, incl: p.incl||0 });
    }
  }
  return out;
}

function buildAxes(domain){
  const grp = new THREE.Group();
  const {xMin,xMax,zMin,zMax, W, H} = domain;
  // Axes lines (X and Z)
  const mat = new THREE.LineBasicMaterial({ color: 0x6f85c9, transparent:true, opacity:0.6 });
  const xgeo = new THREE.BufferGeometry().setFromPoints([ new THREE.Vector3(-W/2,0,-H/2), new THREE.Vector3(W/2,0,-H/2) ]);
  const zgeo = new THREE.BufferGeometry().setFromPoints([ new THREE.Vector3(-W/2,0,-H/2), new THREE.Vector3(-W/2,0,H/2) ]);
  grp.add(new THREE.Line(xgeo, mat));
  grp.add(new THREE.Line(zgeo, mat));
  // tick marks (log decades)
  const tickMat = new THREE.LineBasicMaterial({ color: 0x3f558f, transparent:true, opacity:0.5 });
  function addTickX(x){
    const gx = new THREE.BufferGeometry().setFromPoints([ new THREE.Vector3(x,0,-H/2), new THREE.Vector3(x,0,-H/2-6) ]);
    grp.add(new THREE.Line(gx, tickMat));
  }
  function addTickZ(z){
    const gz = new THREE.BufferGeometry().setFromPoints([ new THREE.Vector3(-W/2,0,z), new THREE.Vector3(-W/2-6,0,z) ]);
    grp.add(new THREE.Line(gz, tickMat));
  }
  // 10^-1..10^3 for period, 10^-0.3..10^1.5 for radius
  for(let d=-1; d<=3; d++){ const t = (d - (-1)) / (3 - (-1)); const x = -W/2 + t*W; addTickX(x); }
  for(let d=-0.3; d<=1.5; d+=0.3){ const t = (d - (-0.3)) / (1.5 - (-0.3)); const z = -H/2 + t*H; addTickZ(z); }
  return grp;
}

function buildExplore(){
  // Clear any system view
  clearSystem();
  if(explore.mesh){ scene.remove(explore.mesh); explore.mesh.geometry.dispose(); explore.mesh.material.dispose(); explore.mesh=null; }
  if(explore.axis){ scene.remove(explore.axis); explore.axis=null; }
  mode='explore'; ui.modeBadge.textContent = 'Explore'; ui.titleSystem.textContent='All Planets'; ui.titleSub.textContent='Click a planet to open its system';
  // Flatten
  const idx = flattenPlanets(systems);
  // Compute domains
  const P = idx.filter(d=>d.P>0).map(d=>Math.log10(d.P));
  const R = idx.filter(d=>d.R>0).map(d=>Math.log10(d.R));
  const xMin = Math.min(-1, Math.min(...P)), xMax = Math.max(3, Math.max(...P));
  const zMin = Math.min(-0.3, Math.min(...R)), zMax = Math.max(1.5, Math.max(...R));
  const W = 1800, H = 1000;
  const mapX = v => -W/2 + ( (Math.log10(v)-xMin)/(xMax-xMin) ) * W;
  const mapZ = v => -H/2 + ( (Math.log10(v)-zMin)/(zMax-zMin) ) * H;

  // Geometry & material
  const count = Math.min(idx.length, 6000);
  const geom = new THREE.CircleGeometry(1, 24);
  const mat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent:true, opacity:0.95, depthWrite:false });
  const mesh = new THREE.InstancedMesh(geom, mat, count);
  mesh.rotation.x = -Math.PI/2;
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  const color = new THREE.Color();

  // Fill instances
  for(let i=0;i<count;i++){
    const d = idx[i];
    const s = Math.max(0.8, Math.sqrt(d.R||1) * 1.2); // perceptual size
    const x = mapX(d.P);
    const z = mapZ(d.R);
    const m = new THREE.Matrix4().compose(
      new THREE.Vector3(x, 0, z),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI/2, 0, 0)),
      new THREE.Vector3(s, s, 1)
    );
    mesh.setMatrixAt(i, m);
    // color by star teff
    const c = teffColor(d.teff, THREE);
    mesh.setColorAt(i, c);
  }
  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.needsUpdate = true;
  scene.add(mesh);
  explore.mesh = mesh;
  explore.idx = idx;
  explore.domain = {xMin,xMax,zMin,zMax,W,H};

  // Axes
  const axis = buildAxes(explore.domain);
  scene.add(axis);
  explore.axis = axis;

  // Camera top/center
  camera.position.set(0, 900, 1);
  controls.target.set(0,0,0); controls.update();
}


// --- Helper: scene radius from planet radii with size scale ---
function planetSceneRadius(p){
  const scaleCtl = ui.sizeScale ? parseFloat(ui.sizeScale.value || '0.12') : 0.12;
  const exaggerated = ui.exSizes && ui.exSizes.checked;
  // Base multipliers tuned for TOI (Earth radii); keep very small by default
  const base = (p.radius_rj ? (p.radius_rj * (exaggerated ? 3.0 : 0.6))
                            : ((p.radius_re||1) * (exaggerated ? 0.35 : 0.10)));
  // Final scale with clamp
  return Math.max(0.15, base * scaleCtl);
}

// -------------------- Overview (mini-systems grid) --------------------
let overview = { groups:[], pickers:[], root:null };

function clearOverview(){
  overview.groups.forEach(g=>clearSceneGroup(g));
  overview.pickers.forEach(obj=>{ scene.remove(obj); obj.geometry?.dispose(); obj.material?.dispose?.(); });
  if(overview.root){ clearSceneGroup(overview.root); overview.root=null; }
  overview = { groups:[], pickers:[], root:null };
}

function buildOverview(list){
  // list: array of systems; if omitted, use all systems
  clearOverview(); clearSystem(); mode='overview'; if(ui.modeBadge) ui.modeBadge.textContent='Overview';
  const arr = (list && list.length ? list : systems);
  ui.titleSystem.textContent = 'Overview';
  ui.titleSub.textContent = `${arr.length} systems • click a system`;
  const g = new THREE.Group(); scene.add(g); overview.root = g;

  const N = Math.min(arr.length, 48); // show up to 48 mini systems
  const cols = 8, rows = Math.ceil(N/cols);
  const cell = 85; const desired = 14; const miniColor = 0x6ea9ff;

  for(let i=0;i<N;i++){
    const sys = arr[i];
    const gg = new THREE.Group(); g.add(gg);
    const cx = (i % cols) - (cols-1)/2;
    const cy = Math.floor(i/cols) - (rows-1)/2;
    gg.position.set(cx*cell, 0, cy*cell);

    // star
    const sc = teffColor(sys.star?.teff ?? 5500, THREE);
    const sgeom = new THREE.SphereGeometry(1.4, 18, 12);
    const smat  = new THREE.MeshStandardMaterial({ emissive: sc, emissiveIntensity: 1.2, color:0x222233, roughness:0.6, metalness:0.1 });
    gg.add(new THREE.Mesh(sgeom, smat));

    // scale to fit
    const aMax = Math.max(...sys.planets.map(p=>p.a_au || 0.02), 0.02);
    const base = scaleAU(aMax, true);
    const k = base > 0 ? desired / base : 1;

    sys.planets.slice(0,10).forEach((p)=>{
      const e = Math.min(0.95, Math.max(0, p.e || 0));
      const aU = scaleAU(Math.max(0.004, p.a_au||0.02), true) * k;
      const bU = aU * Math.sqrt(1-e*e);
      const cU = Math.sqrt(Math.max(0, aU*aU - bU*bU));
      const Np = 96; const pts = [];
      for(let j=0;j<=Np;j++){ const th = j/Np*2*Math.PI; const x = aU*Math.cos(th)-cU; const y = bU*Math.sin(th); pts.push(new THREE.Vector3(x,0,y)); }
      const ogeom = new THREE.BufferGeometry().setFromPoints(pts);
      const omat  = new THREE.LineBasicMaterial({ color: 0x2c4376, transparent:true, opacity: 0.6 });
      const orbit = new THREE.LineLoop(ogeom, omat); orbit.rotation.x = -Math.PI/2; gg.add(orbit);

      const rScene = Math.max(0.4, (p.radius_rj? p.radius_rj*0.5 : (p.radius_re||1)*0.15));
      const pgeom = new THREE.SphereGeometry(rScene, 16, 12);
      const pmat  = new THREE.MeshStandardMaterial({ color: miniColor, metalness:0.2, roughness:0.5 });
      const pm    = new THREE.Mesh(pgeom, pmat);
      const v = Math.random()*Math.PI*2;
      const r = aU*(1-e*e)/(1+e*Math.cos(v));
      pm.position.set(r*Math.cos(v), 0, r*Math.sin(v));
      gg.add(pm);
    });

    const pickGeom = new THREE.RingGeometry(desired*0.2, desired*1.2, 32);
    const pickMat  = new THREE.MeshBasicMaterial({ color: 0x00ffff, transparent:true, opacity: 0.0, side:THREE.DoubleSide });
    const pickMesh = new THREE.Mesh(pickGeom, pickMat); pickMesh.rotation.x = -Math.PI/2; pickMesh.userData.sys = sys;
    gg.add(pickMesh); overview.pickers.push(pickMesh);
    overview.groups.push(gg);
  }

  // camera to see grid
  const span = Math.max(cols, rows);
  camera.position.set(0, span*32, span*62);
  controls.target.set(0,0,0); controls.update();
}
// -------------------- System view --------------------
function clearSceneGroup(g){
  if(!g) return;
  scene.remove(g);
  g.traverse(obj=>{ if(obj.geometry) obj.geometry.dispose(); if(obj.material) obj.material.dispose?.(); });
}
function clearSystem(){ clearSceneGroup(current.group); current = { group:null, star:null, planets:[], orbits:[], hz:null }; }

function buildSystem(sys, selectPi=null){
  // Remove explore cloud but keep to allow back navigation
  if(explore.mesh){ scene.remove(explore.mesh); }
  if(explore.axis){ scene.remove(explore.axis); }
  clearSystem(); mode='system'; ui.modeBadge.textContent = 'System';

  const g = new THREE.Group(); scene.add(g); current.group = g;

  const starColor = teffColor(sys.star?.teff ?? 5500, THREE);
  const starGeom = new THREE.SphereGeometry( ui.exSizes.checked ? 10 : Math.max(2, (sys.star?.radius_rs ?? 1)*2.5), 32, 16 );
  const starMat = new THREE.MeshStandardMaterial({ emissive: starColor, emissiveIntensity: 1.5, color:0x222233, roughness:0.4, metalness:0.1 });
  const starMesh = new THREE.Mesh(starGeom, starMat); g.add(starMesh); current.star = starMesh;
  const lamp = new THREE.PointLight(starColor.getHex(), 2.2, 0, 2); lamp.position.set(0,0,0); g.add(lamp);

  if(ui.showHZ.checked){
    const hz = new THREE.RingGeometry(scaleAU(0.75, ui.logScale.checked), scaleAU(1.77, ui.logScale.checked), 128);
    const hzmat = new THREE.MeshBasicMaterial({ color:0x66ffcc, transparent:true, opacity:0.08, side:THREE.DoubleSide });
    const hzmesh = new THREE.Mesh(hz, hzmat); hzmesh.rotation.x = -Math.PI/2; g.add(hzmesh); current.hz = hzmesh;
  }

  sys.planets.forEach((p, idx)=>{
    const a = Math.max(0.004, p.a_au || 0.02); const e = Math.min(0.95, Math.max(0, p.e||0));
    const aU = scaleAU(a, ui.logScale.checked); const bU = aU * Math.sqrt(1-e*e); const cU = Math.sqrt(Math.max(0, aU*aU - bU*bU));
    const inc = THREE.MathUtils.degToRad(p.incl||0);

    const N = 256; const pts = [];
    for(let i=0;i<=N;i++){ const th = i/N*2*Math.PI; const x = aU*Math.cos(th)-cU; const y = bU*Math.sin(th); pts.push(new THREE.Vector3(x,0,y)); }
    const orbGeom = new THREE.BufferGeometry().setFromPoints(pts);
    const orbMat = new THREE.LineBasicMaterial({ color: ui.dimOrbits.checked? 0x28406f : 0x4066aa, transparent:true, opacity: ui.dimOrbits.checked? 0.35 : 0.8 });
    const orbit = new THREE.LineLoop(orbGeom, orbMat); orbit.rotation.x = -Math.PI/2; orbit.rotation.z = inc; orbit.userData.index = idx; g.add(orbit); current.orbits.push(orbit);

    const rScene = planetSceneRadius(p);
    const geom = new THREE.SphereGeometry(rScene, 24, 16);
    const color = teffColor(sys.star?.teff ?? 5500, THREE).clone().offsetHSL(0,0,-0.15);
    const mat = new THREE.MeshStandardMaterial({ color, metalness:0.2, roughness:0.4 });
    const m = new THREE.Mesh(geom, mat);
    m.userData = { ...p, aU, bU, cU, inc, M: Math.random()*Math.PI*2, name: `${sys.name} ${p.name}`, index: idx };
    g.add(m); current.planets.push(m);
  });

  ui.titleSystem.textContent = sys.name; ui.titleSub.textContent = `${sys.planets.length} planet(s) • star T_eff ${(sys.star?.teff ?? 5500)}K`;

  // Auto-fit camera to include all orbits at once
  const maxA = Math.max(...current.planets.map(m => m.userData.aU), 60);
  const z = Math.max(180, maxA * 2.2);
  const y = Math.max(80,  maxA * 0.9);
  camera.position.set(0, y, z);
  controls.target.set(0,0,0); controls.update();

  // Optionally preselect a planet
  if(selectPi!=null && current.planets[selectPi]) selectPlanet(current.planets[selectPi]);
}

// List/search UI (left)

function renderList(){
  const q = ui.q?.value?.trim().toLowerCase() || '';
  if(ui.sysList){
    ui.sysList.innerHTML = '';
    const matches = systems.filter(s => !q || s.name.toLowerCase().includes(q) || s.planets.some(p=>(`${s.name} ${p.name}`).toLowerCase().includes(q)));
    matches.forEach(s => {
      const div = document.createElement('div'); div.className='sys';
      div.innerHTML = `<div class="name">${s.name}</div><div class="meta">${s.planets.length} planets • T<sub>eff</sub> ${(s.star?.teff ?? 5500)}K</div>`;
      div.onclick = ()=> { buildSystem(s); showToast(`Loaded ${s.name}`); };
      ui.sysList.appendChild(div);
    });
    // Auto view logic:
    if(matches.length === 0){
      // nothing typed or no matches -> overview all (better than empty)
      buildOverview(systems);
    } else if(matches.length <= 3){
      // auto open best match (first)
      buildSystem(matches[0]);
    } else {
      // many matches -> compact overview of only matches
      buildOverview(matches);
    }
  }
}

ui.q?.addEventListener('input', renderList);

['exSizes','logScale','showHZ'].forEach(id => ui[id]?.addEventListener('change', ()=> {
  if(mode==='system' && current.group) buildSystem(activeSystem());
  if(mode==='explore') buildExplore();
}));
ui.dimOrbits?.addEventListener('change', ()=> current.orbits.forEach(o => o.material.opacity = ui.dimOrbits.checked? 0.35 : 0.8));

function activeSystem(){ return systems.find(s => s.name === ui.titleSystem.textContent) || systems[0]; }

// Buttons
ui.exploreBtn?.addEventListener('click', ()=> buildExplore());
ui.resetCam?.addEventListener('click', ()=> { 
  if(mode==='explore'){ camera.position.set(0, 900, 1); controls.target.set(0,0,0); controls.update(); }
  if(mode==='system'){ 
    const maxA = Math.max(...current.planets.map(m => m.userData.aU), 60);
    const z = Math.max(180, maxA * 2.2);
    const y = Math.max(80,  maxA * 0.9);
    camera.position.set(0, y, z); controls.target.set(0,0,0); controls.update();
  }
});

// Picking
const ray = new THREE.Raycaster(); const mouse = new THREE.Vector2();
renderer.domElement.addEventListener('pointerdown', (e)=>{
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX-rect.left)/rect.width)*2 - 1;
  mouse.y = -((e.clientY-rect.top)/rect.height)*2 + 1;
  ray.setFromCamera(mouse, camera);

  if(mode==='overview'){
    const hits = ray.intersectObjects(overview.pickers, false);
    if(hits.length){ const sys = hits[0].object.userData.sys; buildSystem(sys); }
    return;
  }

  if(mode==='explore' && explore.mesh){
    const hits = ray.intersectObject(explore.mesh, false);
    if(hits.length){
      const id = hits[0].instanceId;
      if(id!=null){
        const d = explore.idx[id];
        buildSystem(systems[d.si], d.pi);
      }
    }
    return;
  }

  if(mode==='system'){
    const hits = ray.intersectObjects(current.planets, false);
    if(hits.length){ selectPlanet(hits[0].object); }
  }
});

function selectPlanet(m){
  const p = m.userData;
  current.orbits.forEach(o=>{
    if(o.userData.index === p.index){ o.material.color.setHex(0xffd27f); o.material.opacity = 1.0; }
    else { o.material.color.setHex(ui.dimOrbits.checked? 0x28406f : 0x4066aa); o.material.opacity = ui.dimOrbits.checked? 0.28 : 0.65; }
  });
  controls.target.lerp(new THREE.Vector3(0,0,0), 0.2);
  camera.position.lerp(new THREE.Vector3(0, Math.max(60, p.aU*0.25), Math.max(120, p.aU*0.6)), 0.2);
  const rows = [
    ['Planet', p.name],
    ['Orbital period', p.period_days?.toFixed ? p.period_days.toFixed(3)+' days' : (p.period_days+' days')],
    ['Semi-major (a)', (p.a_au ?? 0).toFixed(4)+' AU'],
    ['Eccentricity', (p.e||0).toFixed(3)],
    ['Inclination', (p.incl||0).toFixed(1)+'°'],
    ['Radius', (p.radius_rj? p.radius_rj.toFixed(2)+' Rj' : (p.radius_re??1).toFixed(2)+' R⊕')]
  ];
  ui.details.innerHTML = rows.map(([k,v])=>`<div class="row"><div>${k}</div><div>${v}</div></div>`).join('') +
    `<div style="margin-top:8px"><span class="pill">Evidence</span> <span class="pill">MAST link</span> <span class="pill">Archive row</span></div>`;
}

// Animation
const clock = new THREE.Clock();
function tick(){
  requestAnimationFrame(tick);
  const dt = clock.getDelta(); controls.update();
  ui.speedVal.textContent = `${parseFloat(ui.speed.value).toFixed(2)}×`;
  if(ui.sizeScale && ui.sizeVal){ ui.sizeVal.textContent = `${parseFloat(ui.sizeScale.value).toFixed(2)}×`; }
  const timeFactor = (parseFloat(ui.speed.value)||1);

  if(mode==='system'){
    for(const m of current.planets){
      const p = m.userData; const e = Math.min(0.95, Math.max(0, p.e||0));
      p.M += timeFactor * dt * 2*Math.PI / (p.period_days || 10);
      const v = trueAnomaly(p.M, e);
      const a = p.aU; const r = a*(1-e*e)/(1+e*Math.cos(v));
      const x = r*Math.cos(v); const y = r*Math.sin(v);
      m.position.set(x, 0, y).applyAxisAngle(new THREE.Vector3(0,1,0), THREE.MathUtils.degToRad(0)).applyAxisAngle(new THREE.Vector3(1,0,0), THREE.MathUtils.degToRad(p.incl||0));
    }
  }

  renderer.render(scene, camera);
}

// Init
renderList();
reportTests(ui.testBadge);
tick();
