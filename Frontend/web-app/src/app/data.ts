export const systems = [
  {
    name: 'TRAPPIST-1', star: { teff: 2550, radius_rs: 0.121, mass_ms: 0.089 },
    planets: [
      { name:'b', a_au:0.0115, e:0.01, period_days:1.51, radius_re:1.12, incl:1 },
      { name:'c', a_au:0.0158, e:0.01, period_days:2.42, radius_re:1.10, incl:0.5 },
      { name:'d', a_au:0.0223, e:0.01, period_days:4.05, radius_re:0.79, incl:-0.6 },
      { name:'e', a_au:0.0293, e:0.01, period_days:6.10, radius_re:0.92, incl:0.8 },
      { name:'f', a_au:0.0385, e:0.01, period_days:9.21, radius_re:1.05, incl:0.2 },
      { name:'g', a_au:0.0469, e:0.01, period_days:12.35, radius_re:1.13, incl:-0.3 },
      { name:'h', a_au:0.0619, e:0.01, period_days:18.77, radius_re:0.77, incl:0.4 }
    ]
  },
  {
    name: 'Kepler-90', star: { teff: 5930, radius_rs: 1.2, mass_ms: 1.2 },
    planets: [
      { name:'b', a_au:0.074, e:0.02, period_days:7.0, radius_re:1.3, incl:0.5 },
      { name:'c', a_au:0.089, e:0.02, period_days:8.7, radius_re:1.2, incl:-0.2 },
      { name:'i', a_au:0.106, e:0.03, period_days:10.9, radius_re:2.7, incl:0.4 },
      { name:'d', a_au:0.32, e:0.02, period_days:59.7, radius_re:3.3, incl:-0.6 },
      { name:'e', a_au:0.42, e:0.02, period_days:91.9, radius_re:2.7, incl:0.6 },
      { name:'f', a_au:0.49, e:0.03, period_days:124.9, radius_re:2.9, incl:-0.4 },
      { name:'g', a_au:0.71, e:0.05, period_days:210.6, radius_re:8.1, incl:0.3 },
      { name:'h', a_au:1.01, e:0.05, period_days:331.6, radius_re:11.3, incl:0.1 }
    ]
  },
  {
    name: 'Kepler-10', star: { teff: 5708, radius_rs: 1.07, mass_ms: 0.91 },
    planets: [
      { name:'b', a_au:0.0169, e:0.05, period_days:0.84, radius_re:1.47, incl:0.4 },
      { name:'c', a_au:0.24, e:0.0, period_days:45.3, radius_re:2.35, incl:-0.3 }
    ]
  },
  {
    name: 'WASP-121', star: { teff: 6460, radius_rs: 1.46, mass_ms: 1.35 },
    planets: [ { name:'b', a_au:0.0254, e:0.01, period_days:1.27, radius_rj:1.81, incl:0.2 } ]
  }
] as const;

export type Planet = typeof systems[number]['planets'][number];
export type System = typeof systems[number];
