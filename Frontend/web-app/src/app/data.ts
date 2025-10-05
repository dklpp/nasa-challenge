export type Planet = {
  name: string;
  a_au: number;
  period_days: number;
  e?: number;
  radius_re?: number;
  radius_rj?: number;
  incl?: number;
  pl_eqt?: number;
};

export type System = {
  name: string; 
  star: {
    teff: number; 
    radius_rs?: number;
    mass_ms?: number;
  };
  planets: Planet[];
  ra?: number;
  dec?: number;
  distance_pc?: number;
};

export type ExoplanetRecord = {
  pl_name: string;
  hostname: string;
  default_flag?: number;
  disposition?: string;
  disp_refname?: string;
  sy_snum?: number;
  sy_pnum?: number;
  discoverymethod?: string;
  disc_year?: number;
  disc_facility?: string;
  soltype?: string;
  pl_controv_flag?: number;
  pl_refname?: string;
  pl_orbper?: number;
  pl_orbpererr1?: number;
  pl_orbpererr2?: number;
  pl_orbsmax?: number;
  pl_rade?: number;
  pl_bmasse?: number;
  pl_orbeccen?: number;
  pl_insol?: number;
  pl_eqt?: number;
  st_teff?: number;
  st_rad?: number;
  st_mass?: number;
  st_met?: number;
  ra?: number;
  dec?: number;
  sy_dist?: number;
  sy_vmag?: number;
  sy_kmag?: number;
  sy_gaiamag?: number;
  rowupdate?: string;
  pl_pubdate?: string;
  releasedate?: string;
};