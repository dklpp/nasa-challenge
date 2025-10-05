export type Planet = {
  name: string; // usually a letter like 'b', 'c', ...
  a_au: number; // semi-major axis in AU
  period_days: number;
  e?: number;
  radius_re?: number; // Earth radii
  radius_rj?: number; // Jupiter radii
  incl?: number; // orbital inclination in degrees
  pl_eqt?: number; // planet equilibrium temperature in Kelvin
};

export type System = {
  name: string; 
  star: {
    teff: number; 
    radius_rs?: number;
    mass_ms?: number;
  };
  planets: Planet[];
  ra?: number; // Right Ascension in degrees
  dec?: number; // Declination in degrees
  distance_pc?: number; // Distance in parsecs (if available)
};

export type ExoplanetRecord = {
  pl_name: string; // Planet name
  hostname: string; // Host star name
  default_flag?: number;
  disposition?: string; // e.g., "CONFIRMED"
  disp_refname?: string;
  sy_snum?: number; // Number of stars in system
  sy_pnum?: number; // Number of planets in system
  discoverymethod?: string;
  disc_year?: number;
  disc_facility?: string;
  soltype?: string;
  pl_controv_flag?: number;
  pl_refname?: string;
  pl_orbper?: number; // Orbital period in days
  pl_orbpererr1?: number;
  pl_orbpererr2?: number;
  pl_orbsmax?: number; // Semi-major axis in AU
  pl_rade?: number; // Planet radius in Earth radii
  pl_bmasse?: number; // Planet mass in Earth masses
  pl_orbeccen?: number; // Orbital eccentricity
  pl_insol?: number; // Insolation flux
  pl_eqt?: number; // Equilibrium temperature in K
  st_teff?: number; // Stellar effective temperature in K
  st_rad?: number; // Stellar radius in solar radii
  st_mass?: number; // Stellar mass in solar masses
  st_met?: number; // Stellar metallicity
  ra?: number; // Right Ascension in degrees
  dec?: number; // Declination in degrees
  sy_dist?: number; // Distance in parsecs
  sy_vmag?: number; // V magnitude
  sy_kmag?: number; // K magnitude
  sy_gaiamag?: number; // Gaia magnitude
  rowupdate?: string;
  pl_pubdate?: string;
  releasedate?: string;
};