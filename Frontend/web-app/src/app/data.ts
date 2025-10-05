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
  name: string; // host star / system name
  star: {
    teff: number; // stellar effective temperature (K)
    radius_rs?: number; // stellar radius in solar radii (R☉) from st_rad in K2 dataset
    mass_ms?: number; // stellar mass (M_sun)
  };
  planets: Planet[];
  ra?: number; // Right Ascension in degrees
  dec?: number; // Declination in degrees
  distance_pc?: number; // Distance in parsecs (if available)
};