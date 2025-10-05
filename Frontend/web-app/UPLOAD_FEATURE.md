# Custom System Upload Feature

This feature allows you to add custom exoplanet systems to the 3D visualization by uploading a JSON file. **Custom systems are highlighted with special visual effects!**

## How to Use

1. **Click the + Button**: Look for the bright blue circular + button in the bottom-right corner of the left sidebar (Systems panel)

2. **Upload Your JSON File**: 
   - Click the upload area or drag & drop a JSON file
   - The file will be validated automatically
   - A preview will show the system details

3. **Review and Add**: 
   - Check the preview to ensure your data is correct
   - Click "Add System" to add it to the map
   - The new system will appear in the systems list and on the 3D map

## Visual Highlighting

Custom uploaded systems are highlighted with special visual effects to make them easy to find:

### In the 3D Solar System View:
- **Pulsing Green Glow**: A bright green aura pulses around the star
- **Rotating Highlight Ring**: A green ring rotates around the system
- **Brighter Star**: The star emits more light than standard systems
- **Green Orbits**: Planet orbits are colored green instead of blue
- **Custom Light**: A green point light illuminates the area

### In the Systems List:
- **Star Badge (★)**: A glowing star icon appears next to the system name
- **Green Border**: The system card has a green border with glow effect
- **Special Background**: A subtle darker background distinguishes custom systems
- **Pulsing Animation**: The star badge pulses to draw attention

## JSON Format

Your JSON file must follow this structure:

```json
{
  "name": "System Name",
  "star": {
    "teff": 5778,
    "radius_rs": 1.0,
    "mass_ms": 1.0
  },
  "planets": [
    {
      "name": "b",
      "a_au": 1.0,
      "period_days": 365.25,
      "e": 0.0167,
      "radius_re": 1.0,
      "radius_rj": 0.0892,
      "incl": 0.0,
      "pl_eqt": 288
    }
  ],
  "ra": 0.0,
  "dec": 0.0,
  "distance_pc": 10.0
}
```

## Required Fields

### System Level:
- `name` (string): Name of the star system
- `star` (object): Stellar properties
- `planets` (array): Array of planet objects (at least one)

### Star Properties:
- `teff` (number): Effective temperature in Kelvin

### Optional Star Properties:
- `radius_rs` (number): Stellar radius in solar radii
- `mass_ms` (number): Stellar mass in solar masses

### Planet Properties (required):
- `name` (string): Planet designation (e.g., "b", "c", "d")
- `a_au` (number): Semi-major axis in AU
- `period_days` (number): Orbital period in days

### Optional Planet Properties:
- `e` (number): Orbital eccentricity (0-1)
- `radius_re` (number): Planet radius in Earth radii
- `radius_rj` (number): Planet radius in Jupiter radii
- `incl` (number): Orbital inclination in degrees
- `pl_eqt` (number): Equilibrium temperature in Kelvin

### Optional System Properties:
- `ra` (number): Right Ascension in degrees
- `dec` (number): Declination in degrees
- `distance_pc` (number): Distance in parsecs

## Example File

See `example-system.json` in the web-app directory for a complete example based on the K2-18 system.

## Tips

- Make sure your JSON is properly formatted (use a JSON validator if needed)
- If the system has celestial coordinates (RA/Dec), it will be positioned correctly in the Earth-centered view
- Without coordinates, the system will still appear in the systems list and can be viewed individually
- The visualization uses realistic orbital mechanics and scales

## Data Sources

You can create custom systems from:
- K2 mission data
- Your own research
- Hypothetical systems for educational purposes
- Any exoplanet database following the same format
