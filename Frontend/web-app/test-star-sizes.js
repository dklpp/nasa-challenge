


const testStars = [
  { name: "Tiny Red Dwarf", st_rad: 0.12 },
  { name: "Small Red Dwarf", st_rad: 0.31 },
  { name: "Medium Red Dwarf", st_rad: 0.57 },
  { name: "Sun-like Star", st_rad: 1.0 },
  { name: "Larger Star", st_rad: 1.6 },
  { name: "Giant Star", st_rad: 3.17 },
  { name: "Large Giant", st_rad: 10.0 },
  { name: "Supergiant", st_rad: 25.0 },
  { name: "Massive Supergiant", st_rad: 85.0 },
];

console.log("\n=== STAR SIZE CALCULATIONS ===\n");
console.log("K2 Dataset st_rad range: 0.12 R☉ to 85 R☉\n");

testStars.forEach(star => {
  const stellarRadius = star.st_rad;
  

  const systemSizeExaggerated = Math.max(3, Math.min(stellarRadius * 6, 40 + Math.log(stellarRadius) * 8));
  

  const systemSizeNormal = Math.max(1.5, Math.min(stellarRadius * 1.8, 12 + Math.log(stellarRadius) * 3));
  

  const exploreSizeExaggerated = Math.max(1.5, Math.min(stellarRadius * 2.5, 15 + Math.log(stellarRadius) * 5));
  

  const exploreSizeNormal = Math.max(1, Math.min(stellarRadius * 1, 8 + Math.log(stellarRadius) * 2));
  
  console.log(`${star.name} (${star.st_rad} R☉):`);
  console.log(`  System View:  ${systemSizeExaggerated.toFixed(2)} units (exaggerated) | ${systemSizeNormal.toFixed(2)} units (normal)`);
  console.log(`  Explore View: ${exploreSizeExaggerated.toFixed(2)} units (exaggerated) | ${exploreSizeNormal.toFixed(2)} units (normal)`);
  console.log('');
});

console.log("\n=== SIZE RATIOS ===\n");
const sunRad = 1.0;
const dwarfRad = 0.12;
const giantRad = 10.0;
const supergiantRad = 85.0;

console.log("True physical ratios:");
console.log(`  Dwarf to Sun: ${(dwarfRad / sunRad).toFixed(2)}x`);
console.log(`  Sun to Giant: ${(giantRad / sunRad).toFixed(2)}x`);
console.log(`  Giant to Supergiant: ${(supergiantRad / giantRad).toFixed(2)}x`);

const sunSize = Math.max(3, Math.min(sunRad * 6, 40 + Math.log(sunRad) * 8));
const dwarfSize = Math.max(3, Math.min(dwarfRad * 6, 40 + Math.log(dwarfRad) * 8));
const giantSize = Math.max(3, Math.min(giantRad * 6, 40 + Math.log(giantRad) * 8));
const supergiantSize = Math.max(3, Math.min(supergiantRad * 6, 40 + Math.log(supergiantRad) * 8));

console.log("\nVisualization ratios (System View, exaggerated):");
console.log(`  Dwarf to Sun: ${(dwarfSize / sunSize).toFixed(2)}x`);
console.log(`  Sun to Giant: ${(giantSize / sunSize).toFixed(2)}x`);
console.log(`  Giant to Supergiant: ${(supergiantSize / giantSize).toFixed(2)}x`);

console.log("\nNote: Logarithmic scaling preserves differences while keeping extreme stars manageable\n");
