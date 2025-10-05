# Custom System Highlighting Feature

## Overview
When users upload a custom exoplanet system, it is visually highlighted in multiple ways to make it stand out from the existing systems in the database.

## Visual Effects Implemented

### 1. 3D Solar System View (Earth-Centered Mode)

#### Star Highlighting
- **Increased Emissive Intensity**: Custom stars glow 33% brighter (2.0 vs 1.5)
- **Pulsing Outer Glow Sphere**: A translucent green sphere (2x star size) pulses around the star
  - Color: `#00ff88` (bright cyan-green)
  - Opacity: Pulses between 0.15 and 0.45
  - Scale: Pulses between 0.9x and 1.1x
  - Animation speed: 2 Hz sine wave
  - Uses additive blending for glow effect

#### Highlight Ring
- **Rotating Ring**: A flat ring orbits the star in the XY plane
  - Inner radius: 2.5x star size
  - Outer radius: 3x star size
  - Color: `#00ffaa` (bright green-cyan)
  - Opacity: Pulses between 0.4 and 0.8
  - Rotation speed: 0.5 radians/second
  - Uses additive blending

#### Point Light
- **Custom Green Light**: A bright point light at the star's position
  - Color: `#00ff88`
  - Intensity: 2.0
  - Distance: 100 units
  - Decay: 2

#### Orbit Highlighting
- **Green Orbits**: Planet orbits are rendered in green instead of blue
  - Color: `#00ffaa` vs `#5588dd` for standard systems
  - Opacity: 0.9 vs 0.7 (more visible)

### 2. Systems List (Sidebar)

#### Visual Indicators
- **Star Badge**: A ★ symbol appears before the system name
  - Color: `#00ffaa` (bright green)
  - Font size: 16px
  - Pulsing animation (2s cycle)

#### Card Styling
- **Border Color**: Green border instead of blue
  - Color: `#00aa77`
- **Box Shadow**: Green glow effect
  - Base: `0 0 8px rgba(0,170,119,0.2)`
  - Hover: `0 0 12px rgba(0,170,119,0.4)`
- **Background**: Slightly different shade
  - Base: `#0d1930` vs `#0b142e`
  - Hover: `#0f1f3f` vs `#0e1a3c`

#### Animation
- **Badge Pulse**: The star badge pulses continuously
  - Keyframes: opacity 1→0.7→1, scale 1→1.1→1
  - Duration: 2 seconds
  - Easing: ease-in-out
  - Infinite loop

### 3. User Feedback

#### Toast Notification
When a system is uploaded, a special toast message appears:
```
✨ [System Name] added! Look for the glowing green highlight in the map!
```
- Duration: 3000ms (3 seconds)
- Includes sparkle emoji for visual appeal
- Explicitly tells users to look for the highlight

## Technical Implementation

### State Management
- `customSystemNames`: A Set<string> tracking all uploaded system names
- Persisted across rebuilds of the 3D scene
- Used to conditionally apply highlighting effects

### Animation Loop
Custom system effects are animated in the `tick()` function:
```typescript
// Animate custom system highlights
if (current.group) {
  const time = clock.getElapsedTime();
  current.group.children.forEach((child: any) => {
    if (child.userData.customGlow) {
      // Pulsing glow effect
      const glowPulse = Math.sin(time * 2) * 0.15 + 0.3;
      child.userData.customGlow.material.opacity = glowPulse;
      
      // Scale pulsing
      const scalePulse = 1 + Math.sin(time * 2) * 0.1;
      child.userData.customGlow.scale.set(scalePulse, scalePulse, scalePulse);
    }
    
    if (child.userData.customRing) {
      // Rotate the highlight ring
      child.userData.customRing.rotation.z += dt * 0.5;
      
      // Pulsing opacity
      const ringPulse = Math.sin(time * 3) * 0.2 + 0.6;
      child.userData.customRing.material.opacity = ringPulse;
    }
  });
}
```

### Performance Considerations
- Additive blending is used for glow effects (no overdraw issues)
- Simple sine waves for animations (minimal CPU usage)
- Glow and ring are only added to custom systems (not all systems)
- Point lights are limited to custom systems to avoid exceeding WebGL limits

## Color Scheme
All custom system highlights use a consistent green/cyan color palette:
- Primary: `#00ffaa` (bright cyan-green)
- Secondary: `#00ff88` (bright green-cyan)
- Border: `#00aa77` (darker green)

This creates a cohesive visual language that clearly distinguishes custom systems from:
- Standard systems (blue/white)
- The Sun (orange/yellow)
- Earth (blue)

## User Experience Benefits
1. **Immediate Visual Feedback**: Users can instantly see their uploaded system
2. **Easy Navigation**: The pulsing effects make it easy to find custom systems in a crowded star field
3. **Consistent Highlighting**: Same visual language in both 3D view and list
4. **Non-Intrusive**: Highlighting is prominent but doesn't obscure other systems
5. **Professional Look**: Smooth animations and cohesive color scheme
