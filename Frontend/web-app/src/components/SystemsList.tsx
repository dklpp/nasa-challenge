"use client";

import type { System } from "../app/data";

export default function SystemsList({
  systems,
  onSelect,
  customSystemNames,
}: {
  systems: ReadonlyArray<System>;
  onSelect: (s: System) => void;
  customSystemNames?: Set<string>;
}) {
  return (
    <div id="sysList" className="list">
      {systems.map((s) => {
        const isCustom = customSystemNames?.has(s.name);
        return (
          <div 
            key={s.name} 
            className={`sys ${isCustom ? 'custom-system' : ''}`}
            onClick={() => onSelect(s)}
          >
            <div className="name">
              {isCustom && <span className="custom-badge">★</span>}
              {s.name}
            </div>
            <div className="meta">
              {s.planets.length} planets • T<sub>eff</sub> {s.star.teff}K
            </div>
          </div>
        );
      })}
    </div>
  );
}
