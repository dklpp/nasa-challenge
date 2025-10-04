"use client";

import type { System } from "../app/data";

export default function SystemsList({
  systems,
  onSelect,
}: {
  systems: ReadonlyArray<System>;
  onSelect: (s: System) => void;
}) {
  return (
    <div id="sysList" className="list">
      {systems.map((s) => (
        <div key={s.name} className="sys" onClick={() => onSelect(s)}>
          <div className="name">{s.name}</div>
          <div className="meta">
            {s.planets.length} planets • T<sub>eff</sub> {s.star.teff}K
          </div>
        </div>
      ))}
    </div>
  );
}
