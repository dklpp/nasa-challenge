"use client";
import React from "react";

type Props = {
  exSizes: boolean;
  logScale: boolean;
  showHZ: boolean;
  dimOrbits: boolean;
  onChange: (patch: Partial<Pick<Props, "exSizes" | "logScale" | "showHZ" | "dimOrbits">>) => void;
};

export function FiltersPanel({ exSizes, logScale, showHZ, dimOrbits, onChange }: Props) {
  return (
    <div className="filters">
      <label className="toggle">
        <input
          type="checkbox"
          id="exSizes"
          checked={exSizes}
          onChange={(e) => onChange({ exSizes: e.target.checked })}
        />
        Exaggerate sizes
      </label>
      <label className="toggle">
        <input
          type="checkbox"
          id="logScale"
          checked={logScale}
          onChange={(e) => onChange({ logScale: e.target.checked })}
        />
        Log orbit scale
      </label>
      <label className="toggle">
        <input
          type="checkbox"
          id="showHZ"
          checked={showHZ}
          onChange={(e) => onChange({ showHZ: e.target.checked })}
        />
        Habitable zone
      </label>
      <label className="toggle">
        <input
          type="checkbox"
          id="dimOrbits"
          checked={dimOrbits}
          onChange={(e) => onChange({ dimOrbits: e.target.checked })}
        />
        Dim orbits
      </label>
    </div>
  );
}

export default FiltersPanel;
