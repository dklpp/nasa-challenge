"use client";

type Props = {
  logScale: boolean;
  showHZ: boolean;
  onChange: (patch: Partial<Pick<Props, "logScale" | "showHZ">>) => void;
};

export function FiltersPanel({ logScale, showHZ, onChange }: Props) {
  return (
    <div className="filters">
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
    </div>
  );
}

export default FiltersPanel;
