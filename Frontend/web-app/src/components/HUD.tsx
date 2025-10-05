"use client";

export function HUD({
  speed,
  onSpeed,
  speedVal,
  onReset,
  modeLabel,
  onToggleDetails,
  detailsVisible,
  onBackToMain,
  showBackButton,
}: {
  speed: number;
  onSpeed: (v: number) => void;
  speedVal: string;
  onReset: () => void;
  modeLabel: string;
  onToggleDetails?: () => void;
  detailsVisible?: boolean;
  onBackToMain?: () => void;
  showBackButton?: boolean;
}) {
  return (
    <div id="hud">
      <label>Speed</label>
      <input
        id="speed"
        type="range"
        min={0}
        max={5}
        step={0.01}
        value={speed}
        onChange={(e) => onSpeed(parseFloat(e.target.value))}
      />
      <span id="speedVal" className="badge">
        {speedVal}
      </span>
      <span id="modeBadge" className="badge whitespace-nowrap">
        {modeLabel}
      </span>
      {onToggleDetails && (
        <button id="toggleDetails" onClick={onToggleDetails}>
          {detailsVisible ? "Hide details" : "Show details"}
        </button>
      )}
      {showBackButton && onBackToMain && (
        <button
          id="backToMain"
          className="whitespace-nowrap"
          onClick={onBackToMain}
        >
          ← Back to main view
        </button>
      )}
      <button id="resetCam" className="whitespace-nowrap" onClick={onBackToMain}>
        Reset view
      </button>
    </div>
  );
}

export default HUD;
