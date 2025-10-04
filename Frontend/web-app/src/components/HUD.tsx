"use client";
import React from "react";

type Props = {
  speed: number;
  onSpeed: (v: number) => void;
  speedVal: string;
  onReset: () => void;
  sizeScale: number;
  onSizeScale: (v: number) => void;
  sizeVal: string;
  modeLabel: string;
  onExplore: () => void;
};

export function HUD({ speed, onSpeed, speedVal, onReset, sizeScale, onSizeScale, sizeVal, modeLabel, onExplore }: Props) {
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
      <label style={{ marginLeft: 8 }}>Size</label>
      <input
        id="sizeScale"
        type="range"
        min={0.05}
        max={1}
        step={0.01}
        value={sizeScale}
        onChange={(e) => onSizeScale(parseFloat(e.target.value))}
      />
      <span id="sizeVal" className="badge">
        {sizeVal}
      </span>
      <span id="testBadge" className="badge">
        Tests: …
      </span>
      <span id="modeBadge" className="badge">{modeLabel}</span>
      <button id="exploreBtn" onClick={onExplore}>Explore</button>
      <button id="resetCam" onClick={onReset}>
        Reset view
      </button>
    </div>
  );
}

export default HUD;
