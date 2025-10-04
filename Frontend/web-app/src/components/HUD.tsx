"use client";
import React from "react";

type Props = {
  speed: number;
  onSpeed: (v: number) => void;
  speedVal: string;
  onReset: () => void;
};

export function HUD({ speed, onSpeed, speedVal, onReset }: Props) {
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
      <span id="testBadge" className="badge">
        Tests: …
      </span>
      <button id="resetCam" onClick={onReset}>
        Reset view
      </button>
    </div>
  );
}

export default HUD;
