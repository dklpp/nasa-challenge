"use client";
import React from "react";

type Props = {
  detailsRef: React.RefObject<HTMLDivElement | null>;
};

export function DetailsPanel({ detailsRef }: Props) {
  return (
    <>
      <h2>Details</h2>
      <div id="details" ref={detailsRef}>
        Select a planet to see parameters and evidence.
      </div>
      <div className="thumbs" id="thumbs">
        <div className="thumb">Phase‑folded light curve (placeholder)</div>
        <div className="thumb">BLS periodogram (placeholder)</div>
      </div>
    </>
  );
}

export default DetailsPanel;
