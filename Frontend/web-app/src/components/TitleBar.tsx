"use client";
import React from "react";

type Props = { system: string | null; sub: string };

export function TitleBar({ system, sub }: Props) {
  return (
    <div id="titlebar">
      <div className="system">{system ?? "—"}</div>
      <div className="sub">{sub}</div>
    </div>
  );
}

export default TitleBar;
