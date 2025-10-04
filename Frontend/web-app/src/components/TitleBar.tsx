"use client";

export default function TitleBar({
  system,
  sub,
}: {
  system: string | null;
  sub: string;
}) {
  return (
    <div id="titlebar">
      <div className="system">{system ?? "—"}</div>
      <div className="sub">{sub}</div>
    </div>
  );
}
