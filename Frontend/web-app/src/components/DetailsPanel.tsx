"use client";

export default function DetailsPanel({
  detailsRef,
}: {
  detailsRef: React.RefObject<HTMLDivElement | null>;
}) {
  return (
    <>
      <h2>Details</h2>
      <div id="details" ref={detailsRef}>
        Select a planet to see parameters and evidence.
      </div>
    </>
  );
}