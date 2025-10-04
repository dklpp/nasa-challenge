"use client";

export default function Legend({ line1, line2 }: {
  line1?: React.ReactNode;
  line2?: React.ReactNode;
}) {
  return (
    <div id="legend">
      {line1 ?? (
        <>
          Size ∝ radius • Color ∝ T<sub>eff</sub>
        </>
      )}
      <br />
      {line2 ?? <>Speed via Kepler's 2nd law</>}
    </div>
  );
}