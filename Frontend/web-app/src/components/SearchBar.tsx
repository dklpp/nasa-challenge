"use client";

export default function SearchBar({
  q,
  onChange,
}: {
  q: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="search">
      <input
        id="q"
        placeholder="Search system or planet..."
        value={q}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}