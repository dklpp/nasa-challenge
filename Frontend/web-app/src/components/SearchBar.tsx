"use client";

export function SearchBar({
  q,
  onChange,
  onTour,
}: {
  q: string;
  onChange: (v: string) => void;
  onTour: () => void;
}) {
  return (
    <div className="search">
      <input
        id="q"
        placeholder="Search system or planet..."
        value={q}
        onChange={(e) => onChange(e.target.value)}
      />
      <button id="tourBtn" title="Auto tour" onClick={onTour}>
        Tour
      </button>
    </div>
  );
}

export default SearchBar;
