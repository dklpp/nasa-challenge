"use client";

import { useState, useRef } from "react";
import type { System } from "../app/data";

export default function UploadModal({
  isOpen,
  onClose,
  onUpload,
}: {
  isOpen: boolean;
  onClose: () => void;
  onUpload: (system: System) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [previewData, setPreviewData] = useState<System | null>(null);
  const [error, setError] = useState<string>("");
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const validateSystem = (
    data: any
  ): { valid: boolean; error?: string; system?: System } => {
    if (!data.name || typeof data.name !== "string") {
      return { valid: false, error: "Missing or invalid 'name' field" };
    }

    if (!data.star || typeof data.star !== "object") {
      return { valid: false, error: "Missing or invalid 'star' object" };
    }

    if (typeof data.star.teff !== "number") {
      return {
        valid: false,
        error: "Missing or invalid 'star.teff' (stellar temperature)",
      };
    }

    if (!Array.isArray(data.planets)) {
      return { valid: false, error: "Missing or invalid 'planets' array" };
    }

    if (data.planets.length === 0) {
      return { valid: false, error: "System must have at least one planet" };
    }

    // Validate each planet
    for (let i = 0; i < data.planets.length; i++) {
      const planet = data.planets[i];
      if (!planet.name || typeof planet.name !== "string") {
        return {
          valid: false,
          error: `Planet ${i + 1}: Missing or invalid 'name'`,
        };
      }
      if (typeof planet.a_au !== "number" || planet.a_au <= 0) {
        return {
          valid: false,
          error: `Planet ${i + 1}: Missing or invalid 'a_au' (semi-major axis)`,
        };
      }
      if (typeof planet.period_days !== "number" || planet.period_days <= 0) {
        return {
          valid: false,
          error: `Planet ${i + 1}: Missing or invalid 'period_days'`,
        };
      }
    }

    // Build a valid System object
    const system: System = {
      name: data.name,
      star: {
        teff: data.star.teff,
        radius_rs: data.star.radius_rs,
        mass_ms: data.star.mass_ms,
      },
      planets: data.planets.map((p: any) => ({
        name: p.name,
        a_au: p.a_au,
        period_days: p.period_days,
        e: p.e,
        radius_re: p.radius_re,
        radius_rj: p.radius_rj,
        incl: p.incl,
        pl_eqt: p.pl_eqt,
      })),
      ra: data.ra,
      dec: data.dec,
      distance_pc: data.distance_pc,
    };

    return { valid: true, system };
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      processFile(selectedFile);
    }
  };

  const processFile = async (selectedFile: File) => {
    setFile(selectedFile);
    setError("");
    setPreviewData(null);

    try {
      const text = await selectedFile.text();
      const json = JSON.parse(text);

      const result = validateSystem(json);
      if (!result.valid) {
        setError(result.error || "Invalid data format");
        return;
      }

      setPreviewData(result.system!);
    } catch (err: any) {
      setError(`Failed to parse JSON: ${err.message}`);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleUploadClick = () => {
    if (previewData) {
      onUpload(previewData);
      handleClose();
    }
  };

  const handleClose = () => {
    setFile(null);
    setPreviewData(null);
    setError("");
    setDragActive(false);
    onClose();
  };

  return (
    <div className="modal-overlay" onClick={handleClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Upload Custom System</h2>
          <button className="close-btn" onClick={handleClose}>
            ×
          </button>
        </div>

        <div className="modal-body">
          <div
            className={`upload-zone ${dragActive ? "drag-active" : ""}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleFileChange}
              style={{ display: "none" }}
            />
            <div className="upload-icon">📁</div>
            <p>Click to upload or drag & drop</p>
            <p className="upload-hint">JSON file with star and planet data</p>
          </div>

          {file && (
            <div className="file-info">
              <strong>File:</strong> {file.name} (
              {(file.size / 1024).toFixed(2)} KB)
            </div>
          )}

          {error && (
            <div className="error-message">
              <strong>Error:</strong> {error}
            </div>
          )}

          {previewData && (
            <div className="preview-section">
              <h3>Preview</h3>
              <div className="preview-content">
                <div className="preview-row">
                  <span className="preview-label">System Name:</span>
                  <span className="preview-value">{previewData.name}</span>
                </div>
                <div className="preview-row">
                  <span className="preview-label">
                    Star T<sub>eff</sub>:
                  </span>
                  <span className="preview-value">
                    {previewData.star.teff} K
                  </span>
                </div>
                {previewData.star.radius_rs && (
                  <div className="preview-row">
                    <span className="preview-label">Star Radius:</span>
                    <span className="preview-value">
                      {previewData.star.radius_rs.toFixed(3)} R☉
                    </span>
                  </div>
                )}
                {previewData.star.mass_ms && (
                  <div className="preview-row">
                    <span className="preview-label">Star Mass:</span>
                    <span className="preview-value">
                      {previewData.star.mass_ms.toFixed(3)} M☉
                    </span>
                  </div>
                )}
                <div className="preview-row">
                  <span className="preview-label">Number of Planets:</span>
                  <span className="preview-value">
                    {previewData.planets.length}
                  </span>
                </div>

                <div className="planets-list">
                  <h4>Planets:</h4>
                  {previewData.planets.map((planet, idx) => (
                    <div key={idx} className="planet-item">
                      <strong>{planet.name}</strong>
                      <span>a = {planet.a_au.toFixed(3)} AU</span>
                      <span>P = {planet.period_days.toFixed(2)} days</span>
                      {planet.radius_rj && (
                        <span>R = {planet.radius_rj.toFixed(2)} Rj</span>
                      )}
                      {planet.radius_re && (
                        <span>R = {planet.radius_re.toFixed(2)} R⊕</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          <div className="example-format">
            <details>
              <summary>Example JSON format</summary>
              <pre>{`{
  "name": "TRAPPIST-1",
  "star": {
    "teff": 2566,
    "radius_rs": 0.121,
    "mass_ms": 0.089
  },
  "planets": [
    {
      "name": "b",
      "a_au": 0.01154,
      "period_days": 1.51087,
      "e": 0.006,
      "radius_re": 1.116,
      "incl": 89.56,
      "pl_eqt": 400
    },
    {
      "name": "c",
      "a_au": 0.01580,
      "period_days": 2.42182,
      "e": 0.007,
      "radius_re": 1.097,
      "incl": 89.67,
      "pl_eqt": 342
    }
  ],
  "ra": 346.6223,
  "dec": -5.0413,
  "distance_pc": 12.43
}`}</pre>
            </details>
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn-cancel" onClick={handleClose}>
            Cancel
          </button>
          <button
            className="btn-upload"
            onClick={handleUploadClick}
            disabled={!previewData}
          >
            Add System
          </button>
        </div>
      </div>
    </div>
  );
}
