"use client";

import { useState, useRef } from "react";
import type { System, ExoplanetRecord } from "../app/data";

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
  const [originalRecords, setOriginalRecords] = useState<ExoplanetRecord[]>([]);
  const [error, setError] = useState<string>("");
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const validateExoplanetData = (
    data: any
  ): { valid: boolean; error?: string; system?: System; records?: ExoplanetRecord[] } => {
    if (!Array.isArray(data)) {
      return { valid: false, error: "Data must be an array of exoplanet records" };
    }

    if (data.length === 0) {
      return { valid: false, error: "Array must contain at least one exoplanet record" };
    }

    const systemMap = new Map<string, ExoplanetRecord[]>();
    
    for (let i = 0; i < data.length; i++) {
      const record = data[i];
      
      if (!record.pl_name || typeof record.pl_name !== "string") {
        return {
          valid: false,
          error: `Record ${i + 1}: Missing or invalid 'pl_name' field`,
        };
      }
      
      if (!record.hostname || typeof record.hostname !== "string") {
        return {
          valid: false,
          error: `Record ${i + 1}: Missing or invalid 'hostname' field`,
        };
      }

      const hostname = record.hostname;
      if (!systemMap.has(hostname)) {
        systemMap.set(hostname, []);
      }
      systemMap.get(hostname)!.push(record);
    }

    // For now, take the first system found
    const [systemName, records] = Array.from(systemMap.entries())[0];
    const firstRecord = records[0];

    // Build a System object from the exoplanet records
    const system: System = {
      name: systemName,
      star: {
        teff: firstRecord.st_teff || 5778, // Default to Sun-like if missing
        radius_rs: firstRecord.st_rad,
        mass_ms: firstRecord.st_mass,
      },
      planets: records.map((record) => ({
        name: record.pl_name.replace(record.hostname, '').trim(), // Remove host name to get planet designation
        a_au: record.pl_orbsmax || 1.0, // Default to 1 AU if missing
        period_days: record.pl_orbper || 365.25, // Default to 1 year if missing
        e: record.pl_orbeccen,
        radius_re: record.pl_rade,
        incl: undefined, // Not directly available in this format
        pl_eqt: record.pl_eqt,
      })),
      ra: firstRecord.ra,
      dec: firstRecord.dec,
      distance_pc: firstRecord.sy_dist,
    };

    return { valid: true, system, records: data };
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
    setOriginalRecords([]);

    try {
      const text = await selectedFile.text();
      const json = JSON.parse(text);

      const result = validateExoplanetData(json);
      if (!result.valid) {
        setError(result.error || "Invalid data format");
        return;
      }

      setPreviewData(result.system!);
      setOriginalRecords(result.records || []);
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
    setOriginalRecords([]);
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
            <p className="upload-hint">JSON file with NASA Exoplanet Archive format</p>
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

          {previewData && originalRecords.length > 0 && (
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
                {previewData.distance_pc && (
                  <div className="preview-row">
                    <span className="preview-label">Distance:</span>
                    <span className="preview-value">
                      {previewData.distance_pc.toFixed(2)} pc
                    </span>
                  </div>
                )}
                <div className="preview-row">
                  <span className="preview-label">Records Loaded:</span>
                  <span className="preview-value">
                    {originalRecords.length}
                  </span>
                </div>

                <div className="planets-list">
                  <h4>Planets:</h4>
                  {originalRecords.map((record, idx) => (
                    <div key={idx} className="planet-item">
                      <strong>{record.pl_name}</strong>
                      {record.pl_orbsmax && (
                        <span>a = {record.pl_orbsmax.toFixed(3)} AU</span>
                      )}
                      {record.pl_orbper && (
                        <span>P = {record.pl_orbper.toFixed(2)} days</span>
                      )}
                      {record.pl_rade && (
                        <span>R = {record.pl_rade.toFixed(2)} R⊕</span>
                      )}
                      {record.pl_eqt && (
                        <span>T = {record.pl_eqt.toFixed(0)} K</span>
                      )}
                      {record.disposition && (
                        <span className="disposition">{record.disposition}</span>
                      )}
                      {record.discoverymethod && (
                        <span className="method">{record.discoverymethod}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          <div className="example-format">
            <details>
              <summary>Example JSON format (NASA Exoplanet Archive)</summary>
              <pre>{`{
    "pl_name": "Wolf 503 b",
    "hostname": "Wolf 503",
    "default_flag": 1,
    "disposition": "CONFIRMED",
    "disp_refname": "Peterson et al. 2018",
    "sy_snum": 1.0,
    "sy_pnum": 1.0,
    "discoverymethod": "Transit",
    "disc_year": 2018.0,
    "disc_facility": "K2",
    "soltype": "Published Confirmed",
    "pl_controv_flag": 0.0,
    "pl_refname": "<a refstr=BONOMO_ET_AL__2023 href=https://ui.adsabs.harvard.edu/abs/2023arXiv230405773B/abstract target=ref>Bonomo et al. 2023</a>",
    "pl_orbper": 6.00127,
    "pl_orbpererr1": 0.000021,
    "pl_orbpererr2": -0.000021,
    "pl_orbsmax": 0.05712,
    "pl_rade": 2.043,
    "pl_bmasse": 6.27,
    "pl_orbeccen": 0.409,
    "pl_insol": 64.7,
    "pl_eqt": 789.0,
    "st_teff": 4716.0,
    "st_rad": 0.689,
    "st_mass": 0.688,
    "st_met": -0.47,
    "ra": 206.8461979,
    "dec": -6.1393369,
    "sy_dist": 44.526,
    "sy_vmag": 10.27,
    "sy_kmag": 7.617,
    "sy_gaiamag": 9.89816,
    "rowupdate": "2023-04-17",
    "pl_pubdate": "2023-04",
    "releasedate": "2023-04-17"
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
