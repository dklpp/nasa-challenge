"use client";

import { useState, useEffect } from "react";

export default function ModelPredictionModal({
  isOpen,
  onClose,
  onShowInSpace,
}: {
  isOpen: boolean;
  onClose: () => void;
  onShowInSpace: () => void;
}) {
  const [stage, setStage] = useState<"loading" | "result">("loading");
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (isOpen) {
      setStage("loading");
      setProgress(0);

      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) {
            setStage("result");
            clearInterval(interval);
            return 100;
          }
          return prev + 2; 
        });
      }, 100);

      return () => clearInterval(interval);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="model-prediction-overlay">
      <div className="model-prediction-modal">
        <div className="modal-content">
          {stage === "loading" ? (
            <div className="loading-stage">
              <div className="loading-icon">
                <div className="spinner"></div>
              </div>
              <h2>AI Model Processing</h2>
              <p>Model predicting exoplanets probability...</p>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <span className="progress-text">{progress}%</span>
            </div>
          ) : (
            <div className="result-stage">
              <div className="success-icon">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" fill="#10B981" />
                  <path
                    d="M9 12l2 2 4-4"
                    stroke="white"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </div>
              <h2>Exoplanet Confirmed!</h2>
              <div className="accuracy-display">
                <span className="accuracy-label">Probability:</span>
                <span className="accuracy-value">92.67%</span>
              </div>
              <p>Would you like to show it in the space?</p>
              <div className="button-group">
                <button 
                  className="btn-secondary" 
                  onClick={onClose}
                >
                  Not Now
                </button>
                <button 
                  className="btn-primary" 
                  onClick={() => {
                    onShowInSpace();
                    onClose();
                  }}
                >
                  Show in Space
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}