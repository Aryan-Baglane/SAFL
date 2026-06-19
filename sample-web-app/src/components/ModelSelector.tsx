import { useState } from 'react';
import { ENTERPRISE_MODELS } from '../lib/models';

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (model: string) => void;
}

export function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');

  const filtered = ENTERPRISE_MODELS.filter(
    (m) =>
      m.name.toLowerCase().includes(search.toLowerCase()) ||
      m.id.toLowerCase().includes(search.toLowerCase())
  );

  const trimmed = search.trim();
  const showCustom =
    trimmed.length > 0 &&
    !ENTERPRISE_MODELS.some((m) => m.id === trimmed) &&
    trimmed !== selectedModel;

  const displayName =
    ENTERPRISE_MODELS.find((m) => m.id === selectedModel)?.name ?? selectedModel;

  return (
    <div className="model-selector">
      <label className="field-label">Target model endpoint</label>
      <button
        type="button"
        className="model-trigger"
        onClick={() => setOpen(!open)}
      >
        <span className="model-trigger-text">{selectedModel}</span>
        <span className="model-trigger-chevron">▼</span>
      </button>
      <p className="model-display-name">{displayName}</p>

      {open && (
        <div className="model-dropdown">
          <input
            type="text"
            className="input"
            placeholder="Search or type custom model (e.g. cohere/north-mini-code:free)"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onClick={(e) => e.stopPropagation()}
          />
          <div className="model-list">
            {showCustom && (
              <button
                type="button"
                className="model-option model-option-custom"
                onClick={() => {
                  onModelChange(trimmed);
                  setSearch('');
                  setOpen(false);
                }}
              >
                + Use custom: &quot;{trimmed}&quot;
              </button>
            )}
            {filtered.map((m) => (
              <button
                key={m.id}
                type="button"
                className={`model-option ${selectedModel === m.id ? 'selected' : ''}`}
                onClick={() => {
                  onModelChange(m.id);
                  setSearch('');
                  setOpen(false);
                }}
              >
                <span className="model-option-name">{m.name}</span>
                <span className="model-option-id">{m.id}</span>
              </button>
            ))}
          </div>
          <div className="model-custom-row">
            <input
              type="text"
              className="input"
              placeholder="Custom model ID"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <button
              type="button"
              className="btn-secondary btn-sm"
              disabled={!trimmed}
              onClick={() => {
                if (trimmed) {
                  onModelChange(trimmed);
                  setSearch('');
                  setOpen(false);
                }
              }}
            >
              Apply
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
