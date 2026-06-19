import { ModelSelector } from './ModelSelector';
import { ConnectionStatus, KeyMetadata } from '../types';

interface ConfigPanelProps {
  apiKey: string;
  showApiKey: boolean;
  onApiKeyChange: (v: string) => void;
  onToggleShowKey: () => void;
  connectionStatus: ConnectionStatus;
  connectionError: string;
  keyMetadata: KeyMetadata | null;
  onTestConnection: () => void;
  selectedModel: string;
  onModelChange: (v: string) => void;
}

export function ConfigPanel({
  apiKey,
  showApiKey,
  onApiKeyChange,
  onToggleShowKey,
  connectionStatus,
  connectionError,
  keyMetadata,
  onTestConnection,
  selectedModel,
  onModelChange,
}: ConfigPanelProps) {
  return (
    <section className="card">
      <div className="card-header">
        <h2>
          <span className="dot dot-indigo" />
          OpenRouter model router
        </h2>
        <span className="tag-config">CONFIG</span>
      </div>

      <div className="field">
        <div className="field-row">
          <label className="field-label">API authentication key</label>
          {apiKey ? (
            <button
              type="button"
              className={`key-status key-status-${connectionStatus}`}
              onClick={onTestConnection}
              disabled={connectionStatus === 'testing'}
            >
              {connectionStatus === 'connected'
                ? 'KEY VALIDATED ✓'
                : connectionStatus === 'testing'
                ? 'VERIFYING...'
                : connectionStatus === 'error'
                ? 'UNVERIFIED ✕'
                : 'VALIDATE KEY'}
            </button>
          ) : (
            <span className="key-status key-status-offline">NO KEY</span>
          )}
        </div>
        <div className="input-row">
          <input
            type={showApiKey ? 'text' : 'password'}
            placeholder="Enter OpenRouter Key (sk-or-v1-...)"
            value={apiKey}
            onChange={(e) => onApiKeyChange(e.target.value)}
            className="input"
          />
          <button type="button" className="btn-ghost btn-sm" onClick={onToggleShowKey}>
            {showApiKey ? 'HIDE' : 'SHOW'}
          </button>
        </div>
        {connectionError && <p className="error-text">{connectionError}</p>}
        <div className="usage-row">
          <span>Usage: ${keyMetadata ? keyMetadata.usage.toFixed(4) : '0.0000'}</span>
          <span>Limit: {keyMetadata && keyMetadata.limit > 0 ? `$${keyMetadata.limit.toFixed(2)}` : 'Unlimited'}</span>
        </div>
      </div>

      <ModelSelector selectedModel={selectedModel} onModelChange={onModelChange} />
    </section>
  );
}
