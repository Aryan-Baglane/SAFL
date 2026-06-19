import { ATTACK_PRESETS } from '../lib/attackPresets';

interface PresetPanelProps {
  crescendoIndex: number;
  onPreset: (prompt: string) => void;
  onCrescendo: () => void;
  disabled?: boolean;
}

export function PresetPanel({ crescendoIndex, onPreset, onCrescendo, disabled }: PresetPanelProps) {
  return (
    <section className="card preset-card-wrap">
      <h2 className="preset-heading">Simulation preset injection (fills chat input)</h2>
      <div className="preset-grid">
        {ATTACK_PRESETS.filter((p) => p.id !== 'safe').map((p) => (
          <button
            key={p.id}
            type="button"
            className="preset-tile"
            onClick={() => onPreset(p.prompt)}
            disabled={disabled}
          >
            <span className="preset-tile-title">{p.label}</span>
            <span className="preset-tile-desc">{p.description}</span>
          </button>
        ))}
        <button type="button" className="preset-tile" onClick={onCrescendo} disabled={disabled}>
          <span className="preset-tile-title">
            Crescendo <span className="step-pill">S {crescendoIndex + 1}</span>
          </span>
          <span className="preset-tile-desc">Multi-turn drift</span>
        </button>
        <button
          type="button"
          className="preset-tile"
          onClick={() => onPreset(ATTACK_PRESETS.find((p) => p.id === 'safe')!.prompt)}
          disabled={disabled}
        >
          <span className="preset-tile-title">Safe prompt</span>
          <span className="preset-tile-desc">Should allow</span>
        </button>
      </div>
    </section>
  );
}
