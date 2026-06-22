interface ReputationCurveProps {
  reputation: number;
  reputationTime: number;
  decayStartReputation: number;
  onTimeChange: (t: number) => void;
}

export function ReputationCurve({
  reputation,
  reputationTime,
  decayStartReputation,
  onTimeChange,
}: ReputationCurveProps) {
  const points: string[] = [];
  const r0 = decayStartReputation;
  for (let t = 0; t <= 24; t += 1) {
    const val = t < 6 ? r0 : r0 * 0.5;
    const x = (t / 24) * 400;
    const y = 90 - val * 80;
    points.push(`${x},${y}`);
  }

  const cx = (reputationTime / 24) * 400;
  const score = reputationTime < 6 ? r0 : r0 * 0.5;
  const cy = 90 - score * 80;

  return (
    <section className="card chart-card">
      <div className="card-header">
        <h2>Reputation decay preview</h2>
      </div>
      <p className="chart-subtitle">
        SDK-aligned model: no per-turn decay; halve score once after 6 idle hours.
      </p>
      <svg className="chart-svg chart-svg-sm" viewBox="0 0 400 100">
        <line x1="0" y1="90" x2="400" y2="90" stroke="#cbd5e1" strokeWidth="1" />
        <line x1="0" y1="0" x2="0" y2="100" stroke="#cbd5e1" strokeWidth="1" />
        <path d={`M ${points.join(' ')}`} fill="none" stroke="#8b5cf6" strokeWidth="2.5" />
        <circle cx={cx} cy={cy} r="6" fill="#8b5cf6" stroke="#fff" strokeWidth="1.5" />
      </svg>
      <div className="rep-slider-row">
        <span className="chart-subtitle">idle={reputationTime}h</span>
        <span className="rep-score">R-SCORE: {reputation.toFixed(4)}</span>
      </div>
      <input
        type="range"
        min={0}
        max={24}
        value={reputationTime}
        onChange={(e) => onTimeChange(Number(e.target.value))}
        className="rep-range"
      />
      <div className="turn-labels">
        <span>0h</span>
        <span>6h decay gate</span>
        <span>24h</span>
      </div>
    </section>
  );
}
