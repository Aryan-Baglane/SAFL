interface CentroidDriftChartProps {
  data: number[];
}

export function CentroidDriftChart({ data }: CentroidDriftChartProps) {
  const points = data
    .map((val, idx) => {
      const x = (idx / Math.max(data.length - 1, 1)) * 400;
      const y = 150 - val * 150;
      return `${x},${y}`;
    })
    .join(' ');

  const areaPath = `M ${points} L 400,150 L 0,150 Z`;

  return (
    <section className="card chart-card">
      <div className="card-header">
        <h2>Conversational risk &amp; centroid drift</h2>
      </div>
      <p className="chart-subtitle">Rolling 5-turn attack drift correlation</p>
      <svg className="chart-svg" viewBox="0 0 400 150" preserveAspectRatio="none">
        <line x1="0" y1="45" x2="400" y2="45" stroke="#ef4444" strokeWidth="1.2" strokeDasharray="4 2" />
        <text x="330" y="38" fill="#ef4444" fontSize="8" fontFamily="monospace" fontWeight="bold">LIMIT: 0.70</text>
        <path d={areaPath} fill="rgba(79,70,229,0.05)" />
        <path d={`M ${points}`} fill="none" stroke="#4f46e5" strokeWidth="2.5" />
        {data.map((val, idx) => {
          const x = (idx / Math.max(data.length - 1, 1)) * 400;
          const y = 150 - val * 150;
          return (
            <circle key={idx} cx={x} cy={y} r={idx === data.length - 1 ? 5 : 3.5} fill="#4f46e5" stroke="#fff" strokeWidth="1.5" />
          );
        })}
      </svg>
      <div className="turn-labels">
        {data.map((_, i) => (
          <span key={i}>Turn {i + 1}</span>
        ))}
      </div>
    </section>
  );
}
