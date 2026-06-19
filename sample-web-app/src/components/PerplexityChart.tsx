interface PerplexityChartProps {
  data: number[];
}

export function PerplexityChart({ data }: PerplexityChartProps) {
  const points = data
    .map((val, idx) => {
      const x = (idx / Math.max(data.length - 1, 1)) * 400;
      const y = 150 - val * 150;
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <section className="card chart-card">
      <div className="card-header">
        <h2>Token-by-token perplexity waves</h2>
      </div>
      <p className="chart-subtitle">Y-Axis: Perplexity risk index (0.0 to 1.0)</p>
      <svg className="chart-svg" viewBox="0 0 400 150" preserveAspectRatio="none">
        <line x1="0" y1="0" x2="400" y2="0" stroke="#f1f5f9" strokeWidth="1" strokeDasharray="2 3" />
        <line x1="0" y1="75" x2="400" y2="75" stroke="#f1f5f9" strokeWidth="1" strokeDasharray="2 3" />
        <line x1="0" y1="150" x2="400" y2="150" stroke="#e2e8f0" strokeWidth="1" />
        <text x="5" y="12" fill="#94a3b8" fontSize="9" fontFamily="monospace" fontWeight="bold">1.0</text>
        <text x="5" y="78" fill="#94a3b8" fontSize="9" fontFamily="monospace" fontWeight="bold">0.5</text>
        <text x="5" y="145" fill="#94a3b8" fontSize="9" fontFamily="monospace" fontWeight="bold">0.0</text>
        <path d={`M ${points}`} fill="none" stroke="#4f46e5" strokeWidth="2.5" />
      </svg>
    </section>
  );
}
