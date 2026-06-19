interface SmoothLLMChartProps {
  data: number[];
}

export function SmoothLLMChart({ data }: SmoothLLMChartProps) {
  return (
    <section className="card chart-card">
      <div className="card-header">
        <h2>SmoothLLM ensemble stability index</h2>
      </div>
      <p className="chart-subtitle">10 alphanumeric mutations / variance checks</p>
      <div className="bar-chart">
        {data.map((val, idx) => {
          const height = val * 100;
          let barClass = 'bar-stable';
          if (val > 0.7) barClass = 'bar-danger';
          else if (val > 0.3) barClass = 'bar-warn';
          return (
            <div key={idx} className="bar-col">
              <div className={`bar ${barClass}`} style={{ height: `${height}%` }} />
              <span className="bar-label">{idx + 1}</span>
            </div>
          );
        })}
      </div>
      <div className="chart-legend">
        <span><i className="legend-dot legend-blue" /> Stable (&lt; 0.20)</span>
        <span><i className="legend-dot legend-red" /> High variance</span>
      </div>
    </section>
  );
}
