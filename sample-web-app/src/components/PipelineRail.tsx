import { PipelineNodeState } from '../lib/telemetry';

const NODES = [
  { label: 'Intake', code: 'INT' },
  { label: 'Heuristics', code: 'HEUR' },
  { label: 'ONNX MiniLM', code: 'ONNX' },
  { label: 'Temporal BiLSTM', code: 'LSTM' },
  { label: 'Redis VSIM', code: 'VSIM' },
  { label: 'Aggregator', code: 'AGGR' },
];

interface PipelineRailProps {
  states: PipelineNodeState[];
  activeNode: number | null;
  latencyMs: number;
  isScanning: boolean;
}

export function PipelineRail({ states, activeNode, latencyMs, isScanning }: PipelineRailProps) {
  return (
    <section className="card">
      <div className="card-header">
        <h2>
          <span className="dot dot-indigo" />
          Active pipeline telemetry rail
        </h2>
        <span className="card-meta">LATENCY: ~{latencyMs}ms</span>
      </div>

      <div className="pipeline-rail">
        <div className="pipeline-track" />
        {NODES.map((node, idx) => {
          const state = states[idx] ?? 'idle';
          const isActive = activeNode === idx || (isScanning && activeNode === null && idx === 0);
          return (
            <div key={node.code} className="pipeline-node">
              <div className={`pipeline-circle state-${state} ${isActive ? 'pulse' : ''}`}>
                {node.code}
              </div>
              <span className={`pipeline-label ${isActive ? 'active' : ''}`}>{node.label}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}
