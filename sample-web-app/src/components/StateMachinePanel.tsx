import { SessionStatus } from '../types';

const STATES: SessionStatus[] = [
  'NORMAL',
  'SUSPICIOUS',
  'ACTIVE_ATTACK',
  'BLOCKED',
  'COOLDOWN',
];

interface StateMachinePanelProps {
  status: SessionStatus;
}

export function StateMachinePanel({ status }: StateMachinePanelProps) {
  const ringOffset =
    status === 'BLOCKED' ? 0 :
    status === 'ACTIVE_ATTACK' ? 0.25 :
    status === 'SUSPICIOUS' ? 0.5 :
    status === 'COOLDOWN' ? 0.75 : 1;

  const ringColor =
    status === 'BLOCKED' ? '#ef4444' :
    status === 'ACTIVE_ATTACK' ? '#f97316' :
    status === 'SUSPICIOUS' ? '#f59e0b' :
    status === 'COOLDOWN' ? '#8b5cf6' : '#2563eb';

  const circumference = 2 * Math.PI * 18;

  return (
    <section className="card state-card">
      <div className="state-card-body">
        <div>
          <h2>Redis 8 state machine</h2>
          <p className="chart-subtitle">Tracks user history and reputation across sessions.</p>
          <div className="state-badges">
            {STATES.map((st) => (
              <span key={st} className={`state-badge ${status === st ? `active-${st.toLowerCase()}` : ''}`}>
                {st}
              </span>
            ))}
          </div>
        </div>
        <div className="state-ring-wrap">
          <svg className="state-ring" viewBox="0 0 48 48">
            <circle cx="24" cy="24" r="18" stroke="#f1f5f9" strokeWidth="4.5" fill="transparent" />
            <circle
              cx="24" cy="24" r="18"
              stroke={ringColor}
              strokeWidth="4.5"
              fill="transparent"
              strokeDasharray={circumference}
              strokeDashoffset={circumference * ringOffset}
              strokeLinecap="round"
              transform="rotate(-90 24 24)"
            />
          </svg>
          <span className="state-ring-label">{status} STATE</span>
        </div>
      </div>
    </section>
  );
}
