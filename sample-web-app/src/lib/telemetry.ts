import { GuardrailResult } from 'safellmkit-js';
import { SessionSnapshot } from './sessionMemory';

export type PipelineNodeState = 'idle' | 'active' | 'warning' | 'danger';

export interface TelemetrySnapshot {
  perplexityData: number[];
  mutationBars: number[];
  centroidDrift: number[];
  pipelineStates: PipelineNodeState[];
  latencyMs: number;
}

const DEFAULT_TELEMETRY: TelemetrySnapshot = {
  perplexityData: [0.15, 0.12, 0.16, 0.14, 0.15, 0.13, 0.14, 0.15, 0.13, 0.14],
  mutationBars: [0.08, 0.11, 0.09, 0.12, 0.1, 0.07, 0.11, 0.09, 0.12, 0.08],
  centroidDrift: [0.05, 0.08, 0.06, 0.07, 0.05],
  pipelineStates: ['idle', 'idle', 'idle', 'idle', 'idle', 'idle'],
  latencyMs: 15,
};

export function buildTelemetry(
  inspection: GuardrailResult,
  prompt: string,
  session: SessionSnapshot,
  prevDrift: number[],
  firewallActive: boolean
): TelemetrySnapshot {
  if (!firewallActive) {
    return { ...DEFAULT_TELEMETRY, latencyMs: 8 };
  }

  const risk = inspection.riskScore / 100;
  const specialChars = (prompt.match(/[^a-zA-Z0-9\s]/g) || []).length;
  const specialRatio = specialChars / Math.max(prompt.length, 1);
  const isGcg = specialRatio > 0.12 && prompt.length > 25;
  const hasMl = inspection.findings.some((f) => f.category === 'ML_CLASSIFIER');
  const hasJailbreak = inspection.findings.some((f) => f.category === 'PROMPT_INJECTION');

  let basePpl = 0.12 + risk * 0.35 + specialRatio * 1.2;
  if (isGcg) basePpl = Math.max(basePpl, 0.85);
  if (hasJailbreak) basePpl = Math.max(basePpl, 0.55);

  const perplexityData = Array.from({ length: 10 }, (_, i) => {
    const wave = basePpl + Math.sin(i * 0.9) * 0.04 + (i / 9) * risk * 0.15;
    return Number(Math.min(0.98, Math.max(0.08, wave)).toFixed(2));
  });

  let baseSmooth = 0.08 + risk * 0.45 + (hasMl ? 0.25 : 0);
  if (isGcg) baseSmooth = Math.max(baseSmooth, 0.82);

  const mutationBars = Array.from({ length: 10 }, (_, i) => {
    const v = baseSmooth + Math.cos(i * 1.1) * 0.06;
    return Number(Math.min(0.98, Math.max(0.05, v)).toFixed(2));
  });

  let driftValue = 0.05 + risk * 0.55 + session.crescendoScore * 0.35;
  if (inspection.action === 'BLOCK') driftValue = Math.max(driftValue, 0.78);
  const centroidDrift = [...prevDrift.slice(1), Number(driftValue.toFixed(2))];

  const pipelineStates: PipelineNodeState[] = [
    'active',
    hasJailbreak ? 'danger' : risk > 0.35 ? 'warning' : 'active',
    hasMl || risk > 0.5 ? (risk > 0.7 ? 'danger' : 'warning') : 'active',
    session.crescendoScore > 0.4 ? (session.crescendoScore > 0.65 ? 'danger' : 'warning') : 'active',
    driftValue > 0.5 ? 'warning' : 'active',
    inspection.action === 'BLOCK' ? 'danger' : inspection.action === 'SANITIZE' ? 'warning' : 'active',
  ];

  const latencyMs = Math.round(12 + risk * 18 + (hasMl ? 8 : 0));

  return { perplexityData, mutationBars, centroidDrift, pipelineStates, latencyMs };
}

export { DEFAULT_TELEMETRY };
