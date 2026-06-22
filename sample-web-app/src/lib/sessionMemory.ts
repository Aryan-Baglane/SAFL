import { SessionStatus } from '../types';

const SUSPICIOUS_MARKERS = [
  'bomb', 'weapon', 'chlorine', 'explosive', 'hack', 'bypass', 'jailbreak',
  'ignore', 'system prompt', 'override', 'bleach', 'gas',
];

const REPUTATION_BLOCK_THRESHOLD = 0.85;
const REPUTATION_ACTIVE_ATTACK_THRESHOLD = 0.55;
const REPUTATION_SUSPICIOUS_THRESHOLD = 0.25;
const REPUTATION_COOLDOWN_THRESHOLD = 0.1;
const REPUTATION_DECAY_INACTIVITY_MS = 6 * 60 * 60 * 1000;
const REPUTATION_DECAY_FACTOR = 0.5;

export interface SessionSnapshot {
  status: SessionStatus;
  reputation: number;
  turnCount: number;
  suspiciousTurns: number;
  crescendoScore: number;
}

interface MemoryState {
  turns: string[];
  suspiciousTurns: number;
  reputation: number;
  blocked: boolean;
  lastUpdatedMs: number;
}

export class SessionMemory {
  private turns: string[] = [];
  private suspiciousTurns = 0;
  private reputation = 0;
  private blocked = false;
  private lastUpdatedMs = 0;

  reset() {
    this.turns = [];
    this.suspiciousTurns = 0;
    this.reputation = 0;
    this.blocked = false;
    this.lastUpdatedMs = 0;
  }

  previewTurn(text: string, attackSignal: number, blocked: boolean, timestampMs = Date.now()): SessionSnapshot {
    return this.toSnapshot(this.evolveRawState(this.captureState(), text, attackSignal, blocked, timestampMs));
  }

  recordTurn(text: string, attackSignal: number, blocked: boolean, timestampMs = Date.now()): SessionSnapshot {
    const nextState = this.evolveRawState(this.captureState(), text, attackSignal, blocked, timestampMs);
    this.turns = nextState.turns;
    this.suspiciousTurns = nextState.suspiciousTurns;
    this.reputation = nextState.reputation;
    this.blocked = nextState.blocked;
    this.lastUpdatedMs = nextState.lastUpdatedMs;
    return this.toSnapshot(nextState);
  }

  shouldEscalateToBlock(crescendoScore: number, reputation = this.reputation): boolean {
    return this.blocked || crescendoScore >= 0.8 || reputation >= REPUTATION_BLOCK_THRESHOLD;
  }

  previewReputationAfterInactivity(hoursIdle: number): number {
    const idleMs = hoursIdle * 60 * 60 * 1000;
    if (idleMs < REPUTATION_DECAY_INACTIVITY_MS) return Number(this.reputation.toFixed(4));
    return Number((this.reputation * REPUTATION_DECAY_FACTOR).toFixed(4));
  }

  snapshot(crescendoScore = this.computeCrescendoScore(this.turns.length, this.suspiciousTurns, this.reputation)): SessionSnapshot {
    return this.toSnapshot(this.captureState(), crescendoScore);
  }

  private captureState(): MemoryState {
    return {
      turns: [...this.turns],
      suspiciousTurns: this.suspiciousTurns,
      reputation: this.reputation,
      blocked: this.blocked,
      lastUpdatedMs: this.lastUpdatedMs,
    };
  }

  private evolveRawState(state: MemoryState, text: string, attackSignal: number, blocked: boolean, timestampMs: number): MemoryState {
    const turns = [...state.turns, text];
    const lower = text.toLowerCase();
    const markerSuspicious = SUSPICIOUS_MARKERS.some((m) => lower.includes(m));
    const normalizedSignal = Math.max(0, Math.min(1, attackSignal));
    const isSuspicious = markerSuspicious || normalizedSignal >= 0.35;

    const inactivityMs = state.lastUpdatedMs > 0 ? timestampMs - state.lastUpdatedMs : 0;
    const decayedReputation =
      state.lastUpdatedMs > 0 && inactivityMs >= REPUTATION_DECAY_INACTIVITY_MS
        ? state.reputation * REPUTATION_DECAY_FACTOR
        : state.reputation;

    const reputation = normalizedSignal > 0
      ? decayedReputation + normalizedSignal * (1 - decayedReputation)
      : decayedReputation;

    return {
      turns,
      suspiciousTurns: state.suspiciousTurns + (isSuspicious ? 1 : 0),
      reputation: Math.max(0, Math.min(1, reputation)),
      blocked: state.blocked || blocked,
      lastUpdatedMs: timestampMs,
    };
  }

  private toSnapshot(
    state: MemoryState,
    crescendoScore = this.computeCrescendoScore(state.turns.length, state.suspiciousTurns, state.reputation)
  ): SessionSnapshot {
    let status: SessionStatus = 'NORMAL';
    if (state.blocked || state.reputation >= REPUTATION_BLOCK_THRESHOLD) status = 'BLOCKED';
    else if (crescendoScore >= 0.7 || state.reputation >= REPUTATION_ACTIVE_ATTACK_THRESHOLD || state.suspiciousTurns >= 3) status = 'ACTIVE_ATTACK';
    else if (crescendoScore >= 0.35 || state.reputation >= REPUTATION_SUSPICIOUS_THRESHOLD || state.suspiciousTurns >= 1) status = 'SUSPICIOUS';
    else if (state.reputation > REPUTATION_COOLDOWN_THRESHOLD) status = 'COOLDOWN';

    return {
      status,
      reputation: Number(state.reputation.toFixed(3)),
      turnCount: state.turns.length,
      suspiciousTurns: state.suspiciousTurns,
      crescendoScore: Number(crescendoScore.toFixed(2)),
    };
  }

  private computeCrescendoScore(turnCount: number, suspiciousTurns: number, reputation: number): number {
    return Math.min(1, suspiciousTurns / 4 + (turnCount > 3 ? 0.15 : 0) + reputation * 0.35);
  }
}
