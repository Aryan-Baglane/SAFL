import { SessionStatus } from '../types';

const SUSPICIOUS_MARKERS = [
  'bomb', 'weapon', 'chlorine', 'explosive', 'hack', 'bypass', 'jailbreak',
  'ignore', 'system prompt', 'override', 'bleach', 'gas',
];

export interface SessionSnapshot {
  status: SessionStatus;
  reputation: number;
  turnCount: number;
  suspiciousTurns: number;
  crescendoScore: number;
}

export class SessionMemory {
  private turns: string[] = [];
  private suspiciousTurns = 0;
  private reputation = 1.0;
  private blocked = false;

  reset() {
    this.turns = [];
    this.suspiciousTurns = 0;
    this.reputation = 1.0;
    this.blocked = false;
  }

  recordTurn(text: string, blocked: boolean): SessionSnapshot {
    this.turns.push(text);
    const lower = text.toLowerCase();
    const isSuspicious = SUSPICIOUS_MARKERS.some((m) => lower.includes(m));

    if (isSuspicious) this.suspiciousTurns += 1;
    if (blocked) {
      this.blocked = true;
      this.reputation = Math.max(0.05, this.reputation - 0.35);
    } else if (isSuspicious) {
      this.reputation = Math.max(0.05, this.reputation - 0.12);
    } else {
      this.reputation = Math.min(1, this.reputation + 0.02);
    }

    const crescendoScore = Math.min(1, this.suspiciousTurns / 4 + (this.turns.length > 3 ? 0.2 : 0));

    return this.snapshot(crescendoScore);
  }

  shouldEscalateToBlock(crescendoScore: number): boolean {
    return this.blocked || crescendoScore >= 0.75 || this.reputation < 0.25;
  }

  snapshot(crescendoScore = 0): SessionSnapshot {
    let status: SessionStatus = 'NORMAL';
    if (this.blocked || this.reputation < 0.25) status = 'BLOCKED';
    else if (crescendoScore >= 0.65 || this.suspiciousTurns >= 3) status = 'ACTIVE_ATTACK';
    else if (crescendoScore >= 0.35 || this.suspiciousTurns >= 1) status = 'SUSPICIOUS';
    else if (this.reputation < 0.85) status = 'COOLDOWN';

    return {
      status,
      reputation: Number(this.reputation.toFixed(3)),
      turnCount: this.turns.length,
      suspiciousTurns: this.suspiciousTurns,
      crescendoScore: Number(crescendoScore.toFixed(2)),
    };
  }
}
