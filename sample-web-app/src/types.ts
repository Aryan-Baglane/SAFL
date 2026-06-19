import { GuardrailAction, GuardrailResult } from 'safellmkit-js';

export type ConnectionStatus = 'offline' | 'testing' | 'connected' | 'error';

export type SessionStatus = 'NORMAL' | 'SUSPICIOUS' | 'ACTIVE_ATTACK' | 'BLOCKED' | 'COOLDOWN';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  action: GuardrailAction | 'BYPASSED' | 'FLAG';
  riskScore: number;
  blocked?: boolean;
  providerCalled?: boolean;
  findings?: GuardrailResult['findings'];
}

export interface SendResult {
  blocked: boolean;
  userMessage: ChatMessage;
  assistantMessage?: ChatMessage;
  inspection: GuardrailResult;
  providerCalled: boolean;
  bypassed?: boolean;
  sessionStatus: SessionStatus;
  reputation: number;
  turnCount: number;
}

export interface KeyMetadata {
  limit: number;
  usage: number;
  label: string;
}
