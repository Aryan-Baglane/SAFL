import { useEffect, useRef } from 'react';
import { GuardrailAction } from 'safellmkit-js';
import { ChatMessage } from '../types';

interface ChatPanelProps {
  messages: ChatMessage[];
  input: string;
  onInputChange: (v: string) => void;
  onSend: () => void;
  isLoading: boolean;
  modelReady: boolean;
  firewallActive: boolean;
  turnCount: number;
}

export function ChatPanel({
  messages,
  input,
  onInputChange,
  onSend,
  isLoading,
  modelReady,
  firewallActive,
  turnCount,
}: ChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <section className="card chat-card">
      <div className="card-header">
        <h2>
          <span className="dot dot-indigo" />
          Active chat session
        </h2>
        <span className="tag-turns">TURNS: {turnCount}</span>
      </div>

      <div className="chat-scroll">
        {messages.map((msg) => {
          if (msg.role === 'system') return null;
          const isUser = msg.role === 'user';
          const blocked = msg.blocked || msg.action === GuardrailAction.BLOCK;
          const flagged = msg.action === 'FLAG' || msg.action === GuardrailAction.SANITIZE;

          return (
            <div key={msg.id} className={`chat-bubble-row ${isUser ? 'end' : 'start'}`}>
              <div className="chat-meta">
                <span>{isUser ? 'CLIENT' : 'FIREWALL'}</span>
                <span>•</span>
                <span className={`action-${String(msg.action).toLowerCase()}`}>{msg.action}</span>
                {blocked && msg.providerCalled === false && (
                  <>
                    <span>•</span>
                    <span className="provider-skip">provider skipped</span>
                  </>
                )}
              </div>
              <div
                className={`chat-bubble ${isUser ? 'user' : ''} ${blocked ? 'blocked' : ''} ${flagged ? 'flagged' : ''}`}
              >
                {msg.content}
              </div>
              {msg.riskScore > 0 && (
                <div className="risk-tag">
                  RISK:{' '}
                  <strong className={msg.riskScore > 70 ? 'high' : msg.riskScore > 40 ? 'med' : 'low'}>
                    {(msg.riskScore <= 1 ? msg.riskScore * 100 : msg.riskScore).toFixed(0)}%
                  </strong>
                </div>
              )}
            </div>
          );
        })}
        {isLoading && (
          <div className="chat-bubble-row start">
            <div className="chat-meta">
              <span>SECURITY AGENT</span>
              <span className="ping-dot" />
            </div>
            <div className="chat-bubble loading">Evaluating prompt through guardrail pipeline…</div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-compose">
        <textarea
          className="chat-textarea"
          placeholder={
            firewallActive
              ? 'Type prompt (blocked inputs never reach OpenRouter)…'
              : 'Type prompt (firewall bypassed)…'
          }
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              onSend();
            }
          }}
          disabled={isLoading}
          rows={3}
        />
        <div className="chat-compose-footer">
          <span className="system-tag">SYSTEM: {modelReady ? 'ONLINE' : 'BOOTING'}</span>
          <button
            type="button"
            className="btn-primary"
            onClick={onSend}
            disabled={isLoading || !input.trim()}
          >
            {isLoading ? 'SCANNING…' : 'SEND'}
          </button>
        </div>
      </div>
    </section>
  );
}
