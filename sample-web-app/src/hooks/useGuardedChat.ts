import { useCallback, useEffect, useRef, useState } from 'react';
import { GuardrailAction, GuardrailResult, SafeLLMKit, OnnxClassifier } from 'safellmkit-js';
import { chatCompletion } from '../lib/openRouter';
import { SessionMemory } from '../lib/sessionMemory';
import { buildTelemetry, DEFAULT_TELEMETRY, TelemetrySnapshot } from '../lib/telemetry';
import { ChatMessage, SendResult, SessionStatus } from '../types';

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function blockedMessage(result: GuardrailResult, stage: 'input' | 'output' | 'session') {
  const top = result.findings[0]?.message;
  if (stage === 'session') {
    return `Security Advisory: Session memory escalated to BLOCK. Repeated suspicious turns detected. Reason: ${top || 'Crescendo pattern'}`;
  }
  if (stage === 'output') {
    return `Security Advisory: Model output blocked by SafeLLMKit firewall. ${top || ''}`.trim();
  }
  return `Security Advisory: Input blocked before provider call. ${top || 'Policy violation detected.'}`.trim();
}

async function animatePipeline(
  setActive: (n: number | null) => void,
  setStates: (s: TelemetrySnapshot['pipelineStates']) => void,
  isDanger: boolean,
  risk: number,
  finalAction: string
) {
  const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));
  const states: TelemetrySnapshot['pipelineStates'] = ['idle', 'idle', 'idle', 'idle', 'idle', 'idle'];

  for (let i = 0; i < 6; i++) {
    setActive(i);
    if (i === 0) states[0] = 'active';
    else if (i === 1) states[1] = isDanger ? 'danger' : risk > 0.4 ? 'warning' : 'active';
    else if (i === 2) states[2] = isDanger ? 'danger' : risk > 0.4 ? 'warning' : 'active';
    else if (i === 3) states[3] = risk > 0.6 ? 'danger' : 'active';
    else if (i === 4) states[4] = isDanger || risk > 0.6 ? 'danger' : risk > 0.3 ? 'warning' : 'active';
    else states[5] = finalAction === 'BLOCK' ? 'danger' : finalAction === 'SANITIZE' ? 'warning' : 'active';
    setStates([...states]);
    await delay(100);
  }
  setActive(null);
}

export function useGuardedChat(
  firewallActive: boolean,
  apiKey: string,
  model: string,
  canCallProvider: boolean
) {
  const guardRef = useRef<SafeLLMKit | null>(null);
  const memoryRef = useRef(new SessionMemory());
  const providerCallsRef = useRef(0);
  const driftRef = useRef(DEFAULT_TELEMETRY.centroidDrift);

  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'sys-1',
      role: 'system',
      content: 'SafeLLMKit AI Firewall active. Gateway channels initialized.',
      action: GuardrailAction.ALLOW,
      riskScore: 0,
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [providerCallCount, setProviderCallCount] = useState(0);
  const [lastInspection, setLastInspection] = useState<GuardrailResult | null>(null);
  const [session, setSession] = useState<SessionStatus>('NORMAL');
  const [reputation, setReputation] = useState(1);
  const [crescendoScore, setCrescendoScore] = useState(0);
  const [telemetry, setTelemetry] = useState<TelemetrySnapshot>(DEFAULT_TELEMETRY);
  const [pipelineActiveNode, setPipelineActiveNode] = useState<number | null>(null);
  const [reputationTime, setReputationTime] = useState(31);
  const [decayStartReputation, setDecayStartReputation] = useState(1);

  useEffect(() => {
    const classifier = new OnnxClassifier('/jailbreak_classifier.onnx');
    guardRef.current = new SafeLLMKit(undefined, classifier);
    classifier
      .init()
      .then(() => setModelReady(true))
      .catch(() => {
        guardRef.current = new SafeLLMKit();
        setModelReady(true);
      });
  }, []);

  useEffect(() => {
    const r0 = decayStartReputation;
    const t = reputationTime;
    const next = r0 + (1.0 - r0) * (1.0 - Math.exp(-t / 30));
    setReputation(Number(next.toFixed(4)));
  }, [reputationTime, decayStartReputation]);

  const clearChat = useCallback(() => {
    memoryRef.current.reset();
    providerCallsRef.current = 0;
    driftRef.current = DEFAULT_TELEMETRY.centroidDrift;
    setProviderCallCount(0);
    setLastInspection(null);
    setSession('NORMAL');
    setReputation(1);
    setCrescendoScore(0);
    setTelemetry(DEFAULT_TELEMETRY);
    setPipelineActiveNode(null);
    setReputationTime(31);
    setDecayStartReputation(1);
    setMessages([
      {
        id: 'sys-1',
        role: 'system',
        content: 'SafeLLMKit AI Firewall active. Gateway channels initialized.',
        action: GuardrailAction.ALLOW,
        riskScore: 0,
      },
    ]);
  }, []);

  const send = useCallback(
    async (prompt: string): Promise<SendResult | null> => {
      const trimmed = prompt.trim();
      if (!trimmed || isLoading) return null;

      setIsLoading(true);
      const guard = guardRef.current!;

      try {
        if (!firewallActive) {
          const inspection = await guard.validateAsync(trimmed);
          const snap = memoryRef.current.snapshot();
          const tel = buildTelemetry(inspection, trimmed, snap, driftRef.current, false);
          driftRef.current = tel.centroidDrift;
          setTelemetry(tel);

          const userMessage: ChatMessage = {
            id: uid(),
            role: 'user',
            content: trimmed,
            action: 'BYPASSED',
            riskScore: inspection.riskScore,
            providerCalled: false,
            findings: inspection.findings,
          };
          setMessages((m) => [...m, userMessage]);
          setLastInspection(inspection);

          let assistantMessage: ChatMessage;
          if (canCallProvider) {
            providerCallsRef.current += 1;
            setProviderCallCount(providerCallsRef.current);
            const text = await chatCompletion(apiKey, model, trimmed);
            assistantMessage = {
              id: uid(),
              role: 'assistant',
              content: text,
              action: 'BYPASSED',
              riskScore: 0,
              providerCalled: true,
            };
          } else {
            assistantMessage = {
              id: uid(),
              role: 'assistant',
              content:
                '[Unprotected] Firewall bypassed — prompt sent without gate enforcement. Add OpenRouter key for live completions.',
              action: 'BYPASSED',
              riskScore: 0,
              providerCalled: false,
            };
          }
          setMessages((m) => [...m, assistantMessage]);
          return {
            blocked: false,
            userMessage,
            assistantMessage,
            inspection,
            providerCalled: canCallProvider,
            bypassed: true,
            sessionStatus: snap.status,
            reputation: snap.reputation,
            turnCount: snap.turnCount,
          };
        }

        const inputInspection = await guard.validateAsync(trimmed);
        const snapBefore = memoryRef.current.recordTurn(trimmed, false);
        const escalate = memoryRef.current.shouldEscalateToBlock(snapBefore.crescendoScore);
        const inputBlocked = inputInspection.action === GuardrailAction.BLOCK || escalate;

        const effectiveInspection: GuardrailResult = escalate
          ? {
              ...inputInspection,
              action: GuardrailAction.BLOCK,
              riskScore: Math.max(inputInspection.riskScore, 92),
              findings: [
                ...inputInspection.findings,
                {
                  category: 'SESSION_MEMORY',
                  rule: 'CRESCENDO_ESCALATION',
                  severity: 10,
                  message: `Multi-turn escalation (score=${snapBefore.crescendoScore})`,
                },
              ],
            }
          : inputInspection;

        const isDanger = effectiveInspection.action === GuardrailAction.BLOCK;
        const riskNorm = effectiveInspection.riskScore / 100;

        await animatePipeline(
          setPipelineActiveNode,
          (states) => setTelemetry((t) => ({ ...t, pipelineStates: states })),
          isDanger,
          riskNorm,
          effectiveInspection.action
        );

        const tel = buildTelemetry(
          effectiveInspection,
          trimmed,
          snapBefore,
          driftRef.current,
          true
        );
        driftRef.current = tel.centroidDrift;
        setTelemetry(tel);
        setLastInspection(effectiveInspection);
        setSession(snapBefore.status);
        setCrescendoScore(snapBefore.crescendoScore);

        if (session !== 'BLOCKED') {
          const newRep = Math.max(0.08, 1.0 - riskNorm * 0.9);
          setDecayStartReputation(newRep);
          setReputationTime(0);
        }

        const displayAction =
          effectiveInspection.action === GuardrailAction.SANITIZE
            ? 'FLAG'
            : effectiveInspection.action;

        const userMessage: ChatMessage = {
          id: uid(),
          role: 'user',
          content: trimmed,
          action: displayAction as ChatMessage['action'],
          riskScore: effectiveInspection.riskScore,
          blocked: inputBlocked,
          providerCalled: false,
          findings: effectiveInspection.findings,
        };
        setMessages((m) => [...m, userMessage]);

        if (inputBlocked) {
          memoryRef.current.recordTurn(trimmed, true);
          const assistantMessage: ChatMessage = {
            id: uid(),
            role: 'assistant',
            content: blockedMessage(effectiveInspection, escalate ? 'session' : 'input'),
            action: GuardrailAction.BLOCK,
            riskScore: effectiveInspection.riskScore,
            blocked: true,
            providerCalled: false,
          };
          setMessages((m) => [...m, assistantMessage]);
          const snap = memoryRef.current.snapshot(snapBefore.crescendoScore);
          setSession(snap.status);
          return {
            blocked: true,
            userMessage,
            assistantMessage,
            inspection: effectiveInspection,
            providerCalled: false,
            sessionStatus: snap.status,
            reputation: snap.reputation,
            turnCount: snap.turnCount,
          };
        }

        const promptForProvider =
          effectiveInspection.action === GuardrailAction.SANITIZE
            ? effectiveInspection.sanitizedInput
            : trimmed;

        let modelText: string;
        if (canCallProvider) {
          providerCallsRef.current += 1;
          setProviderCallCount(providerCallsRef.current);
          modelText = await chatCompletion(apiKey, model, promptForProvider);
        } else {
          modelText = `[Simulation] Prompt cleared SafeLLMKit firewall. Configure OpenRouter key to fetch live completions from ${model}.`;
        }

        const outputInspection = await guard.validateAsync(modelText);
        if (outputInspection.action === GuardrailAction.BLOCK) {
          const assistantMessage: ChatMessage = {
            id: uid(),
            role: 'assistant',
            content: blockedMessage(outputInspection, 'output'),
            action: GuardrailAction.BLOCK,
            riskScore: outputInspection.riskScore,
            blocked: true,
            providerCalled: canCallProvider,
            findings: outputInspection.findings,
          };
          setMessages((m) => [...m, assistantMessage]);
          setLastInspection(outputInspection);
          return {
            blocked: true,
            userMessage,
            assistantMessage,
            inspection: outputInspection,
            providerCalled: canCallProvider,
            sessionStatus: snapBefore.status,
            reputation: snapBefore.reputation,
            turnCount: snapBefore.turnCount,
          };
        }

        const assistantMessage: ChatMessage = {
          id: uid(),
          role: 'assistant',
          content: modelText,
          action: GuardrailAction.ALLOW,
          riskScore: outputInspection.riskScore,
          providerCalled: canCallProvider,
        };
        setMessages((m) => [...m, assistantMessage]);

        return {
          blocked: false,
          userMessage,
          assistantMessage,
          inspection: effectiveInspection,
          providerCalled: canCallProvider,
          sessionStatus: snapBefore.status,
          reputation: snapBefore.reputation,
          turnCount: snapBefore.turnCount,
        };
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Unknown error';
        setMessages((m) => [
          ...m,
          {
            id: uid(),
            role: 'assistant',
            content: `Gateway error: ${msg}`,
            action: GuardrailAction.BLOCK,
            riskScore: 50,
            blocked: true,
            providerCalled: false,
          },
        ]);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [apiKey, canCallProvider, firewallActive, isLoading, model, session]
  );

  return {
    messages,
    isLoading,
    modelReady,
    providerCallCount,
    lastInspection,
    session,
    reputation,
    crescendoScore,
    telemetry,
    pipelineActiveNode,
    reputationTime,
    setReputationTime,
    decayStartReputation,
    send,
    clearChat,
  };
}
