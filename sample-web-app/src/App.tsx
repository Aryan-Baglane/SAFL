import { useCallback, useEffect, useState } from 'react';
import './App.css';
import { ConfigPanel } from './components/ConfigPanel';
import { ChatPanel } from './components/ChatPanel';
import { PresetPanel } from './components/PresetPanel';
import { PipelineRail } from './components/PipelineRail';
import { PerplexityChart } from './components/PerplexityChart';
import { SmoothLLMChart } from './components/SmoothLLMChart';
import { StateMachinePanel } from './components/StateMachinePanel';
import { CentroidDriftChart } from './components/CentroidDriftChart';
import { ReputationCurve } from './components/ReputationCurve';
import { useGuardedChat } from './hooks/useGuardedChat';
import { validateApiKey } from './lib/openRouter';
import { CRESCENDO_STEPS } from './lib/attackPresets';
import { ConnectionStatus, KeyMetadata } from './types';

export default function App() {
  const [firewallActive, setFirewallActive] = useState(true);
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [selectedModel, setSelectedModel] = useState('cohere/north-mini-code:free');
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('offline');
  const [connectionError, setConnectionError] = useState('');
  const [keyMetadata, setKeyMetadata] = useState<KeyMetadata | null>(null);
  const [input, setInput] = useState('');
  const [crescendoIndex, setCrescendoIndex] = useState(0);

  const canCallProvider = connectionStatus === 'connected' && !!apiKey;

  const {
    messages,
    isLoading,
    modelReady,
    providerCallCount,
    session,
    reputation,
    telemetry,
    pipelineActiveNode,
    reputationTime,
    setReputationTime,
    decayStartReputation,
    send,
    clearChat,
  } = useGuardedChat(firewallActive, apiKey, selectedModel, canCallProvider);

  const turnCount = messages.filter((m) => m.role !== 'system').length;

  useEffect(() => {
    if (!apiKey) {
      setConnectionStatus('offline');
      setKeyMetadata(null);
      setConnectionError('');
    } else if (connectionStatus === 'connected') {
      setConnectionStatus('offline');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiKey]);

  const handleTestConnection = useCallback(async () => {
    if (!apiKey) return;
    setConnectionStatus('testing');
    setConnectionError('');
    try {
      const meta = await validateApiKey(apiKey);
      setKeyMetadata(meta);
      setConnectionStatus('connected');
    } catch (err: unknown) {
      setConnectionStatus('error');
      setConnectionError(err instanceof Error ? err.message : 'Verification failed');
      setKeyMetadata(null);
    }
  }, [apiKey]);

  const handleSend = useCallback(async () => {
    const prompt = input.trim();
    if (!prompt) return;
    setInput('');
    await send(prompt);
  }, [input, send]);

  const handleCrescendo = useCallback(() => {
    setInput(CRESCENDO_STEPS[crescendoIndex]);
    setCrescendoIndex((i) => (i + 1) % CRESCENDO_STEPS.length);
  }, [crescendoIndex]);

  const gatewayLabel =
    connectionStatus === 'connected'
      ? `● LIVE ONLINE (${selectedModel})`
      : connectionStatus === 'testing'
      ? '○ VERIFYING KEY…'
      : connectionStatus === 'error'
      ? '● KEY UNVERIFIED (SIM MOCK)'
      : `● SIM MODE (${selectedModel})`;

  return (
    <div className={`app ${session === 'BLOCKED' && firewallActive ? 'session-blocked' : ''}`}>
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">🛡️</div>
          <div>
            <h1>SafeLLMKit AI Firewall</h1>
            <span className="version-tag">CONSOLE V0.1.0</span>
          </div>
        </div>

        <div className="topbar-controls">
          <div className="guard-toggle-wrap">
            <span className={`guard-label ${firewallActive ? 'on' : ''}`}>
              SDK GUARD: {firewallActive ? 'ON' : 'OFF'}
            </span>
            <button
              type="button"
              className={`toggle ${firewallActive ? 'on' : ''}`}
              onClick={() => setFirewallActive((v) => !v)}
              aria-pressed={firewallActive}
            >
              <span className="toggle-knob" />
            </button>
          </div>

          <div className="provider-pill" title="OpenRouter calls this session">
            <span className="provider-pill-label">Provider calls</span>
            <span className="provider-pill-value">{providerCallCount}</span>
          </div>

          <div className="gateway-badge">{gatewayLabel}</div>

          <button type="button" className="btn-ghost" onClick={clearChat}>
            Clear data
          </button>
        </div>
      </header>

      <main className="dashboard">
        {/* Column 1 — Config & Chat */}
        <div className="col col-left">
          <ConfigPanel
            apiKey={apiKey}
            showApiKey={showApiKey}
            onApiKeyChange={setApiKey}
            onToggleShowKey={() => setShowApiKey((v) => !v)}
            connectionStatus={connectionStatus}
            connectionError={connectionError}
            keyMetadata={keyMetadata}
            onTestConnection={handleTestConnection}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
          />

          <ChatPanel
            messages={messages}
            input={input}
            onInputChange={setInput}
            onSend={handleSend}
            isLoading={isLoading}
            modelReady={modelReady}
            firewallActive={firewallActive}
            turnCount={turnCount}
          />

          <PresetPanel
            crescendoIndex={crescendoIndex}
            onPreset={setInput}
            onCrescendo={handleCrescendo}
            disabled={isLoading}
          />
        </div>

        {/* Column 2 — Pipeline telemetry */}
        <div className="col col-mid">
          <PipelineRail
            states={telemetry.pipelineStates}
            activeNode={pipelineActiveNode}
            latencyMs={telemetry.latencyMs}
            isScanning={isLoading}
          />
          <PerplexityChart data={telemetry.perplexityData} />
          <SmoothLLMChart data={telemetry.mutationBars} />
        </div>

        {/* Column 3 — Memory & reputation */}
        <div className="col col-right">
          <StateMachinePanel status={session} />
          <CentroidDriftChart data={telemetry.centroidDrift} />
          <ReputationCurve
            reputation={reputation}
            reputationTime={reputationTime}
            decayStartReputation={decayStartReputation}
            onTimeChange={setReputationTime}
          />
        </div>
      </main>
    </div>
  );
}
