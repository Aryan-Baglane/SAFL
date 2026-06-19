export async function validateApiKey(apiKey: string) {
  const response = await fetch('https://openrouter.ai/api/v1/auth/key', {
    method: 'GET',
    headers: { Authorization: `Bearer ${apiKey}` },
  });

  if (!response.ok) {
    throw new Error(`Unauthorized (HTTP ${response.status}) — check your API key.`);
  }

  const data = await response.json();
  if (!data?.data) {
    throw new Error('Unexpected authentication response.');
  }

  return {
    limit: data.data.limit || 0,
    usage: data.data.usage || 0,
    label: data.data.label || 'Default',
  };
}

export async function chatCompletion(
  apiKey: string,
  model: string,
  prompt: string
): Promise<string> {
  const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
      'HTTP-Referer': window.location.origin,
      'X-Title': 'SafeLLMKit Demo',
    },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: prompt }],
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData?.error?.message || `HTTP ${response.status}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || 'No response from model.';
}
