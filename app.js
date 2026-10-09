let currentMode = 'react';

const REACT_PROMPTS = [
  "Calculate (45.5 * 12) + sqrt(144) ^ 3",
  "Who is the CEO of Google?",
  "What is today's date and current time?",
  "What is our company annual vacation leave policy?",
  "Find Q3 net profit from financial documents and calculate 20% of it"
];

const RAG_PROMPTS = [
  "What is Priya Sharma's attendance and overall mark?",
  "Who has the highest overall mark?",
  "Which students belong to the CSE department?",
  "List students whose attendance is below 75%",
  "Who scored above 90 in AI?",
  "What is the phone number of Fathima N?"
];

document.addEventListener('DOMContentLoaded', () => {
  renderQuickPrompts();
  loadApiKey();
});

function loadApiKey() {
  const saved = localStorage.getItem('gemini_api_key');
  if (saved) {
    const el = document.getElementById('api-key-input');
    if (el) el.value = saved;
  }
}

function saveApiKey() {
  const el = document.getElementById('api-key-input');
  const val = (el ? el.value : '').trim();
  if (val) {
    localStorage.setItem('gemini_api_key', val);
    alert('API Key saved successfully!');
  } else {
    localStorage.removeItem('gemini_api_key');
    alert('API Key cleared.');
  }
}

function switchMode(mode) {
  currentMode = mode;
  document.getElementById('mode-react').classList.toggle('active', mode === 'react');
  document.getElementById('mode-rag').classList.toggle('active', mode === 'rag');

  const label = document.getElementById('current-mode-label');
  if (mode === 'react') {
    label.innerText = 'Mode: ReAct Multi-Tool Agent (4 Tools)';
  } else {
    label.innerText = 'Mode: Student Database RAG (Dense Vector Search)';
  }

  renderQuickPrompts();
}

function renderQuickPrompts() {
  const container = document.getElementById('quick-prompts-container');
  container.innerHTML = '';

  const prompts = currentMode === 'react' ? REACT_PROMPTS : RAG_PROMPTS;

  prompts.forEach(p => {
    const btn = document.createElement('button');
    btn.className = 'prompt-chip';
    btn.innerText = p;
    btn.onclick = () => {
      document.getElementById('user-input').value = p;
      document.getElementById('user-input').focus();
    };
    container.appendChild(btn);
  });
}

function handleKeyDown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    handleSend(event);
  }
}

async function handleSend(event) {
  event.preventDefault();
  const inputEl = document.getElementById('user-input');
  const query = inputEl.value.trim();
  if (!query) return;

  inputEl.value = '';
  appendUserMessage(query);

  const sendBtn = document.getElementById('send-btn');
  sendBtn.disabled = true;
  sendBtn.innerHTML = '<div class="spinner"></div><span>Thinking...</span>';

  // Loading message placeholder
  const loadingId = appendLoadingMessage();

  try {
    const endpoint = currentMode === 'react' ? '/api/chat' : '/api/rag';
    const apiKey = localStorage.getItem('gemini_api_key') || '';
    const payload = currentMode === 'react' 
      ? { message: query, api_key: apiKey } 
      : { query: query, api_key: apiKey };

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'X-Gemini-API-Key': apiKey
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    removeLoadingMessage(loadingId);

    if (data.error) {
      if (data.error.includes("UNAUTHENTICATED") || data.error.includes("401") || data.error.includes("invalid authentication")) {
        appendErrorMessage("⚠️ API Key Error: Your Gemini API key is missing or expired. Please paste your fresh API key from https://aistudio.google.com/app/apikey into the 'GOOGLE GEMINI API KEY' box on the left sidebar and click Save.");
      } else {
        appendErrorMessage(data.error);
      }
      return;
    }

    if (currentMode === 'react') {
      appendAgentReactMessage(data);
    } else {
      appendAgentRAGMessage(data);
    }

  } catch (err) {
    removeLoadingMessage(loadingId);
    appendErrorMessage("Error connecting to backend: " + err.message);
  } finally {
    sendBtn.disabled = false;
    sendBtn.innerHTML = `<span>Send</span><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>`;
  }
}

function appendUserMessage(text) {
  const container = document.getElementById('chat-messages');
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = `
    <div class="avatar">👤</div>
    <div class="message-body">
      <div class="user-bubble">${escapeHtml(text)}</div>
    </div>
  `;
  container.appendChild(row);
  scrollToBottom();
}

function appendLoadingMessage() {
  const container = document.getElementById('chat-messages');
  const id = 'loading-' + Date.now();
  const row = document.createElement('div');
  row.className = 'message-row agent';
  row.id = id;
  row.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="message-body">
      <div class="agent-bubble">
        <div style="display:flex; align-items:center; gap:8px; font-size:13px; color:var(--text-muted)">
          <div class="spinner"></div> Reasoning & executing tools...
        </div>
      </div>
    </div>
  `;
  container.appendChild(row);
  scrollToBottom();
  return id;
}

function removeLoadingMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendAgentReactMessage(data) {
  const container = document.getElementById('chat-messages');
  const row = document.createElement('div');
  row.className = 'message-row agent';

  // Build ReAct steps HTML if present
  let reactStepsHtml = '';
  if (data.react_steps && data.react_steps.length > 0) {
    reactStepsHtml = '<div class="react-steps-container">';
    data.react_steps.forEach(step => {
      reactStepsHtml += `
        <div class="react-step-card step-thought">
          <span class="step-badge">💭 THOUGHT (Step ${step.step})</span>
          <div class="step-content">${escapeHtml(step.thought)}</div>
        </div>
        <div class="react-step-card step-action">
          <span class="step-badge">⚡ ACTION</span>
          <div class="step-content">${escapeHtml(step.action)}</div>
        </div>
        <div class="react-step-card step-obs">
          <span class="step-badge">👁️ OBSERVATION</span>
          <div class="step-content">${escapeHtml(step.observation)}</div>
        </div>
      `;
    });
    reactStepsHtml += '</div>';
  }

  // Answer text
  const answerText = data.final_answer || data.explanation || (typeof data.answer === 'string' ? data.answer : JSON.stringify(data));
  const jsonString = JSON.stringify(data, null, 2);

  row.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="message-body">
      <div class="agent-bubble">
        <div class="agent-answer"><strong>${escapeHtml(answerText)}</strong></div>
        ${reactStepsHtml}
        <div class="json-viewer-box">
          <div class="json-header">
            <span>STRUCTURED JSON OUTPUT</span>
            <button class="json-copy-btn" onclick="copyToClipboard('${encodeURIComponent(jsonString)}')">📋 Copy JSON</button>
          </div>
          <pre class="json-content"><code>${escapeHtml(jsonString)}</code></pre>
        </div>
      </div>
    </div>
  `;

  container.appendChild(row);
  scrollToBottom();
}

function appendAgentRAGMessage(data) {
  const container = document.getElementById('chat-messages');
  const row = document.createElement('div');
  row.className = 'message-row agent';

  const answerText = data.answer || JSON.stringify(data);
  const jsonString = JSON.stringify(data, null, 2);

  let sourcesHtml = '';
  if (data.retrieved_sources && data.retrieved_sources.length > 0) {
    const list = data.retrieved_sources.map(s => `${s.student_id} (${s.name})`).join(', ');
    sourcesHtml = `<div style="font-size:12px; color:var(--accent-cyan); margin-top:4px;">🔍 Retrieved Sources: <strong>${escapeHtml(list)}</strong></div>`;
  }

  row.innerHTML = `
    <div class="avatar">📚</div>
    <div class="message-body">
      <div class="agent-bubble">
        <div class="agent-answer"><strong>${escapeHtml(answerText)}</strong></div>
        ${sourcesHtml}
        <div class="json-viewer-box">
          <div class="json-header">
            <span>RAG RETRIEVAL JSON</span>
            <button class="json-copy-btn" onclick="copyToClipboard('${encodeURIComponent(jsonString)}')">📋 Copy JSON</button>
          </div>
          <pre class="json-content"><code>${escapeHtml(jsonString)}</code></pre>
        </div>
      </div>
    </div>
  `;

  container.appendChild(row);
  scrollToBottom();
}

function appendErrorMessage(msg) {
  const container = document.getElementById('chat-messages');
  const row = document.createElement('div');
  row.className = 'message-row agent';
  row.innerHTML = `
    <div class="avatar" style="border-color:var(--accent-rose)">⚠️</div>
    <div class="message-body">
      <div class="agent-bubble" style="border-color:var(--accent-rose); color:var(--accent-rose)">
        ${escapeHtml(msg)}
      </div>
    </div>
  `;
  container.appendChild(row);
  scrollToBottom();
}

function clearChat() {
  document.getElementById('chat-messages').innerHTML = `
    <div class="welcome-card">
      <div class="welcome-icon">🧹</div>
      <h3>Conversation Cleared</h3>
      <p>Choose an agent mode from the left or select a quick prompt below to start.</p>
    </div>
  `;
}

function copyToClipboard(encoded) {
  const text = decodeURIComponent(encoded);
  navigator.clipboard.writeText(text).then(() => {
    alert("JSON copied to clipboard!");
  });
}

function scrollToBottom() {
  const container = document.getElementById('chat-messages');
  container.scrollTop = container.scrollHeight;
}

function escapeHtml(str) {
  if (typeof str !== 'string') return String(str);
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
