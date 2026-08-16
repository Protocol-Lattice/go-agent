// ============================================================================
// go-agent WebUI Core Client
// ============================================================================

(function () {
  'use strict';

  // --- DOM Elements ---
  const messagesEl = document.getElementById('messages');
  const composerForm = document.getElementById('composer');
  const inputEl = document.getElementById('input');
  const sessionInput = document.getElementById('session');
  const sendBtn = document.getElementById('send');
  const statusBadge = document.getElementById('statusBadge');
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');
  const modelSelect = document.getElementById('modelSelect');

  const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
  const sidebar = document.getElementById('sidebar');
  const sidebarBackdrop = document.getElementById('sidebarBackdrop');

  const historyEl = document.getElementById('history');
  const historySearch = document.getElementById('historySearch');
  const newChatBtn = document.getElementById('newChat');

  const skillsEl = document.getElementById('skills');
  const skillCountEl = document.getElementById('skillCount');
  const toolsEl = document.getElementById('tools');
  const toolCountEl = document.getElementById('toolCount');
  const subagentsEl = document.getElementById('subagents');
  const createSubagentBtn = document.getElementById('createSubagent');

  const subagentModal = document.getElementById('subagentModal');
  const closeSubagentBtn = document.getElementById('closeSubagent');
  const cancelSubagentBtn = document.getElementById('cancelSubagent');
  const subagentForm = document.getElementById('subagentForm');
  const subagentName = document.getElementById('subagentName');
  const subagentProvider = document.getElementById('subagentProvider');
  const subagentModel = document.getElementById('subagentModel');
  const subagentDescription = document.getElementById('subagentDescription');
  const subagentPrompt = document.getElementById('subagentPrompt');
  const subagentError = document.getElementById('subagentError');

  const fileInput = document.getElementById('fileInput');
  const attachBtn = document.getElementById('attachBtn');
  const stagedAttachmentsEl = document.getElementById('stagedAttachments');

  const stopContainer = document.getElementById('stopContainer');
  const stopGenerationBtn = document.getElementById('stopGenerationBtn');
  const toastContainer = document.getElementById('toastContainer');

  // --- Constants & State ---
  const HISTORY_KEY_V2 = 'go-agent.webui.chat-history.v2';
  const HISTORY_KEY_V1 = 'go-agent.webui.chat-history.v1';
  const STREAM_TIMEOUT_MS = 90000;

  let activeSession = sessionInput.value.trim() || 'web';
  let activeAbortController = null;
  let isStreaming = false;
  let stagedFiles = [];
  let currentModelCatalog = [];
  let userHasScrolledUp = false;

  // --- Toast Notifications ---
  function showToast(message, isError = false) {
    const toast = document.createElement('div');
    toast.className = `toast ${isError ? 'toast-error' : ''}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(8px)';
      toast.style.transition = 'opacity 0.25s, transform 0.25s';
      setTimeout(() => toast.remove(), 250);
    }, 3500);
  }

  // --- HTML Escaping & Markdown Parser ---
  function escapeHTML(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function renderMarkdown(md) {
    if (!md) return '';
    let text = String(md);

    // 1. Code blocks: ```lang ... ```
    const codeBlocks = [];
    text = text.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, function (_, lang, code) {
      const id = `__CODE_BLOCK_${codeBlocks.length}__`;
      const cleanLang = lang.trim() || 'code';
      const cleanCode = code.replace(/\n$/, '');
      codeBlocks.push({ lang: cleanLang, code: cleanCode });
      return id;
    });

    // 2. Inline code: `code`
    const inlineCodes = [];
    text = text.replace(/`([^`\n]+)`/g, function (_, code) {
      const id = `__INLINE_CODE_${inlineCodes.length}__`;
      inlineCodes.push(code);
      return id;
    });

    // Escape remaining text
    text = escapeHTML(text);

    // 3. Tables
    const lines = text.split('\n');
    const processedLines = [];
    let inTable = false;
    let tableRows = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (line.startsWith('|') && line.endsWith('|')) {
        if (!inTable) {
          inTable = true;
          tableRows = [];
        }
        tableRows.push(line);
      } else {
        if (inTable) {
          processedLines.push(renderMarkdownTable(tableRows));
          inTable = false;
          tableRows = [];
        }
        processedLines.push(lines[i]);
      }
    }
    if (inTable) {
      processedLines.push(renderMarkdownTable(tableRows));
    }
    text = processedLines.join('\n');

    // 4. Headers
    text = text.replace(/^#### (.*?)$/gm, '<h4>$1</h4>');
    text = text.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
    text = text.replace(/^## (.*?)$/gm, '<h2>$1</h2>');
    text = text.replace(/^# (.*?)$/gm, '<h1>$1</h1>');

    // 5. Blockquotes
    text = text.replace(/^>\s?(.*?)$/gm, '<blockquote>$1</blockquote>');

    // 6. Bold, Italic, Strikethrough
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/__(.*?)__/g, '<strong>$1</strong>');
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    text = text.replace(/_(.*?)_/g, '<em>$1</em>');
    text = text.replace(/~~(.*?)~~/g, '<del>$1</del>');

    // 7. Links: [text](url)
    text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

    // 8. Lists
    text = text.replace(/^[\*\-]\s+(.*?)$/gm, '<li>$1</li>');
    text = text.replace(/^\d+\.\s+(.*?)$/gm, '<li class="ordered">$1</li>');
    text = text.replace(/(<li>.*?<\/li>\n?)+/g, '<ul>$&</ul>');

    // 9. Paragraphs (lines with double breaks)
    text = text.split(/\n\n+/).map(p => {
      p = p.trim();
      if (!p) return '';
      if (/^(<h\d|<ul|<blockquote|<div|<table)/.test(p)) return p;
      return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join('\n');

    // 10. Restore Inline Code
    for (let i = 0; i < inlineCodes.length; i++) {
      text = text.replace(`__INLINE_CODE_${i}__`, `<code>${escapeHTML(inlineCodes[i])}</code>`);
    }

    // 11. Restore Code Blocks
    for (let i = 0; i < codeBlocks.length; i++) {
      const { lang, code } = codeBlocks[i];
      const escapedRawCode = encodeURIComponent(code);
      const blockHTML = `
        <pre><div class="code-header"><span class="code-lang">${escapeHTML(lang)}</span><button type="button" class="copy-code-btn" data-code="${escapedRawCode}">Copy</button></div><code>${escapeHTML(code)}</code></pre>
      `;
      text = text.replace(`__CODE_BLOCK_${i}__`, blockHTML);
    }

    return text;
  }

  function renderMarkdownTable(rows) {
    if (rows.length < 2) return rows.join('\n');
    let html = '<div class="markdown-table-wrapper"><table>';
    let isHeader = true;

    for (let i = 0; i < rows.length; i++) {
      const row = rows[i].trim();
      // Skip separator line |---|---|
      if (/^\|[\s\-:|]+\|$/.test(row)) {
        isHeader = false;
        continue;
      }
      const cells = row.slice(1, -1).split('|').map(c => c.trim());
      html += '<tr>';
      for (const cell of cells) {
        const tag = isHeader ? 'th' : 'td';
        html += `<${tag}>${cell}</${tag}>`;
      }
      html += '</tr>';
    }
    html += '</table></div>';
    return html;
  }

  // --- Copy to Clipboard Handler ---
  messagesEl.addEventListener('click', e => {
    const copyBtn = e.target.closest('.copy-code-btn');
    if (!copyBtn) return;
    const rawCode = decodeURIComponent(copyBtn.dataset.code || '');
    if (!rawCode) return;

    navigator.clipboard.writeText(rawCode).then(() => {
      const origText = copyBtn.textContent;
      copyBtn.textContent = '✓ Copied';
      copyBtn.style.color = 'var(--accent-primary)';
      setTimeout(() => {
        copyBtn.textContent = origText;
        copyBtn.style.color = '';
      }, 2000);
    }).catch(() => {
      showToast('Failed to copy code to clipboard', true);
    });
  });

  // --- History & Storage Management ---
  function migrateOldHistory() {
    try {
      const oldRaw = localStorage.getItem(HISTORY_KEY_V1);
      if (oldRaw && !localStorage.getItem(HISTORY_KEY_V2)) {
        localStorage.setItem(HISTORY_KEY_V2, oldRaw);
      }
    } catch (_) {}
  }

  function loadAllHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY_V2);
      const data = raw ? JSON.parse(raw) : {};
      return data && typeof data === 'object' && !Array.isArray(data) ? data : {};
    } catch (_) {
      return {};
    }
  }

  function saveAllHistory(data) {
    try {
      localStorage.setItem(HISTORY_KEY_V2, JSON.stringify(data));
    } catch (_) {}
  }

  function getSessionMessages(sessionId = activeSession) {
    const all = loadAllHistory();
    return Array.isArray(all[sessionId]) ? all[sessionId] : [];
  }

  function persistMessage(role, text, files = [], sessionId = activeSession) {
    const all = loadAllHistory();
    if (!Array.isArray(all[sessionId])) all[sessionId] = [];
    all[sessionId].push({
      role,
      text: String(text || ''),
      files: files.map(f => ({ name: f.name, size: f.size, mime: f.mime })),
      timestamp: new Date().toISOString()
    });
    // Keep max 200 messages per session
    all[sessionId] = all[sessionId].slice(-200);
    saveAllHistory(all);
    renderHistorySidebar();
  }

  function deleteSession(sessionId, event) {
    if (event) event.stopPropagation();
    const all = loadAllHistory();
    if (Object.prototype.hasOwnProperty.call(all, sessionId)) {
      delete all[sessionId];
      saveAllHistory(all);
      if (sessionId === activeSession) {
        activeSession = 'web';
        sessionInput.value = 'web';
        renderSession('web');
      }
      renderHistorySidebar();
      showToast(`Deleted conversation "${sessionId}"`);
    }
  }

  function renderHistorySidebar() {
    const all = loadAllHistory();
    const filter = (historySearch.value || '').trim().toLowerCase();
    const sessions = Object.entries(all)
      .filter(([id, items]) => Array.isArray(items) && items.length > 0)
      .filter(([id, items]) => {
        if (!filter) return true;
        if (id.toLowerCase().includes(filter)) return true;
        return items.some(m => (m.text || '').toLowerCase().includes(filter));
      })
      .sort((a, b) => {
        const tA = a[1].at(-1)?.timestamp || '';
        const tB = b[1].at(-1)?.timestamp || '';
        return tB.localeCompare(tA);
      });

    historyEl.replaceChildren();
    if (!sessions.length) {
      historyEl.innerHTML = '<div class="empty-state">No conversations found</div>';
      return;
    }

    for (const [id, items] of sessions) {
      const lastMsg = items.at(-1);
      const row = document.createElement('div');
      row.className = `history-row ${id === activeSession ? 'active' : ''}`;

      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'history-btn';

      const title = document.createElement('strong');
      title.textContent = id;

      const preview = document.createElement('span');
      preview.textContent = lastMsg?.text || '(Empty message)';

      btn.append(title, preview);
      btn.addEventListener('click', () => switchSession(id));

      const delBtn = document.createElement('button');
      delBtn.type = 'button';
      delBtn.className = 'history-delete-btn';
      delBtn.innerHTML = '&times;';
      delBtn.title = `Delete ${id}`;
      delBtn.setAttribute('aria-label', `Delete conversation ${id}`);
      delBtn.addEventListener('click', e => deleteSession(id, e));

      row.append(btn, delBtn);
      historyEl.appendChild(row);
    }
  }

  function renderSession(sessionId) {
    messagesEl.replaceChildren();
    const items = getSessionMessages(sessionId);

    if (!items.length) {
      const welcome = document.createElement('div');
      welcome.className = 'welcome-screen';
      welcome.innerHTML = `
        <div class="welcome-glyph">λ</div>
        <h1>What would you like to build?</h1>
        <p class="welcome-subtitle">
          Execute agentic workflows, orchestrate UTCP tools, run Go CodeMode pipelines, and delegate to specialist sub-agents.
        </p>
        <div class="quick-prompts">
          <button class="prompt-chip" data-prompt="What skills and tools are currently available in the active runtime?">
            <span class="chip-icon">⚡</span>
            <span class="chip-text">Explore active capabilities</span>
          </button>
          <button class="prompt-chip" data-prompt="Inspect the workspace files and suggest a refactoring plan.">
            <span class="chip-icon">🔍</span>
            <span class="chip-text">Analyze workspace files</span>
          </button>
          <button class="prompt-chip" data-prompt="Run a multi-step CodeMode workflow to test UTCP tool integration.">
            <span class="chip-icon">💻</span>
            <span class="chip-text">Execute CodeMode workflow</span>
          </button>
        </div>
      `;
      messagesEl.appendChild(welcome);
      bindQuickPrompts();
      return;
    }

    for (const item of items) {
      addMessage(item.role, item.text, item.files || [], false);
    }
    scrollToBottom(true);
  }

  function switchSession(sessionId) {
    activeSession = sessionId;
    sessionInput.value = sessionId;
    renderSession(sessionId);
    renderHistorySidebar();
    inputEl.focus();
  }

  function createNewSession() {
    const timestamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '');
    let id = `chat-${timestamp}`;
    const all = loadAllHistory();
    let counter = 2;
    while (all[id]) {
      id = `chat-${timestamp}-${counter++}`;
    }
    switchSession(id);
  }

  function bindQuickPrompts() {
    document.querySelectorAll('.prompt-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        inputEl.value = chip.dataset.prompt || '';
        inputEl.focus();
        inputEl.dispatchEvent(new Event('input'));
      });
    });
  }

  // --- Scroll Management ---
  messagesEl.addEventListener('scroll', () => {
    const distFromBottom = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight;
    userHasScrolledUp = distFromBottom > 120;
  });

  function scrollToBottom(force = false) {
    if (force || !userHasScrolledUp) {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }

  // --- Message & Tool UI Builders ---
  function addMessage(role, text = '', files = [], persist = true) {
    document.querySelector('.welcome-screen')?.remove();

    const row = document.createElement('div');
    row.className = `msg-row ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = role === 'user' ? 'YOU' : 'λ';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble markdown-body';

    // If there were attached files, render badges
    if (files && files.length > 0) {
      const filesWrap = document.createElement('div');
      filesWrap.className = 'staged-attachments';
      filesWrap.style.marginBottom = '8px';
      for (const file of files) {
        const chip = document.createElement('div');
        chip.className = 'attachment-chip';
        chip.innerHTML = `<span class="attachment-name">📎 ${escapeHTML(file.name)}</span>`;
        filesWrap.appendChild(chip);
      }
      bubble.appendChild(filesWrap);
    }

    const contentDiv = document.createElement('div');
    contentDiv.className = 'msg-text-content';
    contentDiv.innerHTML = renderMarkdown(text);
    bubble.appendChild(contentDiv);

    row.append(avatar, bubble);
    messagesEl.appendChild(row);
    scrollToBottom();

    if (persist && text) {
      persistMessage(role, text, files);
    }
    return { row, bubble, contentDiv };
  }

  function addThinkingIndicator() {
    document.querySelector('.welcome-screen')?.remove();

    const row = document.createElement('div');
    row.className = 'msg-row assistant thinking-row';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = 'λ';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble markdown-body';

    const thinkingWrap = document.createElement('div');
    thinkingWrap.className = 'thinking-state';
    thinkingWrap.innerHTML = `
      <span>Thinking</span>
      <span class="thinking-dots"><i></i><i></i><i></i></span>
    `;

    bubble.appendChild(thinkingWrap);
    row.append(avatar, bubble);
    messagesEl.appendChild(row);
    scrollToBottom(true);

    return {
      row,
      bubble,
      thinkingWrap,
      workflow: null,
      toolSteps: new Map()
    };
  }

  function ensureWorkflowPanel(thinking) {
    if (thinking.workflow) return thinking.workflow;

    const card = document.createElement('div');
    card.className = 'tool-workflow-card';

    const header = document.createElement('div');
    header.className = 'workflow-header';

    const title = document.createElement('span');
    title.className = 'workflow-title';
    title.textContent = 'Tool & CodeMode Workflow';

    const badge = document.createElement('span');
    badge.className = 'workflow-badge';
    badge.textContent = '0 steps · running';

    const stepsList = document.createElement('div');
    stepsList.className = 'workflow-steps';

    header.append(title, badge);
    card.append(header, stepsList);

    // Insert workflow card before or replace thinking indicator
    thinking.bubble.replaceChildren(card);
    thinking.workflow = {
      card,
      badge,
      stepsList,
      stepsMap: new Map()
    };
    return thinking.workflow;
  }

  function updateWorkflowSummary(thinking, isComplete = false) {
    if (!thinking.workflow) return;
    const total = thinking.workflow.stepsMap.size;
    const state = isComplete ? 'complete' : 'running';
    thinking.workflow.badge.textContent = `${total} ${total === 1 ? 'step' : 'steps'} · ${state}`;
  }

  function ensureToolStep(thinking, toolName, args = null) {
    toolName = String(toolName || '').trim();
    if (!toolName) return null;

    const wf = ensureWorkflowPanel(thinking);
    let stepCount = wf.stepsMap.size + 1;
    let stepKey = `${toolName}#${stepCount}`;

    const stepItem = document.createElement('div');
    stepItem.className = 'tool-step-item';

    const headerBtn = document.createElement('button');
    headerBtn.type = 'button';
    headerBtn.className = 'tool-step-header';

    const icon = document.createElement('span');
    icon.className = 'tool-step-icon';
    icon.textContent = '⚙';

    const name = document.createElement('span');
    name.className = 'tool-step-name';
    name.textContent = `apply ${toolName}`;

    const status = document.createElement('span');
    status.className = 'tool-step-status running';
    status.textContent = 'RUNNING';

    const chevron = document.createElement('span');
    chevron.className = 'tool-step-chevron';
    chevron.textContent = '›';

    headerBtn.append(icon, name, status, chevron);

    const body = document.createElement('pre');
    body.className = 'tool-step-body';
    body.style.display = 'none';

    let initialBody = 'Executing tool in runtime…';
    if (args && Object.keys(args).length > 0) {
      try {
        initialBody = `Arguments:\n${JSON.stringify(args, null, 2)}`;
      } catch (_) {}
    }
    body.textContent = initialBody;

    headerBtn.addEventListener('click', () => {
      const isVisible = body.style.display !== 'none';
      body.style.display = isVisible ? 'none' : 'block';
      stepItem.classList.toggle('open', !isVisible);
      chevron.textContent = !isVisible ? '⌄' : '›';
    });

    stepItem.append(headerBtn, body);
    wf.stepsList.appendChild(stepItem);

    const stepData = {
      name: toolName,
      item: stepItem,
      status,
      body,
      chevron,
      isDone: false
    };

    wf.stepsMap.set(stepKey, stepData);
    thinking.toolSteps.set(toolName, stepData);
    updateWorkflowSummary(thinking, false);
    scrollToBottom();
    return stepData;
  }

  function setToolResult(thinking, toolName, result, errorMsg = '') {
    toolName = String(toolName || '').trim();
    let step = thinking.toolSteps.get(toolName);
    if (!step) {
      step = ensureToolStep(thinking, toolName);
    }
    if (!step) return;

    step.isDone = true;
    if (errorMsg) {
      step.status.className = 'tool-step-status error';
      step.status.textContent = 'ERROR';
      step.body.textContent = `Error:\n${errorMsg}`;
    } else {
      step.status.className = 'tool-step-status complete';
      step.status.textContent = 'OK';
      let text = result;
      if (typeof result === 'object' && result !== null) {
        try { text = JSON.stringify(result, null, 2); } catch (_) { text = String(result); }
      }
      step.body.textContent = `Result:\n${String(text ?? 'Success')}`;
    }
    updateWorkflowSummary(thinking, false);
    scrollToBottom();
  }

  function finalizeAllToolSteps(thinking) {
    if (!thinking.workflow) return;
    for (const step of thinking.workflow.stepsMap.values()) {
      if (!step.isDone) {
        step.isDone = true;
        step.status.className = 'tool-step-status complete';
        step.status.textContent = 'DONE';
      }
    }
    updateWorkflowSummary(thinking, true);
  }

  // --- SSE Streaming Chat Client ---
  async function streamChat(session, message, files, thinking) {
    activeAbortController = new AbortController();
    isStreaming = true;
    stopContainer.classList.remove('hidden');

    let idleTimer = null;
    const resetIdleTimer = () => {
      if (idleTimer) clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        if (activeAbortController) activeAbortController.abort();
      }, STREAM_TIMEOUT_MS);
    };
    resetIdleTimer();

    const payload = {
      session,
      message,
      files: files.map(f => ({ name: f.name, mime: f.mime, data: f.data })),
      provider: modelSelect.dataset.provider || '',
      model: modelSelect.value !== 'default' ? modelSelect.value : ''
    };

    let response;
    try {
      response = await fetch('/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify(payload),
        signal: activeAbortController.signal
      });
    } catch (err) {
      clearTimeout(idleTimer);
      if (err.name === 'AbortError') {
        throw new Error('Generation stopped by user or timed out.');
      }
      throw err;
    }

    if (!response.ok) {
      clearTimeout(idleTimer);
      let errMsg = `Request failed (${response.status})`;
      try {
        const errJson = await response.json();
        if (errJson.error) errMsg = errJson.error;
      } catch (_) {}
      throw new Error(errMsg);
    }

    if (!response.body) {
      clearTimeout(idleTimer);
      throw new Error('Streaming response body not supported by browser.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let accumulatedText = '';
    let textContainer = null;

    const ensureTextContainer = () => {
      if (!textContainer) {
        textContainer = document.createElement('div');
        textContainer.className = 'msg-text-content';
        thinking.bubble.appendChild(textContainer);
      }
      return textContainer;
    };

    try {
      while (true) {
        const { value, done } = await reader.read();
        resetIdleTimer();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split(/\n\n/);
        buffer = parts.pop() || '';

        for (const part of parts) {
          const lines = part.split('\n');
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const dataStr = line.slice(6).trim();
            if (!dataStr) continue;

            try {
              const eventData = JSON.parse(dataStr);
              if (eventData.error) {
                throw new Error(eventData.error);
              }

              // Event: tool_start
              if (eventData.type === 'tool_start') {
                ensureToolStep(thinking, eventData.tool, eventData.arguments);
                continue;
              }

              // Event: tool_result
              if (eventData.type === 'tool_result') {
                setToolResult(thinking, eventData.tool, eventData.result, eventData.error);
                continue;
              }

              // Event: delta
              if (eventData.delta !== undefined && eventData.delta !== null) {
                accumulatedText += String(eventData.delta);
                const targetEl = ensureTextContainer();
                targetEl.innerHTML = renderMarkdown(accumulatedText);
                scrollToBottom();
              }

              // Event: done
              if (eventData.done) {
                clearTimeout(idleTimer);
                finalizeAllToolSteps(thinking);
                if (accumulatedText) {
                  persistMessage('assistant', accumulatedText, [], session);
                }
                return;
              }
            } catch (jsonErr) {
              if (jsonErr instanceof Error && jsonErr.message !== 'Unexpected token') {
                throw jsonErr;
              }
            }
          }
        }
      }
    } finally {
      clearTimeout(idleTimer);
      finalizeAllToolSteps(thinking);
      isStreaming = false;
      activeAbortController = null;
      stopContainer.classList.add('hidden');
    }

    if (accumulatedText) {
      persistMessage('assistant', accumulatedText, [], session);
    }
  }

  // --- Form Submission Handler ---
  async function handleSend() {
    const text = inputEl.value.trim();
    const filesToSend = [...stagedFiles];

    if ((!text && filesToSend.length === 0) || isStreaming) return;

    const session = sessionInput.value.trim() || activeSession || 'web';
    activeSession = session;
    sessionInput.value = session;

    // Reset input and clear staged files
    inputEl.value = '';
    inputEl.style.height = 'auto';
    stagedFiles = [];
    renderStagedAttachments();

    // Render user message bubble
    addMessage('user', text, filesToSend);

    // Render thinking assistant placeholder
    const thinking = addThinkingIndicator();

    sendBtn.disabled = true;
    inputEl.disabled = true;

    try {
      await streamChat(session, text, filesToSend, thinking);
    } catch (err) {
      finalizeAllToolSteps(thinking);
      const errDiv = document.createElement('div');
      errDiv.className = 'msg-text-content';
      errDiv.style.color = 'var(--danger)';
      errDiv.textContent = `Error: ${err.message}`;
      thinking.bubble.appendChild(errDiv);
      persistMessage('assistant', `Error: ${err.message}`, [], session);
      showToast(err.message, true);
    } finally {
      sendBtn.disabled = false;
      inputEl.disabled = false;
      inputEl.focus();
      renderHistorySidebar();
    }
  }

  // --- Stop Generation ---
  stopGenerationBtn.addEventListener('click', () => {
    if (activeAbortController) {
      activeAbortController.abort();
      showToast('Generation cancelled');
    }
  });

  // --- File Upload & Staging Handlers ---
  function renderStagedAttachments() {
    if (stagedFiles.length === 0) {
      stagedAttachmentsEl.classList.add('hidden');
      stagedAttachmentsEl.replaceChildren();
      return;
    }

    stagedAttachmentsEl.classList.remove('hidden');
    stagedAttachmentsEl.replaceChildren();

    for (let i = 0; i < stagedFiles.length; i++) {
      const file = stagedFiles[i];
      const chip = document.createElement('div');
      chip.className = 'attachment-chip';

      const name = document.createElement('span');
      name.className = 'attachment-name';
      name.textContent = file.name;

      const size = document.createElement('span');
      size.className = 'attachment-size';
      size.textContent = `(${(file.size / 1024).toFixed(1)} KB)`;

      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'attachment-remove';
      removeBtn.innerHTML = '&times;';
      removeBtn.setAttribute('aria-label', `Remove attachment ${file.name}`);
      removeBtn.addEventListener('click', () => {
        stagedFiles.splice(i, 1);
        renderStagedAttachments();
      });

      chip.append(name, size, removeBtn);
      stagedAttachmentsEl.appendChild(chip);
    }
  }

  function handleFilesSelected(fileList) {
    for (const file of fileList) {
      if (file.size > 20 * 1024 * 1024) {
        showToast(`File "${file.name}" exceeds 20MB limit`, true);
        continue;
      }
      const reader = new FileReader();
      reader.onload = e => {
        stagedFiles.push({
          name: file.name,
          mime: file.type || 'text/plain',
          size: file.size,
          data: e.target.result
        });
        renderStagedAttachments();
      };
      if (file.type.startsWith('image/')) {
        reader.readAsDataURL(file);
      } else {
        reader.readAsDataURL(file);
      }
    }
  }

  attachBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => {
    if (e.target.files?.length) {
      handleFilesSelected(e.target.files);
      fileInput.value = '';
    }
  });

  // Drag and drop onto composer or messages
  ['dragenter', 'dragover'].forEach(eventName => {
    document.addEventListener(eventName, e => {
      e.preventDefault();
      e.stopPropagation();
    });
  });

  document.addEventListener('drop', e => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer?.files?.length) {
      handleFilesSelected(e.dataTransfer.files);
    }
  });

  // --- Model Catalog Management ---
  async function loadModels() {
    try {
      const r = await fetch('/api/models');
      if (!r.ok) throw new Error();
      const data = await r.json();
      currentModelCatalog = data.models || [];

      modelSelect.replaceChildren();

      const defaultOpt = document.createElement('option');
      defaultOpt.value = 'default';
      defaultOpt.textContent = `Default (${data.current_provider || 'Gateway'}: ${data.current_model || 'default'})`;
      modelSelect.appendChild(defaultOpt);

      // Group models by provider
      const groups = {};
      for (const m of currentModelCatalog) {
        const prov = m.provider || 'Other';
        if (!groups[prov]) groups[prov] = [];
        groups[prov].push(m);
      }

      for (const [prov, list] of Object.entries(groups)) {
        const optGroup = document.createElement('optgroup');
        optGroup.label = prov.toUpperCase();
        for (const m of list) {
          const opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = `${m.name} (${m.id})`;
          opt.dataset.provider = m.provider;
          optGroup.appendChild(opt);
        }
        modelSelect.appendChild(optGroup);
      }

      // Restore saved model preference
      const savedModel = localStorage.getItem('go-agent.webui.selected-model');
      if (savedModel && Array.from(modelSelect.options).some(o => o.value === savedModel)) {
        modelSelect.value = savedModel;
        const opt = modelSelect.selectedOptions[0];
        if (opt) modelSelect.dataset.provider = opt.dataset.provider || '';
      }
    } catch (_) {
      modelSelect.innerHTML = '<option value="default">Default Model</option>';
    }
  }

  modelSelect.addEventListener('change', () => {
    const opt = modelSelect.selectedOptions[0];
    modelSelect.dataset.provider = opt?.dataset?.provider || '';
    localStorage.setItem('go-agent.webui.selected-model', modelSelect.value);
    showToast(`Model set to ${opt ? opt.textContent : 'Default'}`);
  });

  // --- Capabilities Loaders ---
  function renderCapabilityCard(item, type) {
    const el = document.createElement('div');
    el.className = 'capability-item';

    const name = document.createElement('strong');
    name.textContent = item.name;

    const desc = document.createElement('span');
    desc.textContent = type === 'skill' ? (item.description || item.instructions || 'Skill') : (item.description || 'UTCP tool');

    el.append(name, desc);

    if (type === 'skill' && item.tags?.length) {
      const tags = document.createElement('small');
      tags.textContent = item.tags.slice(0, 3).join(' · ');
      el.append(tags);
    }
    return el;
  }

  async function loadSkillsAndTools() {
    try {
      const [skillsRes, toolsRes] = await Promise.all([
        fetch('/api/skills'),
        fetch('/api/tools')
      ]);

      if (skillsRes.ok) {
        const skillsData = await skillsRes.json();
        const skills = skillsData.skills || [];
        skillCountEl.textContent = skills.length;
        skillsEl.replaceChildren(...skills.map(s => renderCapabilityCard(s, 'skill')));
        if (!skills.length) skillsEl.innerHTML = '<div class="empty-state">No skills available</div>';
      }

      if (toolsRes.ok) {
        const toolsData = await toolsRes.json();
        const tools = toolsData.tools || [];
        toolCountEl.textContent = tools.length;
        toolsEl.replaceChildren(...tools.map(t => renderCapabilityCard(t, 'tool')));
        if (!tools.length) toolsEl.innerHTML = '<div class="empty-state">No tools registered</div>';
      }
    } catch (_) {
      skillsEl.innerHTML = '<div class="empty-state">Failed to load skills</div>';
      toolsEl.innerHTML = '<div class="empty-state">Failed to load tools</div>';
    }
  }

  // --- Sub-agents Management ---
  function renderSubagentCard(item) {
    const el = document.createElement('div');
    el.className = 'capability-item subagent-card';

    const name = document.createElement('strong');
    name.textContent = item.name;

    const desc = document.createElement('span');
    desc.textContent = item.description || 'Specialist UTCP sub-agent';

    el.append(name, desc);

    el.addEventListener('click', () => {
      inputEl.value = `Delegate this task to the ${item.name} sub-agent: `;
      inputEl.focus();
      inputEl.dispatchEvent(new Event('input'));
    });
    return el;
  }

  async function loadSubagents() {
    try {
      const r = await fetch('/api/subagents');
      if (!r.ok) throw new Error();
      const data = await r.json();
      const items = data.subagents || [];
      subagentsEl.replaceChildren(...items.map(renderSubagentCard));
      if (!items.length) {
        subagentsEl.innerHTML = '<div class="empty-state">No sub-agents registered</div>';
      }
    } catch (_) {
      subagentsEl.innerHTML = '<div class="empty-state">Failed to load sub-agents</div>';
    }
  }

  function openSubagentModal() {
    subagentError.textContent = '';
    subagentForm.reset();
    subagentModal.classList.remove('hidden');
    setTimeout(() => subagentName.focus(), 50);
  }

  function closeSubagentModal() {
    subagentModal.classList.add('hidden');
  }

  async function submitSubagent(e) {
    e.preventDefault();
    subagentError.textContent = '';

    const payload = {
      name: subagentName.value.trim(),
      provider: subagentProvider.value.trim(),
      model: subagentModel.value.trim(),
      description: subagentDescription.value.trim(),
      system_prompt: subagentPrompt.value.trim()
    };

    if (!payload.name) {
      subagentError.textContent = 'Name is required';
      return;
    }

    const submitBtn = subagentForm.querySelector('button[type="submit"]');
    submitBtn.disabled = true;

    try {
      const res = await fetch('/api/subagents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Server returned ${res.status}`);

      closeSubagentModal();
      await loadSubagents();
      showToast(`Created sub-agent "${data.name}"`);
    } catch (err) {
      subagentError.textContent = err.message;
    } finally {
      submitBtn.disabled = false;
    }
  }

  // --- Health Check ---
  async function checkHealth() {
    try {
      const r = await fetch('/health');
      if (!r.ok) throw new Error();
      statusDot.classList.add('ok');
      statusText.textContent = 'Connected';
    } catch (_) {
      statusDot.classList.remove('ok');
      statusText.textContent = 'Disconnected';
    }
  }

  // --- Responsive Drawer Toggle ---
  toggleSidebarBtn.addEventListener('click', () => {
    const isOpen = sidebar.classList.toggle('open');
    sidebarBackdrop.classList.toggle('hidden', !isOpen);
  });

  sidebarBackdrop.addEventListener('click', () => {
    sidebar.classList.remove('open');
    sidebarBackdrop.classList.add('hidden');
  });

  // --- Event Listeners Initialization ---
  composerForm.addEventListener('submit', e => {
    e.preventDefault();
    handleSend();
  });

  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      composerForm.requestSubmit();
    }
  });

  inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = `${Math.min(inputEl.scrollHeight, 200)}px`;
  });

  newChatBtn.addEventListener('click', createNewSession);

  sessionInput.addEventListener('change', () => {
    const val = sessionInput.value.trim() || 'web';
    switchSession(val);
  });

  historySearch.addEventListener('input', () => {
    renderHistorySidebar();
  });

  createSubagentBtn.addEventListener('click', openSubagentModal);
  closeSubagentBtn.addEventListener('click', closeSubagentModal);
  cancelSubagentBtn.addEventListener('click', closeSubagentModal);
  subagentModal.addEventListener('click', e => {
    if (e.target === subagentModal) closeSubagentModal();
  });
  subagentForm.addEventListener('submit', submitSubagent);

  // Close modals on Escape key
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !subagentModal.classList.contains('hidden')) {
      closeSubagentModal();
    }
  });

  // --- App Startup ---
  migrateOldHistory();
  checkHealth();
  loadModels();
  loadSkillsAndTools();
  loadSubagents();
  renderHistorySidebar();
  renderSession(activeSession);

  // Polling intervals
  setInterval(checkHealth, 15000);
  setInterval(loadSkillsAndTools, 30000);
  setInterval(loadSubagents, 12000);

})();