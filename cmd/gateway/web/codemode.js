const CodeModeUI = (() => {
  const state = { steps: [], active: null };

  function ensurePanel() {
    let panel = document.getElementById('codemodeWorkflow');
    if (panel) return panel;
    panel = document.createElement('section');
    panel.id = 'codemodeWorkflow';
    panel.className = 'codemode-workflow hidden';
    panel.innerHTML = `
      <div class="codemode-header">
        <div>
          <span class="codemode-eyebrow">CODEMODE</span>
          <strong>Tool workflow</strong>
        </div>
        <span id="codemodeStatus" class="codemode-status">idle</span>
      </div>
      <div id="codemodeSteps" class="codemode-steps"></div>
    `;
    const messages = document.getElementById('messages');
    messages?.prepend(panel);
    return panel;
  }

  function setStatus(text, kind = '') {
    const el = document.getElementById('codemodeStatus');
    if (!el) return;
    el.textContent = text;
    el.className = `codemode-status ${kind}`.trim();
  }

  function render() {
    const panel = ensurePanel();
    panel.classList.remove('hidden');
    const list = document.getElementById('codemodeSteps');
    if (!list) return;
    list.replaceChildren(...state.steps.map((step, index) => {
      const row = document.createElement('article');
      row.className = `codemode-step ${step.status || ''}`.trim();
      const marker = document.createElement('span');
      marker.className = 'codemode-step-marker';
      marker.textContent = step.status === 'done' ? '✓' : step.status === 'error' ? '!' : String(index + 1);
      const body = document.createElement('div');
      body.className = 'codemode-step-body';
      const title = document.createElement('strong');
      title.textContent = step.title;
      const meta = document.createElement('span');
      meta.textContent = step.meta || '';
      body.append(title, meta);
      if (step.code) {
        const code = document.createElement('pre');
        code.className = 'codemode-code';
        code.textContent = step.code;
        body.append(code);
      }
      if (step.output) {
        const output = document.createElement('pre');
        output.className = 'codemode-output';
        output.textContent = step.output;
        body.append(output);
      }
      row.append(marker, body);
      return row;
    }));
  }

  function addStep(title, meta, code = '') {
    const step = { title, meta, code, status: 'running' };
    state.steps.push(step);
    state.active = step;
    setStatus(`${state.steps.length} step${state.steps.length === 1 ? '' : 's'} · running`, 'running');
    render();
    return step;
  }

  function finishStep(output = '') {
    if (!state.active) return;
    state.active.status = 'done';
    if (output) state.active.output = output;
    state.active = null;
    setStatus(`${state.steps.length} step${state.steps.length === 1 ? '' : 's'} · complete`, 'done');
    render();
  }

  function fail(message) {
    if (state.active) {
      state.active.status = 'error';
      state.active.output = message;
      state.active = null;
    }
    setStatus('workflow failed', 'error');
    render();
  }

  function reset() {
    state.steps.length = 0;
    state.active = null;
    const panel = document.getElementById('codemodeWorkflow');
    panel?.classList.add('hidden');
    setStatus('idle');
  }

  function extractCode(text) {
    const fenced = String(text || '').match(/```(?:go)?\s*([\s\S]*?)```/i);
    if (fenced) return fenced[1].trim();
    const code = String(text || '').match(/\b(?:package|func|fmt\.|client\.|tools\.).*[\s\S]*/i);
    return code ? code[0].trim() : '';
  }

  function inspectToolPanel(panel) {
    const title = panel.querySelector('.tool-activity-title')?.textContent || '';
    if (!/codemode\.run_code/i.test(title)) return;
    const output = panel.querySelector('.tool-activity-output')?.textContent || '';
    const code = extractCode(output);
    if (!state.active) addStep('Execute CodeMode program', 'codemode.run_code', code);
    else if (code && !state.active.code) { state.active.code = code; render(); }
    if (output && output !== 'Running…' && output !== 'No output returned') finishStep(output);
  }

  function observe() {
    const messages = document.getElementById('messages');
    if (!messages) return;
    const observer = new MutationObserver(mutations => {
      for (const mutation of mutations) {
        for (const node of mutation.addedNodes) {
          if (!(node instanceof Element)) continue;
          node.querySelectorAll?.('.tool-activity').forEach(inspectToolPanel);
          if (node.matches?.('.tool-activity')) inspectToolPanel(node);
        }
        if (mutation.type === 'characterData') return;
      }
      messages.querySelectorAll('.tool-activity').forEach(inspectToolPanel);
    });
    observer.observe(messages, { childList: true, subtree: true, characterData: true });
  }

  function hookComposer() {
    const form = document.getElementById('composer');
    form?.addEventListener('submit', reset, { capture: true });
  }

  function init() {
    ensurePanel();
    observe();
    hookComposer();
  }

  return { init, reset, addStep, finishStep, fail };
})();

document.readyState === 'loading'
  ? document.addEventListener('DOMContentLoaded', () => CodeModeUI.init())
  : CodeModeUI.init();
