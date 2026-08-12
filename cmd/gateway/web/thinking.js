(() => {
  const style = document.createElement('link');
  style.rel = 'stylesheet';
  style.href = '/web/thinking.css';
  document.head.appendChild(style);

  const phases = ['Understanding request', 'Planning tool execution', 'Executing tools', 'Synthesizing result'];
  const phaseState = new WeakMap();

  function addInspector(row) {
    if (!row || phaseState.has(row)) return;
    const bubble = row.querySelector('.bubble');
    if (!bubble) return;
    const inspector = document.createElement('div');
    inspector.className = 'thinking-inspector';
    const header = document.createElement('div');
    header.className = 'thinking-inspector-header';
    header.innerHTML = '<span class="thinking-inspector-icon">✦</span><span>Thinking summary</span>';
    const list = document.createElement('div');
    list.className = 'thinking-inspector-list';
    phases.forEach((phase, i) => {
      const item = document.createElement('div');
      item.className = 'thinking-phase' + (i === 0 ? ' active' : '');
      item.dataset.phase = String(i);
      item.innerHTML = `<span class="thinking-phase-marker">${i === 0 ? '●' : '○'}</span><span>${phase}</span>`;
      list.appendChild(item);
    });
    const note = document.createElement('div');
    note.className = 'thinking-note';
    note.textContent = 'Showing execution summaries, not private chain-of-thought.';
    inspector.append(header, list, note);
    bubble.appendChild(inspector);
    phaseState.set(row, { inspector, list });
  }

  function update(row) {
    const state = phaseState.get(row);
    if (!state) return;
    const hasWorkflow = !!row.querySelector('.tool-workflow');
    const tools = row.querySelectorAll('.tool-step');
    const running = row.querySelector('.tool-running');
    const complete = row.querySelector('.tool-complete');
    const error = row.querySelector('.tool-error');
    let phase = hasWorkflow ? 2 : 0;
    if (hasWorkflow && !running && (complete || error)) phase = 3;
    [...state.list.children].forEach((el, i) => {
      el.classList.toggle('active', i === phase);
      el.classList.toggle('done', i < phase);
      el.querySelector('.thinking-phase-marker').textContent = i < phase ? '✓' : i === phase ? '●' : '○';
    });
    const header = row.querySelector('.tool-workflow-header strong');
    if (header && tools.length) header.textContent = `Tool workflow · ${tools.length} step${tools.length === 1 ? '' : 's'}`;
  }

  const observer = new MutationObserver(() => {
    document.querySelectorAll('.thinking-row').forEach(row => {
      addInspector(row);
      update(row);
    });
  });
  observer.observe(document.getElementById('messages') || document.body, { childList: true, subtree: true, attributes: true });
  document.querySelectorAll('.thinking-row').forEach(row => { addInspector(row); update(row); });
})();
