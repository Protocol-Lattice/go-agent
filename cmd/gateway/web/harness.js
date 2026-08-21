(() => {
  const stages = [
    ['planner', 'Planner', 'task graph'],
    ['context', 'Context Builder', 'grounded context'],
    ['editor', 'Editor', 'mutations'],
    ['validator', 'Validator', 'checks'],
    ['repairer', 'Repairer', 'self-healing'],
  ];
  const capabilities = [
    'AST workspace index',
    'symbol-aware editing',
    'approval-gated tools',
    'workspace diff previews',
    'parallel validation',
    'bounded dynamic workflows',
  ];

  const shell = document.createElement('section');
  shell.className = 'harness-shell';
  shell.innerHTML = `
    <div class="harness-head">
      <div class="harness-title">
        <span class="harness-badge">Harness</span>
        <div><strong>Agent engineering workflow</strong><span>go-harness-inspired orchestration view</span></div>
      </div>
      <div class="harness-mode" role="group" aria-label="Workflow mode">
        <button type="button" data-mode="static" class="active">Static</button>
        <button type="button" data-mode="dynamic">Dynamic</button>
      </div>
    </div>
    <div class="harness-pipeline" id="harnessPipeline"></div>
    <div class="harness-meta" id="harnessMeta"></div>
    <div class="harness-note" id="harnessNote">UI model based on <code>go-harness</code>: Planner → Context Builder → Editor → Validator → Repairer. Runtime execution remains in go-agent.</div>
  `;

  const chat = document.querySelector('.chat');
  const messages = document.getElementById('messages');
  const composer = document.getElementById('composer');
  if (!chat || !messages || !composer) return;
  chat.insertBefore(shell, messages);

  const pipeline = shell.querySelector('#harnessPipeline');
  const meta = shell.querySelector('#harnessMeta');
  const note = shell.querySelector('#harnessNote');

  pipeline.replaceChildren(...stages.map(([id, title, subtitle]) => {
    const el = document.createElement('div');
    el.className = 'harness-stage';
    el.dataset.stage = id;
    el.innerHTML = `<div class="harness-stage-dot"></div><strong>${title}</strong><span>${subtitle}</span>`;
    return el;
  }));

  meta.replaceChildren(...capabilities.map(text => {
    const el = document.createElement('span');
    el.className = 'harness-chip';
    el.innerHTML = `<strong>✓</strong> ${text}`;
    return el;
  }));

  function setStage(id) {
    const index = stages.findIndex(x => x[0] === id);
    document.querySelectorAll('.harness-stage').forEach((el, i) => {
      el.classList.toggle('active', i === index);
      el.classList.toggle('done', i < index);
    });
  }

  function resetStages() {
    document.querySelectorAll('.harness-stage').forEach(el => el.classList.remove('active', 'done'));
    setStage('planner');
  }

  function finishStages() {
    document.querySelectorAll('.harness-stage').forEach(el => {
      el.classList.remove('active');
      el.classList.add('done');
    });
  }

  composer.addEventListener('submit', () => {
    resetStages();
    setTimeout(() => setStage('context'), 250);
  }, true);

  const observer = new MutationObserver(() => {
    if (messages.querySelector('.tool-thinking')) {
      setStage('editor');
      const outputs = messages.querySelectorAll('.tool-activity-output[data-ready="true"]');
      if (outputs.length) setStage('validator');
    }
    const thinking = messages.querySelector('.thinking-row');
    if (!thinking && messages.querySelector('.msg.assistant')) {
      const last = messages.lastElementChild;
      if (last?.classList.contains('msg') && last.classList.contains('assistant')) finishStages();
    }
  });
  observer.observe(messages, { childList: true, subtree: true, attributes: true });

  shell.querySelectorAll('[data-mode]').forEach(button => {
    button.addEventListener('click', () => {
      shell.querySelectorAll('[data-mode]').forEach(b => b.classList.remove('active'));
      button.classList.add('active');
      if (button.dataset.mode === 'dynamic') {
        note.innerHTML = 'Dynamic mode is a UI workflow hint. For full <code>go-harness -ultra</code> execution, connect the harness runtime to this gateway.';
      } else {
        note.innerHTML = 'UI model based on <code>go-harness</code>: Planner → Context Builder → Editor → Validator → Repairer. Runtime execution remains in go-agent.';
      }
    });
  });
})();
