(() => {
  const originalFetch = window.fetch.bind(window);
  const originalAddThinking = window.addThinking;
  const originalEnsureWorkflowPanel = window.ensureWorkflowPanel;

  function setWorkflowThinking(thinking) {
    window.__goAgentWorkflowThinking = thinking;
    return thinking;
  }

  if (typeof originalAddThinking === 'function') {
    window.addThinking = function (...args) {
      return setWorkflowThinking(originalAddThinking(...args));
    };
  }

  window.ensureWorkflowPanel = function (thinking) {
    const workflow = originalEnsureWorkflowPanel(thinking);
    const title = workflow?.container?.querySelector('.tool-workflow-header strong');
    if (title) title.textContent = 'Agent workflow';
    return workflow;
  };

  function addSkillStep(thinking, name) {
    if (!thinking || !name) return;
    const workflow = window.ensureWorkflowPanel(thinking);
    const key = `skill:${name}`;
    if (workflow.stepsMap.has(key)) return;

    const step = document.createElement('div');
    step.className = 'tool-activity';

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'tool-activity-toggle';
    button.setAttribute('aria-expanded', 'false');

    const icon = document.createElement('span');
    icon.className = 'tool-activity-icon';
    icon.textContent = '⚡';

    const title = document.createElement('span');
    title.className = 'tool-activity-title';
    title.textContent = `apply skill ${name}`;

    const chevron = document.createElement('span');
    chevron.className = 'tool-activity-chevron';
    chevron.textContent = '›';

    button.append(icon, title, chevron);

    const output = document.createElement('pre');
    output.className = 'tool-activity-output';
    output.hidden = true;
    output.textContent = 'Skill active';
    output.dataset.ready = 'true';

    button.addEventListener('click', () => {
      const open = output.hidden;
      output.hidden = !open;
      button.setAttribute('aria-expanded', String(open));
      chevron.textContent = open ? '⌄' : '›';
    });

    step.append(button, output);
    workflow.stepsList.appendChild(step);
    workflow.stepsMap.set(key, { name, tool: step, button, output, chevron });
    workflow.countEl.textContent = `${workflow.stepsMap.size} ${workflow.stepsMap.size === 1 ? 'step' : 'steps'} · running`;
  }

  window.__goAgentWorkflowEvent = (event) => {
    if (!event || event.type !== 'skill_start') return;
    addSkillStep(window.__goAgentWorkflowThinking, String(event.skill || '').trim());
  };

  window.fetch = async function (...args) {
    const response = await originalFetch(...args);
    const requestURL = typeof args[0] === 'string' ? args[0] : args[0]?.url || '';
    if (!requestURL.endsWith('/stream') || !response.body || typeof response.body.tee !== 'function') {
      return response;
    }

    const [eventsStream, clientStream] = response.body.tee();
    (async () => {
      const reader = eventsStream.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const chunks = buffer.split(/\n\n/);
          buffer = chunks.pop() || '';
          for (const chunk of chunks) {
            for (const line of chunk.split('\n')) {
              if (!line.startsWith('data: ')) continue;
              try {
                window.__goAgentWorkflowEvent(JSON.parse(line.slice(6)));
              } catch (_) {}
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    })();

    return new Response(clientStream, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  };
})();
