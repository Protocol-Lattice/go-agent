(() => {
  const skillsRoot = document.getElementById('skills');
  if (!skillsRoot) return;

  const applySkill = (card) => {
    const name = card.querySelector('strong')?.textContent?.trim();
    if (!name) return;
    if (typeof setPrompt === 'function') {
      setPrompt(`Use the ${name} skill for this task.`);
    }
    card.classList.add('skill-applied');
    window.setTimeout(() => card.classList.remove('skill-applied'), 900);
  };

  const enhance = () => {
    skillsRoot.querySelectorAll('.capability').forEach((card) => {
      if (card.dataset.skillApplyReady === 'true') return;
      card.dataset.skillApplyReady = 'true';

      const action = document.createElement('span');
      action.className = 'skill-apply-action';
      action.textContent = 'Apply skill';
      card.appendChild(action);

      card.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopImmediatePropagation();
        applySkill(card);
      }, true);
    });
  };

  const style = document.createElement('style');
  style.textContent = `
    #skills .capability { cursor: pointer; position: relative; }
    #skills .skill-apply-action {
      display: inline-flex;
      align-items: center;
      margin-top: 7px;
      padding: 3px 7px;
      border: 1px solid #3c4e2f;
      border-radius: 6px;
      background: #182017;
      color: #b8f36a;
      font-size: 9px;
      font-weight: 800;
      letter-spacing: .02em;
      transition: background .18s, border-color .18s, transform .18s;
    }
    #skills .capability:hover .skill-apply-action {
      background: #202b1b;
      border-color: #587441;
    }
    #skills .capability.skill-applied .skill-apply-action {
      background: #2a421f;
      border-color: #7aaa50;
      transform: scale(1.03);
    }
    #skills .capability.skill-applied {
      background: #182017;
      border-color: #405431;
      box-shadow: inset 2px 0 #b8f36a;
    }
  `;
  document.head.appendChild(style);

  enhance();
  new MutationObserver(enhance).observe(skillsRoot, { childList: true, subtree: true });
})();
