(() => {
  const skillsEl = document.getElementById('skills');
  if (!skillsEl) return;

  const style = document.createElement('style');
  style.textContent = `
    .capability.skill-capability { position: relative; padding-right: 72px; }
    .skill-apply {
      position: absolute; right: 8px; top: 50%; transform: translateY(-50%);
      border: 1px solid rgba(255,255,255,.14); border-radius: 7px;
      background: rgba(255,255,255,.06); color: inherit; padding: 4px 8px;
      font: inherit; font-size: 11px; cursor: pointer; z-index: 2;
    }
    .skill-apply:hover { background: rgba(255,255,255,.12); }
    .skill-apply.active { background: rgba(100,160,255,.18); border-color: rgba(100,160,255,.45); }
  `;
  document.head.appendChild(style);

  let activeSkill = '';

  function applySkill(name, button) {
    activeSkill = String(name || '').trim();
    document.querySelectorAll('.skill-apply').forEach((el) => {
      el.classList.toggle('active', el === button);
      el.textContent = el === button ? 'Applied' : 'Apply';
    });

    const input = document.getElementById('input');
    if (!input) return;
    const prefix = `Use the ${activeSkill} skill for this task.\n\n`;
    if (!input.value.trim() || input.value.trim().startsWith('Use the ')) input.value = prefix;
    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
    input.dispatchEvent(new Event('input'));
  }

  function decorate() {
    skillsEl.querySelectorAll('.capability:not(.skill-decorated)').forEach((card) => {
      card.classList.add('skill-capability', 'skill-decorated');
      const name = card.querySelector('strong')?.textContent?.trim();
      if (!name) return;
      const apply = document.createElement('span');
      apply.className = 'skill-apply';
      apply.textContent = activeSkill === name ? 'Applied' : 'Apply';
      apply.title = `Apply ${name} skill`;
      apply.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        applySkill(name, apply);
      });
      card.appendChild(apply);
    });
  }

  const observer = new MutationObserver(decorate);
  observer.observe(skillsEl, { childList: true, subtree: true });
  decorate();
})();
