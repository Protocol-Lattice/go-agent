(() => {
  const skillsEl = document.getElementById('skills');
  const input = document.getElementById('input');
  if (!skillsEl) return;

  const style = document.createElement('style');
  style.textContent = `
    .capability.skill-capability { position: relative; padding-right: 80px; }
    .skill-apply {
      position: absolute; right: 8px; top: 50%; transform: translateY(-50%);
      border: 1px solid rgba(255,255,255,.14); border-radius: 7px;
      background: rgba(255,255,255,.06); color: inherit; padding: 4px 8px;
      font: inherit; font-size: 11px; cursor: pointer; z-index: 3;
      user-select: none;
    }
    .skill-apply:hover { background: rgba(255,255,255,.12); }
    .skill-apply.active { background: rgba(100,160,255,.18); border-color: rgba(100,160,255,.45); }
  `;
  document.head.appendChild(style);

  let activeSkill = '';

  function applySkill(name) {
    activeSkill = String(name || '').trim();
    if (!activeSkill || !input) return;

    document.querySelectorAll('.skill-apply').forEach((el) => {
      const active = el.dataset.skill === activeSkill;
      el.classList.toggle('active', active);
      el.textContent = active ? 'Applied' : 'Apply';
    });

    const prefix = `Use the ${activeSkill} skill for this task.\n\n`;
    const value = input.value.trim();
    if (!value || value.startsWith('Use the ')) {
      input.value = prefix;
    } else {
      input.value = `${prefix}${input.value}`;
    }

    input.focus();
    input.setSelectionRange(input.value.length, input.value.length);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }

  function decorate() {
    skillsEl.querySelectorAll('.capability:not(.skill-decorated)').forEach((card) => {
      const name = card.querySelector('strong')?.textContent?.trim();
      if (!name) return;

      card.classList.add('skill-capability', 'skill-decorated');

      const apply = document.createElement('span');
      apply.className = 'skill-apply';
      apply.dataset.skill = name;
      apply.setAttribute('role', 'button');
      apply.setAttribute('tabindex', '0');
      apply.textContent = activeSkill === name ? 'Applied' : 'Apply';
      apply.title = `Apply ${name} skill`;
      card.appendChild(apply);
    });
  }

  skillsEl.addEventListener('click', (event) => {
    const apply = event.target.closest('.skill-apply');
    if (!apply || !skillsEl.contains(apply)) return;
    event.preventDefault();
    event.stopPropagation();
    applySkill(apply.dataset.skill);
  });

  skillsEl.addEventListener('keydown', (event) => {
    const apply = event.target.closest('.skill-apply');
    if (!apply || !skillsEl.contains(apply)) return;
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    event.stopPropagation();
    applySkill(apply.dataset.skill);
  });

  const observer = new MutationObserver(decorate);
  observer.observe(skillsEl, { childList: true, subtree: true });
  decorate();
})();
