// The composer stays editable while the agent is thinking or running CodeMode.
// Only the Send action is locked by the active request.
(() => {
  const input = document.getElementById('input');
  const send = document.getElementById('send');
  if (!input || !send) return;

  const unlockComposer = () => {
    if (input.disabled) input.disabled = false;
  };

  unlockComposer();

  // app.js intentionally locks Send during execution; do not let that
  // transient state make the text editor itself unusable.
  const observer = new MutationObserver(unlockComposer);
  observer.observe(input, { attributes: true, attributeFilter: ['disabled'] });

  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !send.disabled) {
      // app.js owns submission; this only guarantees the input remains usable.
      unlockComposer();
    }
  });
})();
