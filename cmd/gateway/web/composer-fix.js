// Keep the composer usable throughout streaming requests.
// Enter submits, Shift+Enter inserts a newline.
(() => {
  const input = document.getElementById('input');
  const form = document.getElementById('composer');
  const send = document.getElementById('send');
  if (!input || !form || !send) return;

  const unlockComposer = () => {
    input.disabled = false;
    input.readOnly = false;
    input.removeAttribute('disabled');
    input.removeAttribute('readonly');
    input.style.pointerEvents = 'auto';
  };

  unlockComposer();

  // app.js currently toggles input.disabled while a request is running.
  // Keep the editor interactive without changing the request/send lock.
  const observer = new MutationObserver(unlockComposer);
  observer.observe(input, { attributes: true, attributeFilter: ['disabled', 'readonly'] });

  // Capture Enter before app.js so the prompt is submitted even if another
  // handler temporarily changed the textarea state. Shift+Enter stays newline.
  input.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    unlockComposer();
    if (!send.disabled && input.value.trim()) form.requestSubmit();
  }, true);
})();
