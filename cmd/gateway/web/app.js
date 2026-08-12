const messages = document.getElementById('messages');
const form = document.getElementById('composer');
const input = document.getElementById('input');
const sessionInput = document.getElementById('session');
const send = document.getElementById('send');
const statusText = document.getElementById('statusText');
const statusDot = document.getElementById('statusDot');

function addMessage(role, text = '') {
  document.querySelector('.welcome')?.remove();
  const row = document.createElement('div');
  row.className = `msg ${role}`;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'YOU' : 'λ';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  row.append(avatar, bubble);
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

async function health() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error('unhealthy');
    statusDot.classList.add('ok');
    statusText.textContent = 'Gateway connected';
  } catch {
    statusDot.classList.remove('ok');
    statusText.textContent = 'Gateway unavailable';
  }
}

async function streamChat(session, message, bubble) {
  const res = await fetch('/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
    body: JSON.stringify({ session, message })
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try { detail = (await res.json()).error || detail; } catch {}
    throw new Error(detail);
  }
  if (!res.body) throw new Error('Streaming is not supported by this browser');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split(/\n\n/);
    buffer = events.pop() || '';
    for (const event of events) {
      for (const line of event.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6);
        if (data === '[DONE]') continue;
        if (data.startsWith('error: ')) throw new Error(data.slice(7));
        bubble.textContent += data;
        messages.scrollTop = messages.scrollHeight;
      }
    }
  }
}

async function submit(message) {
  message = message.trim();
  if (!message || send.disabled) return;
  const session = sessionInput.value.trim() || 'web';
  input.value = '';
  input.style.height = 'auto';
  addMessage('user', message);
  const bubble = addMessage('assistant');
  send.disabled = true;
  input.disabled = true;
  try {
    await streamChat(session, message, bubble);
  } catch (err) {
    bubble.textContent = `Error: ${err.message}`;
  } finally {
    send.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  submit(input.value);
});

input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
});

document.querySelectorAll('[data-prompt]').forEach(button => {
  button.addEventListener('click', () => {
    input.value = button.dataset.prompt;
    input.focus();
    input.dispatchEvent(new Event('input'));
  });
});

health();
setInterval(health, 15000);
