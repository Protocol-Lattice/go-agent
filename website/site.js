const root = document.documentElement;
const menuToggle = document.querySelector('.menu-toggle');
const mainNav = document.querySelector('#main-nav');
const themeToggle = document.querySelector('#theme-toggle');
const copyInstall = document.querySelector('#copy-install');
const installCommand = document.querySelector('#install-command');
const copyFeedback = document.querySelector('#copy-feedback');

function setTheme(theme) {
  root.dataset.theme = theme;
  window.localStorage.setItem('go-agent-theme', theme);
  themeToggle.textContent = theme === 'dark' ? 'Use light theme' : 'Use dark theme';
}

const savedTheme = window.localStorage.getItem('go-agent-theme');
if (savedTheme === 'light' || savedTheme === 'dark') {
  setTheme(savedTheme);
} else {
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  themeToggle.textContent = prefersDark ? 'Use light theme' : 'Use dark theme';
}

menuToggle.addEventListener('click', () => {
  const open = menuToggle.getAttribute('aria-expanded') === 'true';
  menuToggle.setAttribute('aria-expanded', String(!open));
  mainNav.dataset.open = String(!open);
});

mainNav.addEventListener('click', () => {
  menuToggle.setAttribute('aria-expanded', 'false');
  mainNav.dataset.open = 'false';
});

themeToggle.addEventListener('click', () => {
  const isDark = root.dataset.theme === 'dark' || (!root.dataset.theme && window.matchMedia('(prefers-color-scheme: dark)').matches);
  setTheme(isDark ? 'light' : 'dark');
});

copyInstall.addEventListener('click', async () => {
  const command = installCommand.textContent.trim();
  try {
    await navigator.clipboard.writeText(command);
    copyInstall.textContent = 'Copied';
    copyFeedback.textContent = 'Install command copied to your clipboard.';
  } catch {
    copyFeedback.textContent = 'Copy was blocked. Select the command and copy it manually.';
  }

  window.setTimeout(() => {
    copyInstall.textContent = 'Copy install';
  }, 1600);
});
