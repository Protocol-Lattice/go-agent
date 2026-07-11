const THEME_STORAGE_KEY = 'go-agent-theme';

function prefersDarkTheme() {
  return typeof window.matchMedia === 'function'
    && window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function readStoredTheme() {
  try {
    const theme = window.localStorage.getItem(THEME_STORAGE_KEY);
    return theme === 'light' || theme === 'dark' ? theme : null;
  } catch {
    return null;
  }
}

function writeStoredTheme(theme) {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Theme switching still works when storage is unavailable.
  }
}

function updateThemeLabel(toggle, isDark) {
  toggle.textContent = isDark ? 'Use light theme' : 'Use dark theme';
}

function initTheme() {
  const root = document.documentElement;
  const toggle = document.querySelector('#theme-toggle');
  if (!root || !toggle) {
    return;
  }

  const storedTheme = readStoredTheme();
  if (storedTheme) {
    root.dataset.theme = storedTheme;
  }
  updateThemeLabel(toggle, storedTheme ? storedTheme === 'dark' : prefersDarkTheme());

  toggle.addEventListener('click', () => {
    const isDark = root.dataset.theme === 'dark'
      || (!root.dataset.theme && prefersDarkTheme());
    const nextTheme = isDark ? 'light' : 'dark';
    root.dataset.theme = nextTheme;
    writeStoredTheme(nextTheme);
    updateThemeLabel(toggle, nextTheme === 'dark');
  });
}

function setMenuOpen(toggle, nav, open) {
  toggle.setAttribute('aria-expanded', String(open));
  nav.dataset.open = String(open);
}

function initMenu() {
  const toggle = document.querySelector('.menu-toggle');
  const nav = document.querySelector('#main-nav');
  if (!toggle || !nav) {
    return;
  }

  toggle.addEventListener('click', () => {
    const open = toggle.getAttribute('aria-expanded') !== 'true';
    setMenuOpen(toggle, nav, open);
  });
  nav.addEventListener('click', () => setMenuOpen(toggle, nav, false));
}

function initCopyInstall() {
  const button = document.querySelector('#copy-install');
  const command = document.querySelector('#install-command');
  const feedback = document.querySelector('#copy-feedback');
  if (!button || !command) {
    return;
  }

  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(command.textContent.trim());
      button.textContent = 'Copied';
      if (feedback) {
        feedback.textContent = 'Install command copied to your clipboard.';
      }
    } catch {
      if (feedback) {
        feedback.textContent = 'Copy was blocked. Select the command and copy it manually.';
      }
    }

    window.setTimeout(() => {
      button.textContent = 'Copy install';
    }, 1600);
  });
}

initMenu();
initTheme();
initCopyInstall();
