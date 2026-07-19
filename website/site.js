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

function initTiltSurfaces() {
  if (typeof document.querySelectorAll !== 'function'
    || typeof window.matchMedia !== 'function'
    || !window.matchMedia('(pointer: fine) and (prefers-reduced-motion: no-preference)').matches) {
    return;
  }

  const surfaces = document.querySelectorAll('[data-tilt]');
  surfaces.forEach((surface) => {
    if (!surface.style || typeof surface.getBoundingClientRect !== 'function') {
      return;
    }

    const strength = Number(surface.dataset.tiltStrength) || 5;

    surface.addEventListener('pointermove', (event) => {
      const bounds = surface.getBoundingClientRect();
      if (!bounds.width || !bounds.height) {
        return;
      }

      const x = Math.min(Math.max((event.clientX - bounds.left) / bounds.width, 0), 1);
      const y = Math.min(Math.max((event.clientY - bounds.top) / bounds.height, 0), 1);

      surface.dataset.tilting = 'true';
      surface.style.setProperty('--tilt-x', `${((0.5 - y) * strength).toFixed(2)}deg`);
      surface.style.setProperty('--tilt-y', `${((x - 0.5) * strength).toFixed(2)}deg`);
      surface.style.setProperty('--pointer-x', `${(x * 100).toFixed(1)}%`);
      surface.style.setProperty('--pointer-y', `${(y * 100).toFixed(1)}%`);
    });

    surface.addEventListener('pointerleave', () => {
      surface.dataset.tilting = 'false';
      surface.style.setProperty('--tilt-x', '0deg');
      surface.style.setProperty('--tilt-y', '0deg');
      surface.style.setProperty('--pointer-x', '50%');
      surface.style.setProperty('--pointer-y', '50%');
    });
  });
}

initMenu();
initTheme();
initCopyInstall();
initTiltSurfaces();
