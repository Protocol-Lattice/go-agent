const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, 'site.js'), 'utf8');

function createElement(textContent = '') {
  const attributes = new Map();
  const listeners = new Map();
  return {
    dataset: {},
    textContent,
    addEventListener(type, listener) {
      listeners.set(type, listener);
    },
    getAttribute(name) {
      return attributes.get(name) ?? null;
    },
    setAttribute(name, value) {
      attributes.set(name, value);
    },
    dispatch(type) {
      return listeners.get(type)?.();
    },
  };
}

function runSite({ elements = {}, storage, prefersDark = false, clipboard } = {}) {
  const root = createElement();
  const sandbox = {
    document: {
      documentElement: root,
      querySelector(selector) {
        return elements[selector] ?? null;
      },
    },
    navigator: {
      clipboard: clipboard ?? { writeText: async () => {} },
    },
    window: {
      localStorage: storage ?? {
        getItem: () => null,
        setItem: () => {},
      },
      matchMedia: () => ({ matches: prefersDark }),
      setTimeout(callback) {
        callback();
      },
    },
  };
  vm.runInNewContext(source, sandbox);
  return root;
}

test('menu state is updated through one guarded initializer', () => {
  const toggle = createElement();
  const nav = createElement();
  toggle.setAttribute('aria-expanded', 'false');

  runSite({
    elements: {
      '.menu-toggle': toggle,
      '#main-nav': nav,
    },
  });

  toggle.dispatch('click');
  assert.equal(toggle.getAttribute('aria-expanded'), 'true');
  assert.equal(nav.dataset.open, 'true');

  nav.dispatch('click');
  assert.equal(toggle.getAttribute('aria-expanded'), 'false');
  assert.equal(nav.dataset.open, 'false');
});

test('theme controls survive unavailable storage', () => {
  const toggle = createElement();
  const storage = {
    getItem() {
      throw new Error('storage blocked');
    },
    setItem() {
      throw new Error('storage blocked');
    },
  };

  const root = runSite({
    elements: { '#theme-toggle': toggle },
    storage,
    prefersDark: true,
  });

  assert.equal(toggle.textContent, 'Use light theme');
  assert.doesNotThrow(() => toggle.dispatch('click'));
  assert.equal(root.dataset.theme, 'light');
  assert.equal(toggle.textContent, 'Use dark theme');
});

test('copy feedback handles clipboard success and failure', async (t) => {
  await t.test('success', async () => {
    const button = createElement('Copy install');
    const command = createElement('  go get example/module  ');
    const feedback = createElement();
    let copied = '';

    runSite({
      elements: {
        '#copy-install': button,
        '#install-command': command,
        '#copy-feedback': feedback,
      },
      clipboard: {
        async writeText(value) {
          copied = value;
        },
      },
    });

    await button.dispatch('click');
    assert.equal(copied, 'go get example/module');
    assert.equal(feedback.textContent, 'Install command copied to your clipboard.');
    assert.equal(button.textContent, 'Copy install');
  });

  await t.test('blocked', async () => {
    const button = createElement('Copy install');
    const command = createElement('go get example/module');
    const feedback = createElement();

    runSite({
      elements: {
        '#copy-install': button,
        '#install-command': command,
        '#copy-feedback': feedback,
      },
      clipboard: {
        async writeText() {
          throw new Error('clipboard blocked');
        },
      },
    });

    await button.dispatch('click');
    assert.equal(feedback.textContent, 'Copy was blocked. Select the command and copy it manually.');
  });
});
