/* Re-colour code blocks with Shiki, using VS Code's own One Dark Pro theme.

   Pygments cannot reproduce an editor theme: it has its own token types, so a
   "One Dark" Pygments style still paints every name red. Shiki runs VS Code's
   TextMate grammars and theme files, so a block reads exactly as it does in the
   editor (short of Pylance's semantic colouring, which needs a language server).

   Pygments still renders first and stays as the fallback: if the CDN is
   unreachable, nothing here runs and the page keeps its Pygments colours.

   Both themes are emitted as CSS variables (defaultColor: false); custom.css picks
   one per Furo theme. Source listings from sphinx.ext.viewcode are left alone —
   their line anchors and back-links live inside the <pre> this would replace. */

import { codeToHtml } from 'https://cdn.jsdelivr.net/npm/shiki@1.29.2/+esm';

// Sphinx's lexer name (from the `highlight-<lang>` class) -> Shiki language id.
// `default` is an unlabelled fence (directory trees, plain output): left as is.
const LANGS = {
  python: 'python',
  python3: 'python',
  py: 'python',
  yaml: 'yaml',
  bash: 'bash',
  shell: 'bash',
  console: 'shellsession',
  json: 'json',
  toml: 'toml',
  bibtex: 'bibtex',
};

const THEMES = { light: 'github-light', dark: 'one-dark-pro' };

async function recolour(block) {
  const lexer = (block.className.match(/highlight-(\S+)/) || [])[1];
  const lang = LANGS[lexer];
  const pre = block.querySelector('pre');
  if (!lang || !pre) return;
  const html = await codeToHtml(pre.textContent.replace(/\n$/, ''), {
    lang,
    themes: THEMES,
    defaultColor: false,
  });
  const shiki = new DOMParser().parseFromString(html, 'text/html').body.firstElementChild;
  if (shiki) pre.replaceWith(shiki);
}

if (!location.pathname.includes('/_modules/')) {
  const blocks = document.querySelectorAll('div[class*="highlight-"]');
  Promise.all([...blocks].map((block) => recolour(block).catch(() => {})));
}
