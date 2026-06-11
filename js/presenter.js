/**
 * Modo presentación: la prosa y los widgets aparecen paso a paso con un
 * fundido, controlados por teclado (espacio / flechas / clicker PageDown),
 * para presentar la página en vivo sin diapositivas.
 *
 * Activación: ?presentar en la URL, o la tecla P en cualquier momento.
 * Solo alterna clases CSS (.reveal / .reveal--shown); no toca el estado ni
 * los widgets, así que todo sigue siendo interactivo durante la charla.
 */

const SECTION_LABELS = {
  llm: '¿Qué es un LLM?',
  tokens: 'Tokens',
  embeddings: 'Embeddings',
  posicion: 'Posición',
  atencion: 'Atención',
  bloque: 'El bloque',
  pipeline: 'El pipeline',
  logits: 'Logits',
  muestreo: 'Muestreo',
  bucle: 'El bucle',
  entrenamiento: 'Entrenamiento',
  limites: 'Límites',
};

let steps = [];   // [{ els, section, sectionStart, label }]
let cursor = 0;   // cuántos pasos están visibles
let active = false;
let hud = null;

export function isActive() {
  return active;
}

export function init() {
  hud = document.createElement('div');
  hud.className = 'presenter-hud mono';
  hud.hidden = true;
  document.body.appendChild(hud);

  window.addEventListener('keydown', onKey);
  document.addEventListener('click', onClick);

  if (new URLSearchParams(location.search).has('presentar')) enter();
}

/**
 * Orden de aparición por sección: título → definición → widget → analogía →
 * párrafos → Profundizar → En el mundo real → Pruébalo → remates marcados
 * con [data-reveal] en el widget.
 */
function collectSteps() {
  steps = [];
  document.querySelectorAll('.stage').forEach(section => {
    const label = SECTION_LABELS[section.dataset.stage];
    const prose = section.querySelector('.stage__prose');
    const widget = section.querySelector('.stage__widget');
    if (!label || !prose || !widget) return; // el hero queda siempre visible

    const push = (els, sectionStart = false) => {
      els = els.filter(Boolean);
      if (els.length) steps.push({ els, section, sectionStart, label });
    };

    push([prose.querySelector('.kicker'), prose.querySelector('h2')], true);
    push([prose.querySelector('.definicion')]);
    push([widget]);
    prose.querySelectorAll('.punto').forEach(p => push([p]));
    push([prose.querySelector('.analogia')]);
    [...prose.children]
      .filter(el => el.matches('p') && !el.matches('.kicker, .tryit, .analogia'))
      .forEach(p => push([p]));
    push([prose.querySelector('details.deeper')]);
    push([prose.querySelector('.mundo-real')]);
    push([prose.querySelector('.tryit')]);
    widget.querySelectorAll('[data-reveal]').forEach(el => push([el]));
  });
}

function enter() {
  if (active) return;
  active = true;
  cursor = 0;
  collectSteps();
  steps.forEach(step => step.els.forEach(el => el.classList.add('reveal')));
  hud.hidden = false;
  updateHud();
}

function exit() {
  if (!active) return;
  active = false;
  steps.forEach(step =>
    step.els.forEach(el => el.classList.remove('reveal', 'reveal--shown')));
  hud.hidden = true;
}

function advance() {
  if (cursor >= steps.length) return;
  const step = steps[cursor];
  cursor += 1;
  step.els.forEach(el => el.classList.add('reveal--shown'));
  if (step.sectionStart) scrollSection(step.section);
  else ensureVisible(step.els[step.els.length - 1]);
  updateHud();
}

function back() {
  if (cursor === 0) return;
  cursor -= 1;
  steps[cursor].els.forEach(el => el.classList.remove('reveal--shown'));
  const prev = steps[cursor - 1];
  if (prev) {
    if (prev.sectionStart) scrollSection(prev.section);
    else ensureVisible(prev.els[prev.els.length - 1]);
  }
  updateHud();
}

function updateHud() {
  if (cursor === 0) {
    hud.textContent = 'presentación · espacio avanza · ← retrocede · P sale';
  } else {
    hud.textContent = `${steps[cursor - 1].label} · ${cursor}/${steps.length} · P sale`;
  }
}

// ─── Teclado: espacio/→/PageDown avanzan (los clickers mandan PageDown) ───

function onKey(e) {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const typing = e.target.matches?.('input, textarea, select');

  if ((e.key === 'p' || e.key === 'P') && !typing) {
    e.preventDefault();
    active ? exit() : enter();
    return;
  }
  if (!active || typing) return;

  if (e.key === ' ' || e.key === 'ArrowRight' || e.key === 'PageDown') {
    e.preventDefault();
    // Que el espacio no re-active el botón con foco (p.ej. «Comenzar»)
    if (e.target instanceof HTMLElement && e.target.matches('button, summary, a')) e.target.blur();
    e.shiftKey ? back() : advance();
  } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
    e.preventDefault();
    back();
  } else if (e.key === 'Escape') {
    exit();
  }
}

// Clic en zonas muertas avanza; widgets, controles y navegación siguen
// siendo interactivos sin avanzar la presentación.
function onClick(e) {
  if (!active) return;
  if (e.target.closest('button, a, input, label, select, summary, details, .stage__widget, .rail, .context-bar, .hero, .presenter-hud')) return;
  advance();
}

// ─── Scroll con el mismo fallback que main.js (smooth puede no animar) ───

function scrollSection(section) {
  section.scrollIntoView({ behavior: 'smooth' });
  setTimeout(() => {
    if (Math.abs(section.getBoundingClientRect().top) > 120) {
      section.scrollIntoView({ behavior: 'instant' });
    }
  }, 700);
}

function isInView(el) {
  const r = el.getBoundingClientRect();
  const vh = window.innerHeight;
  const fits = r.height < vh - 140;
  return fits
    ? r.top >= 50 && r.bottom <= vh - 24
    : r.top >= -10 && r.top <= vh * 0.4; // muy alto: basta ver su inicio
}

function ensureVisible(el) {
  if (isInView(el)) return;
  const block = el.getBoundingClientRect().height < window.innerHeight - 140 ? 'center' : 'start';
  el.scrollIntoView({ behavior: 'smooth', block });
  setTimeout(() => {
    if (!isInView(el)) el.scrollIntoView({ behavior: 'instant', block });
  }, 700);
}
