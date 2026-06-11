/**
 * §5 El bloque transformer — diagrama interactivo de la pieza que se
 * repite N veces (hover explica cada parte). Estático: solo usa el config.
 */

let captionEl;

export function init(rootEl, cfg) {
  const d = cfg.hidden_dim, f = cfg.ffn_dim, L = cfg.layers, h = cfg.heads;

  const PARTS = [
    { cls: 'blk-box--io', text: `entran ${d} números por token`, info:
      `la salida del paso anterior: un vector de ${d} dims por cada token del texto` },
    { cls: 'blk-box--norm', text: 'LayerNorm', info:
      'normaliza los valores del vector para que el entrenamiento no explote ni se apague' },
    { cls: 'blk-box--attn', text: `Atención multi-cabeza (${h} cabezas)`, info:
      `la del paso anterior: cada token mezcla información de los anteriores, en ${h} cabezas paralelas que captan relaciones distintas` },
    { cls: 'blk-plus', text: '⊕ residual', info:
      'la entrada se suma a la salida: la capa solo añade una corrección, no reemplaza — así la señal nunca se pierde' },
    { cls: 'blk-box--norm', text: 'LayerNorm', info:
      'otra normalización antes de la segunda mitad del bloque' },
    { cls: 'blk-box--ffn', text: `Feed-forward ${d} → ${f.toLocaleString('es')} → ${d}`, info:
      `una mini red neuronal aplicada a cada posición por separado: expande el vector ×4, lo procesa y lo comprime — aquí "piensa" cada token` },
    { cls: 'blk-plus', text: '⊕ residual', info:
      'de nuevo: lo nuevo se suma a lo que había' },
    { cls: 'blk-box--io', text: 'salen vectores refinados → siguiente capa', info:
      `el mismo formato que entró (${d} dims por token), pero con más contexto destilado — listo para la capa siguiente` },
  ];

  rootEl.innerHTML = `
    <div class="blk" id="blk">
      ${PARTS.map((p, i) => p.cls === 'blk-plus'
        ? `<div class="blk-plus mono" data-i="${i}">${p.text}</div>`
        : `<div class="blk-box ${p.cls}" data-i="${i}">${p.text}</div>`
      ).join('<div class="blk-arrow">↓</div>')}
    </div>
    <div class="blk-repeat" id="blk-repeat">
      ${Array.from({ length: L }, (_, i) => `<div class="blk-layer" data-l="${i + 1}"></div>`).join('')}
    </div>
    <p class="widget-caption" id="blk-caption">este bloque, apilado × ${L} en ${cfg.name} · × 96 en GPT-3</p>`;

  captionEl = rootEl.querySelector('#blk-caption');
  const base = `este bloque, apilado × ${L} en ${cfg.name} · × 96 en GPT-3`;

  rootEl.querySelectorAll('[data-i]').forEach(el => {
    el.addEventListener('mouseenter', () => {
      captionEl.textContent = PARTS[Number(el.dataset.i)].info;
    });
  });
  rootEl.querySelectorAll('.blk-layer').forEach(el => {
    el.addEventListener('mouseenter', () => {
      const l = Number(el.dataset.l);
      captionEl.textContent = l <= Math.ceil(L / 2)
        ? `capa ${l} de ${L}: las capas tempranas captan patrones locales y sintaxis`
        : `capa ${l} de ${L}: las capas profundas captan semántica y relaciones abstractas`;
    });
  });
  rootEl.querySelector('#blk').addEventListener('mouseleave', () => {
    captionEl.textContent = base;
  });
  rootEl.querySelector('#blk-repeat').addEventListener('mouseleave', () => {
    captionEl.textContent = base;
  });
}

export function update() { /* estático: nada que actualizar por ciclo */ }
