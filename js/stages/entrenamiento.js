/**
 * §Detrás de escena — Entrenamiento: pre-training → fine-tuning → inferencia.
 * Diagrama estático de 3 etapas con hover; solo usa el config del modelo.
 */

let captionEl;

const STAGES = [
  {
    title: 'Pre-entrenamiento',
    tag: 'el conocimiento',
    rows: [
      ['tarea', 'predecir el siguiente token'],
      ['datos', 'billones de tokens (web, libros, código)'],
      ['tiempo', 'meses · miles de GPUs'],
      ['costo', 'millones de USD'],
      ['resultado', 'modelo base: solo continúa texto'],
    ],
    info: 'el mismo juego de esta página, jugado billones de veces: predecir, medir el error, ajustar los pesos con backpropagation',
  },
  {
    title: 'Fine-tuning + RLHF',
    tag: 'el asistente',
    rows: [
      ['tarea', 'de continuar texto a conversar'],
      ['datos', 'miles–millones de ejemplos curados'],
      ['tiempo', 'días–semanas'],
      ['resultado', 'así GPT-3 se volvió ChatGPT'],
    ],
    info: 'ejemplos de conversación + humanos puntuando respuestas (RLHF): el modelo base aprende el formato pregunta-respuesta y a ser útil',
  },
  {
    title: 'Inferencia',
    tag: 'estás aquí',
    here: true,
    rows: [
      ['tarea', 'usar el modelo: solo lectura'],
      ['datos', 'tu prompt'],
      ['tiempo', 'milisegundos por token'],
      ['aprende', 'nada — pesos congelados'],
    ],
    info: 'lo que hiciste hoy en esta página: una pasada hacia adelante por pesos congelados. El modelo no aprende nada de tu uso',
  },
];

export function init(rootEl) {
  rootEl.innerHTML = `
    <div class="train">
      ${STAGES.map((s, i) => `
        <div class="train-card ${s.here ? 'train-card--here' : ''}" data-i="${i}">
          <div class="train-card__head">
            <span class="train-card__title">${s.title}</span>
            <span class="train-card__tag mono ${s.here ? 'is-here' : ''}">${s.tag}</span>
          </div>
          <div class="train-card__rows mono">
            ${s.rows.map(([k, v]) => `<div><span class="k">${k}</span><span>${v}</span></div>`).join('')}
          </div>
        </div>`).join('<div class="blk-arrow">↓</div>')}
    </div>
    <p class="widget-caption" id="train-caption">Pasa el cursor por una etapa.</p>`;

  captionEl = rootEl.querySelector('#train-caption');
  rootEl.querySelectorAll('.train-card').forEach(card => {
    card.addEventListener('mouseenter', () => {
      captionEl.textContent = STAGES[Number(card.dataset.i)].info;
    });
  });
  rootEl.querySelector('.train').addEventListener('mouseleave', () => {
    captionEl.textContent = 'Pasa el cursor por una etapa.';
  });
}

export function update() { /* estático: nada que actualizar por ciclo */ }
