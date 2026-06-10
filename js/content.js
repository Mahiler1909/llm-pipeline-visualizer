/**
 * Prosa educativa de cada sección (español).
 * Funciones template que reciben el config del modelo para que
 * dimensiones, capas y vocabulario sean siempre los reales.
 */

export const PROSE = {
  tokens: {
    step: 'Paso 1 · Tokens',
    title: 'El texto se corta en piezas',
    basic: (cfg) => `
      <p>El modelo no lee palabras: lee <strong>tokens</strong>, fragmentos de texto
      con un ID numérico. Es lo primero que le pasa a tu frase — deja de ser texto
      y se convierte en una lista de números.</p>
      <p>Las palabras comunes son un solo token ("the"), pero las raras se parten
      en pedazos ("tokenization" → "token" + "ization"). Fíjate también en que el
      espacio inicial forma parte del token: <span class="mono">" the"</span> y
      <span class="mono">"the"</span> son tokens distintos.</p>`,
    advanced: (cfg) => `
      <p>GPT-2 usa <strong>BPE (Byte-Pair Encoding)</strong>: parte de los 256 bytes
      posibles y fusiona iterativamente el par de símbolos más frecuente del corpus
      de entrenamiento, unas 50.000 veces. Resultado: un vocabulario de
      ${cfg.vocab_size.toLocaleString('es')} tokens.</p>
      <div class="formula">vocab = 256 bytes
+ ~50.000 fusiones BPE
+ &lt;|endoftext|&gt;</div>
      <p>Por eso una palabra rara o inventada se parte en varios pedazos: sus pares
      de letras nunca fueron lo bastante frecuentes como para fusionarse.</p>`,
    tryIt: (cfg) => `Escribe una palabra inventada como «tokenizometro» en el campo
      de abajo y mira en cuántos pedazos la parte el tokenizer real.`,
  },

  embeddings: {
    step: 'Paso 2 · Embeddings',
    title: 'Cada token se vuelve un vector con significado',
    basic: (cfg) => `
      <p>Un ID como 3139 no dice nada sobre el <em>significado</em>. Por eso cada
      token se convierte en una lista de <strong>${cfg.hidden_dim} números</strong>:
      su <strong>embedding</strong>. Esa lista es el "significado" del token para
      el modelo.</p>
      <p>Palabras parecidas tienen números parecidos: "king" y "queen" quedan cerca
      en este espacio; "king" y "pizza", lejos. Lo que ves a la derecha son los
      valores <strong>reales</strong> del modelo, descargados de HuggingFace.</p>`,
    advanced: (cfg) => `
      <p>Los embeddings forman un espacio vectorial de ${cfg.hidden_dim} dimensiones
      donde la <em>dirección</em> codifica significado. La cercanía se mide con
      <strong>similitud coseno</strong>:</p>
      <div class="formula">cos(θ) = A·B / (|A|·|B|)
+1 = mismo sentido · 0 = sin relación · −1 = opuestos</div>
      <p>La matriz de embeddings (wte) es
      ${cfg.vocab_size.toLocaleString('es')} × ${cfg.hidden_dim} ≈
      <strong>${Math.round(cfg.vocab_size * cfg.hidden_dim / 1e6)}M parámetros</strong>,
      y se reutiliza al final del pipeline para convertir el vector de salida en
      puntajes (<em>weight tying</em>).</p>
      <p>Falta un ingrediente: antes de entrar al transformer, a cada embedding se
      le <strong>suma un embedding de posición</strong> (wpe). El mismo token en la
      posición 3 y en la 7 entra como vectores distintos — sin esa señal, la
      atención no sabría en qué orden están las palabras.</p>`,
    tryIt: (cfg) => `Vuelve arriba y comienza con «The king and the queen» — en la
      matriz de similitud, ¿qué par de tokens sale más alto?`,
  },

  atencion: {
    step: 'Paso 3 · Atención',
    title: 'Cada token mira a los anteriores',
    basic: (cfg) => `
      <p>Aquí ocurre la "comprensión". Un embedding solo dice qué significa un token
      <em>aislado</em>; la <strong>atención</strong> le añade el contexto: cada token
      reparte su atención entre los tokens anteriores para entender de qué se habla.</p>
      <p>En "capital of <em>Spain</em>", el token final necesita mirar a "Spain" para
      saber que la respuesta es Madrid. Lo que ves a la derecha son los pesos de
      atención <strong>reales</strong> de la primera capa del modelo — y esto se
      repite en ${cfg.layers} capas, captando relaciones cada vez más abstractas.</p>`,
    advanced: (cfg) => `
      <div class="formula">Attention(Q,K,V) = softmax(Q·Kᵀ/√d)·V</div>
      <p>Cada token genera tres vectores: <strong>Q</strong>uery (qué busco),
      <strong>K</strong>ey (qué ofrezco) y <strong>V</strong>alue (qué comunico).
      El producto Q·K mide cuánto "encaja" cada par de tokens, como un buscador
      comparando tu consulta con cada documento.</p>
      <p>Esto pasa en <strong>${cfg.heads} cabezas en paralelo</strong> (cada una de
      ${cfg.hidden_dim / cfg.heads} dimensiones) que aprenden relaciones distintas:
      sintaxis, posiciones, referencias… Después, una red feed-forward
      (${cfg.hidden_dim}→${cfg.ffn_dim}→${cfg.hidden_dim}) procesa cada posición, con
      conexiones residuales y LayerNorm estabilizando todo.</p>`,
    tryIt: (cfg) => `Cambia de cabeza con los botones: cada una mira a tokens
      distintos. Y haz clic en otro token para usarlo como punto de vista.`,
  },

  logits: {
    step: 'Paso 4 · Logits',
    title: 'Un puntaje para cada palabra posible',
    basic: (cfg) => `
      <p>Tras pasar por las ${cfg.layers} capas, el modelo produce un
      <strong>puntaje (logit) para cada uno de los ${cfg.vocab_size.toLocaleString('es')}
      tokens</strong> del vocabulario: cuanto más alto, más cree que ese token es el
      siguiente.</p>
      <p>Los puntajes se convierten en <strong>probabilidades</strong> con softmax,
      y la <strong>temperatura</strong> los moldea antes: baja = una opción domina,
      alta = muchas opciones parejas. Muévela y mira las barras reaccionar al instante.</p>
      <p>Los porcentajes son las probabilidades <strong>reales</strong> sobre los
      ${cfg.vocab_size.toLocaleString('es')} tokens: fíjate en que con T=1 ni
      siquiera el favorito suele pasar del 50% — el modelo siempre reparte su
      apuesta entre miles de opciones.</p>`,
    advanced: (cfg) => `
      <div class="formula">pᵢ = e^(zᵢ/T) / Σⱼ e^(zⱼ/T)</div>
      <p>La temperatura <strong>divide los logits antes del softmax</strong>: con
      T&lt;1 las diferencias se amplifican (distribución afilada, casi determinista);
      con T&gt;1 se atenúan (distribución plana, más azar). T cercana a 0 equivale a
      greedy: siempre gana el logit mayor.</p>
      <p>Los logits salen de multiplicar el vector final del último token por la
      matriz de embeddings transpuesta (<em>weight tying</em>): literalmente se mide
      qué token del vocabulario "se parece más" al vector que produjo el transformer.</p>`,
    tryIt: (cfg) => `Baja la temperatura a 0.1: una sola barra domina. Súbela a 2.0
      y compara cómo se aplana la distribución.`,
  },

  muestreo: {
    step: 'Paso 5 · Muestreo',
    title: 'Elegir la siguiente palabra: azar controlado',
    basic: (cfg) => `
      <p>El modelo <strong>no siempre elige la palabra más probable</strong>: tira
      una ruleta donde cada candidato ocupa un espacio proporcional a su probabilidad.
      Por eso cada generación puede ser distinta.</p>
      <p>Antes de girar, se filtra: <strong>top-k</strong> deja solo los k mejores
      candidatos y <strong>top-p</strong> recorta a los que acumulan hasta p de
      probabilidad (el "núcleo"). La ★ marca al ganador de esta tirada.</p>`,
    advanced: (cfg) => `
      <div class="formula">logits / T → top-k → softmax
→ top-p → renormalizar → muestrear</div>
      <p><strong>Greedy</strong> (siempre el más probable) es determinista pero cae
      en bucles repetitivos. El muestreo ponderado introduce variedad controlada.</p>
      <p><strong>Top-p (nucleus)</strong> es adaptativo: si el modelo está muy seguro,
      el 90% acumulado lo cubren 2-3 tokens; si está indeciso, entran muchos más.
      Por eso suele funcionar mejor que un top-k fijo.</p>`,
    tryIt: (cfg) => `Pulsa «Muestrear» varias veces: misma distribución, distinto
      ganador. Activa «Greedy» y la ★ se queda quieta en el más probable.`,
  },

  bucle: {
    step: 'Paso 6 · El bucle',
    title: 'La salida se vuelve entrada',
    basic: (cfg) => `
      <p>El token elegido <strong>se añade al final del texto</strong>… y todo vuelve
      a empezar: tokens, embeddings, atención, logits, muestreo. Una palabra por
      ciclo. Esto se llama <strong>generación autorregresiva</strong>.</p>
      <p>Pulsa el botón y observa la barra superior: el texto crece y cada sección
      de esta página se recalcula con el texto nuevo. Repítelo unas cuantas veces.</p>`,
    advanced: (cfg) => `
      <p>El modelo nunca "planea" la frase completa: solo predice el siguiente token,
      una y otra vez. La coherencia a largo plazo <em>emerge</em> de que cada
      predicción ve todo lo anterior a través de la atención.</p>
      <p>Cada ciclo re-procesa la secuencia completa, por eso generar texto largo es
      caro: el costo de la atención crece con el cuadrado de la longitud. Los modelos
      de producción lo alivian cacheando los K y V de los tokens ya procesados
      (<em>KV-cache</em>).</p>`,
    tryIt: (cfg) => `Repite el ciclo 3 veces con greedy activado, resetea y repítelo
      sin greedy: la primera vez el texto será idéntico; la segunda, distinto.`,
  },
};

const CLOSING = `Esto es todo lo que hace ChatGPT: este mismo ciclo, miles de veces,
con un modelo mil veces más grande.`;

export function getClosing() {
  return CLOSING;
}

/**
 * Inyecta la prosa en cada contenedor [data-prose] del documento.
 */
export function renderProse(cfg) {
  document.querySelectorAll('[data-prose]').forEach(el => {
    const key = el.dataset.prose;
    const s = PROSE[key];
    if (!s) return;
    el.innerHTML = `
      <p class="kicker">${s.step}</p>
      <h2>${s.title}</h2>
      ${s.basic(cfg)}
      <details class="deeper">
        <summary>Profundizar</summary>
        ${s.advanced(cfg)}
      </details>
      <p class="tryit">${s.tryIt(cfg)}</p>`;
  });
}
