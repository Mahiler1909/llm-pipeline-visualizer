/**
 * Contenido de cada sección (español), en formato presentación:
 * poco texto en pantalla, bloques visuales — la narración completa
 * vive en GUION.md.
 *
 * Estructura por sección:
 *   step      kicker («Paso N · Tema»)
 *   title     titular
 *   term/def  tarjeta de definición citable (1-2 líneas)
 *   points    3 bloques de palabras clave [titular, detalle]
 *   analogy   una línea «Piénsalo así:»
 *   advanced  bloque «Profundizar» plegado (teoría densa, para lectores)
 *   world     «En el mundo real»: UNA línea (o mini-tabla)
 *   tryIt     experimento ejecutable en ESA sección
 */

export const PROSE = {
  llm: {
    step: 'Definición · ¿Qué es un LLM?',
    title: 'Una sola tarea: predecir lo que sigue',
    term: 'LLM (Large Language Model)',
    def: (cfg) => `red neuronal con millones —o billones— de parámetros entrenada
      para una única tarea: dada una secuencia de texto, asignar una probabilidad
      a cada posible token siguiente.`,
    points: (cfg) => [
      ['P(siguiente token | contexto)', 'toda la «magia» es esta función'],
      [`${cfg.params} → 175.000M de parámetros`, '«large» es literal: GPT-3 es ~2.000× este modelo'],
      ['más escala → capacidades emergentes', 'instrucciones · razonamiento · aprender del prompt'],
    ],
    analogy: (cfg) => `el autocompletar del teléfono llevado al extremo: tan bueno
      prediciendo lo que sigue que, encadenando predicciones, escribe, traduce y
      programa.`,
    advanced: (cfg) => `
      <div class="formula">P(texto) = P(t₁) · P(t₂|t₁) · P(t₃|t₁t₂) · …</div>
      <p>Un modelo de lenguaje define la probabilidad de un texto completo como el
      producto de las probabilidades de cada token dado lo anterior. Generar es
      recorrer ese producto hacia adelante, un token a la vez. No hay un módulo de
      gramática ni una base de datos de hechos: solo esa función, aprendida
      leyendo billones de palabras.</p>
      <p>Las <em>scaling laws</em> muestran que el error baja de forma predecible
      al aumentar parámetros, datos y cómputo a la vez — la apuesta que produjo
      GPT-3, GPT-4 y todo lo que siguió.</p>`,
    world: (cfg) => `
      <p>ChatGPT, Claude y Copilot son esta misma función, en bucle.</p>`,
    tryIt: (cfg) => `Comparemos tamaños en la escala: ¿cuántas veces cabe este
      modelo en GPT-3?`,
  },

  tokens: {
    step: 'Concepto 1 · Tokens',
    title: 'El texto se corta en piezas',
    term: 'token',
    def: (cfg) => `fragmento de texto —una palabra, un trozo de palabra o un
      signo— con un ID numérico propio en el vocabulario del modelo; la unidad
      mínima que el modelo lee y escribe.`,
    points: (cfg) => [
      ['el modelo no lee palabras', 'lee fragmentos con un ID numérico'],
      ['común = 1 token · raro = varios', '"tokenization" → "token" + "ization"'],
      ['el espacio cuenta', '" the" y "the" son tokens distintos'],
    ],
    analogy: (cfg) => `piezas de Lego del lenguaje: con
      ${cfg.vocab_size.toLocaleString('es')} piezas distintas se arma cualquier
      texto posible, incluso palabras que no existen.`,
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
    world: (cfg) => `
      <p>Las APIs cobran por token · el contexto se mide en tokens · en inglés,
      1 token ≈ ¾ de palabra.</p>`,
    tryIt: (cfg) => `Juguemos a crear tokens: inventen una palabra (como
      «tokenizometro»), la escribimos abajo y vemos en cuántas piezas la corta el
      tokenizer real, en vivo.`,
  },

  embeddings: {
    step: 'Concepto 2 · Embeddings',
    title: 'Cada token se vuelve un vector con significado',
    term: 'embedding',
    def: (cfg) => `vector denso (aquí, ${cfg.hidden_dim} números) que representa
      el significado de un token: las coordenadas de un punto en un espacio donde
      estar cerca es significar parecido.`,
    points: (cfg) => [
      [`ID → ${cfg.hidden_dim} números`, 'el «significado» del token para el modelo'],
      ['parecido = cercano', 'king ≈ queen · king ≉ pizza'],
      ['estos valores son reales', 'descargados de HuggingFace, fila a fila'],
    ],
    analogy: (cfg) => `un mapa gigante de palabras: «pizza» y «pasta» caen cerca;
      «pizza» y «JavaScript», lejos. El significado es la posición en el mapa.`,
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
      puntajes (<em>weight tying</em>). Lo que aún falta aquí es el <em>orden</em>
      de los tokens — eso lo añade el paso siguiente.</p>`,
    world: (cfg) => `
      <p>Búsqueda semántica, recomendadores y RAG = comparar embeddings por coseno
      (&gt;0.7 muy relacionado · &lt;0.3 ruido).</p>`,
    tryIt: (cfg) => `Probemos con «The king and the queen»: ¿qué par sale más alto
      en la matriz de similitud?`,
  },

  posicion: {
    step: 'Concepto 3 · Posición',
    title: 'El orden importa',
    term: 'embedding de posición',
    def: (cfg) => `vector aprendido que codifica «soy el token n.º k de la
      secuencia»; se suma al embedding del token antes de entrar al transformer.`,
    points: (cfg) => [
      ['la atención mira todo a la vez', 'sola, no sabe qué token va primero'],
      ['solución: sumar un vector de posición', 'entrada = wte[token] + wpe[posición]'],
      ['mismo token, posición distinta', '→ entrada distinta al modelo'],
    ],
    analogy: (cfg) => `«dog bites man» y «man bites dog» usan exactamente las
      mismas piezas — solo el orden separa la noticia del accidente.`,
    advanced: (cfg) => `
      <div class="formula">entrada = wte[token] + wpe[posición]</div>
      <p>GPT-2 usa posiciones <strong>aprendidas</strong>: una matriz wpe de
      1024 × ${cfg.hidden_dim} entrenada junto al resto del modelo. De ahí sale su
      límite de contexto: la fila 1025 simplemente no existe.</p>
      <p>El transformer original usaba ondas sinusoidales (sin entrenar); los
      modelos actuales usan <strong>RoPE</strong> — rotaciones aplicadas dentro de
      la atención que escalan mucho mejor a contextos largos.</p>`,
    world: (cfg) => `
      <p>La «ventana de contexto» (8k, 128k, 1M tokens) es esto: hasta qué posición
      sabe codificar el modelo.</p>`,
    tryIt: (cfg) => `Movamos la posición con los chips: el vector del token queda
      quieto, el de posición cambia — y la suma con él.`,
  },

  atencion: {
    step: 'Concepto 4 · Atención',
    title: 'Cada token mira a los anteriores',
    term: 'atención (self-attention)',
    def: (cfg) => `mecanismo por el que cada token calcula cuánto le importa cada
      token anterior y mezcla la información de todos en proporción a ese peso; es
      donde el contexto entra al significado.`,
    points: (cfg) => [
      ['"the bank by the river"', '¿banco u orilla? el contexto desambigua'],
      ['cada token mira a los anteriores', 'y absorbe de ellos lo que necesita'],
      ['pesos reales de la capa 0 →', `esto se repite en ${cfg.layers} capas`],
    ],
    analogy: (cfg) => `una biblioteca: la <strong>Q</strong>uery es la búsqueda,
      cada <strong>K</strong>ey la etiqueta de un libro y cada
      <strong>V</strong>alue su contenido — el resultado es una mezcla de libros
      ponderada por cuánto encaja cada etiqueta con la búsqueda.`,
    advanced: (cfg) => `
      <div class="formula">Attention(Q,K,V) = softmax(Q·Kᵀ/√d)·V</div>
      <p>Cada token genera tres vectores: <strong>Q</strong>uery (qué busco),
      <strong>K</strong>ey (qué ofrezco) y <strong>V</strong>alue (qué comunico).
      El producto Q·K mide cuánto "encaja" cada par de tokens.</p>
      <p>GPT usa atención <strong>causal</strong>: cada token solo mira hacia
      atrás, porque al generar el futuro aún no existe. (BERT, que mira en ambas
      direcciones, entiende texto pero no puede generarlo.)</p>
      <p>Esto pasa en <strong>${cfg.heads} cabezas en paralelo</strong> (cada una de
      ${cfg.hidden_dim / cfg.heads} dimensiones) que aprenden relaciones distintas:
      sintaxis, posiciones, referencias…</p>`,
    world: (cfg) => `
      <p>El costo crece con el cuadrado de la longitud — por eso el contexto largo
      es caro (KV-cache, FlashAttention).</p>`,
    tryIt: (cfg) => `Cambiemos de cabeza con los botones (cada una mira distinto) y
      de token (clic) para usar otro punto de vista.`,
  },

  bloque: {
    step: 'Concepto 5 · El bloque transformer',
    title: 'La pieza que se repite',
    term: 'bloque transformer',
    def: (cfg) => `la unidad de cómputo del modelo — atención multi-cabeza + red
      feed-forward, envueltas en conexiones residuales y normalización. Un LLM es
      este bloque apilado N veces.`,
    points: (cfg) => [
      ['atención: mueve información entre tokens', 'la mitad famosa del bloque'],
      [`feed-forward: procesa cada token`, `${cfg.hidden_dim} → ${cfg.ffn_dim.toLocaleString('es')} → ${cfg.hidden_dim}, posición por posición`],
      [`apilado × ${cfg.layers} aquí · ~× 100 en los grandes`, 'capas tempranas: sintaxis · profundas: semántica'],
    ],
    analogy: (cfg) => `una línea de ensamblaje con ${cfg.layers} estaciones de
      estructura idéntica pero herramientas distintas: cada capa refina lo que le
      dejó la anterior.`,
    advanced: (cfg) => `
      <p>Cerca de dos tercios de los parámetros del modelo viven en las FFN — la
      atención es la parte famosa, no la más pesada. Las conexiones residuales y
      LayerNorm mantienen el entrenamiento estable.</p>
      <p>La misma pieza, tres arquitecturas: <strong>GPT</strong> (decoder-only,
      atención causal → genera), <strong>BERT</strong> (encoder-only, bidireccional
      → entiende y clasifica), <strong>T5</strong> (encoder-decoder → traduce,
      resume). Todos los asistentes de chat actuales son decoder-only.</p>`,
    world: (cfg) => `
      <p>«70B de parámetros» o «120 capas» = cuántas veces se apila el bloque y
      cuán anchas son sus matrices.</p>`,
    tryIt: (cfg) => `Recorramos el diagrama pieza por pieza — el texto de abajo
      explica cada una.`,
  },

  logits: {
    step: 'El pipeline · Logits',
    title: 'Un puntaje para cada palabra posible',
    term: 'logit',
    def: (cfg) => `puntaje crudo —un número sin escala— que el modelo asigna a
      cada uno de los ${cfg.vocab_size.toLocaleString('es')} tokens como candidato
      a continuar el texto; el softmax los convierte en probabilidades.`,
    points: (cfg) => [
      [`un puntaje por cada token`, `${cfg.vocab_size.toLocaleString('es')} candidatos en cada predicción`],
      ['softmax(z/T) → probabilidades reales', 'con T=1 ni el favorito suele pasar del 50%'],
      ['la temperatura afila o aplana', 'T→0 determinista · T alta caos'],
    ],
    analogy: (cfg) => `las puntuaciones de un jurado antes de normalizarlas:
      importan las <em>diferencias</em> entre candidatos, no los valores
      absolutos.`,
    advanced: (cfg) => `
      <div class="formula">pᵢ = e^(zᵢ/T) / Σⱼ e^(zⱼ/T)</div>
      <p>La temperatura <strong>divide los logits antes del softmax</strong>: con
      T&lt;1 las diferencias se amplifican (distribución afilada, casi determinista);
      con T&gt;1 se atenúan (distribución plana, más azar). T cercana a 0 equivale a
      greedy: siempre gana el logit mayor.</p>
      <p>Los logits salen de multiplicar el vector final del último token por la
      matriz de embeddings transpuesta (<em>weight tying</em>): literalmente se mide
      qué token del vocabulario "se parece más" al vector que produjo el transformer.</p>`,
    world: (cfg) => `
      <p>Los <span class="mono">logprobs</span> de las APIs son esto · el parámetro
      <span class="mono">temperature</span> es este divisor.</p>`,
    tryIt: (cfg) => `Bajemos la temperatura a 0.1 (una sola barra domina) y
      subámosla a 2.0 (todo se aplana).`,
  },

  muestreo: {
    step: 'El pipeline · Muestreo',
    title: 'Elegir la siguiente palabra: azar controlado',
    term: 'muestreo (sampling)',
    def: (cfg) => `elegir el siguiente token al azar, ponderado por su
      probabilidad, en vez de tomar siempre el más probable; top-k y top-p
      recortan la lista de candidatos antes de tirar el dado.`,
    points: (cfg) => [
      ['no siempre gana el más probable', 'se tira una ruleta ponderada'],
      ['top-k y top-p recortan candidatos', 'antes de girar'],
      ['★ = ganador de esta tirada', 'repetir la tirada → otro resultado'],
    ],
    analogy: (cfg) => `una ruleta donde cada candidato ocupa un arco proporcional
      a su probabilidad. Top-k y top-p deciden quién entra; la temperatura, qué
      tan parejos son los arcos.`,
    advanced: (cfg) => `
      <div class="formula">logits / T → top-k → softmax
→ top-p → renormalizar → muestrear</div>
      <p><strong>Greedy</strong> (siempre el más probable) es determinista pero cae
      en bucles repetitivos. El muestreo ponderado introduce variedad controlada.</p>
      <p><strong>Top-p (nucleus)</strong> es adaptativo: si el modelo está muy seguro,
      el 90% acumulado lo cubren 2-3 tokens; si está indeciso, entran muchos más.
      Por eso suele funcionar mejor que un top-k fijo.</p>`,
    world: (cfg) => `
      <table class="mini-table">
        <tr><td>chat (ChatGPT)</td><td class="mono">T≈0.7 · top-p 0.9</td></tr>
        <tr><td>código</td><td class="mono">T≈0.2</td></tr>
        <tr><td>escritura creativa</td><td class="mono">T≈1.2</td></tr>
        <tr><td>tests reproducibles</td><td class="mono">greedy (T=0)</td></tr>
      </table>`,
    tryIt: (cfg) => `Muestreemos varias veces: misma distribución, distinto
      ganador. Con «Greedy», la ★ se queda quieta.`,
  },

  bucle: {
    step: 'El pipeline · El bucle',
    title: 'La salida se vuelve entrada',
    term: 'generación autorregresiva',
    def: (cfg) => `el bucle donde el token elegido se añade al contexto y el
      modelo entero vuelve a ejecutarse para elegir el siguiente; el texto se
      escribe de a un token.`,
    points: (cfg) => [
      ['el token elegido se añade al texto', 'y la barra superior crece'],
      ['todo vuelve a ejecutarse', 'una palabra por ciclo'],
      ['la coherencia emerge', 'el modelo nunca «planea» la frase completa'],
    ],
    analogy: (cfg) => `escribir con una única regla — «lee todo lo anterior y
      añade la palabra siguiente» — repetida cientos de veces por respuesta.`,
    advanced: (cfg) => `
      <p>El modelo nunca "planea" la frase completa: solo predice el siguiente token,
      una y otra vez. La coherencia a largo plazo <em>emerge</em> de que cada
      predicción ve todo lo anterior a través de la atención.</p>
      <p>Cada ciclo re-procesa la secuencia completa, por eso generar texto largo es
      caro: el costo de la atención crece con el cuadrado de la longitud. Los modelos
      de producción lo alivian cacheando los K y V de los tokens ya procesados
      (<em>KV-cache</em>).</p>`,
    world: (cfg) => `
      <p>El «streaming» de ChatGPT es este bucle en vivo · el output se cobra por
      token porque cada token es una pasada completa.</p>`,
    tryIt: (cfg) => `Repitamos el ciclo varias veces y veamos crecer el texto en la
      barra superior — con greedy la secuencia se vuelve determinista.`,
  },

  entrenamiento: {
    step: 'Concepto 6 · Entrenamiento',
    title: '¿De dónde salen los pesos?',
    term: 'pre-entrenamiento',
    def: (cfg) => `ajustar los millones (o billones) de parámetros del modelo para
      que prediga bien el siguiente token sobre un corpus gigante; todo su
      «conocimiento» es el residuo de ese ajuste.`,
    points: (cfg) => [
      ['el mismo juego, billones de veces', 'predicción → error → backpropagation'],
      ['base → fine-tuning + RLHF → asistente', 'así GPT-3 se volvió ChatGPT'],
      ['usar ≠ aprender', 'en inferencia los pesos están congelados'],
    ],
    analogy: (cfg) => `comprimir internet en una función: para predecir bien lo
      que sigue hay que haber capturado los patrones — y comprimir es, en cierto
      modo, entender.`,
    advanced: (cfg) => `
      <p>Por eso un modelo base como DistilGPT-2 no «responde» preguntas: las
      continúa como continuaría cualquier texto. Un modelo <em>instruct</em> fue
      afinado además para tratar el texto como una conversación con turnos.</p>
      <p>Clave para no confundirse: el modelo <strong>no aprende al usarlo</strong>.
      Cada conversación parte de los mismos pesos congelados; la «memoria» entre
      turnos es solo el contexto, que se le reenvía completo cada vez.</p>`,
    world: (cfg) => `
      <p>De aquí salen la fecha de corte, que el chat de ayer no le enseñó nada, y
      las dos vías para conocimiento propio: RAG o fine-tuning.</p>`,
    tryIt: (cfg) => `Comparemos las tres etapas del diagrama: datos, tiempo y
      costo.`,
  },

  pipeline: {
    step: 'El pipeline · Las piezas, juntas',
    title: 'De un texto a la siguiente palabra',
    term: 'pipeline de inferencia',
    def: (cfg) => `la cadena completa que el modelo ejecuta en cada predicción:
      texto → tokens → vectores (+posición) → ${cfg.layers} bloques transformer →
      un puntaje por token → elegir uno → repetir.`,
    points: (cfg) => [
      ['ya conocemos todas las piezas', 'ahora, encadenadas de principio a fin'],
      ['esto ocurre en cada token generado', 'milisegundos por pasada completa'],
      ['faltan las tres etapas finales', 'logits → muestreo → el bucle'],
    ],
    analogy: (cfg) => `una línea de producción: entra texto crudo, cada estación
      lo transforma, y sale una única palabra — lista para volver a entrar.`,
    advanced: (cfg) => `
      <div class="formula">texto → tokens → embeddings (+posición)
→ × ${cfg.layers} bloques → logits → muestreo
→ token nuevo → repetir</div>
      <p>Las primeras cinco etapas ya las vimos como conceptos. Lo que sigue son
      las tres últimas — donde el modelo pasa de representar a <em>decidir</em>:
      puntuar candidatos, elegir uno y volver a empezar.</p>`,
    world: (cfg) => `
      <p>Cada token de cada respuesta de ChatGPT recorre exactamente esta
      cadena.</p>`,
    tryIt: (cfg) => `El mapa es clicable: saltemos a cualquier etapa para
      repasarla.`,
  },

  limites: {
    step: 'Para terminar · Límites',
    title: 'Lo que un LLM no es',
    term: 'alucinación',
    def: (cfg) => `salida generada con la misma mecánica —y la misma confianza—
      que un hecho correcto, pero falsa. El modelo optimiza plausibilidad, no
      verdad.`,
    points: (cfg) => [
      ['nada en el pipeline verifica hechos', 'verdad e invención: misma mecánica'],
      ['corte de conocimiento · contexto finito', 'y cero acceso al mundo'],
      ['mitigación: RAG con citas + herramientas', 'y validar lo crítico'],
    ],
    analogy: (cfg) => `un estudiante brillante en un examen oral: cuando no sabe
      la respuesta, improvisa una que suena impecable, sin cambiar el tono de
      voz.`,
    advanced: (cfg) => `
      <p>El modelo no tiene un registro interno de «esto lo sé» vs «esto lo
      invento»: solo probabilidades. Los modelos grandes alucinan menos (comprimen
      mejor los hechos frecuentes), pero ninguno llega a cero.</p>
      <p>Mitigaciones reales: <strong>RAG con citas</strong> (que el dato venga del
      contexto, no de los pesos), <strong>herramientas</strong> (buscar, calcular,
      ejecutar) y validación posterior. Este DistilGPT-2 de ${cfg.params} alucina
      muchísimo — ideal para ver el fenómeno a simple vista.</p>`,
    world: (cfg) => `
      <p>Un LLM sin fuentes es un generador de texto plausible — no una base de
      datos.</p>`,
    tryIt: (cfg) => `Pulsa los prompts de la derecha: el modelo completa hechos
      futuros o discutibles con total confianza.`,
  },
};

const CLOSING = `Esto es todo lo que hace ChatGPT: este mismo ciclo, miles de veces,
con un modelo mil veces más grande.`;

export function getClosing() {
  return CLOSING;
}

/**
 * Inyecta el contenido en cada contenedor [data-prose] del documento.
 */
export function renderProse(cfg) {
  document.querySelectorAll('[data-prose]').forEach(el => {
    const key = el.dataset.prose;
    const s = PROSE[key];
    if (!s) return;
    el.innerHTML = `
      <p class="kicker">${s.step}</p>
      <h2>${s.title}</h2>
      ${s.term ? `<div class="definicion"><span class="definicion__term">${s.term}</span> — ${s.def(cfg)}</div>` : ''}
      ${s.points ? `<div class="puntos">${s.points(cfg).map(([k, d]) => `
        <div class="punto">
          <span class="punto__k">${k}</span>
          ${d ? `<span class="punto__d">${d}</span>` : ''}
        </div>`).join('')}</div>` : ''}
      ${s.analogy ? `<p class="analogia"><strong>Piénsalo así:</strong> ${s.analogy(cfg)}</p>` : ''}
      <details class="deeper">
        <summary>Profundizar</summary>
        ${s.advanced(cfg)}
      </details>
      ${s.world ? `<div class="mundo-real"><span class="mundo-real__label">En el mundo real</span>${s.world(cfg)}</div>` : ''}
      <p class="tryit">${s.tryIt(cfg)}</p>`;
  });
}
