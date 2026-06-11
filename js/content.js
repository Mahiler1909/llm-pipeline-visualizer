/**
 * Prosa educativa de cada sección (español).
 * Funciones template que reciben el config del modelo para que
 * dimensiones, capas y vocabulario sean siempre los reales.
 *
 * Estructura por sección:
 *   step      kicker («Paso N · Tema»)
 *   title     titular
 *   term/def  tarjeta de definición citable (1-2 líneas)
 *   analogy   una línea «Piénsalo así:»
 *   basic     2 párrafos de intuición
 *   advanced  bloque «Profundizar» (teoría densa)
 *   world     bloque «En el mundo real» (conexión con APIs/productos)
 *   tryIt     experimento ejecutable en ESA sección
 */

export const PROSE = {
  llm: {
    step: 'Antes de empezar · ¿Qué es un LLM?',
    title: 'Una sola tarea: predecir lo que sigue',
    term: 'LLM (Large Language Model)',
    def: (cfg) => `red neuronal con millones —o billones— de parámetros entrenada
      para una única tarea: dada una secuencia de texto, asignar una probabilidad
      a cada posible token siguiente.`,
    analogy: (cfg) => `el autocompletar de tu teléfono llevado al extremo. Es tan
      bueno prediciendo lo que sigue que, encadenando predicciones, escribe,
      traduce, programa y aparenta razonar.`,
    basic: (cfg) => `
      <p>Todo lo que hace un LLM se reduce a una función:
      <span class="mono">P(siguiente token | contexto)</span>. No hay un módulo de
      gramática, ni una base de datos de hechos, ni reglas escritas a mano —
      solo esa función, aprendida leyendo billones de palabras.</p>
      <p>La palabra «large» es literal: el DistilGPT-2 que corre en esta página
      tiene ${cfg.params} de parámetros; GPT-3 tiene 175.000 millones (~2.000×
      más). Y al escalar no solo mejora: <strong>emergen</strong> capacidades
      nuevas — seguir instrucciones, razonar por pasos, aprender de los ejemplos
      del propio prompt.</p>`,
    advanced: (cfg) => `
      <div class="formula">P(texto) = P(t₁) · P(t₂|t₁) · P(t₃|t₁t₂) · …</div>
      <p>Un modelo de lenguaje define la probabilidad de un texto completo como el
      producto de las probabilidades de cada token dado lo anterior. Generar es
      recorrer ese producto hacia adelante, un token a la vez.</p>
      <p>Las <em>scaling laws</em> muestran que el error baja de forma predecible
      al aumentar parámetros, datos y cómputo a la vez — la apuesta que produjo
      GPT-3, GPT-4 y todo lo que siguió.</p>`,
    world: (cfg) => `
      <p>Cada producto de «IA generativa» que usa tu equipo —ChatGPT, Claude,
      Copilot— es esta misma función en bucle. La diferencia está en la escala y
      en el entrenamiento posterior, no en una arquitectura secreta.</p>`,
    tryIt: (cfg) => `El diagrama de la derecha es el mapa del viaje: haz clic en
      cualquier etapa para saltar a ella. Abajo, compara el tamaño de este modelo
      con los famosos.`,
  },

  tokens: {
    step: 'Paso 1 · Tokens',
    title: 'El texto se corta en piezas',
    term: 'token',
    def: (cfg) => `fragmento de texto —una palabra, un trozo de palabra o un
      signo— con un ID numérico propio en el vocabulario del modelo; la unidad
      mínima que el modelo lee y escribe.`,
    analogy: (cfg) => `piezas de Lego del lenguaje: con
      ${cfg.vocab_size.toLocaleString('es')} piezas distintas se arma cualquier
      texto posible, incluso palabras que no existen.`,
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
    world: (cfg) => `
      <p>Por esto las APIs cobran por token (no por palabra), la «ventana de
      contexto» se mide en tokens, y un modelo puede cortarse a media palabra: el
      token es su única unidad. Regla rápida en inglés: 1 token ≈ 4 caracteres
      ≈ ¾ de palabra.</p>`,
    tryIt: (cfg) => `Escribe una palabra inventada como «tokenizometro» en el campo
      de abajo y mira en cuántos pedazos la parte el tokenizer real.`,
  },

  embeddings: {
    step: 'Paso 2 · Embeddings',
    title: 'Cada token se vuelve un vector con significado',
    term: 'embedding',
    def: (cfg) => `vector denso (aquí, ${cfg.hidden_dim} números) que representa
      el significado de un token: las coordenadas de un punto en un espacio donde
      estar cerca es significar parecido.`,
    analogy: (cfg) => `un mapa gigante de palabras. «pizza» y «pasta» caen cerca;
      «pizza» y «JavaScript», lejos. El significado de una palabra es su posición
      en el mapa.`,
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
      puntajes (<em>weight tying</em>). Lo que aún falta aquí es el <em>orden</em>
      de los tokens — eso lo añade el paso siguiente.</p>`,
    world: (cfg) => `
      <p>La búsqueda semántica, los recomendadores y RAG funcionan exactamente
      así: convierten tu consulta y tus documentos en embeddings y devuelven los
      más cercanos por coseno. Regla práctica: &gt;0.7 muy relacionado, ~0.5 algo,
      &lt;0.3 ruido.</p>`,
    tryIt: (cfg) => `Vuelve arriba y comienza con «The king and the queen» — en la
      matriz de similitud, ¿qué par de tokens sale más alto?`,
  },

  posicion: {
    step: 'Paso 3 · Posición',
    title: 'El orden importa',
    term: 'embedding de posición',
    def: (cfg) => `vector aprendido que codifica «soy el token n.º k de la
      secuencia»; se suma al embedding del token antes de entrar al transformer.`,
    analogy: (cfg) => `«dog bites man» y «man bites dog» usan exactamente las
      mismas piezas — solo el orden separa la noticia del accidente.`,
    basic: (cfg) => `
      <p>La atención (el paso siguiente) mira todos los tokens <em>a la vez</em>,
      en paralelo. Ese es su superpoder y su defecto: por sí sola no sabe cuál va
      primero. Sin ayuda, «dog bites man» y «man bites dog» serían la misma
      entrada.</p>
      <p>La solución es de una simpleza brutal: a cada embedding se le
      <strong>suma</strong> un segundo vector que depende solo de su posición. El
      mismo token en la posición 1 y en la 5 entra al modelo como dos vectores
      distintos. A la derecha: filas <strong>reales</strong> de la matriz de
      posición (wpe) de este modelo.</p>`,
    advanced: (cfg) => `
      <div class="formula">entrada = wte[token] + wpe[posición]</div>
      <p>GPT-2 usa posiciones <strong>aprendidas</strong>: una matriz wpe de
      1024 × ${cfg.hidden_dim} entrenada junto al resto del modelo. De ahí sale su
      límite de contexto: la fila 1025 simplemente no existe.</p>
      <p>El transformer original usaba ondas sinusoidales (sin entrenar); los
      modelos actuales usan <strong>RoPE</strong> — rotaciones aplicadas dentro de
      la atención que escalan mucho mejor a contextos largos.</p>`,
    world: (cfg) => `
      <p>La «ventana de contexto» que comparas entre modelos (8k, 128k, 1M tokens)
      es en el fondo esto: hasta qué posición sabe codificar el modelo. RoPE y sus
      variantes son los trucos que la estiran.</p>`,
    tryIt: (cfg) => `Cambia la posición en el widget: el vector del token no se
      mueve, el de posición sí — y la suma, que es lo que entra al modelo, cambia
      con él.`,
  },

  atencion: {
    step: 'Paso 4 · Atención',
    title: 'Cada token mira a los anteriores',
    term: 'atención (self-attention)',
    def: (cfg) => `mecanismo por el que cada token calcula cuánto le importa cada
      token anterior y mezcla la información de todos en proporción a ese peso; es
      donde el contexto entra al significado.`,
    analogy: (cfg) => `una biblioteca. Tu <strong>Q</strong>uery es lo que buscas,
      cada <strong>K</strong>ey es la etiqueta de un libro y cada
      <strong>V</strong>alue su contenido: te llevas una mezcla de libros ponderada
      por cuánto encaja cada etiqueta con tu búsqueda.`,
    basic: (cfg) => `
      <p>En «the bank by the river», ¿"bank" es banco u orilla? El embedding solo
      da el significado <em>aislado</em>; la <strong>atención</strong> añade el
      contexto: cada token reparte su atención entre los anteriores y absorbe de
      ellos lo que necesita para desambiguarse.</p>
      <p>En "capital of <em>Spain</em>", el token final necesita mirar a "Spain"
      para saber que la respuesta es Madrid. Lo que ves a la derecha son los pesos
      de atención <strong>reales</strong> de la primera capa del modelo — y esto se
      repite en ${cfg.layers} capas, captando relaciones cada vez más abstractas.</p>`,
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
      <p>El costo de la atención crece con el <em>cuadrado</em> de la longitud:
      cada token nuevo se compara con todos los anteriores. Buena parte de la
      ingeniería de LLMs (KV-cache, FlashAttention, ventanas deslizantes) existe
      para pagar esa cuenta — por eso el contexto largo es caro.</p>`,
    tryIt: (cfg) => `Cambia de cabeza con los botones: cada una mira a tokens
      distintos. Y haz clic en otro token para usarlo como punto de vista.`,
  },

  bloque: {
    step: 'Paso 5 · El bloque transformer',
    title: 'La pieza que se repite',
    term: 'bloque transformer',
    def: (cfg) => `la unidad de cómputo del modelo — atención multi-cabeza + red
      feed-forward, envueltas en conexiones residuales y normalización. Un LLM es
      este bloque apilado N veces.`,
    analogy: (cfg) => `una línea de ensamblaje con ${cfg.layers} estaciones de
      estructura idéntica pero herramientas distintas: cada capa refina la
      representación que le dejó la anterior.`,
    basic: (cfg) => `
      <p>Ya viste la atención; el bloque la completa con una red
      <strong>feed-forward</strong> (${cfg.hidden_dim}→${cfg.ffn_dim}→${cfg.hidden_dim})
      que procesa cada posición por separado. La división del trabajo: la atención
      <em>mueve información entre tokens</em>; la FFN <em>procesa cada token</em>.
      Las conexiones residuales y LayerNorm mantienen todo estable.</p>
      <p>Este bloque se repite <strong>${cfg.layers} veces</strong> en este modelo —
      y hasta ~100 en los grandes. Las capas tempranas captan sintaxis; las
      profundas, semántica y relaciones abstractas. Cuando pulsaste «Comenzar»,
      esto acaba de ocurrir ${cfg.layers} veces en tu navegador.</p>`,
    advanced: (cfg) => `
      <p>Cerca de dos tercios de los parámetros del modelo viven en las FFN — la
      atención es la parte famosa, no la más pesada.</p>
      <p>La misma pieza, tres arquitecturas: <strong>GPT</strong> (decoder-only,
      atención causal → genera), <strong>BERT</strong> (encoder-only, bidireccional
      → entiende y clasifica), <strong>T5</strong> (encoder-decoder → traduce,
      resume). Todos los asistentes de chat actuales son decoder-only.</p>`,
    world: (cfg) => `
      <p>Cuando leas «70B de parámetros» o «120 capas», es esto: cuántas veces se
      apila el bloque (profundidad) y cuán anchas son sus matrices. El «tamaño» de
      un modelo no tiene más misterio.</p>`,
    tryIt: (cfg) => `Pasa el cursor por cada pieza del diagrama para ver qué hace.`,
  },

  logits: {
    step: 'Paso 6 · Logits',
    title: 'Un puntaje para cada palabra posible',
    term: 'logit',
    def: (cfg) => `puntaje crudo —un número sin escala— que el modelo asigna a
      cada uno de los ${cfg.vocab_size.toLocaleString('es')} tokens como candidato
      a continuar el texto; el softmax los convierte en probabilidades.`,
    analogy: (cfg) => `las puntuaciones de un jurado antes de normalizarlas:
      importan las <em>diferencias</em> entre candidatos, no los valores
      absolutos.`,
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
    world: (cfg) => `
      <p>Si pides <span class="mono">logprobs</span> a una API, estás viendo
      exactamente esto: el log de estas probabilidades. Y el parámetro
      <span class="mono">temperature</span> de OpenAI/Anthropic es literalmente el
      divisor que estás a punto de mover.</p>`,
    tryIt: (cfg) => `Baja la temperatura a 0.1: una sola barra domina. Súbela a 2.0
      y compara cómo se aplana la distribución.`,
  },

  muestreo: {
    step: 'Paso 7 · Muestreo',
    title: 'Elegir la siguiente palabra: azar controlado',
    term: 'muestreo (sampling)',
    def: (cfg) => `elegir el siguiente token al azar, ponderado por su
      probabilidad, en vez de tomar siempre el más probable; top-k y top-p
      recortan la lista de candidatos antes de tirar el dado.`,
    analogy: (cfg) => `una ruleta donde cada candidato ocupa un arco proporcional
      a su probabilidad. Top-k y top-p deciden quién entra a la ruleta; la
      temperatura, qué tan parejos son los arcos.`,
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
    world: (cfg) => `
      <p>Las configuraciones típicas que verás en producción:</p>
      <table class="mini-table">
        <tr><td>chat (ChatGPT)</td><td class="mono">T≈0.7 · top-p 0.9</td></tr>
        <tr><td>código</td><td class="mono">T≈0.2</td></tr>
        <tr><td>escritura creativa</td><td class="mono">T≈1.2</td></tr>
        <tr><td>tests reproducibles</td><td class="mono">greedy (T=0)</td></tr>
      </table>`,
    tryIt: (cfg) => `Pulsa «Muestrear» varias veces: misma distribución, distinto
      ganador. Activa «Greedy» y la ★ se queda quieta en el más probable.`,
  },

  bucle: {
    step: 'Paso 8 · El bucle',
    title: 'La salida se vuelve entrada',
    term: 'generación autorregresiva',
    def: (cfg) => `el bucle donde el token elegido se añade al contexto y el
      modelo entero vuelve a ejecutarse para elegir el siguiente; el texto se
      escribe de a un token.`,
    analogy: (cfg) => `escribir con una única regla — «lee todo lo que llevas y
      añade la palabra siguiente» — repetida cientos de veces por respuesta.`,
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
    world: (cfg) => `
      <p>El «streaming» de ChatGPT —palabras llegando una a una— no es un efecto
      visual: estás viendo este bucle en vivo. Y el output se cobra por token
      porque cada token cuesta una pasada completa del modelo.</p>`,
    tryIt: (cfg) => `Repite el ciclo 3 veces con greedy activado, resetea y repítelo
      sin greedy: la primera vez el texto será idéntico; la segunda, distinto.`,
  },

  entrenamiento: {
    step: 'Detrás de escena · Entrenamiento',
    title: '¿De dónde salen los pesos?',
    term: 'pre-entrenamiento',
    def: (cfg) => `ajustar los millones (o billones) de parámetros del modelo para
      que prediga bien el siguiente token sobre un corpus gigante; todo su
      «conocimiento» es el residuo de ese ajuste.`,
    analogy: (cfg) => `comprimir internet en una función. Para predecir bien lo que
      sigue en cualquier texto hay que haber capturado sus patrones — gramática,
      hechos, estilos. Comprimir es, en cierto modo, entender.`,
    basic: (cfg) => `
      <p>El juego es el mismo que viste en esta página: predecir el siguiente
      token. En entrenamiento se juega billones de veces: el modelo predice, se
      mide el error contra el texto real, y <em>backpropagation</em> ajusta cada
      peso un poquito. Meses de eso, con miles de GPUs y millones de dólares.</p>
      <p>El resultado es un modelo <strong>base</strong> — como este DistilGPT-2:
      solo continúa texto. El <strong>fine-tuning</strong> (con RLHF) lo convierte
      en asistente: ejemplos de conversación más humanos puntuando respuestas.
      Así GPT-3 se volvió ChatGPT. Lo que hiciste hoy es la tercera fase,
      <strong>inferencia</strong>: solo lectura.</p>`,
    advanced: (cfg) => `
      <p>Por eso DistilGPT-2 no «responde» tus preguntas: las continúa como
      continuaría cualquier texto. Un modelo <em>instruct</em> fue afinado además
      para tratar el texto como una conversación con turnos.</p>
      <p>Clave para no confundirse: el modelo <strong>no aprende al usarlo</strong>.
      Cada conversación parte de los mismos pesos congelados; la «memoria» entre
      turnos es solo el contexto, que se le reenvía completo cada vez.</p>`,
    world: (cfg) => `
      <p>De aquí salen tres realidades de producción: la <strong>fecha de corte</strong>
      de conocimiento (el corpus termina un día concreto), que tu chat de ayer no
      le enseñó nada, y que para darle conocimiento propio hay dos vías —
      <strong>RAG</strong> (ponérselo en el contexto) o <strong>fine-tuning</strong>
      (ajustar los pesos).</p>`,
    tryIt: (cfg) => `Recorre las tres etapas del diagrama y compara las escalas de
      datos, tiempo y costo.`,
  },

  limites: {
    step: 'Para terminar · Límites',
    title: 'Lo que un LLM no es',
    term: 'alucinación',
    def: (cfg) => `salida generada con la misma mecánica —y la misma confianza—
      que un hecho correcto, pero falsa. El modelo optimiza plausibilidad, no
      verdad.`,
    analogy: (cfg) => `un estudiante brillante en un examen oral: cuando no sabe la
      respuesta, improvisa una que suena impecable, sin cambiar el tono de voz.`,
    basic: (cfg) => `
      <p>Nada en el pipeline que acabas de recorrer verifica hechos. «París»
      después de "the capital of France is" y un dato inventado salen del
      <em>mismo mecanismo exacto</em>: la distribución de probabilidad del paso 6.
      Si el patrón es plausible, el token sale — sea verdad o no.</p>
      <p>De ahí los límites prácticos: <strong>corte de conocimiento</strong> (no
      sabe nada posterior a su corpus), <strong>ventana de contexto finita</strong>,
      y <strong>cero acceso al mundo</strong> salvo lo que le pongas en el contexto.
      A la derecha puedes provocar el fenómeno en vivo.</p>`,
    advanced: (cfg) => `
      <p>El modelo no tiene un registro interno de «esto lo sé» vs «esto lo
      invento»: solo probabilidades. Los modelos grandes alucinan menos (comprimen
      mejor los hechos frecuentes), pero ninguno llega a cero.</p>
      <p>Mitigaciones reales: <strong>RAG con citas</strong> (que el dato venga del
      contexto, no de los pesos), <strong>herramientas</strong> (buscar, calcular,
      ejecutar) y validación posterior. Este DistilGPT-2 de ${cfg.params} alucina
      muchísimo — ideal para ver el fenómeno a simple vista.</p>`,
    world: (cfg) => `
      <p>Regla de oro para tu equipo: un LLM sin fuentes es un generador de texto
      plausible, no una base de datos. En producción: dale los hechos en el
      contexto (RAG), pídele citas y valida lo crítico. La misma máquina que
      acabas de recorrer, usada con sus límites a la vista, es una herramienta
      extraordinaria.</p>`,
    tryIt: (cfg) => `Pulsa los prompts de la derecha: el modelo completa hechos
      futuros o discutibles con total confianza. La confianza no distingue verdad
      de invención.`,
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
      ${s.term ? `<div class="definicion"><span class="definicion__term">${s.term}</span> — ${s.def(cfg)}</div>` : ''}
      ${s.analogy ? `<p class="analogia"><strong>Piénsalo así:</strong> ${s.analogy(cfg)}</p>` : ''}
      ${s.basic(cfg)}
      <details class="deeper">
        <summary>Profundizar</summary>
        ${s.advanced(cfg)}
      </details>
      ${s.world ? `<div class="mundo-real"><span class="mundo-real__label">En el mundo real</span>${s.world(cfg)}</div>` : ''}
      <p class="tryit">${s.tryIt(cfg)}</p>`;
  });
}
