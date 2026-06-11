# Guion de exposición — «Anatomía de una predicción»

**Presenta:** Fernando Chullo Mamani · **Duración estimada:** 35–45 min
**Setup:** abrir `…/llm-pipeline-visualizer/?presentar` · avanzar con **espacio / →** (o clicker: PageDown) · retroceder con **←** · salir del modo con **P**

**Arco de la charla:** definición → 6 conceptos con ejemplos en vivo → el pipeline (las piezas, juntas) → cómo decide y genera → límites.

Cada parada tiene: **[Pantalla]** lo que el público ve · **[Decir]** la narración · **[Demo]** qué tocar en vivo.

---

## 0 · Portada

**[Pantalla]** Título, autor, campo de texto con «The capital of Spain is».

**[Decir]** Cuando ChatGPT escribe, no hace magia: convierte un texto en números, los procesa y elige la siguiente palabra. Hoy vamos a abrir esa caja negra — y no con diapositivas: en esta página hay un modelo de lenguaje real, DistilGPT-2, ejecutándose dentro del navegador. Cada número que van a ver es real, calculado en vivo. Primero definiremos qué es un LLM, después conoceremos sus piezas una por una —token, embedding, atención— con ejemplos que podemos tocar, y al final juntaremos todo para verlo decidir una palabra.

**[Demo]** Escribir (o dejar) el prompt y pulsar «Comenzar el viaje». El modelo (~165 MB) baja en segundo plano; los conceptos no lo necesitan.

---

## 1 · ¿Qué es un LLM?

**[Pantalla]** Definición · P(siguiente token | contexto) en grande · escala de tamaños.

**[Decir]** La definición que lo cambia todo: un LLM hace UNA sola cosa — dada una secuencia de texto, asigna una probabilidad a cada posible token siguiente. Esa fórmula de la derecha es todo: no hay módulo de gramática, ni base de datos de hechos, ni reglas escritas a mano. Es el autocompletar del teléfono llevado al extremo: tan bueno prediciendo lo que sigue que, encadenando predicciones, escribe, traduce y programa. ¿Y «large»? Es literal — miren la escala: este modelo tiene 82 millones de parámetros; GPT-3, 175 mil millones — dos mil veces más. Y al escalar no solo mejora: *emergen* capacidades — seguir instrucciones, razonar por pasos. ChatGPT, Claude y Copilot son exactamente esta función, en bucle.

**[Demo]** Hover sobre la escala («GPT-3: ~2.134× el modelo de esta página»).

---

## 2 · Concepto 1: Tokens

**[Pantalla]** La frase coloreada por piezas + sus IDs + mini-tokenizer en vivo.

**[Decir]** Primera pieza de terminología: el token. El modelo no lee palabras — lee *tokens*, fragmentos con un ID numérico en un vocabulario de 50.257. Piezas de Lego del lenguaje: con esas 50 mil piezas se arma cualquier texto, incluso palabras que no existen. Las comunes son una sola pieza; las raras se parten — «tokenization» es «token» + «ization». Y el detalle que confunde a todos: el espacio inicial es parte del token — « the» y «the» son piezas distintas. Consecuencias prácticas: las APIs cobran por token, el contexto se mide en tokens, y por eso un modelo a veces se corta a media palabra.

**[Demo]** ⭐ El juego de crear tokens: pedir al público una palabra inventada (o usar «tokenizometro»), escribirla en el campo y ver al tokenizer real partirla en vivo. Probar 2-3 propuestas — palabras raras dan más piezas.

---

## 3 · Concepto 2: Embeddings

**[Pantalla]** Vector real de 768 dims (barras) + matriz de similitud coseno.

**[Decir]** Un ID como 3139 no dice nada del *significado*. Por eso cada token se convierte en una lista de 768 números: su embedding. Piensen en un mapa gigante de palabras: «pizza» y «pasta» caen cerca, «pizza» y «JavaScript» lejos — el significado ES la posición en el mapa. Esto no es un dibujo: son los valores reales del modelo, descargados de HuggingFace fila por fila. La cercanía se mide con similitud coseno — la matriz compara los tokens de nuestra frase. Y esto les suena: búsqueda semántica, recomendadores y RAG son exactamente esto.

**[Demo]** Hover en las barras (valor por dimensión) y en la matriz (coseno por par). Si hay tiempo: reiniciar con «The king and the queen» y ver qué par gana.

---

## 4 · Concepto 3: Posición

**[Pantalla]** wte[token] + wpe[posición] = suma, con selector de posición.

**[Decir]** Un problema: la atención —que veremos ahora— mira todos los tokens *a la vez*. Es su superpoder y su defecto: por sí sola no sabe cuál va primero. «Dog bites man» y «man bites dog» usan las mismas piezas — solo el orden separa la noticia del accidente. La solución es brutalmente simple: a cada embedding se le SUMA un segundo vector que solo depende de su posición. Miren: el vector del token no cambia, el de posición sí — y la suma, que es lo que entra al modelo, cambia con él. Bonus: la matriz de posiciones de GPT-2 tiene 1024 filas — por eso su contexto máximo es 1024 tokens. La «ventana de contexto» de los modelos es, en el fondo, esto.

**[Demo]** Cambiar la posición con los chips (incluidos pos 100 y 1000). Señalar el badge «pesos reales».

---

## 5 · Concepto 4: Atención

**[Pantalla]** Tokens sombreados por peso de atención real (capa 0), selector de cabezas.

**[Decir]** El concepto estrella, donde ocurre la «comprensión». En «the bank by the river», ¿bank es banco u orilla? Solo el contexto lo decide. La atención es el mecanismo: cada token mira a los anteriores, calcula cuánto le importa cada uno, y absorbe lo que necesita. La analogía: una biblioteca — la Query es la búsqueda, cada Key la etiqueta de un libro, cada Value su contenido; el resultado es una mezcla ponderada por cuánto encaja cada etiqueta. Estos pesos son REALES, de la primera capa de este modelo — y esto se repite en 6 capas con 12 «cabezas» paralelas que aprenden relaciones distintas. Dato de ingeniería: el costo crece con el cuadrado de la longitud — por eso el contexto largo es caro.

**[Demo]** Clic en distintos tokens como «punto de vista». Cambiar de cabeza (Prom, 1…12).

---

## 6 · Concepto 5: El bloque transformer

**[Pantalla]** Diagrama del bloque (LayerNorm → atención → residual → FFN → residual) + 6 capas apiladas.

**[Decir]** La atención es la mitad famosa del bloque; la otra mitad es una red feed-forward que procesa cada posición por separado, expandiendo el vector de 768 a 3072 y de vuelta. La división del trabajo: la atención *mueve información entre tokens*; la FFN *piensa en cada token*. Y este bloque ES el modelo: se apila 6 veces aquí, ~100 en los grandes. Capas tempranas: sintaxis; profundas: semántica. Cuando alguien dice «70B de parámetros, 120 capas», está diciendo solo esto: cuántas veces se apila y cuán ancho es.

**[Demo]** Hover por cada pieza del diagrama y por las 6 capas de abajo.

---

## 7 · Concepto 6: Entrenamiento

**[Pantalla]** Tres tarjetas: Pre-entrenamiento → Fine-tuning/RLHF → Inferencia («estamos aquí»).

**[Decir]** Última pieza de teoría: ¿de dónde salieron esos 82 millones de pesos? Del mismo juego: predecir el siguiente token. En entrenamiento se juega billones de veces — el modelo predice, se mide el error, y backpropagation ajusta cada peso un poquito. Meses, miles de GPUs, millones de dólares. Es comprimir internet en una función. El resultado es un modelo *base* — como este: solo continúa texto. El fine-tuning con RLHF lo convierte en asistente — así GPT-3 se volvió ChatGPT. Y lo que hacemos hoy es la tercera fase: inferencia, solo lectura. Clave: el modelo NO aprende al usarlo — cada conversación parte de los mismos pesos congelados. De ahí la fecha de corte, y las dos vías para conocimiento propio: RAG o fine-tuning.

**[Demo]** Hover por las tres tarjetas comparando datos/tiempo/costo.

---

## 8 · El pipeline: las piezas, juntas

**[Pantalla]** Mapa clicable de la cadena completa (8 etapas).

**[Decir]** Ya tenemos toda la terminología. Ahora juntémosla: esto es el pipeline de inferencia — la cadena que el modelo ejecuta en CADA predicción. Texto → tokens → vectores más posición → 6 bloques transformer → un puntaje por token → elegir uno → repetir. Como una línea de producción: entra texto crudo, cada estación lo transforma, y sale una única palabra, lista para volver a entrar. Las primeras cinco estaciones ya las conocemos. Nos faltan las tres últimas — donde el modelo pasa de representar a *decidir*. Vamos.

**[Demo]** Recorrer el mapa con el cursor; mencionar que es clicable para repasar.

---

## 9 · Pipeline: Logits

**[Pantalla]** Top-15 candidatos: logit crudo · barra · probabilidad real. Slider de temperatura.

**[Decir]** Después de los 6 bloques, el modelo emite un puntaje —un logit— para CADA token del vocabulario: 50.257 candidatos por predicción. Como las notas de un jurado antes de normalizar: importan las diferencias, no los valores. El softmax los convierte en probabilidades — y ojo: estos porcentajes son reales sobre todo el vocabulario; con temperatura 1, ni el favorito suele pasar del 50%. La temperatura divide los logits antes del softmax: baja, las diferencias se amplifican; alta, se aplanan. El `temperature` de las APIs es literalmente este divisor; los `logprobs`, el log de estas barras.

**[Demo]** El momento slider: T a 0.1 → una barra se come todo. T a 2.0 → plano. Volver a 1.0.

---

## 10 · Pipeline: Muestreo

**[Pantalla]** Ruleta de candidatos + eliminados tachados + sliders top-k/top-p + greedy.

**[Decir]** Hay probabilidades — falta elegir. Y la sorpresa: el modelo NO siempre elige la más probable. Tira una ruleta donde cada candidato ocupa un arco proporcional a su probabilidad — por eso la misma pregunta da respuestas distintas. Antes de girar se filtra: top-k deja los k mejores; top-p recorta al «núcleo» que acumula p de probabilidad. Abajo, los eliminados y por qué. Valores típicos: chat T≈0.7 y top-p 0.9; código T≈0.2; creatividad T≈1.2; greedy para reproducibilidad.

**[Demo]** «Muestrear» 4-5 veces: misma distribución, distinto ganador — ESO es el muestreo. Greedy: la ★ se congela y el botón se apaga. Mover top-p y ver candidatos caer.

---

## 11 · Pipeline: El bucle

**[Pantalla]** El texto + token nuevo en acento · historial · botón «Añadir y repetir».

**[Decir]** Y el cierre del pipeline: el token elegido se añade al final del texto… y TODO vuelve a ejecutarse. Tokens, embeddings, atención, bloques, logits, muestreo — una palabra por ciclo. Generación autorregresiva: una única regla, «lee todo lo anterior y añade la palabra siguiente», repetida cientos de veces. El modelo nunca planea la frase — la coherencia *emerge*. El «streaming» de ChatGPT, las palabras llegando una a una, no es un efecto visual: es este bucle en vivo. Y por eso el output se cobra por token: cada palabra es una pasada completa.

**[Demo]** ⭐ Pulsar «Añadir y repetir» 2-3 veces. La página vuelve sola a Tokens, la barra superior crece, todo se recalcula.

---

## 12 · Límites

**[Pantalla]** Demo de alucinación: prompts con trampa + top-5 con confianza.

**[Decir]** Cierro con lo más importante para usarlos bien: lo que un LLM no es. Nada del pipeline verifica hechos. «París» después de «the capital of France is» y un dato inventado salen del MISMO mecanismo. Miren: «Barack Obama was the president of the» → «United», 72% — hecho frecuente, alta confianza. «Steve Jobs was the founder of» → Apple… y Microsoft, casi con la misma confianza, lado a lado. «The 2030 World Cup was won by» → Germany, Brazil — un mundial que no existe, afirmado con total naturalidad. Eso es una alucinación: misma mecánica, misma confianza, cero verdad. Un estudiante brillante en examen oral: si no sabe, improvisa sin cambiar el tono. Regla de oro: un LLM sin fuentes es un generador de texto plausible, no una base de datos. En producción: hechos en el contexto (RAG), citas, y validar lo crítico.

**[Demo]** Clic en los 3 prompts en orden (Obama → Jobs → 2030). Si el público propone uno, usar el campo libre.

**[Cierre]** Esto es todo lo que hace ChatGPT: este mismo ciclo, miles de veces, con un modelo mil veces más grande. Conocer la máquina —y sus límites— es lo que nos deja usarla bien. Gracias.

---

## Si algo falla

- **El modelo no carga** (red): los conceptos 1–5 funcionan igual (tokenizer + pesos por HTTP Range). Saltar logits/muestreo y mostrar el resto.
- **Perdiste el hilo de pasos**: **P** sale del modo presentación y muestra todo; navegar con el rail.
- **Pregunta fuera de orden**: clic en el punto del rail o en el mapa de la sección «El pipeline».
- **Cachear antes**: abrir la página en la red de la sala una vez ese día — el modelo queda en caché del navegador.
