# Guion de exposición — «Anatomía de una predicción»

**Presenta:** Fernando Chullo Mamani · **Duración estimada:** 35–45 min
**Setup:** abrir `…/llm-pipeline-visualizer/?presentar` · avanzar con **espacio / →** (o clicker: PageDown) · retroceder con **←** · salir del modo con **P**

Cada parada tiene: **[Pantalla]** lo que el público ve · **[Decir]** la narración · **[Demo]** qué tocar en vivo.

---

## 0 · Portada

**[Pantalla]** Título, autor, campo de texto con «The capital of Spain is».

**[Decir]** Cuando ChatGPT escribe, no hace magia: convierte el texto en números, los procesa y elige la siguiente palabra. Hoy vamos a abrir esa caja negra — y no con diapositivas: en esta página hay un modelo de lenguaje real, DistilGPT-2, ejecutándose dentro del navegador. Cada número que van a ver es real, calculado en vivo. Vamos a darle una frase sin terminar y a seguirla por dentro del modelo hasta ver cómo elige la palabra que sigue.

**[Demo]** Escribir (o dejar) el prompt y pulsar «Comenzar el viaje». Mientras tanto el modelo (~165 MB) baja en segundo plano — las primeras secciones no lo necesitan.

---

## 1 · ¿Qué es un LLM?

**[Pantalla]** Definición de LLM · mapa del recorrido · escala de tamaños.

**[Decir]** Primero, la definición que lo cambia todo: un LLM hace UNA sola cosa — dada una secuencia de texto, asigna una probabilidad a cada posible token siguiente. No tiene módulo de gramática, ni base de datos de hechos, ni reglas escritas a mano. Es el autocompletar del teléfono llevado al extremo: tan bueno prediciendo lo que sigue que, encadenando predicciones, escribe, traduce y programa. La palabra «large» es literal — miren la escala: el modelo de esta página tiene 82 millones de parámetros; GPT-3, 175 mil millones — dos mil veces más. Y lo sorprendente: al escalar no solo mejora, *emergen* capacidades nuevas — seguir instrucciones, razonar por pasos. ChatGPT, Claude y Copilot son exactamente esta función, en bucle.

**[Demo]** Hover sobre la escala de tamaños («GPT-3: ~2.134× el modelo de esta página»). Mencionar que el mapa es clicable.

---

## 2 · Tokens

**[Pantalla]** La frase coloreada por piezas + sus IDs + mini-tokenizer en vivo.

**[Decir]** Lo primero que le pasa a la frase: deja de ser texto. El modelo no lee palabras — lee *tokens*, fragmentos con un ID numérico en un vocabulario de 50.257. Son como piezas de Lego del lenguaje: con esas 50 mil piezas se arma cualquier texto, incluso palabras que no existen. Las palabras comunes son una sola pieza; las raras se parten — «tokenization» es «token» + «ization». Y un detalle que confunde a todos: el espacio inicial es parte del token — « the» y «the» son piezas distintas. De aquí salen cosas muy prácticas: las APIs cobran por token, la ventana de contexto se mide en tokens, y por eso un modelo a veces se corta a media palabra.

**[Demo]** Hover sobre los tokens para ver IDs. Escribir «tokenizometro» en el campo de abajo: el tokenizer real lo parte en vivo.

---

## 3 · Embeddings

**[Pantalla]** Vector real de 768 dims (barras) + matriz de similitud coseno.

**[Decir]** Un ID como 3139 no dice nada del *significado*. Por eso cada token se convierte en una lista de 768 números: su embedding. Piensen en un mapa gigante de palabras: «pizza» y «pasta» caen cerca, «pizza» y «JavaScript» lejos — el significado ES la posición en el mapa. Lo que ven en pantalla no es un dibujo: son los valores reales del modelo, descargados de HuggingFace fila por fila. La cercanía se mide con similitud coseno — la matriz de abajo compara los tokens de nuestra frase. Y esto les suena: búsqueda semántica, recomendadores y RAG son exactamente esto — convertir consulta y documentos en embeddings y devolver los más cercanos.

**[Demo]** Hover en las barras (valor por dimensión) y en la matriz (coseno por par). Si hay tiempo: reiniciar con «The king and the queen» y ver qué par sale más alto.

---

## 4 · Posición

**[Pantalla]** wte[token] + wpe[posición] = suma, con selector de posición.

**[Decir]** Hay un problema: la atención —el siguiente paso— mira todos los tokens *a la vez*, en paralelo. Es su superpoder y su defecto: por sí sola no sabe cuál va primero. «Dog bites man» y «man bites dog» usan las mismas piezas — solo el orden separa la noticia del accidente. La solución es de una simpleza brutal: a cada embedding se le SUMA un segundo vector que depende solo de su posición. Miren: el vector del token no cambia, el de posición sí — y la suma, que es lo que entra al modelo, cambia con él. Bonus: la matriz de posiciones de GPT-2 tiene 1024 filas — por eso su contexto máximo es 1024 tokens. La «ventana de contexto» que comparan entre modelos es, en el fondo, esto.

**[Demo]** Cambiar la posición con los chips (incluidos pos 100 y pos 1000) y ver la fila wpe cambiar. Señalar el badge «pesos reales».

---

## 5 · Atención

**[Pantalla]** Tokens sombreados por peso de atención real (capa 0), selector de cabezas.

**[Decir]** Aquí ocurre la «comprensión». En «the bank by the river», ¿bank es banco u orilla? Solo el contexto lo decide. La atención es el mecanismo: cada token mira a los anteriores, calcula cuánto le importa cada uno, y absorbe de ellos lo que necesita. La analogía: una biblioteca — la Query es la búsqueda, cada Key es la etiqueta de un libro, cada Value su contenido; te llevas una mezcla ponderada por cuánto encaja cada etiqueta. Lo que ven son los pesos REALES de la primera capa de este modelo — y esto se repite en 6 capas, con 12 «cabezas» paralelas que aprenden relaciones distintas. Dato de ingeniería: el costo crece con el cuadrado de la longitud — por eso el contexto largo es caro y existen el KV-cache y FlashAttention.

**[Demo]** Clic en distintos tokens como «punto de vista». Cambiar de cabeza (Prom, 1…12): cada una mira distinto.

---

## 6 · El bloque transformer

**[Pantalla]** Diagrama del bloque (LayerNorm → atención → residual → FFN → residual) + 6 capas apiladas.

**[Decir]** La atención es la mitad famosa del bloque; la otra mitad es una red feed-forward que procesa cada posición por separado, expandiendo el vector de 768 a 3072 y de vuelta. La división del trabajo: la atención *mueve información entre tokens*; la FFN *piensa en cada token*. Residuales y normalización mantienen todo estable. Y este bloque es TODO el modelo: se apila 6 veces aquí, ~100 en los grandes. Las capas tempranas captan sintaxis; las profundas, semántica. Cuando alguien dice «70B de parámetros, 120 capas», está diciendo solo esto: cuántas veces se apila y cuán ancho es. Cuando pulsamos «Comenzar», esto ocurrió 6 veces en este navegador.

**[Demo]** Hover por cada pieza del diagrama (el caption explica) y por las 6 capas de abajo.

---

## 7 · Logits

**[Pantalla]** Top-15 candidatos: logit crudo · barra · probabilidad real. Slider de temperatura.

**[Decir]** Después de las 6 capas, el modelo emite un puntaje —un logit— para CADA token del vocabulario: 50.257 candidatos en cada predicción. Son como las notas de un jurado antes de normalizar: importan las diferencias, no los valores. El softmax los convierte en probabilidades — y ojo, estos porcentajes son las probabilidades reales sobre todo el vocabulario: con temperatura 1, ni el favorito suele pasar del 50%; el modelo siempre reparte su apuesta. La temperatura divide los logits antes del softmax: baja, y las diferencias se amplifican; alta, y se aplanan. El parámetro `temperature` de las APIs es literalmente este divisor, y los `logprobs` que devuelven son el log de estas barras.

**[Demo]** El momento slider: bajar T a 0.1 → una barra se come todo (99%+). Subir a 2.0 → distribución plana. Volver a 1.0.

---

## 8 · Muestreo

**[Pantalla]** Ruleta de candidatos + eliminados tachados + sliders top-k/top-p + greedy.

**[Decir]** Ya hay probabilidades — falta elegir. Y aquí la sorpresa: el modelo NO siempre elige la más probable. Tira una ruleta donde cada candidato ocupa un arco proporcional a su probabilidad — por eso la misma pregunta da respuestas distintas. Antes de girar se filtra: top-k deja solo los k mejores; top-p recorta a los que acumulan hasta p de probabilidad — el «núcleo». Abajo se ve a los eliminados y por qué. Los valores típicos de producción: chat con T≈0.7 y top-p 0.9; código con T≈0.2; creatividad con T≈1.2; y greedy —siempre el más probable— cuando necesitas reproducibilidad.

**[Demo]** Pulsar «Muestrear» 4-5 veces: misma distribución, distinto ganador — ESO es el muestreo. Activar greedy: la ★ se congela y el botón se apaga («sin azar»). Mover top-p y ver candidatos caer a la lista de eliminados.

---

## 9 · El bucle

**[Pantalla]** El texto + token nuevo en acento · historial · botón «Añadir y repetir».

**[Decir]** Y ahora el secreto completo: el token elegido se añade al final del texto… y TODO lo que vimos vuelve a ejecutarse. Tokens, embeddings, atención, logits, muestreo — una palabra por ciclo. Se llama generación autorregresiva: escribir con una única regla, «lee todo lo anterior y añade la palabra siguiente», repetida cientos de veces. El modelo nunca planea la frase completa — la coherencia *emerge* de que cada predicción ve todo lo anterior. El «streaming» de ChatGPT, las palabras apareciendo una a una, no es un efecto visual: es este bucle en vivo. Y por eso el output se cobra por token: cada palabra es una pasada completa del modelo.

**[Demo]** El momento estrella: pulsar «Añadir y repetir» 2-3 veces. La página vuelve sola a Tokens, la barra superior crece, todo se recalcula. Dejar que se vea el ciclo entero.

---

## 10 · Entrenamiento

**[Pantalla]** Tres tarjetas: Pre-entrenamiento → Fine-tuning/RLHF → Inferencia («estamos aquí»).

**[Decir]** ¿Y de dónde salieron esos 82 millones de pesos? Del mismo juego que acabamos de jugar: predecir el siguiente token. En entrenamiento se juega billones de veces — el modelo predice, se mide el error contra el texto real, y backpropagation ajusta cada peso un poquito. Meses, miles de GPUs, millones de dólares. Es comprimir internet en una función: para predecir bien hay que haber capturado los patrones. El resultado es un modelo *base* — como este: solo continúa texto, no «responde». El fine-tuning con RLHF lo convierte en asistente — así GPT-3 se volvió ChatGPT. Y lo que hicimos hoy es la tercera fase: inferencia, solo lectura. Clave para el equipo: el modelo NO aprende al usarlo — cada conversación parte de los mismos pesos congelados. De aquí salen la fecha de corte, y las dos vías para conocimiento propio: RAG o fine-tuning.

**[Demo]** Hover por las tres tarjetas comparando escalas de datos/tiempo/costo.

---

## 11 · Límites

**[Pantalla]** Demo de alucinación: prompts con trampa + top-5 con confianza.

**[Decir]** Cierro con lo más importante para usarlos bien: lo que un LLM no es. Nada de lo que recorrimos verifica hechos. «París» después de «the capital of France is» y un dato inventado salen del MISMO mecanismo. Miren: «Barack Obama was the president of the» → «United», 72% — hecho frecuente, alta confianza. Ahora «Steve Jobs was the founder of» → Apple… y Microsoft, casi con la misma confianza, lado a lado. Y «The 2030 World Cup was won by» → Germany, Brazil — un mundial que no ha pasado, afirmado con total naturalidad. Eso es una alucinación: misma mecánica, misma confianza, cero verdad. Es un estudiante brillante en examen oral: si no sabe, improvisa sin cambiar el tono. La regla de oro: un LLM sin fuentes es un generador de texto plausible, no una base de datos. En producción: los hechos van en el contexto (RAG), se piden citas y se valida lo crítico.

**[Demo]** Clic en los 3 prompts en ese orden (Obama → Jobs → 2030). Si el público propone uno, usar el campo libre.

**[Cierre]** Esto es todo lo que hace ChatGPT: este mismo ciclo, miles de veces, con un modelo mil veces más grande. Conocer la máquina —y sus límites— es lo que nos deja usarla bien. Gracias.

---

## Si algo falla

- **El modelo no carga** (red): las secciones 1–5 funcionan igual (tokenizer + pesos por HTTP Range). Saltar logits/muestreo y mostrar el resto.
- **Perdiste el hilo de pasos**: **P** sale del modo presentación y muestra todo; navegar con el rail.
- **Pregunta fuera de orden**: clic en el punto del rail o en el mapa de la sección 1.
- **Cachear antes**: abrir la página en la red de la sala una vez ese día — el modelo queda en caché del navegador.
