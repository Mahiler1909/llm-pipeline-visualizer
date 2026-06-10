/**
 * Helpers compartidos por los widgets de etapa.
 */

export function esc(s) {
  return String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

/** Etiqueta legible de un token: espacio inicial visible como ␣ */
export function tokenLabel(text) {
  return text.replace(/^ /, '␣');
}

const HUES = [210, 25, 95, 150, 265, 330, 50, 185];

/** Fondo pastel desaturado para el token i */
export function pastelBg(i) {
  return `hsl(${HUES[i % HUES.length]} 52% 88%)`;
}

/** Color de borde/texto a juego con pastelBg */
export function pastelInk(i) {
  return `hsl(${HUES[i % HUES.length]} 42% 42%)`;
}

/**
 * Escala divergente para valores en [-1, 1]:
 * azul (negativo) ↔ casi blanco (0) ↔ naranja quemado (positivo).
 */
export function divergent(v) {
  const a = Math.min(1, Math.sqrt(Math.abs(v)));
  return v >= 0
    ? `rgba(194, 85, 44, ${a.toFixed(3)})`
    : `rgba(74, 111, 165, ${a.toFixed(3)})`;
}

export function fmtPct(p, decimals = 1) {
  return (p * 100).toFixed(decimals) + '%';
}
