// Guitar constants shared by the page's pipeline (guitab/guitar.py, guitab/audio.py).

export const SR = 22050;
export const FPS = SR / 256;
export const OPEN_PITCH = { 6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64 }; // string number (1 = high E) -> MIDI pitch
export const STRINGS = [6, 5, 4, 3, 2, 1]; // column order of the model outputs: low E .. high E

// Python's round(): halves go to the even neighbour.
export function roundHalfEven(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
