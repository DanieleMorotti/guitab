"""Guitar constants shared by the whole pipeline (standard tuning)."""

OPEN_PITCH = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}  # string number (1 = high E) -> MIDI pitch
STRINGS = [6, 5, 4, 3, 2, 1]  # label column order of the model heads: low E .. high E
N_FRETS = 20  # frets 0..19 -> tab classes 1..20 (0 = string silent)
N_CLASSES = N_FRETS + 1
