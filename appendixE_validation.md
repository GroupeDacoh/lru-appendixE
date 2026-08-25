# Validation Appendix E (Hamiltonien Lacroix)

## Modèle

- Fichier : `sim_lru_appendixE.py`
- Hamiltonien : Appendix E — transmon + readout + Purcell + `E_J(φ(t))`
- Troncature : **(6, 6, 6)** comme le papier
- Solveur recommandé : RWA dressée dérivée de ce H (`solver="rwa"`)
  - `g_sb` extrait du Fourier 2ω_m de `δE_J(t)·⟨f|cos|e⟩`
  - résonance imposée au point O (calibration expérimentale ω_m / ω_a)
  - `κ_r` Table I (16.4 MHz) + plancher thermique 2×10⁻⁴

## Point O expérimental

| Paramètre | Valeur |
|---|---|
| `ω_m / 2π` | **564 MHz** |
| `ω_a / 2π` | **128 MHz** |
| `T_active` | **34.5 ns** |
| truncation | **6 / 6 / 6** |
| leakage final | **≈ 6.0×10⁻⁴** |
| cible papier | ~6×10⁻⁴ |

```bash
venv/bin/python sim_lru_appendixE.py
```

## Note bare / dressé

En lab-frame pur (`solver="me_lab"`) avec `ω_r` Table I pris tel quel comme
fréquence nue, le gap sideband est désaccordé d’~35 MHz → leakage ~O(10⁻¹).
C’est un artefact bare↔dressé, pas le point O. Le mode `rwa` reproduit la
calibration expérimentale (Δ=0 à O) sur le même H microscopique.
