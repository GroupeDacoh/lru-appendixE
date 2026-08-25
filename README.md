# LRU flux-activated — simulations QuTiP (Appendix E)

Code et jeux de données ARFF pour la réduction de fuite (LRU) activée par flux
sur transmon, inspirée de Lacroix et al. (arXiv:2309.07060, Appendix E).

## Contenu

| Fichier / dossier | Rôle |
|---|---|
| `params_appendixE.py` | Paramètres Table I + point O + échelle sideband |
| `sim_lru_appendixE.py` | Simulateur (Hamiltonien Appendix E, solveur RWA) |
| `create_appendixE_full_arff.py` | Génération ARFF pulse-only (~400 lignes) |
| `create_ml_arff_bundle.py` | Bundle de 8 ARFF pour Weka / M5Rules |
| `sweep_pointO_pulse_only.py` | Balayage autour du point O |
| `qutip_param_dataset_appendixE_full.arff` | Dataset principal |
| `arff_ml_bundle/` | 8 ARFF + README + manifest |

## Méthode (résumé)

- Puce **fixée** (Table I, qubit A)
- Variables pulse : `omega_m`, `omega_a`, `sigma`, `T_active`
- Truncation Hilbert `(6,6,6)`
- Cible ML : `leakage_final`
- Point O expérimental en **1re ligne** de chaque ARFF (`leakage_final ≈ 6×10⁻⁴`)

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage rapide

```bash
# Une simulation au point O (params par défaut)
python sim_lru_appendixE.py

# Régénérer le dataset principal
python create_appendixE_full_arff.py

# Régénérer le bundle ML (8 fichiers)
python create_ml_arff_bundle.py
```

## Citation

Lacroix et al., *Fast Flux-Activated Leakage Reduction for Superconducting Quantum Circuits*, arXiv:2309.07060.
