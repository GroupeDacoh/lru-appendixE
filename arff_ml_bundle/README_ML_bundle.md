# Bundle ARFF pulse-only (Appendix E, 6/6/6)


- **Cible ML** : `leakage_final`
- **Puce** : fixée Table I (`omega_r=7129`, `omega_p=7111`, `g_qr=120`, `J=28.8`)
- **Variables** : `omega_m`, `omega_a`, `sigma`, `T_active`
- **Ligne 1** de chaque fichier : point O expérimental

| Fichier | Description | n |
|---|---|---|
| `qutip_01_full_mixed.arff` | Mélange grille+aléatoire autour de O (baseline) | 400 |
| `qutip_02_near_O.arff` | Échantillonnage dense près du point O | 300 |
| `qutip_03_resonance.arff` | Sur la droite de résonance (2Δω_m≈Δω_a) | 300 |
| `qutip_04_explore.arff` | Contrastes hors résonance / hors O | 300 |
| `qutip_05_T_active_sweep.arff` | Balayage systématique de T_active | 118 |
| `qutip_06_omega_m_sweep.arff` | Balayage systématique de ω_m | 190 |
| `qutip_07_omega_a_sweep.arff` | Balayage systématique de ω_a | 190 |
| `qutip_08_sigma_sweep.arff` | Balayage systématique de σ | 100 |

Les attributs puce sont présents mais constants (choix méthodologique pulse-only).

