| site | regime | price scenario | days | safe_uncontrolled | best heuristic | cost | oracle | headroom H |
|---|---|---|---|---|---|---|---|---|
| caltech | stress | duck | 8 | 328.8 | edf_cap_0.8 | 267.5 | 246.1 | 8.0% |
| caltech | stress | flat | 8 | 347.4 | edf_cap_0.8 | 286.3 | 262.8 | 8.2% |
| caltech | stress | solar_glut | 8 | 314.4 | edf_cap_0.8 | 252.3 | 234.0 | 7.3% |
| caltech | stress | volatile | 8 | 334.5 | edf_cap_0.8 | 274.2 | 252.7 | 7.8% |
| jpl | stress | duck | 8 | 716.7 | safe_uncontrolled | 716.7 | 594.4 | 17.1% |
| jpl | stress | flat | 8 | 761.1 | safe_uncontrolled | 761.1 | 635.2 | 16.5% |
| jpl | stress | solar_glut | 8 | 676.4 | safe_uncontrolled | 676.4 | 560.8 | 17.1% |
| jpl | stress | volatile | 8 | 719.0 | safe_uncontrolled | 719.0 | 600.6 | 16.5% |

H = (J_best_heuristic - J_oracle)/J_best_heuristic. All prices are SYNTHETIC scenarios (provenance tag in each run); real sessions + IEEE-33 physics.
