# Glass Transition Temperature Predictor

Random Forest regression model predicting glass transition temperature (Tg)
from oxide composition, trained on ~29,000 samples from the SciGlass database.

Originally built as a Master's thesis project; currently being rebuilt as a
full-stack portfolio app.

## Results

Evaluated on held-out test set:

- **R²:** 0.95  
- **RMSE:** 32.5 °C

## Roadmap

- [x] Phase 1 — ML model (Random Forest baseline, feature engineering)
- [ ] Phase 2 — FastAPI backend (model serving, inference endpoint)
- [ ] Phase 3 — Streamlit frontend (interactive composition input)
- [ ] Phase 4 — Deployment

## Dataset

SciGlass database (~29,000 oxide glass compositions). Not included in repo
due to licensing.
