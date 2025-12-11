# ML-Based Prediction of Soil Stabilization with Cement and Waste Glass Powder

This repository presents a machine learning framework for predicting **Unconfined Compressive Strength (UCS)** of lateritic soil stabilized with Portland cement and waste glass powder.

The models are trained on a comprehensive laboratory dataset:
- **96 UCS measurements** (4 cement levels × 4 glass levels × 2 compaction methods × 3 curing ages)

## Key Results
- **Best UCS Model**: Gradient Boosting (R² = 0.842, RMSE = 89.2 kN)
- **Optimal UCS Mix** (28 days, WAS compaction): **6.5% cement + 4.0% glass** → Predicted 1082.8 kN (close to lab max 1106 kN)
- **Recommended Practical Mix**: **5% cement + 2.5% glass** (28 days, WAS): Predicted UCS ≈ 704 kN, Soaked CBR ≈ 72% (lab: ~739 kN UCS, ~70% CBR)
