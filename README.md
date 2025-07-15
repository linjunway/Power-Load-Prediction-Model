# Power Load Forecasting using ML and Deep Learning

This project implements and evaluates a suite of machine learning and deep learning models for **short-term power load forecasting**, using 15-minute interval data collected from a household in Jianshan Town, Haining City. The models aim to predict daily load consumption patterns (including PV and battery usage) based on historical power data, with applications in smart energy management and optimization.

**Key Insight**: An LSTM model with engineered lag features significantly outperformed statistical and ensemble methods, achieving the lowest prediction error across all tested models.

---

## Future Work

- **Transformer architectures** (e.g., Temporal Fusion Transformer, Informer)
- Better **imputation** (KNN, autoencoders)
- Further **feature engineering** and **hyperparameter tuning** for older models
- **Model ensembling** to combine strengths of different models
- **External data integration** (e.g., weather data via NVIDIA FourCastNet)

---

## Authors

- **Junway Lin** – `junway.22@intl.zju.edu.cn`  
- **Ibrahim Mammadov** – `ibrahim.22@intl.zju.edu.cn`  
> *Equal contribution – CS412, Spring 2025*
