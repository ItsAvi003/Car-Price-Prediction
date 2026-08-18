# 🚗 Used Car Price Valuation & Prediction

An end-to-end Machine Learning web application that predicts used car valuations using an optimized **XGBoost Regressor** pipeline. The model is served via a **FastAPI backend** and provides an interactive user interface built with **Streamlit**, featuring real-time conversion and formatting in Indian Rupees (**INR ₹** in Lakhs and Crores).

---

## 📌 Project Overview

Predicting used car prices is challenging due to high variance across vehicle brands, non-linear depreciation over time, and skewed price distributions. This project implements a full production-ready pipeline that handles data preprocessing, feature engineering, pipeline serialization, API serving, and frontend consumption.

### Key Highlights:
- **Exploratory Data Analysis & Cleaning**: Handled target price skewness with log transformation (`TransformedTargetRegressor`), filtered sensor/entry anomalies, and managed ~80% missing Japanese auction ratings with explicit indicator modeling.
- **Feature Engineering**: Extracted clean brand tokens, top 25 high-volume car models, and engineered interaction terms (`Car_Age`, `Mileage_Per_Year`).
- **High Predictive Accuracy**:
  - **$R^2$ Score**: `0.9321` (~93.2% price variance explained)
  - **Mean Absolute Error (MAE)**: `~5.40 Lakh PKR`
  - **Mean Absolute Percentage Error (MAPE)**: `10.98%`
- **Decoupled Architecture**: FastAPI backend for low-latency inference + Streamlit UI for an intuitive user experience.
- **Currency Support**: Automatic conversion from baseline data to Indian Rupees (INR ₹) with regional financial formatting (Lakhs / Crores).

---

## 🏗️ Architecture & Data Flow

```text
[ Streamlit Web UI ] 
       │
       │  1. GET /options (Fetch dynamic makes, fuel types, transmissions)
       ▼
 [ FastAPI Server ] ─── Loads ───► [ car_price_metadata.pkl ]
       │
       │  2. POST /predict (Send car specifications)
       ▼
[ XGBoost Pipeline ] ─── Encodes & Scales ───► Outputs Price 


⚙️ Tech Stack
    Machine Learning: scikit-learn, xgboost, pandas, numpy, joblib
    Backend API: FastAPI, uvicorn, pydantic
    Frontend UI: Streamlit, requests
    Environment: Python 3.11+


## 📊 Model Performance
| Metric | Random Forest (Baseline) | XGBoost (Final Model) |
| :--- | :---: | :---: |
| **$R^2$ Score** | 0.9193 | **0.9321** |
| **MAPE** | ~12.4% | **10.98%** |
| **MAE** | PKR 583,707 (~₹2.04L) | **PKR 540,312 (~₹1.89L)** |
| **RMSE** | PKR 2,050,708 (~₹7.18L) | **PKR 1,881,364 (~₹6.58L)** |

* **Accuracy**: Explains **93.2%** of price variance with an average error of **~11%**.
* **Key Drivers**: Engine capacity (28.4%), transmission type (36.6%), and vehicle age (25.6%).