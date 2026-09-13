"""
Modul Machine Learning Prediktor Saham Indonesia (IDX).
Melatih model Random Forest Regressor berbasis fitur teknikal dan deret waktu.
Menghasilkan proyeksi harga 5 hari ke depan, arah tren, dan evaluasi akurasi model.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Mempersiapkan fitur-fitur teknikal dan lag harga untuk pelatihan Machine Learning.
    """
    data = df.copy()
    close = data["Close"]

    # Fitur Return Masa Lalu
    data["Return_1d"] = close.pct_change(1)
    data["Return_3d"] = close.pct_change(3)
    data["Return_5d"] = close.pct_change(5)
    data["Return_10d"] = close.pct_change(10)

    # Fitur Rasio Harga terhadap MA
    if "EMA_20" in data.columns:
        data["Ratio_EMA20"] = close / data["EMA_20"].replace(0, np.nan)
    if "SMA_50" in data.columns:
        data["Ratio_SMA50"] = close / data["SMA_50"].replace(0, np.nan)

    # Fitur Posisi Bollinger Band
    if "BB_Upper" in data.columns and "BB_Lower" in data.columns:
        band_range = (data["BB_Upper"] - data["BB_Lower"]).replace(0, np.nan)
        data["BB_Position"] = (close - data["BB_Lower"]) / band_range

    # Fitur Rasio Volatilitas ATR
    if "ATR_14" in data.columns:
        data["ATR_Pct"] = data["ATR_14"] / close.replace(0, np.nan)

    # Fitur Target: Harga Penutupan 1 Hari ke Depan
    data["Target_Next_Close"] = close.shift(-1)

    feature_cols = [
        "Return_1d", "Return_3d", "Return_5d", "Return_10d",
        "Ratio_EMA20", "Ratio_SMA50", "BB_Position", "ATR_Pct"
    ]
    if "RSI_14" in data.columns:
        feature_cols.append("RSI_14")
    if "Volume_Ratio" in data.columns:
        feature_cols.append("Volume_Ratio")

    valid_data = data.dropna(subset=feature_cols + ["Target_Next_Close"])
    X = valid_data[feature_cols]
    y = valid_data["Target_Next_Close"]

    return X, y, valid_data, feature_cols


def train_and_predict_future(
    df: pd.DataFrame, 
    forecast_days: int = 5
) -> Dict[str, Any]:
    """
    Melatih model Random Forest pada data historis dan memproyeksikan harga 5 hari ke depan.
    """
    if len(df) < 80:
        return {
            "success": False,
            "message": "Data historis kurang panjang (minimal butuh 80 hari bursa)."
        }

    X, y, clean_df, feature_cols = prepare_ml_features(df)
    if len(X) < 60:
        return {
            "success": False,
            "message": "Data tidak mencukupi setelah pembersihan fitur."
        }

    # Pembagian Train-Test Berurutan (Time-Series Split: 80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluasi pada data uji (Test Set)
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Hitung Akurasi Arah Gerak (Directional Accuracy)
    actual_direction = (y_test.values - clean_df["Close"].iloc[split_idx:].values) > 0
    pred_direction = (y_pred_test - clean_df["Close"].iloc[split_idx:].values) > 0
    dir_accuracy = np.mean(actual_direction == pred_direction) * 100

    # Latih ulang pada seluruh data untuk proyeksi masa depan
    model.fit(X, y)

    # Proyeksi Rekursif ke Depan
    last_row = X.iloc[-1:].copy()
    current_price = float(clean_df["Close"].iloc[-1])
    projected_prices = []
    temp_price = current_price

    for day in range(1, forecast_days + 1):
        pred_val = float(model.predict(last_row)[0])
        # Batasi perubahan wajar per hari di BEI (Auto Rejection Simetris ~ ±10-25%)
        max_jump = temp_price * 0.08
        clamped_pred = max(temp_price - max_jump, min(temp_price + max_jump, pred_val))
        projected_prices.append(round(clamped_pred))
        temp_price = clamped_pred

    final_pred = projected_prices[-1]
    pct_change = ((final_pred - current_price) / current_price) * 100

    if pct_change >= 2.5:
        trend = "BULLISH"
    elif pct_change <= -2.5:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS / NEUTRAL"

    # Kepentingan Fitur (Feature Importances)
    importances = dict(zip(feature_cols, [round(float(v), 4) for v in model.feature_importances_]))

    return {
        "success": True,
        "current_price": round(current_price),
        "forecast_prices": projected_prices,
        "final_projected_price": round(final_pred),
        "expected_pct_change": round(pct_change, 2),
        "trend_prediction": trend,
        "mae_idr": round(mae, 2),
        "rmse_idr": round(rmse, 2),
        "directional_accuracy_pct": round(dir_accuracy, 1),
        "feature_importances": importances
    }
