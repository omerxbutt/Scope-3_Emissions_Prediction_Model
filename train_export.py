from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


RANDOM_STATE = 42
N_COMPANIES = 1000
MODEL_PATH = Path("scope3_model.onnx")

SECTORS = ["Energy", "Technology", "Manufacturing", "Financials"]
SECTOR_FACTORS = {
    "Energy": 0.0028,
    "Technology": 0.00075,
    "Manufacturing": 0.00185,
    "Financials": 0.00045,
}

FEATURE_COLUMNS = [
    "Revenue_USD",
    "COGS_USD",
    "Employee_Count",
    "COGS_to_Revenue_Ratio",
    "Revenue_per_Employee",
    "Sector_Energy",
    "Sector_Technology",
    "Sector_Manufacturing",
    "Sector_Financials",
]


def generate_synthetic_companies() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    sectors = rng.choice(SECTORS, size=N_COMPANIES, p=[0.2, 0.32, 0.28, 0.2])
    revenue = rng.lognormal(mean=np.log(2.5e8), sigma=1.0, size=N_COMPANIES)
    cogs_ratio = rng.uniform(0.18, 0.82, size=N_COMPANIES)
    cogs = revenue * cogs_ratio
    employees = rng.integers(25, 50000, size=N_COMPANIES)
    sector_factor = np.array([SECTOR_FACTORS[sector] for sector in sectors])
    base_scope_3 = (cogs**1.08) * sector_factor
    noise = rng.normal(loc=0.0, scale=base_scope_3 * 0.08, size=N_COMPANIES)

    # Scope_3_MT = (COGS_USD ** 1.08) * sector_intensity_factor + noise
    scope_3 = base_scope_3 + noise
    scope_3 = np.maximum(scope_3, 0.0)

    return pd.DataFrame(
        {
            "Revenue_USD": revenue,
            "COGS_USD": cogs,
            "Employee_Count": employees,
            "GICS_Sector": sectors,
            "Scope_3_MT": scope_3,
        }
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["COGS_to_Revenue_Ratio"] = np.divide(
        enriched["COGS_USD"],
        enriched["Revenue_USD"],
        out=np.zeros(len(enriched), dtype=float),
        where=enriched["Revenue_USD"].to_numpy() != 0,
    )
    enriched["Revenue_per_Employee"] = np.divide(
        enriched["Revenue_USD"],
        enriched["Employee_Count"],
        out=np.zeros(len(enriched), dtype=float),
        where=enriched["Employee_Count"].to_numpy() != 0,
    )

    for sector in SECTORS:
        enriched[f"Sector_{sector}"] = (enriched["GICS_Sector"] == sector).astype(np.float32)

    return enriched


def main() -> None:
    df = add_engineered_features(generate_synthetic_companies())
    x = df[FEATURE_COLUMNS].astype(np.float32)
    y = df["Scope_3_MT"].astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE: {mae:,.2f}")
    print(f"R2: {r2:.4f}")

    initial_types = [("float_input", FloatTensorType([None, len(FEATURE_COLUMNS)]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=15,
    )

    input_name = onnx_model.graph.input[0].name
    output_name = onnx_model.graph.output[0].name
    print(f"ONNX input node name: {input_name}")
    print(f"ONNX output node name: {output_name}")

    MODEL_PATH.write_bytes(onnx_model.SerializeToString())
    print(f"Saved model: {MODEL_PATH}")
    print(f"File size: {MODEL_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
