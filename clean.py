from pathlib import Path
import numpy as np
import pandas as pd


def clean_raw_csv(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw hospital patient records and adds length_of_stay_days."""
    df = raw_df.copy()

    # 1. Remove duplicate rows and records missing patient_id
    df = df.drop_duplicates().dropna(subset=["patient_id"]).copy()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()

    # 2. Standardize gender
    gender_map = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
    df["gender"] = (
        df["gender"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": np.nan, "nan": np.nan, "none": np.nan})
        .map(gender_map)
    )
    gender_mode = df["gender"].mode()
    df["gender"] = df["gender"].fillna(gender_mode[0] if not gender_mode.empty else "Unknown")

    # 3. Standardize department
    df["department"] = (
        df["department"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": np.nan, "nan": np.nan, "none": np.nan})
        .str.title()
        .fillna("General Medicine")
    )

    # 4. Standardize 30-day readmission flag
    readmit_map = {"yes": "Yes", "y": "Yes", "1": "Yes", "no": "No", "n": "No", "0": "No"}
    df["readmitted_30d"] = (
        df["readmitted_30d"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(readmit_map)
        .fillna("No")
    )

    # 5. Clean age (valid range 0-120)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[(df["age"] < 0) | (df["age"] > 120), "age"] = np.nan
    median_age = df["age"].median()
    df["age"] = df["age"].fillna(median_age if pd.notna(median_age) else 45).round(0).astype(int)

    # 6. Clean billing_amount (remove negatives, cap 99th percentile, impute median)
    df["billing_amount"] = pd.to_numeric(df["billing_amount"], errors="coerce")
    df.loc[df["billing_amount"] < 0, "billing_amount"] = np.nan
    valid_billing = df["billing_amount"].dropna()
    if not valid_billing.empty:
        cap_99 = valid_billing.quantile(0.99)
        df["billing_amount"] = df["billing_amount"].clip(upper=cap_99)
        median_billing = df["billing_amount"].median()
    else:
        median_billing = 0.0
    df["billing_amount"] = df["billing_amount"].fillna(median_billing).round(2)

    # 7. Parse and fix admission & discharge dates
    adm = pd.to_datetime(df["admission_date"], format="mixed", errors="coerce")
    dis = pd.to_datetime(df["discharge_date"], format="mixed", errors="coerce")

    valid_adm = adm.dropna()
    median_adm = valid_adm.median() if not valid_adm.empty else pd.Timestamp("2026-02-01")
    adm = adm.fillna(median_adm)
    dis = dis.fillna(adm + pd.Timedelta(days=3))

    swapped = dis < adm
    dis.loc[swapped] = adm.loc[swapped] + pd.Timedelta(days=1)

    # 8. Derive length_of_stay_days and format dates
    df["length_of_stay_days"] = (dis - adm).dt.days.clip(lower=1).astype(int)
    df["admission_date"] = adm.dt.strftime("%Y-%m-%d")
    df["discharge_date"] = dis.dt.strftime("%Y-%m-%d")

    return df


def main() -> None:
    project_dir = Path(__file__).parent
    raw_path = project_dir / "bk.csv"
    cleaned_path = project_dir / "cleaned_data.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"'{raw_path.name}' not found in {project_dir}.")

    raw_df = pd.read_csv(raw_path)
    cleaned_df = clean_raw_csv(raw_df)
    cleaned_df.to_csv(cleaned_path, index=False)

    print("=" * 65)
    print("DATA CLEANING COMPLETE")
    print("=" * 65)
    print(f"Raw rows     : {len(raw_df)}")
    print(f"Cleaned rows : {len(cleaned_df)}")
    print(f"Saved file   : {cleaned_path.resolve()}")
    print("\nFirst 5 cleaned rows:")
    print(cleaned_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()