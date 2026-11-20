from pathlib import Path
import numpy as np
import pandas as pd


def load_and_clean_csv(csv_path: Path) -> pd.DataFrame:
    """Loads raw CSV (e.g., bk.csv) and cleans it in memory for searching."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find '{csv_path.name}' in {csv_path.parent}.")

    df = pd.read_csv(csv_path).drop_duplicates().dropna(subset=["patient_id"]).copy()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()

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

    df["department"] = (
        df["department"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"": np.nan, "nan": np.nan, "none": np.nan})
        .str.title()
        .fillna("General Medicine")
    )

    readmit_map = {"yes": "Yes", "y": "Yes", "1": "Yes", "no": "No", "n": "No", "0": "No"}
    df["readmitted_30d"] = (
        df["readmitted_30d"].astype(str).str.strip().str.lower().map(readmit_map).fillna("No")
    )

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[(df["age"] < 0) | (df["age"] > 120), "age"] = np.nan
    df["age"] = df["age"].fillna(df["age"].median()).round(0).astype(int)

    df["billing_amount"] = pd.to_numeric(df["billing_amount"], errors="coerce")
    df.loc[df["billing_amount"] < 0, "billing_amount"] = np.nan
    cap_99 = df["billing_amount"].dropna().quantile(0.99)
    df["billing_amount"] = df["billing_amount"].clip(upper=cap_99)
    df["billing_amount"] = df["billing_amount"].fillna(df["billing_amount"].median()).round(2)

    adm = pd.to_datetime(df["admission_date"], format="mixed", errors="coerce")
    dis = pd.to_datetime(df["discharge_date"], format="mixed", errors="coerce")
    adm = adm.fillna(adm.dropna().median())
    dis = dis.fillna(adm + pd.Timedelta(days=3))
    swapped = dis < adm
    dis.loc[swapped] = adm.loc[swapped] + pd.Timedelta(days=1)

    df["length_of_stay_days"] = (dis - adm).dt.days.clip(lower=1).astype(int)
    df["admission_date"] = adm.dt.strftime("%Y-%m-%d")
    df["discharge_date"] = dis.dt.strftime("%Y-%m-%d")
    return df


def search_records(
    df: pd.DataFrame,
    patient_id: str | None = None,
    department: str | None = None,
    gender: str | None = None,
    readmitted_30d: str | None = None,
    min_billing: float | None = None,
    min_stay_days: int | None = None,
) -> pd.DataFrame:
    result = df.copy()

    if patient_id:
        result = result[
            result["patient_id"].astype(str).str.lower().str.contains(patient_id.strip().lower(), na=False)
        ]
    if department:
        result = result[
            result["department"].astype(str).str.lower().str.contains(department.strip().lower(), na=False)
        ]
    if gender:
        result = result[result["gender"].astype(str).str.lower() == gender.strip().lower()]
    if readmitted_30d:
        result = result[result["readmitted_30d"].astype(str).str.lower() == readmitted_30d.strip().lower()]
    if min_billing is not None:
        result = result[pd.to_numeric(result["billing_amount"], errors="coerce") >= min_billing]
    if min_stay_days is not None:
        result = result[pd.to_numeric(result["length_of_stay_days"], errors="coerce") >= min_stay_days]

    return result


def display_search_results(matches: pd.DataFrame, query_label: str) -> None:
    print("=" * 72)
    print(f"TERMINAL SEARCH RESULT — {query_label}")
    print("=" * 72)
    print(f"Matching Records Found: {len(matches)}")

    if matches.empty:
        print("No matching patient records found.")
        return

    print(matches.to_string(index=False))
    print("\nQuick Summary of Matches:")
    print(f"  Total Billing : {matches['billing_amount'].sum():,.2f}")
    print(f"  Avg Stay Days : {matches['length_of_stay_days'].mean():.2f}")
    print(f"  Readmitted %  : {(matches['readmitted_30d'] == 'Yes').mean() * 100:.2f}%")


def main() -> None:
    project_dir = Path(__file__).parent
    csv_path = project_dir / "bk.csv"
    df = load_and_clean_csv(csv_path)

    display_search_results(search_records(df, department="Cardiology"), "department == 'Cardiology'")
    display_search_results(
        search_records(df, readmitted_30d="Yes", min_billing=20000),
        "readmitted_30d == 'Yes' AND billing >= 20000",
    )
    display_search_results(search_records(df, patient_id="PAT-1002"), "patient_id contains 'PAT-1002'")


if __name__ == "__main__":
    main()