"""Build the model features and the target from the cleaned screening sample."""

import numpy as np
import pandas as pd

HBA1C_CUTOFF = 5.7      # ADA threshold for prediabetes (%)
DIABETES_CUTOFF = 6.5   # ADA threshold for diabetes (%)

# Times per week for each frequency unit in the physical activity questions
WEEKLY_FACTOR = {"D": 7, "W": 1, "M": 12 / 52, "Y": 1 / 52}

CVD_COLUMNS = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]
PHQ9_COLUMNS = [f"DPQ0{i}0" for i in range(1, 10)]


def yes_no(series):
    """Turn a 1 = Yes / 2 = No question into 1 / 0. Missing stays missing."""
    return series.map({1: 1, 2: 0})


def weekly_minutes(freq, unit, minutes):
    """Minutes per week from a frequency, its unit (D/W/M/Y), and minutes per session."""
    # pandas reads text columns from SAS files as bytes (b"D"), so decode them first
    unit = unit.map(lambda u: u.decode() if isinstance(u, bytes) else u)
    total = freq * unit.map(WEEKLY_FACTOR) * minutes
    # A frequency of 0 means no activity, even though the unit is blank
    return total.where(freq != 0, 0)


def build_features(df):
    """Return one row per participant with the 19 model input variables."""
    f = pd.DataFrame(index=df.index)

    # Demographics
    f["age"] = df["RIDAGEYR"]
    f["female"] = (df["RIAGENDR"] == 2).astype(int)
    f["race_ethnicity"] = df["RIDRETH3"]            # categorical, one-hot encoded later
    f["income_ratio"] = df["INDFMPIR"]

    # Body measures
    f["bmi"] = df["BMXBMI"]
    f["waist"] = df["BMXWAIST"]
    f["waist_height_ratio"] = df["BMXWAIST"] / df["BMXHT"]

    # Blood pressure: mean of the available readings
    f["systolic_bp"] = df[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
    f["diastolic_bp"] = df[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
    f["told_high_bp"] = yes_no(df["BPQ020"])

    # Any cardiovascular disease: 1 if any condition = Yes, 0 if all five = No
    cvd = df[CVD_COLUMNS]
    f["cvd"] = np.where((cvd == 1).any(axis=1), 1,
                        np.where((cvd == 2).all(axis=1), 0, np.nan))

    # Lifestyle
    f["smoking_status"] = df["smoking_status"]      # categorical, one-hot encoded later
    moderate = weekly_minutes(df["PAD790Q"], df["PAD790U"], df["PAD800"])
    vigorous = weekly_minutes(df["PAD810Q"], df["PAD810U"], df["PAD820"])
    f["activity_log"] = np.log1p(moderate + 2 * vigorous)   # moderate-equivalent minutes
    # Reorder ALQ121 so higher = more often: 0 = never ... 10 = every day
    f["alcohol_freq"] = df["ALQ121"].where(df["ALQ121"] == 0, 11 - df["ALQ121"])
    f["alcohol_drinks"] = df["ALQ130"]
    f["sleep_hours"] = df["SLD012"]

    # Other
    f["phq9"] = df[PHQ9_COLUMNS].sum(axis=1, min_count=9)  # only if all 9 answered
    f["insured"] = yes_no(df["HIQ011"])
    f["health_rating"] = df["HUQ010"]               # 1 = excellent ... 5 = poor
    return f


def make_target(df):
    """1 if HbA1c is 5.7% or higher (prediabetes or diabetes range), else 0."""
    return (df["LBXGH"] >= HBA1C_CUTOFF).astype(int).rename("elevated_hba1c")
