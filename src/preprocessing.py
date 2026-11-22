"""Prepare the merged NHANES table for modeling."""

import pandas as pd


def select_screening_sample(data):
    """Select the screening sample and count participants after each step.

    DIQ010 asks whether a doctor ever said the participant has diabetes
    (1 = Yes, 2 = No, 3 = Borderline, 7 = Refused, 9 = Don't know).
    Only "Yes" is removed. Pregnant participants (RIDEXPRG = 1) and
    participants taking diabetes pills (DIQ070 = 1) are also removed.

    Returns the screening sample and a table with the count after each step.
    """
    steps = [
        ("Aged 18 or older", data["RIDAGEYR"] >= 18),
        ("With an HbA1c result", data["LBXGH"].notna()),
        ("Excluded: reported diabetes diagnosis (DIQ010 = 1)", data["DIQ010"] != 1),
        ("Excluded: pregnant at exam (RIDEXPRG = 1)", data["RIDEXPRG"] != 1),
        ("Excluded: taking diabetes pills (DIQ070 = 1)", data["DIQ070"] != 1),
    ]

    keep = pd.Series(True, index=data.index)
    flow = [("All participants", len(data))]
    for label, condition in steps:
        keep &= condition
        flow.append((label, int(keep.sum())))

    screening = data[keep].copy()
    return screening, pd.DataFrame(flow, columns=["Step", "Participants"])




# Refused / don't know codes for each raw variable used to build features.
# Taken from the NHANES August 2021-August 2023 codebooks.
# Variables not listed here (e.g. age, BMI, blood pressure, sleep hours) have no such codes.
MISSING_CODES = {
    "BPQ020": [7, 9],                                   # told high blood pressure
    "MCQ160B": [7, 9], "MCQ160C": [7, 9], "MCQ160D": [7, 9],
    "MCQ160E": [7, 9], "MCQ160F": [7, 9],               # cardiovascular conditions
    "SMQ020": [7, 9], "SMQ040": [7, 9],                 # smoking
    "PAD790Q": [7777, 9999], "PAD800": [7777, 9999],    # moderate activity
    "PAD810Q": [7777, 9999], "PAD820": [7777, 9999],    # vigorous activity
    "ALQ111": [7, 9], "ALQ121": [77, 99], "ALQ130": [777, 999],  # alcohol
    **{f"DPQ0{i}0": [7, 9] for i in range(1, 10)},     # PHQ-9 items DPQ010-DPQ090
    "HIQ011": [7, 9],                                   # health insurance
    "HUQ010": [7, 9],                                   # self-rated health
}


def recode_missing(df):
    """Replace each variable's refused / don't know codes with missing values."""
    not_found = [col for col in MISSING_CODES if col not in df.columns]
    if not_found:
        raise KeyError(f"Columns not found in the data: {not_found}")

    df = df.copy()
    for col, codes in MISSING_CODES.items():
        df[col] = df[col].where(~df[col].isin(codes))
    return df


def resolve_skip_patterns(df):
    """Fill blanks that NHANES created on purpose through skip patterns.

    - Smoking: combine SMQ020 (100+ cigarettes in life) and SMQ040 (smoke now)
      into one variable: never / former / current.
    - Alcohol: participants who never drank (ALQ111 = 2) skipped ALQ121, so their
      frequency is 0. Participants with no drinks in the past year (ALQ121 = 0)
      skipped ALQ130, so their drinks per drinking day are 0.
    - Physical activity: participants with a frequency of 0 skipped the minutes
      question, so their minutes are 0.
    """
    df = df.copy()

    # Smoking status (stays missing if SMQ020 is refused, don't know, or blank)
    df["smoking_status"] = pd.Series(index=df.index, dtype="object")
    df.loc[df["SMQ020"] == 2, "smoking_status"] = "never"
    df.loc[(df["SMQ020"] == 1) & (df["SMQ040"] == 3), "smoking_status"] = "former"
    df.loc[(df["SMQ020"] == 1) & df["SMQ040"].isin([1, 2]), "smoking_status"] = "current"

    # Alcohol
    df.loc[df["ALQ111"] == 2, "ALQ121"] = 0
    df.loc[df["ALQ121"] == 0, "ALQ130"] = 0

    # Physical activity
    df.loc[df["PAD790Q"] == 0, "PAD800"] = 0
    df.loc[df["PAD810Q"] == 0, "PAD820"] = 0
    return df
