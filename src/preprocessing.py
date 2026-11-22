"""Prepare the merged NHANES table for modeling."""


def select_screening_sample(data):
    """Keep adults (18+) with an HbA1c result who do not report a diabetes diagnosis.

    DIQ010 asks whether a doctor ever said the participant has diabetes
    (1 = Yes, 2 = No, 3 = Borderline, 7 = Refused, 9 = Don't know).
    Only "Yes" is removed.
    """
    adults_with_hba1c = data[(data["RIDAGEYR"] >= 18) & data["LBXGH"].notna()]
    screening = adults_with_hba1c[adults_with_hba1c["DIQ010"] != 1].copy()
    return screening
