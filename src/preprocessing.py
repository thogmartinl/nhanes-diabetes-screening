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
