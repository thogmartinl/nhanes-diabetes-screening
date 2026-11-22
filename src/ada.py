"""The ADA Diabetes Risk Test (Bang et al., 2009), rebuilt from NHANES variables."""

import numpy as np
import pandas as pd

ADA_CUTOFF = 5   # a total score of 5 or more counts as high risk


def ada_risk_score(X):
    """Points for each item of the ADA Diabetes Risk Test, plus the total.

    Uses the model features: age, female, told_high_bp, activity_log and bmi.
    - Gestational diabetes and family history are not available in this NHANES
      cycle and score 0 points, so the highest possible total here is 9 (not 11).
    - Physically active = any leisure-time activity (more than 0 minutes per week).
    - Weight status uses BMI 25 / 30 / 40, which matches the test's height-weight chart.
    - Missing answers score 0 points, like a skipped question on the paper test.
    """
    age, bmi = X["age"], X["bmi"]
    points = pd.DataFrame(index=X.index)
    points["age"] = np.select([age >= 60, age >= 50, age >= 40], [3, 2, 1], default=0)
    points["male"] = (X["female"] == 0).astype(int)
    points["high_bp"] = (X["told_high_bp"] == 1).astype(int)
    points["inactive"] = (X["activity_log"] == 0).astype(int)   # 0 minutes of leisure-time activity
    points["weight"] = np.select([bmi >= 40, bmi >= 30, bmi >= 25], [3, 2, 1], default=0)
    points["total"] = points.sum(axis=1)
    return points
