from __future__ import annotations

from typing import List


def risk_band(probability: float) -> str:
    """
    Convert raw model probability into a simple clinical-style risk band.
    """
    if probability >= 0.85:
        return "high"
    if probability >= 0.50:
        return "medium"
    return "low"


def audit_prediction(
    probability: float,
    prediction: int,
    threshold: float = 0.85,
    quality: str | None = None,
    artifacts: int | float | None = None,
    margin: float = 0.10,
) -> dict:
    """
    Rule-based auditor for the multimodal DR screening model.

    The auditor does not replace the model. It adds a transparent review layer
    that flags uncertain or low-quality cases for manual review.
    """

    reasons: List[str] = []

    lower = threshold - margin
    upper = threshold + margin

    # Rule 1: near operating threshold
    if lower <= probability <= upper:
        reasons.append("near_decision_threshold")

    # Rule 2: very high probability positive case
    if prediction == 1 and probability >= threshold:
        reasons.append("model_positive_high_risk")

    # Rule 3: borderline negative case
    if prediction == 0 and probability >= 0.50:
        reasons.append("borderline_negative_but_elevated_probability")

    # Rule 4: poor image quality if available
    if quality is not None:
        quality_str = str(quality).strip().lower()
        if quality_str in {"bad", "poor", "low", "0", "inadequate"}:
            reasons.append("poor_image_quality")

    # Rule 5: artifacts present if available
    if artifacts is not None:
        try:
            if float(artifacts) >= 2:
                reasons.append("high_artifact_level")
        except Exception:
            pass

    decision = "manual_review" if reasons else "accept_model_output"

    return {
        "audit_decision": decision,
        "audit_reasons": ";".join(reasons) if reasons else "none",
        "risk_band": risk_band(probability),
    }

