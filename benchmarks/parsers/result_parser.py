"""Parse accuracy/MSE/score from LAMBDA execution output."""
import re
from typing import Optional


def parse_accuracy(text: str) -> Optional[float]:
    """Extract accuracy value from execution output text."""
    if not text:
        return None

    # Strategy 1: Structured marker
    m = re.search(r'FINAL_ACCURACY:\s*([0-9]*\.?[0-9]+)', text)
    if m:
        val = float(m.group(1))
        return val if val <= 1.0 else val / 100.0

    # Strategy 2: Common patterns
    patterns = [
        r'(?:accuracy|Accuracy|ACC)\s*[:=]\s*([0-9]*\.?[0-9]+)',
        r'(?:accuracy|Accuracy|ACC)\s+is\s+([0-9]*\.?[0-9]+)',
        r'(?:accuracy|Accuracy|ACC)\s+of\s+([0-9]*\.?[0-9]+)',
        r'(?:test\s+accuracy|Test\s+Accuracy)\s*[:=]\s*([0-9]*\.?[0-9]+)',
        r'(?:mean\s+accuracy|Mean\s+Accuracy)\s*[:=]\s*([0-9]*\.?[0-9]+)',
        r'(?:score|Score)\s*[:=]\s*\[([0-9.,\s]+)\]',  # CV scores array
        r'(?:cv\s+accuracy|CV\s+Accuracy)\s*[:=]\s*([0-9]*\.?[0-9]+)',
        r'Average\s+Score:\s*([0-9]*\.?[0-9]+)',
        r'(?:Best\s+)?(?:Score|score)\s*[:=]\s*([0-9]*\.?[0-9]+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1).strip().rstrip(','))
            if val <= 1.0:
                return val
            elif 1.0 < val <= 100.0:
                return val / 100.0

    # Strategy 3: Look for percentage
    m = re.search(r'([0-9]*\.?[0-9]+)\s*%', text)
    if m:
        return float(m.group(1)) / 100.0

    # Strategy 4: Look for cross_val_score array and average
    m = re.search(r'Score:\s*\[([0-9.,\s]+)\]', text)
    if m:
        scores = [float(x.strip()) for x in m.group(1).split(',') if x.strip()]
        if scores:
            avg = sum(scores) / len(scores)
            if 0.0 <= avg <= 1.0:
                return avg

    return None


def parse_mse(text: str) -> Optional[float]:
    """Extract MSE value from execution output text."""
    if not text:
        return None

    # Strategy 1: Structured marker
    m = re.search(r'FINAL_MSE:\s*([0-9]*\.?[0-9]+)', text)
    if m:
        return float(m.group(1))

    # Strategy 2: Common patterns
    patterns = [
        r'(?:MSE|Mean\s+Squared\s+Error|mse)\s*[:=]\s*([0-9]*\.?[0-9]+)',
        r'(?:MSE|mse)\s+is\s+([0-9]*\.?[0-9]+)',
        r'(?:MSE|mse)\s+of\s+([0-9]*\.?[0-9]+)',
        r'(?:test\s+MSE|Test\s+MSE)\s*[:=]\s*([0-9]*\.?[0-9]+)',
        r'(?:mean\s+MSE|Mean\s+MSE)\s*[:=]\s*([0-9]*\.?[0-9]+)',
        r'(?:negative\s+)?(?:mean\s+)?(?:squared_error| MSE)\s*[:=]?\s*-?([0-9]*\.?[0-9]+)',
        r'Average\s+Score:\s*-?([0-9]*\.?[0-9]+)',  # neg_mean_squared_error
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1).strip())
            if val >= 0:
                return val

    return None


def parse_knowledge_score(text: str, error_occurred: bool) -> float:
    """Score knowledge integration tasks: 0, 0.5, 0.8, or 1.0"""
    if error_occurred:
        if not text or len(text.strip()) == 0:
            return 0.0  # code error and execution error
        return 0.5  # code error but something ran
    # No error - check if output makes sense
    if text and len(text.strip()) > 0:
        return 1.0
    return 0.8  # code successful but execution error due to env
