"""
landmark_features.py — Shared feature engineering (supports 1 or 2 hands)

Feature vector breakdown:
  Per hand (applied to each detected hand):
    1.  Normalised relative XYZ coordinates     → 63
    2.  Tip-to-base distances (all combos)       → 25
    3.  Tip-to-tip distances                     → 10
    4.  Tip-to-wrist distances                   → 5
    5.  Inter-tip cosine angles                  → 10
    6.  Finger curl ratios (MCP→PIP→DIP)         → 5
    7.  Palm normal vector (orientation)         → 3
                                           Total → 121 per hand

  Two-hand feature vector:
    hand_0 (121) + hand_1 (121) + cross-hand wrist distance (1) = 243

  One-hand feature vector:
    hand (121) + zero-padded second slot (121) + placeholder (1) = 243

  Model always receives exactly 243 features — consistent for training & inference.
"""

import numpy as np
from itertools import combinations

# MediaPipe hand landmark indices
WRIST        = 0
FINGER_TIPS  = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
FINGER_BASES = [2, 5,  9, 13, 17]   # thumb, index, middle, ring, pinky bases

# MCP → PIP → DIP triplets for curl ratio (per finger)
CURL_TRIPLETS = [
    (1,  2,  3),   # thumb:  CMC → MCP → IP
    (5,  6,  7),   # index:  MCP → PIP → DIP
    (9,  10, 11),  # middle: MCP → PIP → DIP
    (13, 14, 15),  # ring:   MCP → PIP → DIP
    (17, 18, 19),  # pinky:  MCP → PIP → DIP
]

PALM_PLANE        = (0, 5, 17)   # wrist, index MCP, pinky MCP
FEATURES_PER_HAND = 121          # 63+25+10+5+10+5+3
TOTAL_FEATURES    = 243          # 121*2 + 1


def _normalise(pts: np.ndarray) -> np.ndarray:
    """Translate to wrist origin and normalise by hand span."""
    pts = pts - pts[WRIST]
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-9
    return pts / scale


def _curl_ratio(pts: np.ndarray, a: int, b: int, c: int) -> float:
    """
    Curl ratio for joint triplet A→B→C.
    cos(angle at B): +1 = fully straight, -1 = fully curled.
    """
    v1 = pts[a] - pts[b]
    v2 = pts[c] - pts[b]
    n1 = np.linalg.norm(v1) + 1e-9
    n2 = np.linalg.norm(v2) + 1e-9
    return float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))


def _palm_normal(pts: np.ndarray) -> np.ndarray:
    """Unit normal to palm plane (wrist, index-MCP, pinky-MCP)."""
    a, b, c = PALM_PLANE
    v1 = pts[b] - pts[a]
    v2 = pts[c] - pts[a]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal) + 1e-9
    return normal / norm


def _single_hand_features(raw_21x3: np.ndarray) -> np.ndarray:
    """Extract 121 features from one hand's 21 landmarks (shape 21×3 or flat 63)."""
    pts = _normalise(raw_21x3.reshape(21, 3).astype(np.float64))
    features = []

    # 1. Normalised XYZ (63)
    features.extend(pts.flatten())

    # 2. Tip-to-base distances 5×5 = 25
    for tip in FINGER_TIPS:
        for base in FINGER_BASES:
            features.append(np.linalg.norm(pts[tip] - pts[base]))

    # 3. Tip-to-tip distances C(5,2) = 10
    for i, j in combinations(FINGER_TIPS, 2):
        features.append(np.linalg.norm(pts[i] - pts[j]))

    # 4. Tip-to-wrist distances (5)
    for tip in FINGER_TIPS:
        features.append(np.linalg.norm(pts[tip]))

    # 5. Inter-tip cosine angles C(5,2) = 10
    for i, j in combinations(FINGER_TIPS, 2):
        v1 = pts[i] / (np.linalg.norm(pts[i]) + 1e-9)
        v2 = pts[j] / (np.linalg.norm(pts[j]) + 1e-9)
        features.append(float(np.clip(np.dot(v1, v2), -1.0, 1.0)))

    # 6. Finger curl ratios (5)
    for a, b, c in CURL_TRIPLETS:
        features.append(_curl_ratio(pts, a, b, c))

    # 7. Palm normal vector (3)
    features.extend(_palm_normal(pts).tolist())

    if len(features) != FEATURES_PER_HAND:
        raise RuntimeError(
            f"Feature count mismatch: expected {FEATURES_PER_HAND}, got {len(features)}"
        )
    return np.array(features, dtype=np.float64)


def extract_features(raw_row: np.ndarray) -> np.ndarray:
    """
    Public API — accepts EITHER:
      • flat 63-element array  → single hand
      • flat 126-element array → two hands (hand_0 then hand_1)

    Always returns a 243-D vector for consistent model input.
    """
    raw = np.asarray(raw_row, dtype=np.float64).flatten()

    if raw.size == 63:
        h0_feat = _single_hand_features(raw)
        h1_feat = np.zeros(FEATURES_PER_HAND, dtype=np.float64)
        cross   = np.array([0.0])

    elif raw.size == 126:
        h0_raw  = raw[:63]
        h1_raw  = raw[63:]
        h0_feat = _single_hand_features(h0_raw)
        h1_feat = _single_hand_features(h1_raw)
        wrist0  = h0_raw.reshape(21, 3)[WRIST]
        wrist1  = h1_raw.reshape(21, 3)[WRIST]
        cross   = np.array([np.linalg.norm(wrist0 - wrist1)])

    else:
        raise ValueError(
            f"raw_row must have 63 (1 hand) or 126 (2 hands) values, got {raw.size}"
        )

    result = np.concatenate([h0_feat, h1_feat, cross])
    if result.size != TOTAL_FEATURES:
        raise RuntimeError(
            f"Total feature mismatch: expected {TOTAL_FEATURES}, got {result.size}"
        )
    return result
