"""Persistent session identity is independent of short-lived motion tracks."""
import numpy as np

def normalize(feature):
    if feature is None:
        return None
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 1e-8 else None

def similarity(feature, person):
    if feature is None or not person.gallery:
        return -1.
    # Median prevents one contaminated sample from establishing an identity.
    return float(np.median([np.dot(feature, known) for known in person.gallery]))

def rank(feature, people):
    return sorted(((similarity(feature, p), p.person_id) for p in people), reverse=True)

def select(feature, people, threshold=.45, margin=.08):
    scores = rank(feature, people)
    if not scores:
        return None
    top, pid = scores[0]
    second = scores[1][0] if len(scores)>1 else -1.
    return pid if top >= threshold and top-second >= margin else None
