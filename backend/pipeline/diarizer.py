"""
Speaker diarization — lightweight, no API key / no model download.

Goal: tag each segment with a *speaker* identity so the UI can assign one voice
per speaker (Man 1 / Man 2 / Man 3 / Girl) and have it apply to every line that
speaker says.

Approach (matches the project's "free, no heavy deps" bias):
  1. Load the already-extracted mono WAV once, slice it per segment.
  2. Per segment compute a voice fingerprint: MFCC mean/std + mean pitch (f0).
  3. Decide gender from pitch (same threshold idea as tts._detect_gender_*).
  4. Cluster the MALE segments into up to 3 groups (KMeans, k chosen by
     silhouette) → man_1 / man_2 / man_3, ordered deepest-first by pitch.
  5. All FEMALE segments → a single `girl` speaker.

This is "good enough" for clearly different voices; it can confuse very similar
male voices — see CLAUDE.md note on the lightweight method. Everything is wrapped
so a failure degrades to gender-only labelling rather than breaking extraction.
"""
import numpy as np
from typing import List, Dict, Any

# Mean f0 below this (Hz) is treated as male. Mirrors tts._detect_gender_from_segment.
GENDER_F0_THRESHOLD = 165.0
# Max distinct male speakers to separate (the "Man 1/2/3" buckets).
MAX_MALE_SPEAKERS = 3
# Minimum silhouette score to accept a multi-speaker split; below this we assume
# a single speaker (avoids inventing fake speakers when voices are similar).
MIN_SILHOUETTE = 0.12


def _segment_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Voice fingerprint for one audio slice: (feature_vector, mean_f0)."""
    import librosa
    # MFCCs capture timbre (who is speaking); mean+std over time → fixed vector.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
    f0 = f0[~np.isnan(f0)]
    pitch = float(np.mean(f0)) if len(f0) else 0.0
    return {"feat": feat, "pitch": pitch}


def _cluster_labels(feats: np.ndarray, max_k: int) -> np.ndarray:
    """
    KMeans cluster the feature rows into the best 1..max_k groups.
    Returns an int label per row. Picks k by silhouette; falls back to a single
    cluster (all zeros) if no split is confidently better than one speaker.
    """
    n = len(feats)
    if n <= 1 or max_k <= 1:
        return np.zeros(n, dtype=int)

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as e:
        # No sklearn → can't split; treat all males as one speaker (man_1).
        print(f"[diarizer] sklearn unavailable, single male speaker: {e}")
        return np.zeros(n, dtype=int)

    X = StandardScaler().fit_transform(feats)

    best_labels = np.zeros(n, dtype=int)
    best_score = -1.0
    for k in range(2, min(max_k, n) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score, best_labels = score, labels

    return best_labels if best_score >= MIN_SILHOUETTE else np.zeros(n, dtype=int)


def assign_speakers(audio_path: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Tag each segment with `speaker` (man_1/man_2/man_3/girl) and `gender`.

    Non-destructive: returns new dicts. On any failure (missing libs, unreadable
    audio) falls back to per-segment gender only, leaving `speaker` unset so the
    pipeline keeps working.
    """
    if not segments:
        return segments
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        print(f"[diarizer] audio load failed, skipping diarization: {e}")
        return segments

    # 1) Per-segment fingerprint + gender.
    male_idx: List[int] = []
    info: List[Dict[str, Any]] = []
    for seg in segments:
        a = int(max(0.0, float(seg.get("start", 0.0))) * sr)
        b = int(max(0.0, float(seg.get("end", 0.0))) * sr)
        clip = y[a:b]
        if len(clip) < int(0.1 * sr):           # too short to fingerprint
            info.append({"feat": None, "pitch": 0.0, "gender": "female"})
            continue
        try:
            f = _segment_features(clip, sr)
        except Exception as e:
            print(f"[diarizer] feature extraction failed for a segment: {e}")
            info.append({"feat": None, "pitch": 0.0, "gender": "female"})
            continue
        gender = "male" if 0 < f["pitch"] < GENDER_F0_THRESHOLD else "female"
        info.append({**f, "gender": gender})

    for i, it in enumerate(info):
        if it["gender"] == "male" and it["feat"] is not None:
            male_idx.append(i)

    # 2) Cluster male segments into Man 1/2/3.
    out = [dict(seg) for seg in segments]
    if male_idx:
        male_feats = np.vstack([info[i]["feat"] for i in male_idx])
        labels = _cluster_labels(male_feats, MAX_MALE_SPEAKERS)
        # Order clusters deepest-voice-first so labels are stable across runs:
        # man_1 = lowest mean pitch.
        cluster_pitch: Dict[int, List[float]] = {}
        for j, idx in enumerate(male_idx):
            cluster_pitch.setdefault(int(labels[j]), []).append(info[idx]["pitch"])
        order = sorted(cluster_pitch, key=lambda c: np.mean(cluster_pitch[c]))
        rank = {c: r for r, c in enumerate(order)}
        for j, idx in enumerate(male_idx):
            out[idx]["speaker"] = f"man_{rank[int(labels[j])] + 1}"
            out[idx]["gender"] = "male"

    # 3) All female (or unfingerprinted) segments → a single Girl speaker.
    for i, it in enumerate(info):
        if "speaker" not in out[i]:
            out[i]["speaker"] = "girl"
            out[i]["gender"] = "female"

    n_men = len({out[i]["speaker"] for i in male_idx}) if male_idx else 0
    has_girl = any(o.get("speaker") == "girl" for o in out)
    print(f"[diarizer] {len(segments)} segments → {n_men} male speaker(s)"
          f"{' + girl' if has_girl else ''}")
    return out


# Human-readable labels for the speaker ids this module produces.
SPEAKER_LABELS = {
    "man_1": "Man 1",
    "man_2": "Man 2",
    "man_3": "Man 3",
    "girl":  "Girl",
}
