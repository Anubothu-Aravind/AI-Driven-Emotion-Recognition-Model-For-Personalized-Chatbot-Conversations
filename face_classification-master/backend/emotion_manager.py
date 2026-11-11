"""
Emotion Manager - 1-Minute Emotion Aggregator
==============================================
Collects per-frame emotion labels from the webcam emotion detector.
Every 60 seconds, computes the dominant (most frequent) emotion,
stores it as the "current dominant emotion", and resets the counters.

Thread-safe: uses a Lock so the FastAPI server and the webcam
process can read/write concurrently without race conditions.
"""

import time
import threading
from collections import Counter


class EmotionManager:
    """Singleton-style emotion aggregator with 60-second window."""

    # ── valid emotions from fer2013 ──
    VALID_EMOTIONS = {"angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"}

    def __init__(self, interval_seconds: int = 60):
        self._interval = interval_seconds          # aggregation window
        self._lock = threading.Lock()
        self._counter: Counter = Counter()          # raw frame counts
        self._dominant_emotion: str = "neutral"     # last computed dominant
        self._window_start: float = time.time()     # current window start
        self._total_frames: int = 0                 # frames in current window
        self._history: list[dict] = []              # optional: history log

        # ── Confidence tracking ──
        self._latest_confidence: float = 0.0        # most recent frame confidence
        self._confidence_sum: float = 0.0            # sum for averaging
        self._confidence_count: int = 0              # count for averaging

        # ── Emoji override state ──
        self._emoji_emotion: str | None = None       # manual emoji override

    # ──────────────────────────────────────────────
    #  Called by the webcam/emotion detector every frame
    # ──────────────────────────────────────────────
    def record_emotion(self, emotion: str, confidence: float = 0.0) -> None:
        """Record a single frame's detected emotion and confidence."""
        emotion = emotion.lower().strip()
        if emotion not in self.VALID_EMOTIONS:
            return

        with self._lock:
            self._counter[emotion] += 1
            self._total_frames += 1
            self._latest_confidence = confidence
            self._confidence_sum += confidence
            self._confidence_count += 1

            # Check if the 60-second window has elapsed
            elapsed = time.time() - self._window_start
            if elapsed >= self._interval:
                self._compute_dominant()

    # ──────────────────────────────────────────────
    #  Compute dominant emotion & reset window
    # ──────────────────────────────────────────────
    def _compute_dominant(self) -> None:
        """Must be called while holding self._lock."""
        if self._total_frames > 0:
            dominant = self._counter.most_common(1)[0][0]
            self._dominant_emotion = dominant

            avg_conf = (
                self._confidence_sum / self._confidence_count
                if self._confidence_count > 0 else 0.0
            )

            # Save to history (optional but useful for debugging)
            self._history.append({
                "timestamp": time.time(),
                "dominant": dominant,
                "counts": dict(self._counter),
                "total_frames": self._total_frames,
                "avg_confidence": round(avg_conf, 4),
            })

        # Reset for next window
        self._counter.clear()
        self._total_frames = 0
        self._confidence_sum = 0.0
        self._confidence_count = 0
        self._window_start = time.time()

    # ──────────────────────────────────────────────
    #  Public getters (called by chatbot / API)
    # ──────────────────────────────────────────────
    def get_current_emotion(self) -> str:
        """Return the latest dominant emotion (computed every 60s)."""
        with self._lock:
            return self._dominant_emotion

    def get_latest_confidence(self) -> float:
        """Return the most recent frame's emotion confidence score."""
        with self._lock:
            return self._latest_confidence

    def get_avg_confidence(self) -> float:
        """Return the average confidence in the current window."""
        with self._lock:
            if self._confidence_count == 0:
                return 0.0
            return self._confidence_sum / self._confidence_count

    # ── Emoji override ──
    def set_emoji_emotion(self, emotion: str | None) -> None:
        """Set or clear the emoji override emotion."""
        with self._lock:
            if emotion is None:
                self._emoji_emotion = None
            else:
                emotion = emotion.lower().strip()
                if emotion in self.VALID_EMOTIONS:
                    self._emoji_emotion = emotion

    def get_emoji_emotion(self) -> str | None:
        """Return the current emoji override, or None."""
        with self._lock:
            return self._emoji_emotion

    def get_effective_emotion(self, confidence_threshold: float = 0.6) -> str:
        """
        Priority-based emotion resolution:
          1. Emoji override (if set)  → use emoji emotion
          2. Face emotion (if confidence >= threshold) → use face dominant
          3. Fallback → "neutral"
        """
        with self._lock:
            if self._emoji_emotion is not None:
                return self._emoji_emotion
            if self._latest_confidence >= confidence_threshold:
                return self._dominant_emotion
            return "neutral"

    def get_live_counts(self) -> dict:
        """Return real-time frame counts in the current window (for debug UI)."""
        with self._lock:
            elapsed = round(time.time() - self._window_start, 1)
            avg_conf = (
                self._confidence_sum / self._confidence_count
                if self._confidence_count > 0 else 0.0
            )
            return {
                "dominant_emotion": self._dominant_emotion,
                "current_window_counts": dict(self._counter),
                "total_frames_in_window": self._total_frames,
                "seconds_elapsed": elapsed,
                "window_size_seconds": self._interval,
                "latest_confidence": round(self._latest_confidence, 4),
                "avg_confidence": round(avg_conf, 4),
                "emoji_override": self._emoji_emotion,
                "effective_emotion": (
                    self._emoji_emotion
                    if self._emoji_emotion is not None
                    else (self._dominant_emotion if self._latest_confidence >= 0.6 else "neutral")
                ),
            }

    def get_history(self) -> list[dict]:
        """Return the full aggregation history."""
        with self._lock:
            return list(self._history)

    def force_compute(self) -> str:
        """Force an immediate dominant computation (useful for testing)."""
        with self._lock:
            self._compute_dominant()
            return self._dominant_emotion


# ──────────────────────────────────────────────
#  Global singleton instance
# ──────────────────────────────────────────────
emotion_manager = EmotionManager(interval_seconds=60)
