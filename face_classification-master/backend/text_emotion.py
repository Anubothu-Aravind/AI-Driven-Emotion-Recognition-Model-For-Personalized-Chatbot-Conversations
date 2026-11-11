"""
Text Emotion Analyzer (Optional Enhancement)
=============================================
Uses simple keyword/rule-based analysis to detect emotion
from the user's chat text as a SECONDARY signal.
The primary signal is always the webcam face emotion.

This supplements the facial emotion when:
  - The user writes something clearly emotional ("I'm so frustrated!")
  - The webcam temporarily loses the face

Falls back to facial emotion if text is ambiguous.
"""

from typing import Optional


# ── Keyword dictionaries for rule-based text emotion ──
EMOTION_KEYWORDS: dict[str, list[str]] = {
    "happy": [
        "happy", "glad", "joy", "excited", "great", "wonderful", "amazing",
        "love", "awesome", "fantastic", "excellent", "brilliant", "yay",
        "haha", "lol", "😊", "😄", "🎉", "❤️", "good", "nice", "best",
        "thrilled", "delighted", "cheerful", "elated", "pleased",
    ],
    "sad": [
        "sad", "depressed", "unhappy", "miserable", "crying", "tears",
        "heartbroken", "lonely", "hopeless", "gloomy", "down", "terrible",
        "awful", "worst", "😢", "😭", "💔", "upset", "grief", "sorrow",
        "disappointed", "broken", "hurt", "pain",
    ],
    "angry": [
        "angry", "furious", "mad", "annoyed", "irritated", "frustrated",
        "hate", "rage", "outraged", "pissed", "stupid", "idiot", "damn",
        "hell", "😡", "🤬", "ugh", "sick of", "fed up", "enough",
    ],
    "fear": [
        "scared", "afraid", "fear", "terrified", "anxious", "worried",
        "nervous", "panic", "dread", "frightened", "horror", "scary",
        "😰", "😨", "😱", "stress", "stressed", "tense", "overwhelmed",
    ],
    "surprise": [
        "surprised", "shocked", "wow", "omg", "no way",
        "unbelievable", "incredible", "unexpected", "😮", "😲", "🤯",
        "can't believe", "seriously", "whoa",
    ],
    "disgust": [
        "disgusting", "gross", "nasty", "eww", "yuck", "revolting",
        "repulsive", "horrible", "🤢", "🤮", "sick",
    ],
}


def analyze_text_emotion(text: str) -> Optional[str]:
    """
    Analyze user text for emotional keywords.
    Returns the detected emotion string or None if ambiguous/neutral.
    """
    if not text or not text.strip():
        return None

    text_lower = text.lower()
    scores: dict[str, int] = {}

    for emotion, keywords in EMOTION_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[emotion] = count

    if not scores:
        return None  # No emotional keywords found → use facial emotion

    # Return the emotion with the highest keyword match count
    return max(scores, key=scores.get)


def get_combined_emotion(facial_emotion: str, text: str) -> str:
    """
    Combine facial emotion (primary) with text emotion (secondary).

    Priority rules:
      1. If text has a STRONG emotional signal (2+ keyword hits) → use text emotion
      2. Otherwise → use facial emotion (always available from webcam)
      3. Default → "neutral"
    """
    text_emotion = analyze_text_emotion(text)

    if text_emotion is None:
        return facial_emotion or "neutral"

    # Count how many keywords matched for the text emotion
    text_lower = text.lower()
    hit_count = sum(1 for kw in EMOTION_KEYWORDS.get(text_emotion, []) if kw in text_lower)

    # If text has a strong signal (2+ matches), let it override
    if hit_count >= 2:
        return text_emotion

    # Otherwise, facial emotion is primary
    return facial_emotion or "neutral"
