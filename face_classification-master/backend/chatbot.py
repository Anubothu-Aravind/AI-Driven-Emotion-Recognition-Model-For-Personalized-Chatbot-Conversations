"""
Emotion-Aware Chatbot
=====================
Uses the Groq API (LLaMA 3) to generate chatbot responses.
The chatbot's tone is silently adapted based on the user's dominant
facial emotion — the user is NEVER told they "look sad/angry/etc."

Tone mapping:
  angry    → calm, de-escalating, understanding
  sad      → warm, empathetic, gently encouraging
  fear     → reassuring, supportive, calming
  happy    → positive, energetic, enthusiastic
  surprise → engaging, curious, informative
  disgust  → neutral, respectful, helpful
  neutral  → professional, balanced, helpful
"""

import os
import threading
from groq import Groq

# ── Groq client ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client: Groq | None = None


def _get_client() -> Groq:
    global client
    if client is None:
        api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Set it in your .env file or as an environment variable."
            )
        client = Groq(api_key=api_key)
    return client


# ──────────────────────────────────────────────
#  System prompts per emotion (SILENT adaptation)
# ──────────────────────────────────────────────
_CORE_RULES = (
    "IMPORTANT RULES YOU MUST FOLLOW STRICTLY:\n"
    "- NEVER mention emotions, mood, face, camera, analysis, detection, or AI reasoning.\n"
    "- NEVER say phrases like 'I see you are...', 'You seem...', 'I notice you...', "
    "'I can tell...', or 'It looks like you are...'.\n"
    "- Your behavior must feel natural and human — never robotic or clinical.\n"
    "- Adapt your tone silently and subtly.\n"
    "- Responses must match the emotional context but remain relevant to what the user says.\n"
    "- Focus on WHAT the user says, not WHY they feel a certain way.\n"
    "- Keep replies natural, emotionally aligned, and conversational.\n"
    "- Avoid emojis unless the user uses them first.\n"
    "- Do not overdo emotional language — adaptation should feel subtle, not obvious.\n"
    "- Your goal is to make the conversation feel comfortable, supportive, "
    "and emotionally aligned without revealing how you adapted your tone.\n"
)

SYSTEM_PROMPTS: dict[str, str] = {
    "angry": (
        "You are an adaptive conversational AI assistant.\n\n"
        + _CORE_RULES +
        "\nBEHAVIOR GUIDELINES:\n"
        "- Be calm, polite, and reassuring.\n"
        "- De-escalate tension through respectful, patient language.\n"
        "- Avoid arguing, sounding defensive, or being confrontational.\n"
        "- Use phrases that bring stability and patience.\n"
        "- Validate the user's concerns without being dismissive.\n"
        "- Keep your responses grounded, helpful, and constructive.\n"
    ),
    "sad": (
        "You are an adaptive conversational AI assistant.\n\n"
        + _CORE_RULES +
        "\nBEHAVIOR GUIDELINES:\n"
        "- Be gentle, caring, and emotionally supportive.\n"
        "- Use hopeful and comforting language.\n"
        "- Encourage positivity without sounding preachy or dismissive.\n"
        "- Be warm and understanding in every response.\n"
        "- Offer thoughtful, caring replies that feel human.\n"
        "- Do NOT sound clinical, robotic, or overly formal.\n"
    ),
    "fear": (
        "You are an adaptive conversational AI assistant.\n\n"
        + _CORE_RULES +
        "\nBEHAVIOR GUIDELINES:\n"
        "- Be grounding, reassuring, and steady.\n"
        "- Use clear, simple explanations to reduce uncertainty.\n"
        "- Help the user feel safe and supported.\n"
        "- Avoid sudden, intense, or alarming language.\n"
        "- Respond with confidence and provide structured, helpful information.\n"
        "- Keep your tone calm and stable throughout.\n"
    ),
    "happy": (
        "You are an adaptive conversational AI assistant.\n\n"
        + _CORE_RULES +
        "\nBEHAVIOR GUIDELINES:\n"
        "- Be upbeat, warm, friendly, and positive.\n"
        "- Use light encouragement and enthusiasm.\n"
        "- Maintain a cheerful conversational flow.\n"
        "- Match the positive energy naturally.\n"
        "- Avoid being overly serious or formal.\n"
        "- Keep the conversation lively and engaging.\n"
    ),
    "surprise": (
        "You are an adaptive conversational AI assistant.\n\n"
        + _CORE_RULES +
        "\nBEHAVIOR GUIDELINES:\n"
        "- Be engaging, responsive, and dynamic.\n"
        "- Match curiosity and interest naturally.\n"
        "- Acknowledge novelty without overreacting.\n"
        "- Keep the tone adaptive and flexible.\n"
        "- Share interesting details and keep the conversation stimulating.\n"
    ),
    "disgust": (
        "You are an adaptive conversational AI assistant.\n\n"
        + _CORE_RULES +
        "\nBEHAVIOR GUIDELINES:\n"
        "- Be neutral, composed, and respectful.\n"
        "- Do not amplify negativity in any way.\n"
        "- Redirect gently to constructive conversation.\n"
        "- Keep your tone professional and measured.\n"
        "- Focus on being helpful and providing useful information.\n"
    ),
    "neutral": (
        "You are an adaptive conversational AI assistant.\n\n"
        + _CORE_RULES +
        "\nBEHAVIOR GUIDELINES:\n"
        "- Use a professional, balanced, and helpful tone.\n"
        "- Stay informative and conversational.\n"
        "- Be friendly but not overly casual.\n"
        "- Provide clear, well-structured responses.\n"
    ),
}


# ──────────────────────────────────────────────
#  Conversation history (per-session, in-memory)
# ──────────────────────────────────────────────
conversation_history: list[dict] = []
_history_lock = threading.Lock()
MAX_HISTORY = 20  # keep last N messages to stay within token limits


def reset_conversation():
    """Clear conversation history."""
    global conversation_history
    with _history_lock:
        conversation_history = []


def get_chat_response(user_message: str, dominant_emotion: str) -> str:
    """
    Generate a chatbot response using Groq LLaMA 3.

    Args:
        user_message:     The user's chat message.
        dominant_emotion: The dominant facial emotion (from EmotionManager).

    Returns:
        The chatbot's response string.
    """
    # Normalize emotion
    emotion = dominant_emotion.lower().strip()
    if emotion not in SYSTEM_PROMPTS:
        emotion = "neutral"

    system_prompt = SYSTEM_PROMPTS[emotion]

    with _history_lock:
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_message})

        # Trim history if too long
        if len(conversation_history) > MAX_HISTORY:
            conversation_history[:] = conversation_history[-MAX_HISTORY:]

        # Build messages array
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(list(conversation_history))

    try:
        groq_client = _get_client()
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=512,
            top_p=0.9,
        )
        assistant_message = response.choices[0].message.content.strip()
    except Exception as e:
        assistant_message = (
            "I'm here to help! Could you please try again in a moment? "
            f"(Service temporarily unavailable: {type(e).__name__})"
        )

    # Save assistant reply to history
    with _history_lock:
        conversation_history.append({"role": "assistant", "content": assistant_message})

    return assistant_message
