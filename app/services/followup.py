"""
Post-call follow-up generator.

Uses GPT-4o to write a personalised WhatsApp message that:
- References what the customer ACTUALLY said (not a template)
- Quotes their exact budget, products, features, timeline
- Reads like a human wrote it after a real call
- Is appropriately toned for their lead tier (HOT/WARM/COLD)
- Written in the language the call was conducted in

This is what separates this system from a template-paste bot.
The follow-up proves the system actually listened.
"""
from __future__ import annotations

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.models.call import CallState, Language, LeadTier

logger = get_logger(__name__)

# ─── Follow-up prompt templates ──────────────────────────────────────────────

_FOLLOWUP_SYSTEM_PROMPT = """
You are writing a WhatsApp follow-up message on behalf of Priya from ElevateBox,
a web development agency in Banjara Hills, Hyderabad.

You just finished a sales call. Write a personalised follow-up WhatsApp message.

RULES — these are mandatory:
1. Write in {language_instruction}
2. Reference SPECIFIC things the customer said — their exact budget, what they sell,
   features they asked about, their timeline. Do NOT write a generic message.
3. Write it like a real person — warm, friendly, professional. NOT a template.
4. Keep it under 250 words.
5. NO markdown headers. WhatsApp bold (*word*) is fine.
6. End with a clear next step based on their tier:
   - HOT: "I'll call you shortly to confirm the start date"
   - WARM: "When would be a good time to connect again?"
   - COLD: "No pressure — whenever you're ready, I'm here"
7. Do NOT include phone number or signature — that's added separately.
8. Do NOT say "As per our conversation" or any stiff corporate phrase.
9. Sound like you genuinely remember and care about what they said.
"""

_FOLLOWUP_USER_PROMPT = """
Call details:
- Customer name: {caller_name}
- Lead tier: {tier}
- Language of call: {language}
- What they sell: {product_type}
- Approx product count: {product_count}
- Budget mentioned (exact words): {budget_mentioned}
- Timeline: {timeline}
- Features requested: {features}
- Main barrier (if any): {barrier}
- Key quotes from the customer (exact phrases): {key_quotes}
- Callback booked: {callback_booked}
- Callback time: {callback_time}

Full call transcript excerpt (last 10 turns):
---
{transcript_excerpt}
---

Write the WhatsApp follow-up message now.
"""

_LANGUAGE_INSTRUCTIONS: dict[Language, str] = {
    Language.ENGLISH: "clear, warm Indian English",
    Language.HINDI: "Hindi (Hinglish is natural — mix English product terms freely)",
    Language.TELUGU: "Telugu (mix English product terms naturally where appropriate)",
    Language.UNKNOWN: "clear, warm Indian English",
}


async def generate_followup_message(call_state: CallState) -> str:
    """
    Generate a personalised post-call WhatsApp follow-up message using GPT-4o.

    Returns the message body as a string.
    Falls back to a generic template if the LLM call fails.
    """
    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    disc = call_state.discovery
    score = call_state.score
    lang = call_state.detected_language

    language_instruction = _LANGUAGE_INSTRUCTIONS.get(lang, _LANGUAGE_INSTRUCTIONS[Language.UNKNOWN])

    system_prompt = _FOLLOWUP_SYSTEM_PROMPT.format(
        language_instruction=language_instruction
    )

    user_prompt = _FOLLOWUP_USER_PROMPT.format(
        caller_name=disc.caller_name or "the customer",
        tier=score.tier.value.upper(),
        language=lang.value,
        product_type=disc.product_type or "not specified",
        product_count=disc.product_count or "not specified",
        budget_mentioned=disc.budget_mentioned or "not mentioned",
        timeline=disc.timeline or "not specified",
        features=", ".join(disc.features_requested) if disc.features_requested else "not specified",
        barrier=disc.barrier or "none",
        key_quotes=_format_key_quotes(score.signals_detected),
        callback_booked="Yes" if disc.callback_requested else "No",
        callback_time=disc.callback_time_raw or "N/A",
        transcript_excerpt=call_state.get_recent_transcript(last_n=10),
    )

    logger.info(
        "generating_followup_message",
        tier=score.tier.value,
        language=lang.value,
        transcript_turns=call_state.turn_count,
    )

    try:
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=400,
        )
        message = response.choices[0].message.content or ""
        message = message.strip()

        logger.info(
            "followup_message_generated",
            length=len(message),
            tier=score.tier.value,
        )
        return message

    except Exception as exc:
        logger.error("followup_generation_failed", error=str(exc))
        return _fallback_message(call_state)


def _format_key_quotes(signals: list[str]) -> str:
    """Format signals as readable quotes for the LLM prompt."""
    if not signals:
        return "none captured"
    return "; ".join(f'"{s}"' for s in signals[:5])


def _fallback_message(call_state: CallState) -> str:
    """
    Fallback template if LLM generation fails.
    Uses whatever discovery data we have.
    """
    disc = call_state.discovery
    tier = call_state.score.tier
    name = disc.caller_name or "there"

    if tier == LeadTier.HOT:
        return (
            f"Hi {name}! It was great speaking with you.\n\n"
            f"As discussed, we'd love to build your e-commerce website. "
            f"Based on what you shared, we can have it ready within your timeline.\n\n"
            f"I'll follow up shortly to get started!"
        )
    elif tier == LeadTier.WARM:
        return (
            f"Hi {name}! Thanks for speaking with me today.\n\n"
            f"I've noted down your requirements. Whenever you're ready to move forward, "
            f"we're here to help build your e-commerce website.\n\n"
            f"Feel free to reach out anytime!"
        )
    else:
        return (
            f"Hi {name}! Thanks for your time today.\n\n"
            f"Whenever you're thinking of building your website, ElevateBox is here.\n\n"
            f"I'm sending over our portfolio for reference. No pressure at all!"
        )
