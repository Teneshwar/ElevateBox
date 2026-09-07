"""
Lead qualification engine — Hot / Warm / Cold classifier.

The LLM classifies in real-time via function calls. This module:
1. Receives the LLM's classification data from the webhook
2. Applies rule-based scoring on top for validation/override
3. Updates the call state with the current tier
4. Decides if the mid-call WhatsApp should fire

Scoring system:
  Score >= 50  → HOT  (fire WhatsApp immediately)
  Score 15-49  → WARM (capture barrier, offer callback)
  Score < 15   → COLD (send brochure, close gracefully)

The LLM sends a score; we validate it against hard-coded signal rules
to prevent the LLM from hallucinating an incorrect tier.
"""
from __future__ import annotations

from app.core.logging import get_logger
from app.models.call import CallState, LeadScore, LeadTier

logger = get_logger(__name__)

# ─── Signal scoring table ─────────────────────────────────────────────────────
# These map to exact or paraphrased phrases. The LLM sends the signals list;
# we score them here for a final validated result.

POSITIVE_SIGNALS: dict[str, int] = {
    # Very high intent
    "asks_start_date": 40,          # "kab start karoge?" / "how soon can you start?"
    "asks_price_explicitly": 35,    # "how much does it cost?" / "price kya hai?"
    "mentions_launch_deadline": 30, # "I need this by Diwali / next month"

    # High intent
    "states_budget": 25,            # Actually mentions a number or range
    "says_send_details": 20,        # "details bhejo" / "send me info"
    "asks_portfolio": 15,           # "show me your previous work"
    "asks_timeline": 15,            # "how long will it take?"
    "mentions_product_count": 10,   # Shows they're thinking concretely
    "confirms_decision_maker": 10,  # "haan main hi decide karunga"

    # Moderate interest
    "asks_features": 8,             # Questions about specific features
    "mentions_competitors": 5,      # "I checked Shopify, too expensive"
    "returning_call": 15,           # They called back / knew about ElevateBox
}

NEGATIVE_SIGNALS: dict[str, int] = {
    # Hard blockers
    "explicit_rejection": -60,       # "no thanks", "not interested"
    "already_has_website": -20,      # "we already have a website"

    # Soft blockers
    "no_budget": -20,                # "budget nahi hai abhi"
    "third_party_decides": -15,      # "bhai / partner decide karega"
    "very_far_timeline": -15,        # "shayad 1 saal baad"
    "just_checking": -15,            # "bas dekhna tha"
    "not_ready": -10,                # "abhi nahi"

    # Mild blockers
    "vague_answers": -5,             # Consistently vague, no concrete details
    "short_responses": -3,           # Monosyllabic / minimal engagement
}


def calculate_score(signals: list[str]) -> int:
    """
    Calculate numeric lead score from signal list.
    Clamps to [-100, 100].
    """
    score = 0
    for signal in signals:
        score += POSITIVE_SIGNALS.get(signal, 0)
        score += NEGATIVE_SIGNALS.get(signal, 0)
    return max(-100, min(100, score))


def determine_tier(score: int) -> LeadTier:
    if score >= 50:
        return LeadTier.HOT
    elif score >= 15:
        return LeadTier.WARM
    else:
        return LeadTier.COLD


def qualify_lead(
    call_state: CallState,
    llm_tier: str,
    llm_score: int,
    signals: list[str],
    rationale: str,
    discovery_data: dict | None = None,
) -> tuple[LeadScore, bool]:
    """
    Process a classify_lead function call from the LLM.

    Returns:
        (updated_lead_score, should_fire_whatsapp)

    should_fire_whatsapp is True when:
      - Tier is HOT
      - Mid-call WhatsApp hasn't been sent yet
      - Turn count is high enough to have real context (>= 3 turns)
    """
    # Rule-based score for validation
    rule_score = calculate_score(signals)

    # Reconcile: use LLM score if it agrees directionally, else use rule score
    # This prevents the LLM from under/over-classifying
    if abs(rule_score - llm_score) <= 20:
        # Close enough — use LLM score (it has more context)
        final_score = llm_score
    else:
        # LLM and rules disagree significantly — trust rules
        logger.warning(
            "llm_and_rule_score_disagree",
            llm_score=llm_score,
            rule_score=rule_score,
            signals=signals,
        )
        # Blend: weighted average (rules 60%, LLM 40%)
        final_score = int(rule_score * 0.6 + llm_score * 0.4)

    tier = determine_tier(final_score)

    # Update discovery data if provided
    if discovery_data:
        disc = call_state.discovery
        disc.budget_mentioned = discovery_data.get("budget_mentioned") or disc.budget_mentioned
        disc.product_type = discovery_data.get("product_type") or disc.product_type
        disc.product_count = discovery_data.get("product_count") or disc.product_count
        disc.timeline = discovery_data.get("timeline") or disc.timeline
        disc.barrier = discovery_data.get("barrier") or disc.barrier
        disc.caller_name = discovery_data.get("caller_name") or disc.caller_name
        disc.decision_maker = discovery_data.get("decision_maker")

        features = discovery_data.get("features_requested", [])
        if features:
            existing = set(disc.features_requested)
            disc.features_requested = list(existing | set(features))

    # Build updated score object
    updated_score = LeadScore(
        score=final_score,
        tier=tier,
        signals_detected=signals,
        rationale=rationale,
        whatsapp_triggered=call_state.score.whatsapp_triggered,
        classified_at_turn=call_state.turn_count,
    )

    # Determine if mid-call WhatsApp should fire
    should_fire_whatsapp = (
        tier == LeadTier.HOT
        and not call_state.mid_call_whatsapp_sent
        and call_state.turn_count >= 3  # Need enough context
    )

    logger.info(
        "lead_classified",
        tier=tier.value,
        score=final_score,
        llm_score=llm_score,
        rule_score=rule_score,
        signals_count=len(signals),
        should_fire_whatsapp=should_fire_whatsapp,
        turn=call_state.turn_count,
    )

    return updated_score, should_fire_whatsapp
