"""Prototype phrase banks for the zero-shot semantic classifiers.

Each label maps to a handful of natural-language prototypes describing what
that class looks like. The classifier embeds these once and matches input
text against them by cosine similarity — so paraphrases the keyword stubs
would miss ("wire the funds to keep them safe") still land on the right
class. Category codes + tactic names match the backend vocabulary EXACTLY
(vigiscam-backend nlp-stub.ts).
"""

# ── Scam categories (codes mirror the backend) ───────────────────────────────
SCAM_CATEGORIES: dict[str, list[str]] = {
    "BANK_IMPERSONATION": [
        "your bank account has been compromised, move your money to a safe account",
        "this is your bank's fraud department, we need to verify your account",
        "transfer your funds to a secure account we set up for you",
    ],
    "GIFT_CARD_SCAM": [
        "pay using gift cards and read me the code on the back",
        "buy Amazon or Apple gift cards and send me the numbers",
        "settle the fee with a store gift card right now",
    ],
    "REMOTE_ACCESS_SCAM": [
        "install AnyDesk or TeamViewer so I can fix your computer remotely",
        "let me connect to your device to remove the virus",
        "download remote access software and give me the code",
    ],
    "TECH_SUPPORT_SCAM": [
        "we detected a virus on your computer, this is Microsoft support",
        "your device is infected and your warranty needs renewing",
        "I am a technician and your computer is sending error messages",
    ],
    "GOVERNMENT_IMPERSONATION": [
        "this is the IRS, you owe back taxes and will be arrested",
        "the social security administration has suspended your number",
        "you have a warrant, pay the fine immediately to avoid jail",
    ],
    "CRYPTO_SCAM": [
        "send bitcoin to this wallet for a guaranteed investment return",
        "deposit cryptocurrency now to unlock your trading profits",
        "transfer USDT to verify your crypto account",
    ],
    "ROMANCE_SCAM": [
        "I love you but I'm stuck overseas and need money to come home",
        "my darling, send funds for my emergency, we will be together soon",
        "I need help with a hospital bill before we can finally meet",
    ],
}

# ── Manipulation tactics (names mirror the backend) ──────────────────────────
TACTIC_PROTOTYPES: dict[str, list[str]] = {
    "urgency": [
        "act immediately or you will lose access right now",
        "this expires today, you must do it in the next few minutes",
    ],
    "secrecy": [
        "do not tell anyone, keep this between us",
        "don't discuss this with your family or the bank",
    ],
    "fake-authority": [
        "I am calling from the government and you must comply",
        "this is an official agent and refusal is a crime",
    ],
    "pressure": [
        "if you don't pay now there will be serious consequences",
        "you will be arrested or fined unless you cooperate immediately",
    ],
}

# A benign reference set — text that scores high here and low on every scam
# category should get a low scamScore. Anchors the scam-vs-not calibration.
BENIGN_PROTOTYPES: list[str] = [
    "hi, are we still meeting for coffee tomorrow afternoon?",
    "thanks for the update, talk to you next week",
    "the weather has been lovely, the kids enjoyed the park",
    "your order has shipped and will arrive on tuesday",
]

# ── Fraud journey stages (enum mirrors the backend FraudJourneyStage) ────────
JOURNEY_STAGES: dict[str, list[str]] = {
    "INITIAL_CONTACT": [
        "hello, I am reaching out to you for the first time",
        "you have been selected, please respond to continue",
    ],
    "TRUST_BUILDING": [
        "trust me, I am here to help you and protect you",
        "I understand how you feel, I am on your side",
    ],
    "INFORMATION_GATHERING": [
        "can you confirm your social security number and date of birth",
        "what is your account number and online banking password",
    ],
    "URGENCY_INJECTION": [
        "you must act immediately, this cannot wait",
        "right now or you will lose everything, hurry",
    ],
    "PAYMENT_REQUEST": [
        "please send the payment by wire transfer or gift card",
        "transfer the money now to the account I gave you",
    ],
    "COMPLETED": [
        "the transfer is done and the funds have been sent",
        "payment received, the transaction is complete",
    ],
    "INTERVENED": [
        "the bank stopped the transfer and flagged it as fraud",
        "a family member intervened and blocked the payment",
    ],
}

# ── Victim states (enum mirrors the backend VictimStateLabel) ────────────────
VICTIM_STATES: dict[str, list[str]] = {
    "CALM": ["I am fine and thinking about this clearly", "no worries, I understand"],
    "CONFUSED": ["I don't understand what you mean, why is this happening", "I'm confused"],
    "PRESSURED": ["let me think, wait, I'm not sure about this", "I feel rushed and unsure"],
    "TRUSTING": ["thank you, that makes sense, I believe you", "okay I trust what you say"],
    "ALARMED": ["I'm scared and afraid, this is frightening", "I feel threatened and panicked"],
    "COMPROMISED": ["okay I'll do it, I'll send it now", "alright fine, I will make the payment"],
}

# ── Predicted next move (state-machine over journey stage) ───────────────────
NEXT_MOVE_MAP: dict[str, str] = {
    "INITIAL_CONTACT": "REQUEST_PERSONAL_INFO",
    "TRUST_BUILDING": "REQUEST_PERSONAL_INFO",
    "INFORMATION_GATHERING": "ESCALATE_URGENCY",
    "URGENCY_INJECTION": "REQUEST_PAYMENT",
    "PAYMENT_REQUEST": "REQUEST_GIFT_CARD",
    "COMPLETED": "DROP_OFF",
    "INTERVENED": "DROP_OFF",
}
