"""
sentence_builder.py  — v3
Converts sign language word sequences into natural daily-use English sentences.

Fixes over v2:
  i angry you        → I am angry at you.        (not "I am angry you")
  you me friend      → You and I are friends.     (not "my friend")
  i want you friend  → I want you to be my friend.(not "Is this your friend?")
  you me go work     → You and I are going to work together.
  i angry he         → I am angry at him.
  subject + angry/sad/hurt + person → "at [person]"

Two-tier:
  1. Claude API  — best quality, handles any combo
  2. Local rules — fallback when offline
"""

import time
import urllib.request
import urllib.error
import json

# ─────────────────────────────────────────────────────────────────────────────
# Claude API
# ─────────────────────────────────────────────────────────────────────────────

API_URL = "https://api.anthropic.com/v1/messages"
MODEL   = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a sign language interpreter converting Sign Language Order (SLO) into natural spoken English.

SLO omits grammar words — you must add them naturally. Output ONE sentence only.

STRICT RULES:
- Output ONLY the final sentence. No explanation, no quotes, no alternatives.
- End with exactly one punctuation mark (. or ? or !)
- Make it sound like something a real person would say in daily conversation.
- Use present continuous for ongoing actions (am/is/are + -ing).
- Use simple present for facts and states (I am happy, I have a problem).

IMPORTANT PATTERNS — follow these exactly:

Emotion + person (angry/sad/hurt + you/he/friend/family):
  i angry you        → I am angry at you.
  i sad you          → I am sad because of you.
  i hurt you         → You hurt me.
  why you angry me   → Why are you angry at me?

Two people together (you/me + action):
  you me friend      → You and I are friends.
  you me go work     → You and I are going to work together.
  you me together    → You and I are together.
  you me eat         → Let us eat together.
  you me go home     → Let us go home together.

Want + person + action:
  i want you come    → I want you to come.
  i want you help    → I want you to help me.
  i want you friend  → I want to be your friend.
  i want he come     → I want him to come.

Actions with destination:
  i go work          → I am going to work.
  i go home          → I am going home.
  he go work         → He is going to work.
  you go home        → Are you going home?

Questions:
  how you            → How are you?
  where you go       → Where are you going?
  what your name     → What is your name?
  when you come      → When are you coming?
  why you sad        → Why are you sad?
  who you            → Who are you?
  what time          → What time is it?
  you understand     → Do you understand?
  you okay           → Are you okay?
  you ready          → Are you ready?
  you come home      → Are you coming home?

Feelings/states:
  i sad              → I am sad.
  i angry            → I am angry.
  i good             → I am good.
  i bad              → I am not feeling well.
  i hurt             → I am hurt.
  i sorry            → I am sorry.
  i ready            → I am ready.
  he sad             → He is sad.
  he angry           → He is angry.

Requests:
  please help        → Please help me.
  please repeat      → Please repeat that.
  please slow        → Please speak slowly.
  please come        → Please come here.
  please stop        → Please stop.

Social:
  hello how you      → Hello! How are you?
  good morning       → Good morning!
  no problem         → No problem.
  thanks             → Thank you.
  i no understand    → I do not understand.
  i want help        → I need help.
  i problem          → I have a problem.
  i phone            → I need my phone.
  i hear you         → I can hear you.
  i see you          → I can see you.
  go together        → Let us go together.
  friend come        → My friend is coming.
  i family together  → I am with my family.
  you me friend      → You and I are friends.
  you me go work together → You and I are going to work together.
  i want you friend  → I want to be your friend.
  i want you come    → I want you to come.

The 49 signs: again, angry, bad, bye, clear, come, drink, eat, family, friend,
go, good, he, hear, hello, help, home, how, hurt, i, me, morning, name, no,
phone, please, problem, ready, sad, see, sleep, sorry, speak, stop, thanks,
time, together, understand, want, water, what, when, where, which, who, why,
work, yes, you"""


def _call_claude_api(words: list) -> str | None:
    prompt  = " ".join(words)
    payload = json.dumps({
        "model":      MODEL,
        "max_tokens": 80,
        "system":     SYSTEM_PROMPT,
        "messages":   [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data    = payload,
        method  = "POST",
        headers = {
            "Content-Type":      "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key":         "sk-ant-YOUR-KEY-HERE",   # ← replace with your key
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=4) as resp:
            data     = json.loads(resp.read().decode("utf-8"))
            sentence = data["content"][0]["text"].strip()
            if sentence and len(sentence) < 250:
                return sentence
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Local fallback
# ─────────────────────────────────────────────────────────────────────────────

CONTINUOUS_VERBS = {
    "go", "eat", "drink", "sleep", "work", "speak", "come",
    "help", "see", "hear", "understand", "stop",
}

TO_VERBS = {
    "go", "eat", "drink", "sleep", "work", "speak", "come",
    "help", "see", "hear", "understand",
}

SUBJECTS       = {"i", "me", "you", "he", "she", "we", "they"}
QUESTION_WORDS = {"what", "where", "who", "why", "when", "how", "which"}
ADJECTIVES     = {"good", "bad", "sad", "angry", "sorry", "hurt", "ready", "clear"}

# Emotions that take "at [person]" when followed by a person word
DIRECTED_EMOTIONS = {"angry", "sad", "hurt"}
PERSON_WORDS      = {"you", "me", "he", "she", "friend", "family"}

BE_VERB = {
    "i": "am", "me": "am", "you": "are",
    "he": "is", "she": "is", "we": "are", "they": "are",
}

OBJECT_PRONOUN = {
    "i": "me", "me": "me", "you": "you",
    "he": "him", "she": "her", "we": "us", "they": "them",
}

WORD_MAP = {
    "i": "I", "me": "I", "you": "you", "he": "he", "we": "we",
    "again": "again", "angry": "angry", "bad": "bad", "bye": "goodbye",
    "clear": "clear", "come": "come", "drink": "drink", "eat": "eat",
    "family": "family", "friend": "friend", "go": "go", "good": "good",
    "hear": "hear", "hello": "hello", "help": "help", "home": "home",
    "how": "how", "hurt": "hurt", "morning": "morning", "name": "name",
    "no": "no", "phone": "phone", "please": "please", "problem": "problem",
    "ready": "ready", "sad": "sad", "see": "see", "sleep": "sleep",
    "sorry": "sorry", "speak": "speak", "stop": "stop", "thanks": "thank you",
    "time": "time", "together": "together", "understand": "understand",
    "want": "want", "water": "water", "what": "what", "when": "when",
    "where": "where", "which": "which", "who": "who", "why": "why",
    "work": "work", "yes": "yes",
}

# ── Phrase patterns ───────────────────────────────────────────────────────────
# Covers all the specific cases mentioned + common daily combos
PHRASE_PATTERNS = {

    # ── Single words ──────────────────────────────────────────────────────────
    ("hello",):          "Hello!",
    ("bye",):            "Goodbye!",
    ("thanks",):         "Thank you.",
    ("sorry",):          "I am sorry.",
    ("stop",):           "Stop!",
    ("help",):           "Help!",
    ("yes",):            "Yes.",
    ("no",):             "No.",
    ("again",):          "Please say that again.",
    ("clear",):          "Please speak clearly.",
    ("ready",):          "I am ready.",
    ("morning",):        "Good morning!",
    ("water",):          "I want some water.",
    ("phone",):          "I need my phone.",
    ("together",):       "Let us stay together.",
    ("understand",):     "I understand.",
    ("time",):           "What time is it?",
    ("good",):           "Good.",
    ("bad",):            "That is not good.",
    ("family",):         "My family.",
    ("friend",):         "My friend.",

    # ── Greetings ─────────────────────────────────────────────────────────────
    ("hello", "how", "you"):           "Hello! How are you?",
    ("good", "morning"):               "Good morning!",
    ("good", "morning", "how", "you"): "Good morning! How are you?",
    ("bye", "friend"):                 "Goodbye, my friend!",
    ("bye", "family"):                 "Goodbye, everyone!",

    # ── I + feeling ───────────────────────────────────────────────────────────
    ("i", "good"):    "I am good.",
    ("i", "bad"):     "I am not feeling well.",
    ("i", "sad"):     "I am sad.",
    ("i", "angry"):   "I am angry.",
    ("i", "sorry"):   "I am sorry.",
    ("i", "hurt"):    "I am hurt.",
    ("i", "ready"):   "I am ready.",

    # ── FIXED: emotion directed at a person ───────────────────────────────────
    ("i", "angry", "you"):    "I am angry at you.",
    ("i", "angry", "he"):     "I am angry at him.",
    ("i", "angry", "friend"): "I am angry at my friend.",
    ("i", "angry", "family"): "I am angry at my family.",
    ("i", "sad", "you"):      "I am sad because of you.",
    ("i", "sad", "he"):       "I am sad because of him.",
    ("i", "hurt", "you"):     "You hurt me.",
    ("i", "hurt", "he"):      "He hurt me.",
    ("why", "you", "angry", "me"): "Why are you angry at me?",
    ("why", "you", "sad", "me"):   "Why are you sad at me?",
    ("you", "angry", "me"):   "Are you angry at me?",
    ("you", "angry", "i"):    "Are you angry at me?",
    ("he", "angry", "me"):    "He is angry at me.",
    ("he", "angry", "you"):   "He is angry at you.",

    # ── FIXED: you + me together ──────────────────────────────────────────────
    ("you", "me", "friend"):           "You and I are friends.",
    ("you", "me", "together"):         "You and I are together.",
    ("you", "me", "go", "work"):       "You and I are going to work together.",
    ("you", "me", "go", "work", "together"): "You and I are going to work together.",
    ("you", "me", "go", "home"):       "Let us go home together.",
    ("you", "me", "eat"):              "Let us eat together.",
    ("you", "me", "drink"):            "Let us drink together.",
    ("you", "me", "work"):             "You and I are working together.",
    ("you", "me", "family"):           "You and I are family.",
    ("i", "you", "friend"):            "You and I are friends.",
    ("i", "you", "together"):          "You and I are together.",

    # ── FIXED: i want you/he + action ─────────────────────────────────────────
    ("i", "want", "you", "come"):      "I want you to come.",
    ("i", "want", "you", "help"):      "I want you to help me.",
    ("i", "want", "you", "go"):        "I want you to go.",
    ("i", "want", "you", "stay"):      "I want you to stay.",
    ("i", "want", "you", "friend"):    "I want to be your friend.",
    ("i", "want", "you", "together"):  "I want us to be together.",
    ("i", "want", "he", "come"):       "I want him to come.",
    ("i", "want", "he", "help"):       "I want him to help.",
    ("i", "want", "friend", "come"):   "I want my friend to come.",
    ("i", "want", "family", "come"):   "I want my family to come.",

    # ── I + action ────────────────────────────────────────────────────────────
    ("i", "go", "home"):          "I am going home.",
    ("i", "go", "work"):          "I am going to work.",
    ("i", "go", "friend"):        "I am going to see my friend.",
    ("i", "go", "family"):        "I am going to see my family.",
    ("i", "come"):                "I am coming.",
    ("i", "come", "home"):        "I am coming home.",
    ("i", "eat"):                 "I am eating.",
    ("i", "drink"):               "I am drinking.",
    ("i", "drink", "water"):      "I am drinking water.",
    ("i", "sleep"):               "I am sleeping.",
    ("i", "work"):                "I am working.",
    ("i", "speak"):               "I am speaking.",
    ("i", "understand"):          "I understand.",
    ("i", "no", "understand"):    "I do not understand.",
    ("i", "not", "understand"):   "I do not understand.",
    ("i", "hear", "you"):         "I can hear you.",
    ("i", "see", "you"):          "I can see you.",
    ("i", "no", "hear"):          "I cannot hear.",
    ("i", "no", "see"):           "I cannot see.",

    # ── I + want ──────────────────────────────────────────────────────────────
    ("i", "want", "eat"):             "I want to eat.",
    ("i", "want", "drink"):           "I want to drink.",
    ("i", "want", "drink", "water"):  "I want to drink water.",
    ("i", "want", "water"):           "I want some water.",
    ("i", "want", "sleep"):           "I want to sleep.",
    ("i", "want", "go", "home"):      "I want to go home.",
    ("i", "want", "go", "work"):      "I want to go to work.",
    ("i", "want", "help"):            "I need help.",
    ("i", "want", "see", "friend"):   "I want to see my friend.",
    ("i", "want", "see", "family"):   "I want to see my family.",
    ("i", "want", "speak", "you"):    "I want to talk to you.",
    ("i", "want", "phone"):           "I want my phone.",

    # ── I + have / need ───────────────────────────────────────────────────────
    ("i", "problem"):          "I have a problem.",
    ("i", "have", "problem"):  "I have a problem.",
    ("i", "phone"):            "I need my phone.",
    ("i", "family"):           "I am with my family.",
    ("i", "family", "together"): "I am together with my family.",
    ("i", "friend"):           "I am with my friend.",
    ("i", "together", "family"):  "I am together with my family.",
    ("i", "together", "friend"):  "I am together with my friend.",
    ("i", "name"):             "My name is...",

    # ── He + state/action ─────────────────────────────────────────────────────
    ("he", "good"):    "He is doing well.",
    ("he", "bad"):     "He is not well.",
    ("he", "sad"):     "He is sad.",
    ("he", "angry"):   "He is angry.",
    ("he", "hurt"):    "He is hurt.",
    ("he", "ready"):   "He is ready.",
    ("he", "sorry"):   "He is sorry.",
    ("he", "go", "home"):  "He is going home.",
    ("he", "go", "work"):  "He is going to work.",
    ("he", "come"):        "He is coming.",
    ("he", "come", "home"): "He is coming home.",
    ("he", "eat"):         "He is eating.",
    ("he", "drink"):       "He is drinking.",
    ("he", "sleep"):       "He is sleeping.",
    ("he", "work"):        "He is working.",
    ("he", "understand"):      "He understands.",
    ("he", "no", "understand"): "He does not understand.",
    ("he", "want", "help"):    "He needs help.",
    ("he", "want", "come"):    "He wants to come.",

    # ── You + question ────────────────────────────────────────────────────────
    ("you", "good"):       "Are you doing well?",
    ("you", "bad"):        "Are you not well?",
    ("you", "sad"):        "Are you sad?",
    ("you", "angry"):      "Are you angry?",
    ("you", "hurt"):       "Are you hurt?",
    ("you", "ready"):      "Are you ready?",
    ("you", "okay"):       "Are you okay?",
    ("you", "sorry"):      "Are you sorry?",
    ("you", "understand"): "Do you understand?",
    ("you", "hear"):       "Can you hear me?",
    ("you", "see"):        "Can you see me?",
    ("you", "come"):       "Are you coming?",
    ("you", "come", "home"): "Are you coming home?",
    ("you", "go", "home"):   "Are you going home?",
    ("you", "go", "work"):   "Are you going to work?",
    ("you", "want", "help"):  "Do you need help?",
    ("you", "want", "eat"):   "Do you want to eat?",
    ("you", "want", "drink"): "Do you want to drink?",
    ("you", "want", "water"): "Do you want some water?",
    ("you", "speak"):      "Can you speak?",
    ("you", "name"):       "What is your name?",
    ("you", "phone"):      "Is that your phone?",
    ("you", "family"):     "Is that your family?",
    ("you", "friend"):     "Is that your friend?",

    # ── Wh-questions ──────────────────────────────────────────────────────────
    ("what", "time"):           "What time is it?",
    ("which", "time"):          "What time is it?",
    ("what", "problem"):        "What is the problem?",
    ("what", "name"):           "What is the name?",
    ("what", "you", "name"):    "What is your name?",
    ("what", "your", "name"):   "What is your name?",
    ("where", "you"):           "Where are you?",
    ("where", "you", "go"):     "Where are you going?",
    ("where", "he", "go"):      "Where is he going?",
    ("where", "home"):          "Where is home?",
    ("where", "family"):        "Where is your family?",
    ("where", "friend"):        "Where is your friend?",
    ("where", "phone"):         "Where is the phone?",
    ("who", "you"):             "Who are you?",
    ("who", "he"):              "Who is he?",
    ("why", "you", "sad"):      "Why are you sad?",
    ("why", "you", "angry"):    "Why are you angry?",
    ("why", "you", "stop"):     "Why did you stop?",
    ("why", "he", "sad"):       "Why is he sad?",
    ("why", "he", "angry"):     "Why is he angry?",
    ("when", "you", "come"):    "When are you coming?",
    ("when", "you", "go"):      "When are you going?",
    ("when", "he", "come"):     "When is he coming?",
    ("how", "you"):             "How are you?",
    ("how", "he"):              "How is he?",
    ("how", "family"):          "How is your family?",
    ("how", "friend"):          "How is your friend?",
    ("how", "work"):            "How is work?",

    # ── Please ────────────────────────────────────────────────────────────────
    ("please", "help"):              "Please help me.",
    ("please", "repeat"):            "Please repeat that.",
    ("please", "again"):             "Please say that again.",
    ("please", "slow"):              "Please speak slowly.",
    ("please", "speak", "slow"):     "Please speak slowly.",
    ("please", "speak"):             "Please speak.",
    ("please", "stop"):              "Please stop.",
    ("please", "come"):              "Please come here.",
    ("please", "come", "home"):      "Please come home.",
    ("please", "go"):                "Please go.",
    ("please", "understand"):        "Please understand.",
    ("please", "clear"):             "Please speak clearly.",

    # ── No / yes ──────────────────────────────────────────────────────────────
    ("no", "problem"):    "No problem.",
    ("no", "understand"): "I do not understand.",
    ("no", "stop"):       "Do not stop.",
    ("no", "go"):         "Do not go.",
    ("yes", "good"):      "Yes, that is good.",
    ("yes", "understand"): "Yes, I understand.",
    ("yes", "ready"):     "Yes, I am ready.",
    ("yes", "come"):      "Yes, I am coming.",

    # ── Together / family / friend ────────────────────────────────────────────
    ("go", "together"):       "Let us go together.",
    ("come", "together"):     "Come together.",
    ("work", "together"):     "Let us work together.",
    ("we", "together"):       "We are together.",
    ("family", "together"):   "The family is together.",
    ("friend", "come"):       "My friend is coming.",
    ("friend", "go"):         "My friend is going.",
    ("friend", "help"):       "My friend will help.",

    # ── Water / food / phone ──────────────────────────────────────────────────
    ("drink", "water"):      "Drink some water.",
    ("want", "water"):       "I want some water.",
    ("eat", "together"):     "Let us eat together.",
    ("drink", "together"):   "Let us drink together.",
    ("eat", "family"):       "Let us eat with the family.",

    # ── Morning / social ──────────────────────────────────────────────────────
    ("good", "morning", "family"):  "Good morning, everyone!",
    ("good", "morning", "friend"):  "Good morning, my friend!",
    ("morning", "how", "you"):      "Good morning! How are you?",
    ("thanks", "help"):             "Thank you for helping.",
    ("thanks", "come"):             "Thank you for coming.",
    ("sorry", "problem"):           "I am sorry about the problem.",
    ("sorry", "again"):             "I am sorry, please say that again.",

    # ── Speak / hear / see ────────────────────────────────────────────────────
    ("stop", "speak"):   "Stop speaking.",
    ("speak", "slow"):   "Please speak slowly.",
    ("speak", "again"):  "Please speak again.",
    ("hear", "you"):     "I can hear you.",
    ("see", "you"):      "I can see you.",
    ("you", "no", "hear"): "You cannot hear?",
    ("you", "no", "see"):  "You cannot see?",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_continuous(verb: str) -> str:
    if verb.endswith("ing"):
        return verb
    if verb.endswith("e") and not verb.endswith("ee"):
        return verb[:-1] + "ing"
    if (len(verb) >= 3
            and verb[-1] not in "aeiou"
            and verb[-2] in "aeiou"
            and verb[-3] not in "aeiou"):
        return verb + verb[-1] + "ing"
    return verb + "ing"


def _obj(word: str) -> str:
    """Return object pronoun for a subject word."""
    return OBJECT_PRONOUN.get(word, word)


def _rule_based_grammar(signs: list) -> str:
    """Local fallback grammar for all 49 signs."""
    if not signs:
        return ""
    words = [w.lower().strip() for w in signs if w.strip()]
    if not words:
        return ""

    # ── Single word ───────────────────────────────────────────────────────────
    if len(words) == 1:
        singles = {
            "hello": "Hello!", "bye": "Goodbye!", "thanks": "Thank you.",
            "sorry": "I am sorry.", "stop": "Stop!", "help": "Help!",
            "yes": "Yes.", "no": "No.", "again": "Please say that again.",
            "clear": "Please speak clearly.", "ready": "I am ready.",
            "morning": "Good morning!", "water": "I want some water.",
            "phone": "I need my phone.", "together": "Let us stay together.",
            "understand": "I understand.", "work": "I am working.",
            "home": "I am going home.", "family": "My family.",
            "friend": "My friend.", "time": "What time is it?",
            "good": "Good.", "bad": "That is not good.", "name": "My name is...",
        }
        if words[0] in singles:
            return singles[words[0]]
        return WORD_MAP.get(words[0], words[0]).capitalize() + "."

    # ── Phrase pattern — longest match first ──────────────────────────────────
    for length in range(min(7, len(words)), 0, -1):
        for start in range(len(words) - length + 1):
            pattern = tuple(words[start:start + length])
            if pattern in PHRASE_PATTERNS:
                result    = PHRASE_PATTERNS[pattern]
                remaining = words[start + length:]
                if remaining:
                    extra  = " ".join(WORD_MAP.get(w, w) for w in remaining)
                    result = result.rstrip("?!. ") + " " + extra + "."
                return result

    # ── Special case: you + me at start → "You and I" ────────────────────────
    if len(words) >= 2 and words[0] == "you" and words[1] == "me":
        rest = words[2:]
        if not rest:
            return "You and I."
        if rest[0] == "friend":
            return "You and I are friends."
        if rest[0] == "together":
            return "You and I are together."
        if rest[0] in CONTINUOUS_VERBS:
            action = _make_continuous(WORD_MAP.get(rest[0], rest[0]))
            suffix = " ".join(WORD_MAP.get(w, w) for w in rest[1:])
            return f"You and I are {action}{(' ' + suffix) if suffix else ''}."
        suffix = " ".join(WORD_MAP.get(w, w) for w in rest)
        return f"You and I {suffix}."

    # ── Special case: subject + directed emotion + person ─────────────────────
    # e.g. "i angry you" → "I am angry at you."
    if (len(words) >= 3
            and words[0] in SUBJECTS
            and words[1] in DIRECTED_EMOTIONS
            and words[2] in PERSON_WORDS):
        subj   = "I" if words[0] in ("i", "me") else WORD_MAP.get(words[0], words[0])
        be     = BE_VERB.get(words[0], "am")
        emotion = words[1]
        target  = _obj(words[2])
        if target in ("friend", "family"):
            target = f"my {target}"
        if emotion == "hurt":
            # "i hurt you" → "I hurt you." / "you hurt me" is an assault statement
            return f"{subj} hurt {target}."
        prep = "at" if emotion == "angry" else "because of"
        return f"{subj} {be} {emotion} {prep} {target}."

    # ── Special case: i want you/he + verb/noun ───────────────────────────────
    if (len(words) >= 3
            and words[0] in ("i", "me")
            and words[1] == "want"
            and words[2] in ("you", "he", "friend", "family")):
        target = words[2]
        rest   = words[3:]
        if not rest:
            return f"I want {_obj(target)}."
        action = rest[0]
        if action == "friend":
            return "I want to be your friend."
        if action == "together":
            return "I want us to be together."
        if action in TO_VERBS:
            return f"I want {_obj(target)} to {WORD_MAP.get(action, action)}."
        return f"I want {_obj(target)} to {WORD_MAP.get(action, action)}."

    # ── Rule-based construction ───────────────────────────────────────────────
    result  = []
    i       = 0
    subject = None
    has_be  = False

    while i < len(words):
        word   = words[i]
        mapped = WORD_MAP.get(word, word)

        # Subject
        if word in SUBJECTS and subject is None:
            subject = word
            result.append("I" if word in ("i", "me") else mapped)
            i += 1
            continue

        # Leading question word
        if word in QUESTION_WORDS and i == 0:
            result.append(mapped.capitalize())
            if i + 1 < len(words) and words[i + 1] in SUBJECTS:
                subj = words[i + 1]
                result.append(BE_VERB.get(subj, "are"))
                result.append(WORD_MAP.get(subj, subj))
                i += 2
                continue
            i += 1
            continue

        # Adjective after subject → be + adj
        if subject and word in ADJECTIVES and not has_be:
            # Check if next word is a person (directed emotion)
            if word in DIRECTED_EMOTIONS and i + 1 < len(words) and words[i + 1] in PERSON_WORDS:
                be     = BE_VERB.get(subject, "am")
                target = _obj(words[i + 1])
                if target in ("friend", "family"):
                    target = f"my {target}"
                prep = "at" if word == "angry" else "because of"
                result.append(be)
                result.append(f"{word} {prep} {target}")
                has_be = True
                i += 2
                continue
            result.append(BE_VERB.get(subject, "am"))
            result.append(mapped)
            has_be = True
            i += 1
            continue

        # Continuous verb after subject → be + -ing
        if subject and word in CONTINUOUS_VERBS and not has_be:
            result.append(BE_VERB.get(subject, "am"))
            result.append(_make_continuous(mapped))
            has_be = True
            i += 1
            continue

        # want/need + to_verb
        if word in ("want", "need") and i + 1 < len(words) and words[i + 1] in TO_VERBS:
            result.append(f"{mapped} to")
            result.append(WORD_MAP.get(words[i + 1], words[i + 1]))
            i += 2
            continue

        # no/not + verb/adj → negation
        if word in ("no", "not") and i + 1 < len(words):
            next_w = words[i + 1]
            if next_w in CONTINUOUS_VERBS:
                if not has_be:
                    result.append(BE_VERB.get(subject, "am") if subject else "am")
                    has_be = True
                result.append("not")
                result.append(_make_continuous(WORD_MAP.get(next_w, next_w)))
                i += 2
                continue
            if next_w in ADJECTIVES:
                if not has_be:
                    result.append(BE_VERB.get(subject, "am") if subject else "am")
                    has_be = True
                result.append("not")
                result.append(WORD_MAP.get(next_w, next_w))
                i += 2
                continue
            result.append("not")
            i += 1
            continue

        # family/friend with subject → "with my family/friend"
        if word in ("family", "friend") and subject:
            result.append(f"with my {word}")
            i += 1
            continue

        # together → just "together"
        if word == "together":
            result.append("together")
            i += 1
            continue

        # phone → "my phone"
        if word == "phone":
            result.append("my phone")
            i += 1
            continue

        # water → "some water"
        if word == "water":
            result.append("some water")
            i += 1
            continue

        result.append(mapped)
        i += 1

    sentence = " ".join(result).strip()
    if not sentence:
        return ""
    sentence = sentence[0].upper() + sentence[1:]

    if words[0] in QUESTION_WORDS and not sentence.endswith("?"):
        sentence += "?"
    elif (words[0] in ("you", "he")
          and not sentence.endswith(("?", "!", "."))
          and len(words) <= 4):
        sentence += "?"
    elif not sentence.endswith(("?", "!", ".")):
        sentence += "."

    return sentence


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def correct_grammar(signs: list) -> str:
    if not signs:
        return ""
    api_result = _call_claude_api(signs)
    if api_result:
        print(f"[API]   {signs} → {api_result}")
        return api_result
    local_result = _rule_based_grammar(signs)
    print(f"[LOCAL] {signs} → {local_result}")
    return local_result


# ─────────────────────────────────────────────────────────────────────────────
# SentenceBuilder
# ─────────────────────────────────────────────────────────────────────────────

class SentenceBuilder:
    def __init__(self, timeout: float = 3.0, max_words: int = 10):
        self.timeout   = timeout
        self.max_words = max_words
        self.reset()

    def reset(self):
        self.words          = []
        self.last_sign_time = None
        self.sentence       = ""
        self.sentence_ready = False

    def add_sign(self, sign: str):
        sign = sign.lower().strip()
        if not sign:
            return
        if self.words and self.words[-1] == sign:
            self.last_sign_time = time.time()
            return
        self.words.append(sign)
        self.last_sign_time = time.time()
        self.sentence_ready = False
        if len(self.words) >= self.max_words:
            self._generate()

    def update(self) -> bool:
        if (self.last_sign_time is not None
                and not self.sentence_ready
                and len(self.words) > 0
                and time.time() - self.last_sign_time >= self.timeout):
            self._generate()
            return True
        return False

    def _generate(self):
        self.sentence       = correct_grammar(self.words)
        self.sentence_ready = True

    def get_sentence(self) -> str:
        return self.sentence

    def get_words(self) -> list:
        return self.words.copy()

    def time_since_last_sign(self) -> float:
        if self.last_sign_time is None:
            return 0.0
        return time.time() - self.last_sign_time

    def countdown_remaining(self) -> float:
        if self.sentence_ready:
            return 0.0
        return max(0.0, self.timeout - self.time_since_last_sign())


# ─────────────────────────────────────────────────────────────────────────────
# Self-test — includes all your reported problem cases
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        # ── Your specific problem cases ───────────────────────────────────────
        ["i", "angry", "you"],          # was: "I am angry you"    → now: "I am angry at you."
        ["you", "me", "friend"],        # was: "my friend"         → now: "You and I are friends."
        ["i", "want", "you", "friend"], # was: "Is this your friend?" → now: "I want to be your friend."
        ["you", "me", "go", "work", "together"],  # was ok → "You and I are going to work together."
        # ── More directed emotion tests ───────────────────────────────────────
        ["i", "angry", "he"],
        ["i", "angry", "friend"],
        ["i", "sad", "you"],
        ["i", "hurt", "you"],
        ["you", "angry", "me"],
        ["why", "you", "angry", "me"],
        # ── You + me combos ───────────────────────────────────────────────────
        ["you", "me", "together"],
        ["you", "me", "go", "home"],
        ["you", "me", "eat"],
        ["you", "me", "family"],
        # ── I want you/he + action ────────────────────────────────────────────
        ["i", "want", "you", "come"],
        ["i", "want", "you", "help"],
        ["i", "want", "he", "come"],
        ["i", "want", "you", "together"],
        # ── Everyday sentences ────────────────────────────────────────────────
        ["i", "go", "work"],
        ["i", "sad"],
        ["i", "angry"],
        ["i", "no", "understand"],
        ["how", "you"],
        ["where", "you", "go"],
        ["you", "okay"],
        ["please", "help"],
        ["good", "morning"],
        ["i", "want", "water"],
        ["i", "problem"],
        ["no", "problem"],
        ["go", "together"],
        ["friend", "come"],
        ["i", "hear", "you"],
        ["thanks"],
        ["sorry"],
        ["stop"],
        ["help"],
    ]

    print("Sentence builder — local fallback test:")
    print("=" * 65)
    for t in tests:
        result = _rule_based_grammar(t)
        print(f"  {' + '.join(t):<40} → {result}")
