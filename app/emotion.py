"""Emotion detection for text passages.

Combines a lightweight rule/lexicon analyzer with optional LLM-based
analysis (OpenAI-compatible endpoint) when EMOTION_API_KEY is configured.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, asdict, field

EMOTIONS = [
    "neutral", "happy", "sad", "angry", "fear",
    "excited", "tender", "mysterious", "heroic", "ironic",
]

# ---------------------------------------------------------------------------
# Lexicons (word -> list of emotions it votes for)
# ---------------------------------------------------------------------------
LEX: dict[str, list[str]] = {}


def _add(words: str, *emos: str) -> None:
    for w in words.split():
        LEX.setdefault(w, []).extend(emos)


_add("joy happy glad delighted cheerful wonderful amazing fantastic great love"
     " loved lovely laugh laughing laughed smile smiling smiled enjoy enjoying"
     " enjoyable fun pleasure joyful bliss celebrate celebration joyous merry"
     " amused amusement charming beautiful gorgeous sunny bright hope hopeful",
     "happy", "tender")
_add("excit excited thrill thrilling thrilled thrillers rush adrenaline wow"
     " hooray yay awesome incredible unbelievable extraordinary triumph"
     " triumphant victory won celebrating festival fireworks",
     "excited", "happy")
_add("sad unhappy sorrow sorrowful miserable melancholy cry crying cried tears"
     " tear weep weeping wept mourn mourning grief grieving lonely loneliness"
     " alone despair depressed depression gloom gloomy dreary lament heartbreak"
     " heartbroken loss lost farewell goodbye missed miss dear departed",
     "sad")
_add("anger angry furious rage enraged mad irritated annoyed infuriating hate"
     " hated hatred despise despised scowl scowled shout shouted yell yelled"
     " scream screamed roar roaredDamn damn curse cursed fury revolt rebel",
     "angry")
_add("afraid fear fearful frightened fright terrifying terrified terror panic"
     " panicked dread dreadful horror horrible horrifying nightmare nightmares"
     " shiver shivered trembling trembled shook shake worried worry anxious"
     " anxiety nervous nervously danger dangerous deadly peril flee fled"
     " ran away hid hiding shadow shadows dark darkness creepy creep",
     "fear", "mysterious")
_add("mystery mysterious secret secretly secrets strange strangely weird odd"
     " curious curiosity whisper whispered murmur murmured hushed fog mist"
     " misty midnight eerie uncanny unknown enigma riddle vanished vanish"
     " ghost ghostly phantom silence silent quietly dim gloom",
     "mysterious")
_add("brave bravery hero heroic heroes courage courageous valor gallant"
     " determined determination fought fight battle warrior strength strong"
     " mighty stood rise rose vow swore pledge honor honour defend sacrifice"
     " sacrificed charge conquer conquered",
     "heroic")
_add("sweet gently gentle softly tender tenderly kiss kissed embrace hugged"
     " hug caress warm warmth dear darling honey beloved mother father child"
     " lullaby snuggled held cradle affection fond",
     "tender")
_add("said says replied answered asked shouted whispered exclaimed muttered"
     " cried laughed smiled frowned",
     )  # dialogue verbs are handled separately

# ---------------------------------------------------------------------------
# Russian lexicons (word stems are matched by prefix, so any case/ending works)
# ---------------------------------------------------------------------------
RU_LEX: dict[str, list[str]] = {}


def _add_ru(stems: str, *emos: str) -> None:
    for w in stems.split():
        RU_LEX.setdefault(w, []).extend(emos)


_add_ru("радост счастл весел улыбк смех смея смеёт хохоч лику восторг чудесн"
        " прекрасн замечательн любим люблю любовь ласк нежн праздн красота"
        " солнц светл надежд мечты мечта радость хорошо хорон",
        "happy", "tender")
_add_ru("волнова возбужд трепет адреналин ура удивительн невероятн"
        " триумф побед выиграл фейерверк спешк нетерпен",
        "excited", "happy")
_add_ru("грустн печал тоск плакал плачет плачут заплак слез уныл меланхоли"
        " одинок одиночеств покинут горевал горюет скорб потер потерял"
        " прощай прощан расставан депресс оплакив утрат безутешн"
        " умер смерт похорон мёртв мертв",
        "sad")
_add_ru("злост гнев гневу разгнева взбешен бешен ярост ненавиж ненавист презир"
        " крича крикнул кричит орал рычал руга проклять черт дьявол злобн",
        "angry")
_add_ru("страшн страх испуг боюсь боязнь ужас паник",
        "fear", "mysterious")
_add_ru("таинствен тайн секрет странн необычн шёпот шепот туман полночь"
        " мрак призрачн неизвестн загадк исчез",
        "mysterious")
_add_ru("тихо тишин молча молчал",
        "mysterious", "tender")
_add_ru("храбр герой героич мужеств отвага доблест стоек стойк"
        " сражался битв воин сил могуч боролся защищал клятв долг жертв",
        "heroic")
_add_ru("мило целов обним родн мам пап ребёнок малыш колыбельн"
        " тепл обнимал прижался лобыз",
        "tender")
_add_ru("наивно иронич саркам насмешл издеват шутк ирония",
        "ironic")

# Multi-word phrases ("больше не было") -> emotion votes. Words are joined
# with "_" inside a phrase entry; several alternative phrases separated by "|".
RU_PHRASES: list[tuple[re.Pattern, list[str]]] = []


def _add_ru_phrase(entries: str, *emos: str) -> None:
    for entry in entries.split():
        for alt in entry.split("|"):
            words = alt.split("_")
            pat = re.compile(
                r"\b" + r"[\s,.!?…;:\-]+".join(re.escape(w) for w in words) + r"\b",
                re.UNICODE | re.IGNORECASE)
            RU_PHRASES.append((pat, list(emos)))


_add_ru_phrase("больше_не_был|больше_не_было|больше_не_стало|его_не_стало"
               " её_не_стало|её_больше_не", "sad")
_add_ru_phrase("сердце_разбил|сердце_разбито|разбитое_сердц", "sad")
_add_ru_phrase("ни_жив ни_мёртв", "fear")
_add_ru_phrase("чёрт_возьми|дьявол_побери|чтоб_тебя", "angry")
_add_ru_phrase("ну_конечно|вот_молодец|опять_ты|как_обычно", "ironic")


CAPITAL_WORD = re.compile(r"\b[A-ZА-Я]{3,}\b")
EXCLAIM = re.compile(r"[!！]{1,}")
QUESTION = re.compile(r"[?？]")
ELLIPSIS = re.compile(r"\.{3}|…")
ALLCAPS_SENT = re.compile(r"^[^a-zа-я]*[A-ZА-Я]{2,}[^a-zа-я]*$")
DIALOGUE = re.compile(r"[\"“»«].+?[\"”»«]")


def _lex_score(tok: str) -> list[str]:
    """Emotion votes for a token: exact match in English lexicon,
    longest-prefix match in the Russian stem lexicon."""
    emo = LEX.get(tok)
    if emo:
        return emo
    best_key, best_len = "", 0
    for stem in RU_LEX:
        if tok.startswith(stem) and len(stem) > best_len:
            best_key, best_len = stem, len(stem)
    return RU_LEX.get(best_key, ())


@dataclass
class EmotionProfile:
    emotion: str = "neutral"
    confidence: float = 0.5
    reason: str = ""
    # prosody parameters applied on top of the base voice
    rate: float = 1.0        # speaking length multiplier (higher = slower)
    pitch: float = 0.0       # semitone shift
    volume: float = 1.0      # gain multiplier
    pause_before_ms: int = 250
    tremolo: float = 0.0     # amplitude modulation depth 0..1
    vibrato: float = 0.0     # frequency modulation depth 0..1

    def to_dict(self) -> dict:
        return asdict(self)


# Prosody presets per emotion
PROSODY: dict[str, dict] = {
    "neutral":   dict(rate=1.00, pitch=0.0,  volume=1.00, tremolo=0.00, vibrato=0.00),
    "happy":     dict(rate=1.06, pitch=2.5,  volume=1.06, tremolo=0.00, vibrato=0.15),
    "excited":   dict(rate=1.14, pitch=3.8,  volume=1.14, tremolo=0.00, vibrato=0.30),
    "sad":       dict(rate=0.84, pitch=-2.6, volume=0.90, tremolo=0.18, vibrato=0.08),
    "angry":     dict(rate=1.10, pitch=-1.3, volume=1.26, tremolo=0.05, vibrato=0.10),
    "fear":      dict(rate=1.16, pitch=1.7,  volume=0.90, tremolo=0.40, vibrato=0.22),
    "tender":    dict(rate=0.88, pitch=1.1,  volume=0.84, tremolo=0.06, vibrato=0.10),
    "mysterious":dict(rate=0.86, pitch=-1.7, volume=0.84, tremolo=0.10, vibrato=0.06),
    "heroic":    dict(rate=0.94, pitch=-0.9, volume=1.16, tremolo=0.00, vibrato=0.05),
    "ironic":    dict(rate=0.96, pitch=0.7,  volume=1.00, tremolo=0.00, vibrato=0.25),
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ']+", text.lower())


def analyze_rules(text: str, prev_emotion: str | None = None) -> EmotionProfile:
    scores: dict[str, float] = {e: 0.15 for e in EMOTIONS}  # small prior
    toks = _tokens(text)
    n = max(len(toks), 1)

    for t in toks:
        for emo in _lex_score(t):  # exact EN match or RU stem prefix match
            scores[emo] += 1.0
    lex_hits = sum(scores[e] for e in EMOTIONS) - 0.15 * len(EMOTIONS)

    # multi-word Russian phrases ("больше не было" -> sadness)
    low_text = text.lower()
    phrase_hits = 0
    for pat, emos in RU_PHRASES:
        if pat.search(low_text):
            phrase_hits += 1
            for emo in emos:
                scores[emo] += 1.5
    lex_hits += 1.5 * phrase_hits

    # punctuation cues
    if EXCLAIM.search(text):
        ex_count = len(EXCLAIM.findall(text))
        scores["excited"] += 0.9 * min(ex_count, 3)
        scores["angry"] += 0.4
        scores["heroic"] += 0.3
    if QUESTION.search(text):
        scores["mysterious"] += 0.5
        scores["fear"] += 0.2
    if ELLIPSIS.search(text):
        scores["sad"] += 0.5
        scores["mysterious"] += 0.5
        scores["tender"] += 0.3
    if CAPITAL_WORD.search(text) or any(ALLCAPS_SENT.match(s.strip())
                                        for s in re.split(r"[.!?…]", text) if s.strip()):
        scores["angry"] += 1.0
        scores["excited"] += 0.6
    if DIALOGUE.search(text):
        scores["ironic"] += 0.35  # spoken lines often carry attitude
    # Russian-style dialogue dashes: line starting with «— » after a quote/period
    if re.search(r"(?:^|[.!?…]\s)\s*—\s+\S", text):
        scores["ironic"] += 0.2

    # short punchy sentences feel excited/angry; long flowing ones tender/mysterious
    words_per_sent = n / max(len(re.findall(r"[.!?…]+", text)), 1)
    if words_per_sent < 6:
        scores["excited"] += 0.4
        scores["angry"] += 0.25
    elif words_per_sent > 22:
        scores["tender"] += 0.3
        scores["mysterious"] += 0.25

    # emotional continuity: small bias toward previous emotion
    if prev_emotion in scores:
        scores[prev_emotion] += 0.25

    best = max(scores, key=scores.get)
    total = sum(scores.values())
    conf = scores[best] / total
    if best == "neutral" or lex_hits < 0.5 and conf < 0.22:
        best, conf = "neutral", max(conf, 0.4)

    reasons = []
    if lex_hits >= 0.5:
        top_words = [t for t in toks if _lex_score(t)][:4]
        reasons.append("лексика: " + (", ".join(top_words) or "фразы"))
    if EXCLAIM.search(text):
        reasons.append("восклицания")
    if CAPITAL_WORD.search(text):
        reasons.append("КАПС")
    if not reasons:
        reasons.append("нейтральный тон")

    prof = EmotionProfile(emotion=best, confidence=round(min(conf * 1.6, 0.99), 2),
                          reason="; ".join(reasons))
    return prof


# ---------------------------------------------------------------------------
# Optional LLM analysis (OpenAI-compatible chat completions API)
# ---------------------------------------------------------------------------
LLM_PROMPT = (
    "You are an emotional analyst for an audiobook reader. For the given text "
    "fragment choose ONE emotion from: neutral, happy, sad, angry, fear, "
    "excited, tender, mysterious, heroic, ironic. Respond ONLY with JSON "
    '{"emotion": "...", "confidence": 0..1, "reason": "short reason in Russian"}'
)


def analyze_llm(text: str) -> EmotionProfile | None:
    key = os.environ.get("EMOTION_API_KEY")
    if not key:
        return None
    base = os.environ.get("EMOTION_API_BASE", "https://api.openai.com/v1")
    model = os.environ.get("EMOTION_MODEL", "gpt-4o-mini")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": LLM_PROMPT},
                {"role": "user", "content": text[:2000]},
            ],
            "temperature": 0.2,
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
        content = data["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", content, re.S)
        obj = json.loads(m.group(0)) if m else {}
        emo = obj.get("emotion", "neutral")
        if emo not in EMOTIONS:
            return None
        return EmotionProfile(emotion=emo,
                              confidence=float(obj.get("confidence", 0.8)),
                              reason="ИИ-анализ: " + str(obj.get("reason", ""))[:120])
    except Exception:
        return None


def analyze(text: str, prev_emotion: str | None = None,
            use_llm: bool = True) -> EmotionProfile:
    profile = None
    if use_llm and os.environ.get("EMOTION_API_KEY"):
        profile = analyze_llm(text)
    if profile is None:
        profile = analyze_rules(text, prev_emotion)
    # attach prosody
    p = PROSODY.get(profile.emotion, PROSODY["neutral"])
    for k, v in p.items():
        setattr(profile, k, v)
    # intensity scaling by confidence
    inten = 0.5 + 0.5 * profile.confidence
    profile.rate = 1 + (profile.rate - 1) * inten
    profile.pitch = profile.pitch * inten
    profile.volume = 1 + (profile.volume - 1) * inten
    profile.tremolo *= inten
    profile.vibrato *= inten
    return profile
