"""Word list and synthetic text sampler.

No Hebrew word list exists on this machine, so a compact list of frequent
Hebrew words is shipped with the project.  It can be extended at runtime with
:func:`load_user_lexicon` (``--lexicon path/to/words.txt``).

Pseudo words are generated with Hebrew letter frequencies so that the
recogniser sees every letter - including rare final forms - many times.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

from . import charset
from .config import Config

# --------------------------------------------------------------------------
# Frequent Hebrew words (no niqqud, no final letters inside a word).
# --------------------------------------------------------------------------
_COMMON_WORDS = """
שלום תודה בבקשה כן לא אולי מה מי איפה מתי למה איך כמה זה זאת אלה הוא היא הם הן אני
אתה את אנחנו אתם יש אין היה הייתה יהיה הולך בא רואה שומע אוהב רוצה צריך יכול יודע
חושב מדבר כותב קורא לומד עובד אוכל שותה ישן קם רץ יושב עומד עוזר מחכה פוגש
בית ילד ילדה ילדים אבא אמא אחות אח סבא סבתא דוד דודה חבר חברה מורה תלמיד תלמידה
אדם איש אישה נער נערה בן בת תינוק שכן אורח מלך מלכה רופא חולה שוטר חייל טבח נהג
מוכר קונה מנהל מזכירה מהנדס מתכנת אמן זמר שחקן סופר משורר עיתונאי מדען חוקר סטודנט
ספר מחברת עיפרון עט נייר שולחן כיסא דלת חלון קיר רצפה תקרה חדר מטבח אמבטיה מיטה
כרית שמיכה ארון מנורה שעון מראה תמונה מחשב טלפון מפתח ארנק תיק מעיל חולצה מכנסיים
נעל גרב כובע משקפיים טבעת שרשרת
מים לחם חלב קפה תה שוקולד תפוח בננה תפוז עגבניה מלפפון גזר בצל שום ירק פרי בשר דג
עוף גבינה ביצה סוכר מלח פלפל שמן יין בירה עוגה גלידה מרק סלט אורז פסטה קמח דבש ריבה
מסעדה חנות שוק כסף עבודה לימודים אוניברסיטה מכללה ספריה חניון תחנה
עיר כפר רחוב כביש גן פארק ים חוף הר נהר אגם מדבר יער שדה גינה שמיים
שמש ירח כוכב ענן גשם רוח שלג חורף קיץ אביב סתיו בוקר ערב לילה שבוע חודש
שנה יום שעה דקה שנייה רגע
חתול כלב ציפור סוס פרה כבש עז תרנגול תרנגולת אווז ברווז ארנב עכבר נחש צב
אריה נמר דוב זאב שועל קוף פיל גמל חמור פרפר דבורה נמלה עכביש זבוב
ראש עין אוזן אף פה שן לשון צוואר יד רגל אצבע בטן גב לב שיער פנים
ללכת לבוא לראות לשמוע לאהוב לרצות לישון לאכול לשתות לכתוב לקרוא ללמוד לעבוד
לשחק לדבר לשאול לענות לפתוח לסגור לקנות למכור לנסוע לטוס לרוץ לשבת לקום
לעמוד לבכות לצחוק לחשוב לזכור לשכוח להבין לדעת לעזור לחכות לפגוש לבקר
לחזור לנסות להצליח לסיים להתחיל לבנות לתקן לשמור לנקות לבשל לאפות
לצייר לשיר לרקוד לספר להסביר ללמד
גדול קטן טוב רע יפה מכוער חדש ישן חם קר חכם טיפש חזק חלש מהיר איטי
גבוה נמוך ארוך קצר רחב צר כבד קל מתוק מלוח חמוץ מר עשיר עני שמח עצוב
עייף רעב צמא בריא נקי מלוכלך יקר זול קשה חשוב מעניין משעמם
אדום כחול ירוק צהוב שחור לבן אפור כתום ורוד חום סגול זהב
אחת שתיים שלוש ארבע חמש שש שבע שמונה תשע עשר עשרים שלושים חמישים מאה אלף
אבל או גם רק מאוד קצת הרבה מעט תמיד לפעמים עכשיו אחרי לפני בין עם בלי
על מתחת מעל ליד בתוך בחוץ כאן שם שוב עוד כבר עדיין כמעט בדיוק בערך יחד לבד
לכן אמנם למרות בגלל כדי אשר שלו שלה שלהם שלנו שלך שלי
""".split()


# Approximate relative frequencies of the 22 regular Hebrew letters, so that
# pseudo words look "Hebrew-ish" instead of uniformly random.
LETTER_FREQUENCY: dict[str, float] = {
    "א": 6.0, "ב": 4.5, "ג": 1.3, "ד": 3.0, "ה": 8.5, "ו": 10.0, "ז": 1.2,
    "ח": 2.5, "ט": 1.5, "י": 11.0, "כ": 3.5, "ל": 6.0, "מ": 5.5, "נ": 4.5,
    "ס": 1.8, "ע": 3.5, "פ": 2.0, "צ": 1.3, "ק": 2.5, "ר": 5.5, "ש": 4.5,
    "ת": 5.0,
}

# base letter -> final form
FINAL_OF = {"כ": "ך", "מ": "ם", "נ": "ן", "פ": "ף", "צ": "ץ"}


def default_lexicon() -> list[str]:
    return list(_COMMON_WORDS)


def load_user_lexicon(path: str | Path) -> list[str]:
    """Read a plain text word list (one word per line, or whitespace separated)."""
    text = Path(path).read_text(encoding="utf-8")
    return [word for word in text.split() if word]


def filter_lexicon(words: Iterable[str], allowed: set[str]) -> list[str]:
    """Keep only words that the given font can actually render."""
    return [w for w in words if w and all(ch in allowed for ch in w)]


def _weighted_letter(rng: random.Random, letters: Sequence[str]) -> str:
    pool = [ch for ch in letters if ch in LETTER_FREQUENCY]
    if not pool:
        return rng.choice(list(letters))
    weights = [LETTER_FREQUENCY[ch] for ch in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


def sample_pseudo_word(
    rng: random.Random,
    allowed: set[str],
    min_len: int = 2,
    max_len: int = 12,
    p_final: float = 0.7,
) -> str:
    """Generate a pronounceable-looking Hebrew pseudo word."""
    regular = [ch for ch in charset.REGULAR_LETTERS if ch in allowed]
    if not regular:
        regular = [ch for ch in charset.HEBREW_LETTERS if ch in allowed]
    letters = [_weighted_letter(rng, regular) for _ in range(rng.randint(min_len, max_len))]
    finals = [f for f in charset.FINAL_LETTERS if f in allowed]
    if finals and rng.random() < p_final and letters[-1] in FINAL_OF:
        candidate = FINAL_OF[letters[-1]]
        if candidate in allowed:
            letters[-1] = candidate
    return "".join(letters)


def sample_number(rng: random.Random, allowed: set[str]) -> str:
    digits = [d for d in charset.DIGITS if d in allowed] or list(charset.DIGITS)
    length = rng.randint(1, 4)
    return "".join(rng.choice(digits) for _ in range(length))


def sample_word(
    rng: random.Random,
    cfg: Config,
    allowed: set[str],
    lexicon: Sequence[str],
    max_word_len: int | None = None,
) -> str:
    limit = cfg.max_word_len if max_word_len is None else max(2, int(max_word_len))
    if rng.random() < cfg.p_digits:
        return sample_number(rng, allowed)
    if lexicon and rng.random() < cfg.p_real_word:
        candidates = [w for w in lexicon if len(w) <= limit] if limit < cfg.max_word_len else lexicon
        if candidates:
            return rng.choice(candidates)
    return sample_pseudo_word(rng, allowed, cfg.min_word_len, limit)


def sample_text(
    rng: random.Random,
    cfg: Config,
    allowed: set[str],
    lexicon: Sequence[str],
    max_len: int | None = None,
    max_words: int | None = None,
    max_word_len: int | None = None,
) -> str:
    """Sample a *logical* (typing order) text fragment of at most ``max_len`` chars.

    ``max_words``/``max_word_len`` override the config values; the training loop
    uses them for a length curriculum, because a CTC model fits short sequences
    much faster than long ones.
    """
    max_len = max_len or cfg.max_label_len
    upper = cfg.max_words if max_words is None else max(1, int(max_words))
    words: list[str] = []
    for _ in range(rng.randint(cfg.min_words, upper)):
        word = sample_word(rng, cfg, allowed, lexicon, max_word_len=max_word_len)
        if not word:
            continue
        candidate = " ".join(words + [word])
        if len(candidate) > max_len and words:
            break
        words.append(word)
        if len(" ".join(words)) >= max_len:
            break
    if not words:
        words = [sample_pseudo_word(rng, allowed, 3, 4)]
    text = " ".join(words)
    while len(text) > max_len and len(words) > 1:
        words.pop()
        text = " ".join(words)
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text
