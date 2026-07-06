"""
╔══════════════════════════════════════════════════╗
║   معالجة النصوص — تنظيف العربية والإنجليزية     ║
╚══════════════════════════════════════════════════╝

تشمل:
  • إزالة الروابط، الهاشتاقات، الإيموجي
  • تطبيع الهمزات والألف المقصورة والتشكيل
  • كشف اللغة (عربي / إنجليزي / مختلط)
  • تقطيع النص (Tokenization)
"""

import re
import unicodedata
from typing import Tuple


# ══ أنماط Regex ══
URL_PATTERN      = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN  = re.compile(r"@\w+")
HASHTAG_PATTERN  = re.compile(r"#\w+")
EMOJI_PATTERN    = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # وجوه
    "\U0001F300-\U0001F5FF"   # رموز
    "\U0001F680-\U0001F6FF"   # مركبات
    "\U0001F1E0-\U0001F1FF"   # أعلام
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
MULTI_SPACE  = re.compile(r"\s{2,}")
ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")
ENGLISH_RANGE = re.compile(r"[a-zA-Z]")

# حروف الهمزات والألف للتطبيع
HAMZA_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا",
    "ة": "ه",
    "ى": "ي",
    "ؤ": "و",
    "ئ": "ي",
})

# كلمات التوقف العربية (Stop Words)
ARABIC_STOPWORDS = {
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "التي", "الذي",
    "كان", "كانت", "هو", "هي", "هم", "نحن", "أنا", "أنت", "لكن", "وكذلك",
    "أو", "و", "ثم", "حتى", "لأن", "إن", "إذا", "بعد", "قبل", "عند",
    "ليس", "لم", "لن", "قد", "كل", "بعض", "غير", "جداً", "جدا",
}


def detect_language(text: str) -> str:
    """
    كشف لغة النص
    Returns: "ar" | "en" | "mixed"
    """
    arabic_chars  = len(ARABIC_RANGE.findall(text))
    english_chars = len(ENGLISH_RANGE.findall(text))
    total = arabic_chars + english_chars

    if total == 0:
        return "ar"

    arabic_ratio = arabic_chars / total

    if arabic_ratio > 0.7:
        return "ar"
    elif arabic_ratio < 0.3:
        return "en"
    else:
        return "mixed"


def normalize_arabic(text: str) -> str:
    """
    تطبيع النص العربي:
    - توحيد الهمزات
    - إزالة التشكيل (الحركات)
    - توحيد الألف المقصورة والتاء المربوطة
    """
    # إزالة التشكيل (Harakat)
    text = "".join(ch for ch in text if not unicodedata.category(ch) == "Mn")

    # تطبيع الهمزات والحروف
    text = text.translate(HAMZA_MAP)

    # إزالة التطويل (مثل: رائعةةة → رائعة)
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    return text


def clean_text(text: str, keep_hashtags: bool = False) -> str:
    """
    تنظيف النص الكامل (عربي وإنجليزي)

    Args:
        text: النص الخام
        keep_hashtags: هل نحتفظ بالهاشتاقات؟ (مفيدة أحياناً للسياق)

    Returns:
        نص منظّف جاهز للنموذج
    """
    if not text or not text.strip():
        return ""

    # 1. إزالة الروابط
    text = URL_PATTERN.sub(" ", text)

    # 2. إزالة أو الاحتفاظ بالهاشتاقات
    if keep_hashtags:
        text = HASHTAG_PATTERN.sub(lambda m: m.group().replace("#", " "), text)
    else:
        text = HASHTAG_PATTERN.sub(" ", text)

    # 3. إزالة المنشنات
    text = MENTION_PATTERN.sub(" ", text)

    # 4. إزالة الإيموجي
    text = EMOJI_PATTERN.sub(" ", text)

    # 5. إزالة علامات HTML إن وجدت
    text = re.sub(r"<[^>]+>", " ", text)

    # 6. تطبيع النص العربي
    lang = detect_language(text)
    if lang in ("ar", "mixed"):
        text = normalize_arabic(text)

    # 7. إزالة الأحرف الخاصة (مع الإبقاء على العربية والإنجليزية والأرقام)
    text = re.sub(r"[^\w\u0600-\u06FF\s]", " ", text)

    # 8. إزالة المسافات المتعددة
    text = MULTI_SPACE.sub(" ", text).strip()

    return text


def remove_stopwords(text: str, language: str = "ar") -> str:
    """إزالة كلمات التوقف من النص العربي"""
    if language not in ("ar", "mixed"):
        return text
    tokens = text.split()
    filtered = [t for t in tokens if t not in ARABIC_STOPWORDS]
    return " ".join(filtered)


def preprocess(text: str) -> Tuple[str, str]:
    """
    خط أنابيب المعالجة الكامل

    Returns:
        Tuple[نص_منظف, لغة_مكتشفة]
    """
    language = detect_language(text)
    cleaned  = clean_text(text)
    cleaned  = remove_stopwords(cleaned, language)
    return cleaned, language


# ══ اختبار سريع ══
if __name__ == "__main__":
    samples = [
        "المنتج رائعةةة جداً!! 😍 #منتج_ممتاز https://example.com",
        "This product is absolutely amazing!! @user #great",
        "خدمة سيئة للغاية وما يسوى https://bad.com 😠😠",
        "عادي مو ناقص مو زيادة",
    ]
    for s in samples:
        clean, lang = preprocess(s)
        print(f"Original : {s}")
        print(f"Cleaned  : {clean}")
        print(f"Language : {lang}")
        print("-" * 60)
