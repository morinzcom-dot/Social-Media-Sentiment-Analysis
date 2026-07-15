"""
قاموس مشاعر بسيط (عربي + إنجليزي) يُستخدم كميزة إضافية بجانب TF-IDF.
الفكرة: عدّ الكلمات الإيجابية/السلبية الصريحة بالنص كإشارة مباشرة تُضاف
لتمثيل TF-IDF، بدل الاعتماد فقط على ما تعلّمه النموذج ضمنياً من البيانات.
"""
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

POSITIVE_WORDS = {
    # عربي
    "رائع", "رائعة", "ممتاز", "ممتازة", "جميل", "جميلة", "أحب", "احب",
    "أنصح", "انصح", "أفضل", "افضل", "سعيد", "سعيدة", "شكرا", "شكراً",
    "يجنن", "حلو", "حلوة", "مبدع", "توب", "زين", "عجبني", "استثنائي",
    "موفق", "موفقة", "خطير", "احترافي", "احترافية", "سريع", "نظيف", "نظيفة",
    # English
    "great", "excellent", "amazing", "love", "recommend", "best", "happy",
    "awesome", "fantastic", "perfect", "wonderful", "good", "nice",
}
NEGATIVE_WORDS = {
    # عربي
    "سيء", "سيئ", "سيئة", "مشكلة", "مشاكل", "تأخير", "متأخر", "لايعمل",
    "لا يعمل", "ماشتغل", "ما يشتغل", "زفت", "خايس", "خايسة", "فاشل",
    "فاشلة", "أسوأ", "اسوأ", "محبط", "محبطة", "خسارة", "ضعيف", "ضعيفة",
    "واقع", "معطل", "معطلة", "احتيال", "غالي", "بطيء", "بطيئة", "مقرف",
    # English
    "bad", "terrible", "worst", "awful", "horrible", "disappointing",
    "poor", "broken", "useless", "frustrating", "never", "waste",
}

_word_re = re.compile(r"[\w\u0600-\u06FF]+")


class LexiconFeaturizer(BaseEstimator, TransformerMixin):
    """يحوّل كل نص إلى عمودين: (نسبة الكلمات الإيجابية، نسبة الكلمات السلبية)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            words = _word_re.findall(text.lower())
            n = max(len(words), 1)
            pos = sum(1 for w in words if w in POSITIVE_WORDS)
            neg = sum(1 for w in words if w in NEGATIVE_WORDS)
            # نضخّم الإشارة (×5) لتكون قابلة للمقارنة مع ميزات TF-IDF المتناثرة
            rows.append([pos / n * 5, neg / n * 5])
        return np.array(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(["lexicon_pos", "lexicon_neg"])
