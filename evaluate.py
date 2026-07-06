"""
╔══════════════════════════════════════════════════════════════╗
║   تقييم النموذج وتصور النتائج                               ║
║                                                              ║
║   يُنتج:                                                     ║
║   • تقرير أداء شامل (Accuracy, F1, Precision, Recall)       ║
║   • مصفوفة الارتباك (Confusion Matrix)                      ║
║   • مقارنة بين نماذج متعددة                                 ║
║   • رسوم بيانية للتوزيع والأداء                             ║
╚══════════════════════════════════════════════════════════════╝

الاستخدام:
    python evaluate.py
    python evaluate.py --compare   (مقارنة نماذج مختلفة)
"""

import pickle
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════
#  بيانات الاختبار
# ══════════════════════════════════════════════

TEST_DATA: List[Tuple[str, str]] = [
    # إيجابي
    ("المنتج رائع وأنصح به بشدة", "positive"),
    ("خدمة ممتازة وسريعة جداً", "positive"),
    ("سعيد جداً بالشراء تجربة ممتازة", "positive"),
    ("excellent product love it so much", "positive"),
    ("amazing quality fast delivery", "positive"),
    ("great service highly recommend", "positive"),
    ("اشتريته مرة ثانية بسبب جودته", "positive"),
    ("يستحق كل فلس دفعته", "positive"),
    ("best purchase ever made wonderful", "positive"),
    ("wonderful experience very satisfied", "positive"),
    # سلبي
    ("منتج سيء جداً لن أشتري مرة أخرى", "negative"),
    ("خدمة مزعجة وسيئة للغاية", "negative"),
    ("أسوأ تجربة شراء في حياتي ندمت", "negative"),
    ("terrible product waste of money avoid", "negative"),
    ("horrible experience never again", "negative"),
    ("poor quality broke quickly disappointed", "negative"),
    ("وصل مكسور وغير مناسب أبداً", "negative"),
    ("غير راضٍ إطلاقاً سأطلب الاسترداد", "negative"),
    ("worst purchase I ever made", "negative"),
    ("awful service rude staff unhelpful", "negative"),
    # محايد
    ("عادي يؤدي الغرض لا أكثر", "neutral"),
    ("مقبول للسعر المدفوع معقول", "neutral"),
    ("okay nothing special does the job", "neutral"),
    ("decent average price acceptable", "neutral"),
    ("متوسط لا سيء ولا ممتاز", "neutral"),
    ("not bad could be better though", "neutral"),
    ("يفي بالغرض اليومي البسيط", "neutral"),
    ("so so average product nothing wow", "neutral"),
    ("لا بأس به بالنسبة للسعر", "neutral"),
    ("alright nothing extraordinary here", "neutral"),
]


# ══════════════════════════════════════════════
#  تقييم النموذج المحفوظ
# ══════════════════════════════════════════════

def evaluate_saved_model() -> Dict:
    """تقييم النموذج المحفوظ على بيانات الاختبار"""
    model_path = MODEL_DIR / "logistic_model.pkl"
    if not model_path.exists():
        print("❌ النموذج غير موجود — شغّل: python seeder.py")
        return {}

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    vec = saved["vectorizer"]
    clf = saved["classifier"]

    texts  = [t for t, _ in TEST_DATA]
    labels = [l for _, l in TEST_DATA]

    X      = vec.transform(texts)
    preds  = clf.predict(X)
    probas = clf.predict_proba(X)
    confs  = probas.max(axis=1)

    accuracy = accuracy_score(labels, preds)
    f1       = f1_score(labels, preds, average="weighted")
    prec     = precision_score(labels, preds, average="weighted")
    rec      = recall_score(labels, preds, average="weighted")

    report = classification_report(
        labels, preds,
        target_names=["positive","negative","neutral"],
        output_dict=True,
    )

    print_evaluation_report(accuracy, f1, prec, rec, report,
                             labels, preds, confs, texts)

    return {
        "accuracy": round(accuracy*100, 2),
        "f1":       round(f1*100, 2),
        "report":   report,
    }


def print_evaluation_report(acc, f1, prec, rec, report,
                              labels, preds, confs, texts):
    """طباعة تقرير الأداء بشكل منسق"""

    LABELS_AR = {"positive": "إيجابي 😊", "negative": "سلبي 😠", "neutral": "محايد 😐"}
    W = 58

    print(f"\n{'═'*W}")
    print(f"  📊 تقرير تقييم النموذج — SentimentAI")
    print(f"{'═'*W}")

    # مقاييس رئيسية
    metrics = [
        ("Accuracy  (الدقة الكلية)", acc),
        ("F1-Score  (المتوسط الموزون)", f1),
        ("Precision (الدقة الإيجابية)", prec),
        ("Recall    (الاستدعاء)", rec),
    ]
    print(f"\n  المقاييس الرئيسية:")
    for name, val in metrics:
        bar   = "█" * int(val * 20)
        empty = "░" * (20 - int(val * 20))
        color = "✅" if val >= 0.80 else "⚠️ " if val >= 0.65 else "❌"
        print(f"  {color} {name:<35} {val*100:5.1f}% |{bar}{empty}|")

    # تفصيل لكل تصنيف
    print(f"\n  الأداء لكل تصنيف:")
    print(f"  {'التصنيف':12} {'Precision':>11} {'Recall':>9} {'F1':>9} {'العينات':>9}")
    print(f"  {'-'*52}")
    for lbl in ["positive", "negative", "neutral"]:
        r = report[lbl]
        print(
            f"  {LABELS_AR[lbl]:15}"
            f"  {r['precision']*100:7.1f}%"
            f"  {r['recall']*100:7.1f}%"
            f"  {r['f1-score']*100:7.1f}%"
            f"  {int(r['support']):7}"
        )

    # مصفوفة الارتباك
    print(f"\n  Confusion Matrix:")
    print(f"  {'':12} {'إيجابي':>8} {'سلبي':>8} {'محايد':>8}")
    print(f"  {'-'*38}")
    lbls = ["positive","negative","neutral"]
    cm = confusion_matrix(labels, preds, labels=lbls)
    for i, lbl in enumerate(["إيجابي", "سلبي", "محايد"]):
        row = cm[i]
        cells = []
        for j, v in enumerate(row):
            if i == j:
                cells.append(f"  \033[32m{v:6}\033[0m")
            else:
                cells.append(f"  {v:6}")
        print(f"  {lbl:12}{''.join(cells)}")

    # متوسط الثقة
    avg_conf = confs.mean()
    print(f"\n  متوسط درجة الثقة: {avg_conf*100:.1f}%")

    # أمثلة صحيحة وخاطئة
    wrong = [(texts[i], labels[i], preds[i])
             for i in range(len(labels)) if labels[i] != preds[i]]
    if wrong:
        print(f"\n  ⚠️  الأمثلة الخاطئة ({len(wrong)}):")
        for txt, true, pred in wrong[:5]:
            print(f"    النص    : {txt[:50]}")
            print(f"    الصحيح  : {LABELS_AR[true]}  |  التوقع: {LABELS_AR[pred]}")
            print()
    else:
        print(f"\n  ✅ جميع الأمثلة صحيحة!")

    print(f"{'═'*W}\n")


# ══════════════════════════════════════════════
#  مقارنة النماذج
# ══════════════════════════════════════════════

def compare_models():
    """مقارنة أداء عدة نماذج على نفس البيانات"""

    from seeder import BUILTIN_DATA
    texts  = [t for t, _ in BUILTIN_DATA]
    labels = [l for _, l in BUILTIN_DATA]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    W = 62
    print(f"\n{'═'*W}")
    print(f"  🔬 مقارنة النماذج — Model Comparison")
    print(f"{'═'*W}")
    print(f"  {'النموذج':<28} {'Accuracy':>10} {'F1':>8} {'الوقت':>8}")
    print(f"  {'-'*54}")

    import time

    models = [
        ("Logistic Regression (C=2)",
         Pipeline([
             ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5),
                                       max_features=20000, sublinear_tf=True)),
             ("clf",   LogisticRegression(C=2.0, max_iter=2000, random_state=42)),
         ])),
        ("Logistic Regression (C=5)",
         Pipeline([
             ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5),
                                       max_features=20000, sublinear_tf=True)),
             ("clf",   LogisticRegression(C=5.0, max_iter=2000, random_state=42)),
         ])),
        ("Logistic (word n-gram)",
         Pipeline([
             ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1,3),
                                       max_features=15000, sublinear_tf=True)),
             ("clf",   LogisticRegression(C=2.0, max_iter=2000, random_state=42)),
         ])),
        ("Linear SVM",
         Pipeline([
             ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5),
                                       max_features=20000, sublinear_tf=True)),
             ("clf",   LinearSVC(C=1.0, max_iter=2000, random_state=42)),
         ])),
        ("Naive Bayes (char)",
         Pipeline([
             ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4),
                                       max_features=15000)),
             ("clf",   MultinomialNB(alpha=0.1)),
         ])),
    ]

    results = []
    for name, pipe in models:
        t0 = time.time()
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        elapsed = time.time() - t0

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average="weighted")

        results.append((name, acc, f1, elapsed))
        status = "🥇" if acc == max(r[1] for r in results) else "  "
        print(f"  {status} {name:<28} {acc*100:8.1f}%  {f1*100:6.1f}%  {elapsed:5.2f}s")

    # الفائز
    best = max(results, key=lambda r: r[1])
    print(f"\n  🏆 الأفضل: {best[0]} بدقة {best[1]*100:.1f}%")

    # حفظ نتائج المقارنة
    out = [{
        "model":    r[0],
        "accuracy": round(r[1]*100, 2),
        "f1":       round(r[2]*100, 2),
        "time_sec": round(r[3], 3),
    } for r in results]

    with open(MODEL_DIR / "comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"  ✅ نتائج المقارنة محفوظة: data/models/comparison_results.json")
    print(f"{'═'*W}\n")

    return out


# ══════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SentimentAI — Model Evaluator")
    parser.add_argument("--compare", action="store_true",
                        help="مقارنة عدة نماذج مختلفة")
    args = parser.parse_args()

    if args.compare:
        compare_models()
    else:
        evaluate_saved_model()


if __name__ == "__main__":
    main()
