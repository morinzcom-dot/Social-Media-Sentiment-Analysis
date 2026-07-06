"""
╔══════════════════════════════════════════════════════════════════╗
║          خدمة تحليل المشاعر — المحرك الأساسي                   ║
║                                                                  ║
║  يدعم نموذجَين:                                                  ║
║  1) Logistic Regression  — سريع، خفيف، مناسب للتطوير           ║
║  2) AraBERT              — دقيق، للغة العربية تحديداً           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from app.config import settings
from app.utils.text_processor import clean_text as _light_clean, detect_language as _detect_lang



# ══ ثوابت ══
LABELS    = ["positive", "negative", "neutral"]
LABELS_AR = {"positive": "إيجابي 😊", "negative": "سلبي 😠", "neutral": "محايد 😐"}
MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"


# ══════════════════════════════════════════════
#  النموذج 1: Logistic Regression (خفيف وسريع)
# ══════════════════════════════════════════════

class LogisticSentimentModel:
    """
    نموذج Logistic Regression مع TF-IDF Vectorizer
    - مناسب للتطوير والاختبار السريع
    - يعمل بدون GPU
    - دقة مقبولة على النصوص العربية البسيطة
    """

    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.is_trained  = False
        self._load_or_create()

    def _load_or_create(self):
        """تحميل نموذج محفوظ أو إنشاء نموذج جديد مع بيانات تدريب أساسية"""
        model_path = MODEL_DIR / "logistic_model.pkl"

        if model_path.exists():
            print("📂 تحميل نموذج Logistic Regression المحفوظ...")
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.classifier = data["classifier"]
            self.is_trained  = True
            print("✅ النموذج جاهز")
        else:
            print("🔧 إنشاء نموذج جديد وتدريبه على بيانات أساسية...")
            self._train_baseline()

    def _train_baseline(self):
        """تدريب أولي على بيانات أساسية مضمّنة"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer

        # بيانات تدريب بسيطة (يمكن استبدالها بـ Dataset حقيقي)
        training_data = [
            # إيجابي — عربي
            ("رائع جداً ممتاز", "positive"),
            ("جيد جداً سعيد بالشراء", "positive"),
            ("ممتاز خدمة رائعة أحببته", "positive"),
            ("منتج ممتاز ينصح به", "positive"),
            ("تجربة جميلة مريحة ورائعة", "positive"),
            ("افضل منتج اشتريته", "positive"),
            ("جودة عالية سعر مناسب", "positive"),
            ("خدمة ممتازة فريق محترف", "positive"),
            ("سعيد جداً بالنتيجة", "positive"),
            ("مذهل تجاوز توقعاتي", "positive"),
            # إيجابي — إنجليزي
            ("excellent product amazing quality", "positive"),
            ("great service highly recommend", "positive"),
            ("love it best purchase ever", "positive"),
            ("wonderful experience very satisfied", "positive"),
            ("outstanding quality perfect", "positive"),
            # سلبي — عربي
            ("سيء جداً مشكلة كبيرة", "negative"),
            ("رديء لن أشتري مرة أخرى", "negative"),
            ("خدمة سيئة مزعجة للغاية", "negative"),
            ("جودة ضعيفة مخيب للآمال", "negative"),
            ("تجربة سيئة أسوأ منتج", "negative"),
            ("محبط جداً لا أنصح به", "negative"),
            ("كارثة إهمال وعدم احترام", "negative"),
            ("لا يستحق المال هدر", "negative"),
            # سلبي — إنجليزي
            ("terrible product waste of money", "negative"),
            ("horrible experience very disappointed", "negative"),
            ("worst purchase ever avoid", "negative"),
            ("poor quality broken quickly", "negative"),
            ("awful service rude staff", "negative"),
            # محايد — عربي
            ("عادي مو ناقص مو زيادة", "neutral"),
            ("مقبول يمكن تحسينه", "neutral"),
            ("متوسط لا شيء مميز", "neutral"),
            ("اشتريته وصل بوقت", "neutral"),
            ("منتج معقول بالسعر", "neutral"),
            ("لا بأس يؤدي الغرض", "neutral"),
            # محايد — إنجليزي
            ("okay nothing special average", "neutral"),
            ("decent product does the job", "neutral"),
            ("alright nothing extraordinary", "neutral"),
            ("acceptable quality average price", "neutral"),
        ]

        texts  = [t for t, _ in training_data]
        labels = [l for _, l in training_data]

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",       # مفيد للعربية (n-grams على مستوى الأحرف)
            ngram_range=(2, 4),
            max_features=10000,
            sublinear_tf=True,
        )

        X = self.vectorizer.fit_transform(texts)

        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            
        )
        self.classifier.fit(X, labels)
        self.is_trained = True

        # حفظ النموذج
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / "logistic_model.pkl", "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
            }, f)
        print("💾 تم حفظ النموذج")

    def train(self, texts: List[str], labels: List[str]) -> Dict:
        """إعادة تدريب النموذج على بيانات جديدة"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=15000, sublinear_tf=True,
        )
        X_tr = self.vectorizer.fit_transform(X_train)
        X_te = self.vectorizer.transform(X_test)

        self.classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.classifier.fit(X_tr, y_train)

        preds    = self.classifier.predict(X_te)
        accuracy = accuracy_score(y_test, preds)
        report   = classification_report(y_test, preds, output_dict=True)

        # حفظ النموذج المحدّث
        with open(MODEL_DIR / "logistic_model.pkl", "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, f)

        return {"accuracy": round(accuracy * 100, 2), "report": report}

    def predict(self, text_clean: str) -> Tuple[str, float, Dict[str, float]]:
        """
        تحليل نص واحد

        Returns:
            (label, confidence, scores_dict)
        """
        if not self.is_trained:
            raise RuntimeError("النموذج غير مدرَّب")

        vec   = self.vectorizer.transform([text_clean])
        proba = self.classifier.predict_proba(vec)[0]
        classes = self.classifier.classes_

        scores = {cls: float(p) for cls, p in zip(classes, proba)}
        label  = max(scores, key=scores.get)
        conf   = scores[label]

        return label, conf, scores

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float, Dict]]:
        vec    = self.vectorizer.transform(texts)
        probas = self.classifier.predict_proba(vec)
        classes = self.classifier.classes_
        results = []
        for proba in probas:
            scores = {cls: float(p) for cls, p in zip(classes, proba)}
            label  = max(scores, key=scores.get)
            conf   = scores[label]
            results.append((label, conf, scores))
        return results


# ══════════════════════════════════════════════
#  النموذج 2: AraBERT (دقيق للغة العربية)
# ══════════════════════════════════════════════

class AraBERTSentimentModel:
    """
    نموذج AraBERT المُدرَّب مسبقاً من AraGPT
    - دقة أعلى بكثير على النصوص العربية
    - يتطلب transformers + torch
    - يُنصح بـ GPU لتدريب أسرع
    """

    def __init__(self):
        self.tokenizer = None
        self.model     = None
        self.device    = "cpu"
        self._load()

    def _load(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🤖 تحميل AraBERT على {self.device}...")

            model_name = settings.ARABERT_MODEL_NAME

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # تحميل نموذج fine-tuned إذا وُجد، وإلا النموذج الأساسي
            fine_tuned_path = MODEL_DIR / "arabert_finetuned"
            if fine_tuned_path.exists():
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(fine_tuned_path)
                )
                print("✅ تم تحميل AraBERT المُخصَّص")
            else:
                # النموذج الأساسي — يحتاج fine-tuning
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=3
                )
                print("⚠️  AraBERT الأساسي — يُنصح بإجراء Fine-tuning")

            self.model.to(self.device)
            self.model.eval()

        except ImportError:
            print("⚠️  transformers/torch غير مثبّتَين — الرجوع إلى Logistic Regression")
            raise

    def predict(self, text_clean: str) -> Tuple[str, float, Dict[str, float]]:
        import torch

        inputs = self.tokenizer(
            text_clean,
            return_tensors="pt",
            truncation=True,
            max_length=settings.MAX_TEXT_LENGTH,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            proba  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # ترتيب: positive=0, negative=1, neutral=2 (يعتمد على ترتيب التدريب)
        label_map = {0: "positive", 1: "negative", 2: "neutral"}
        scores    = {label_map[i]: float(p) for i, p in enumerate(proba)}
        label     = max(scores, key=scores.get)
        conf      = scores[label]
        return label, conf, scores

    def fine_tune(self, texts: List[str], labels: List[str], epochs: int = 3) -> Dict:
        """
        Fine-tuning على بيانات مخصصة
        يحتاج GPU وبيانات كافية (500+ مثال على الأقل)
        """
        from transformers import Trainer, TrainingArguments
        import torch
        from torch.utils.data import Dataset

        class SentimentDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, label2id):
                self.encodings = tokenizer(
                    texts, truncation=True, padding=True, max_length=128
                )
                self.labels = [label2id[l] for l in labels]

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        label2id = {"positive": 0, "negative": 1, "neutral": 2}
        dataset  = SentimentDataset(texts, labels, self.tokenizer, label2id)

        args = TrainingArguments(
            output_dir=str(MODEL_DIR / "arabert_finetuned"),
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            warmup_steps=100,
            save_strategy="epoch",
            logging_dir="./logs",
        )

        trainer = Trainer(model=self.model, args=args, train_dataset=dataset)
        trainer.train()
        trainer.save_model(str(MODEL_DIR / "arabert_finetuned"))

        return {"status": "تم Fine-tuning بنجاح", "epochs": epochs}


# ══════════════════════════════════════════════
#  المصنع الموحَّد — SentimentService
# ══════════════════════════════════════════════

class SentimentService:
    """
    الواجهة الموحّدة لتحليل المشاعر
    تختار النموذج المناسب تلقائياً
    """
    _instance: Optional["SentimentService"] = None

    def __init__(self):
        self.model_type = settings.MODEL_TYPE
        self.model      = self._load_model()

    @classmethod
    def get_instance(cls) -> "SentimentService":
        """Singleton — نموذج واحد لكل تشغيل"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        if self.model_type == "arabert":
            try:
                return AraBERTSentimentModel()
            except Exception as e:
                print(f"⚠️  فشل تحميل AraBERT ({e}). الرجوع إلى Logistic Regression")
                self.model_type = "logistic"
                return LogisticSentimentModel()
        return LogisticSentimentModel()

    def analyze(self, text: str) -> Dict:
        """تحليل نص واحد — الدالة الرئيسية"""
        text_clean = _light_clean(text)
        language   = _detect_lang(text)

        if not text_clean.strip():
            text_clean = text.strip()

        label, confidence, scores = self.model.predict(text_clean)

        return {
            "text_original":  text,
            "text_clean":     text_clean,
            "sentiment":      label,
            "sentiment_ar":   LABELS_AR[label],
            "confidence":     round(confidence, 4),
            "confidence_pct": f"{confidence * 100:.1f}%",
            "language":       language,
            "model_used":     self.model_type,
            "scores": {
                "positive": round(scores.get("positive", 0), 4),
                "negative": round(scores.get("negative", 0), 4),
                "neutral":  round(scores.get("neutral",  0), 4),
            },
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """تحليل دفعة من النصوص"""
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results

    def train(self, texts: List[str], labels: List[str]) -> Dict:
        """تدريب/إعادة تدريب النموذج"""
        if isinstance(self.model, LogisticSentimentModel):
            return self.model.train(texts, labels)
        return {"error": "التدريب اليدوي متاح فقط لنموذج Logistic Regression"}


# ── اختبار سريع ──
if __name__ == "__main__":
    service = SentimentService()
    samples = [
        "المنتج رائع جداً وأنصح به بشدة!",
        "خدمة سيئة جداً لن أتعامل معهم مرة أخرى",
        "عادي، لا شيء مميز",
        "This product is absolutely amazing!",
        "Terrible experience, very disappointed",
    ]
    for s in samples:
        result = service.analyze(s)
        print(f"النص    : {s}")
        print(f"النتيجة : {result['sentiment_ar']} ({result['confidence_pct']})")
        print(f"اللغة   : {result['language']}")
        print("-" * 60)
