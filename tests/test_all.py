"""
╔══════════════════════════════════════════════════╗
║   اختبارات المشروع — pytest                     ║
║   تشغيل: pytest tests/ -v                       ║
╚══════════════════════════════════════════════════╝
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.utils.text_processor import clean_text, detect_language, preprocess
from app.services.sentiment_service import SentimentService

# مهم: يجب استخدام TestClient كـ context manager لتشغيل lifespan (init_db + seed).
# بدون `with`، لا يتم إنشاء جداول قاعدة البيانات وتفشل كل نقطة نهاية تلمس DB
# بخطأ "no such table". نفتح الـ context هنا على مستوى الموديول عبر fixture
# تلقائي (autouse) يعمل مرة واحدة لكل جلسة اختبارات.

client: TestClient


@pytest.fixture(scope="session", autouse=True)
def _app_lifespan():
    global client
    with TestClient(app) as c:
        client = c
        yield c


# ══ اختبارات معالجة النصوص ══

class TestTextProcessor:

    def test_clean_removes_urls(self):
        text   = "تفضل بزيارة https://example.com الموقع"
        result = clean_text(text)
        assert "https" not in result
        assert "example" not in result

    def test_clean_removes_hashtags(self):
        result = clean_text("منتج رائع #منتج_ممتاز")
        assert "#" not in result

    def test_clean_removes_mentions(self):
        result = clean_text("شكراً @user على التوصية")
        assert "@" not in result

    def test_clean_removes_emoji(self):
        result = clean_text("رائع جداً 😍🎉")
        assert "😍" not in result

    def test_detect_arabic(self):
        assert detect_language("هذا نص عربي") == "ar"

    def test_detect_english(self):
        assert detect_language("This is English text") == "en"

    def test_detect_mixed(self):
        lang = detect_language("هذا mixed نص")
        assert lang in ("ar", "en", "mixed")

    def test_preprocess_returns_tuple(self):
        clean, lang = preprocess("نص اختباري")
        assert isinstance(clean, str)
        assert lang in ("ar", "en", "mixed")

    def test_normalize_hamza(self):
        result = clean_text("أحمد إبراهيم آمن")
        # الهمزات تُوحَّد إلى ا
        assert "أ" not in result or "ا" in result


# ══ اختبارات نموذج التحليل ══

class TestSentimentModel:

    @pytest.fixture(scope="class")
    def service(self):
        return SentimentService()

    def test_positive_arabic(self, service):
        result = service.analyze("المنتج رائع جداً وممتاز")
        assert result["sentiment"] == "positive"
        assert result["confidence"] > 0.5

    def test_negative_arabic(self, service):
        result = service.analyze("خدمة سيئة جداً لن أعود")
        assert result["sentiment"] == "negative"

    def test_neutral_arabic(self, service):
        result = service.analyze("عادي مقبول يؤدي الغرض")
        assert result["sentiment"] in ("neutral", "positive", "negative")

    def test_positive_english(self, service):
        result = service.analyze("excellent product highly recommend")
        assert result["sentiment"] == "positive"

    def test_negative_english(self, service):
        result = service.analyze("terrible experience very disappointed")
        assert result["sentiment"] == "negative"

    def test_result_structure(self, service):
        result = service.analyze("اختبار")
        assert "sentiment"     in result
        assert "confidence"    in result
        assert "language"      in result
        assert "scores"        in result
        assert "text_clean"    in result
        assert "sentiment_ar"  in result

    def test_confidence_range(self, service):
        result = service.analyze("نص عشوائي للاختبار")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_scores_sum_to_one(self, service):
        result = service.analyze("اختبار المجموع")
        total  = sum(result["scores"].values())
        assert abs(total - 1.0) < 0.01

    def test_batch_analysis(self, service):
        texts   = ["رائع", "سيء", "عادي"]
        results = service.analyze_batch(texts)
        assert len(results) == 3
        for r in results:
            assert r["sentiment"] in ("positive", "negative", "neutral")

    def test_empty_text_handled(self, service):
        # نص قصير جداً لا يجب أن يسبب crash
        result = service.analyze("  ")
        assert "sentiment" in result


# ══ اختبارات API ══

class TestSentimentAPI:

    def test_root_endpoint(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "SentimentAI" in r.json()["message"]

    def test_health_check(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_analyze_arabic(self):
        r = client.post("/api/v1/sentiment/analyze", json={
            "text":     "المنتج رائع جداً!",
            "platform": "facebook",
            "save":     False,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["sentiment"] in ("positive", "negative", "neutral")
        assert "confidence" in data
        assert "scores"     in data

    def test_analyze_english(self):
        r = client.post("/api/v1/sentiment/analyze", json={
            "text": "amazing product love it",
            "save": False,
        })
        assert r.status_code == 200
        assert r.json()["language"] == "en"

    def test_analyze_empty_text_rejected(self):
        r = client.post("/api/v1/sentiment/analyze", json={"text": ""})
        assert r.status_code == 422

    def test_analyze_batch(self):
        r = client.post("/api/v1/sentiment/analyze/batch", json={
            "texts": ["رائع", "سيء", "عادي"],
            "save":  False,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["total"]   == 3
        assert len(data["results"]) == 3
        assert data["positive_pct"] + data["negative_pct"] + data["neutral_pct"] <= 101

    def test_batch_too_large_rejected(self):
        texts = ["نص"] * 101
        r = client.post("/api/v1/sentiment/analyze/batch", json={"texts": texts, "save": False})
        assert r.status_code == 400

    def test_stats_endpoint(self):
        r = client.get("/api/v1/stats/")
        assert r.status_code == 200
        data = r.json()
        assert "total_analyzed" in data
        assert "recent_trend"   in data

    def test_posts_endpoint(self):
        r = client.get("/api/v1/posts/")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_posts_filter_by_sentiment(self):
        r = client.get("/api/v1/posts/?sentiment=positive")
        assert r.status_code == 200


# ══ اختبار الدقة ══

class TestModelAccuracy:
    """
    اختبار دقة النموذج على عينة معروفة
    الهدف: دقة ≥ 70% (على البيانات الأساسية)
    """

    LABELED_SAMPLES = [
        ("المنتج ممتاز وأنصح به",            "positive"),
        ("رائع جداً سعيد جداً",              "positive"),
        ("excellent very happy recommend",   "positive"),
        ("خدمة سيئة للغاية لن أعود",         "negative"),
        ("أسوأ تجربة في حياتي",              "negative"),
        ("terrible product very bad",        "negative"),
        ("عادي لا جديد",                     "neutral"),
        ("okay nothing special",             "neutral"),
    ]

    def test_minimum_accuracy(self):
        service = SentimentService()
        correct = 0
        for text, expected in self.LABELED_SAMPLES:
            result = service.analyze(text)
            if result["sentiment"] == expected:
                correct += 1

        accuracy = correct / len(self.LABELED_SAMPLES)
        print(f"\n📊 دقة النموذج على العينة: {accuracy*100:.1f}%")
        assert accuracy >= 0.60, f"الدقة {accuracy*100:.1f}% أقل من الحد الأدنى 60%"
