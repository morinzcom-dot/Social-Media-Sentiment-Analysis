# SentimentAI — نظام تحليل المشاعر 🧠

نظام متكامل لتحليل مشاعر منشورات وسائل التواصل الاجتماعي بالعربية والإنجليزية.

---

## 🗂️ هيكل المشروع

```
sentiment-api/
├── app/
│   ├── main.py                    # نقطة الدخول الرئيسية
│   ├── config.py                  # الإعدادات (.env)
│   ├── database.py                # اتصال قاعدة البيانات
│   ├── models/
│   │   ├── db_models.py           # جداول SQLAlchemy
│   │   └── schemas.py             # مخططات Pydantic
│   ├── routers/
│   │   ├── sentiment.py           # /api/v1/sentiment
│   │   ├── posts.py               # /api/v1/posts
│   │   └── stats.py               # /api/v1/stats
│   ├── services/
│   │   ├── sentiment_service.py   # محرك التحليل (LR + AraBERT)
│   │   └── facebook_collector.py  # جمع بيانات Facebook
│   └── utils/
│       └── text_processor.py      # تنظيف النصوص
├── data/models/                   # النماذج المحفوظة
├── tests/
│   └── test_all.py                # اختبارات pytest
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚡ تشغيل سريع

```bash
# 1. إنشاء بيئة افتراضية
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows

# 2. تثبيت المتطلبات
pip install -r requirements.txt

# 3. إعداد متغيرات البيئة
cp .env.example .env
# عدّل .env وأضف مفاتيح Facebook إن أردت

# 4. تشغيل الخادم
uvicorn app.main:app --reload --port 8000

# 5. تشغيل الاختبارات
pytest tests/ -v
```

---

## 🔌 نقاط نهاية API

| الطريقة | المسار | الوصف |
|---------|--------|-------|
| POST | `/api/v1/sentiment/analyze` | تحليل نص واحد |
| POST | `/api/v1/sentiment/analyze/batch` | تحليل دفعة نصوص |
| POST | `/api/v1/sentiment/train` | تدريب النموذج |
| GET | `/api/v1/posts/` | جلب المنشورات المحفوظة |
| GET | `/api/v1/stats/` | إحصاءات لوحة التحكم |
| GET | `/docs` | توثيق Swagger التفاعلي |

---

## 🖥️ الواجهة (Frontend)

واجهة ويب بسيطة بصفحة واحدة موجودة في `frontend/index.html` — بدون أي خطوات بناء (build)، تفتح مباشرة بالمتصفح.

**التشغيل:**
1. شغّلوا الـ API أولاً (`uvicorn app.main:app --reload`)
2. افتحوا `frontend/index.html` مباشرة بالمتصفح (أو قدّموه عبر `python -m http.server 5500` من داخل مجلد `frontend`)
3. تأكدوا إن حقل "الخادم" أعلى الصفحة يشير لعنوان الـ API الصحيح (افتراضياً `http://127.0.0.1:8000`)

**المزايا:**
- محلّل مشاعر حي: الصق أي نص عربي واحصل على النتيجة فوراً (يتصل بـ `POST /api/v1/sentiment/analyze`)
- لوحة متابعة: إجمالي المنشورات، توزيع المشاعر، اتجاه زمني، تفصيل حسب المنصة، أحدث المنشورات، الكلمات الأكثر تكراراً (`GET /api/v1/stats/`, `GET /api/v1/posts/`)

⚠️ **CORS:** إذا فتحتوا الملف مباشرة (`file://`) أو من منفذ مختلف، تأكدوا إن `ALLOWED_ORIGINS` بملف `.env` يسمح بمصدر الواجهة، وإلا ستُحجب الطلبات من المتصفح.

```python
import httpx

# تحليل نص عربي
response = httpx.post("http://localhost:8000/api/v1/sentiment/analyze", json={
    "text": "المنتج رائع جداً وأنصح به بشدة!",
    "platform": "facebook",
    "save": True
})

print(response.json())
# {
#   "sentiment": "positive",
#   "sentiment_ar": "إيجابي 😊",
#   "confidence": 0.923,
#   "confidence_pct": "92.3%",
#   "language": "ar",
#   "scores": {"positive": 0.923, "negative": 0.04, "neutral": 0.037}
# }
```

---

## 🧠 النماذج المدعومة

| النموذج | الدقة (بيانات اختبار محجوزة) | السرعة | الاستخدام |
|---------|-------|--------|-----------|
| Logistic Regression | **76.8%** (Cross-Val: 77.3% ± 0.4%) | سريع جداً | `MODEL_TYPE=logistic` |
| AraBERT | ~85-90% (متوقع، لم يُدرَّب في هذه البيئة — راجع القسم أدناه) | متوسط | `MODEL_TYPE=arabert` |

### 📚 بيانات التدريب (31,578 مثال حقيقي — عربي وإنجليزي)

- [ASTD](https://github.com/mahmoudnabil/ASTD) — تغريدات عربية سياسية/اجتماعية
- [HARD](https://github.com/elnagara/HARD-Arabic-Dataset) — تقييمات فنادق عربية حقيقية
- [ArSarcasm](https://github.com/iabufarha/ArSarcasm) (قسم SemEval) — تغريدات عربية متنوعة المواضيع
- **[Twitter US Airline Sentiment](https://github.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-)** — 14,640 تغريدة إنجليزية حقيقية (إيجابي/سلبي/محايد) — تدعم اللغة الإنجليزية بشكل حقيقي بدل الاعتماد على أمثلة يدوية قليلة فقط
- 85 مثال يدوي (عربي + إنجليزي)

**التوزيع اللغوي:** ~76% عربي (23,890) و ~24% إنجليزي حقيقي (7,363) — تحسّن كبير من النسخة السابقة (أقل من 1% إنجليزي).

سكربت البناء: `python seeder.py --source csv --file data/combined_training_data.csv`

### ⚠️ ملاحظة صادقة حول الدقة والتعميم

- **الدقة الرسمية 76.8%** (Cross-Val 77.3%) — دمج لغتين رفع صعوبة المهمة قليلاً مقارنة بنسخة عربي فقط (78.7%)، لكنه رفع جودة الدعم الفعلي للإنجليزية بشكل كبير.
- **اختبار حي على 10 جمل إنجليزية متنوعة:** 90% (9/10) — نتيجة مبنية الآن على 7,363 تغريدة إنجليزية حقيقية بدل تكرار 30 جملة يدوية فقط.
- **نقطة ضعف معروفة:** الجمل ذات المعنى الضمني (سخرية، مديح بصيغة سلبية) لا تزال صعبة على Logistic Regression في كلا اللغتين — هذا حد بنيوي يحتاج AraBERT/نموذج سياقي لتخطيه.

---

## 🚀 النشر (Deployment)

المشروع يحتوي `Dockerfile`, `Procfile`, و`render.yaml` جاهزة للنشر مباشرة.

### ⚠️ ملاحظة مهمة وصادقة حول "المجاني" على Render

- **خدمة الويب (API) مجانية فعلاً** على Render — 750 ساعة تشغيل شهرياً، بدون بطاقة ائتمان. القيد الوحيد: تنطفي بعد 15 دقيقة خمول، وأول طلب بعدها ياخذ ~30-60 ثانية للاستيقاظ.
- **قاعدة بيانات PostgreSQL المُدارة من Render ليست مجانية بشكل دائم** — نسختها التجريبية تُحذف تلقائياً بعد 30-90 يوماً، وبعدها تحتاج خطة مدفوعة (من $7/شهر). هذا سبب شائع لمفاجأة "ما طلع مجاني".

**الحل المعتمد بـ`render.yaml` الحالي:** استخدام **SQLite** بدل قاعدة بيانات Render المدفوعة — مجاني 100% بدون بطاقة ائتمان. القيد الوحيد: التخزين على الخطة المجانية غير دائم، فقاعدة البيانات تُصفَّر مع كل إعادة نشر — مقبول تماماً لعرض تقديمي/مشروع أكاديمي.

**لو احتجتم بيانات دائمة فعلاً**، استخدموا قاعدة بيانات مجانية دائماً من مزوّد خارجي (بدون علاقة بـRender):
- [Neon](https://neon.tech) — PostgreSQL مجاني دائم
- [Supabase](https://supabase.com) — PostgreSQL مجاني دائم

ثم ضعوا رابط الاتصال في متغير `DATABASE_URL` بإعدادات Render بدل قيمة SQLite.

### بديل بدون بطاقة نهائياً: Vercel (للـ API) + GitHub Pages (للواجهة)

لو واجهتوا طلب بطاقة بكل منصة (Render, Railway...)، هذا مسار موثّق رسمياً من Vercel ولا يطلب بطاقة أبداً:

**نشر الـ API على Vercel:**
1. الملفات `main.py` و`vercel.json` بجذر المشروع جاهزة أصلاً (تشير لتطبيق FastAPI الفعلي بـ`app/main.py`)
2. روح [vercel.com](https://vercel.com) → سجّل بحساب GitHub (بدون بطاقة) → **Add New Project** → اختر الـ repo
3. ⚠️ **مهم:** Vercel serverless يعني الحاوية غير دائمة — SQLite المحلي لن يحتفظ بالبيانات بين الطلبات. لازم تستخدموا قاعدة بيانات خارجية دائمة مثل [Neon](https://neon.tech) (مجاني، بدون بطاقة): أنشئوا قاعدة بيانات فيه، وضعوا رابط الاتصال (يبدأ بـ`postgresql://`) بمتغير `DATABASE_URL` من إعدادات Vercel للمشروع
4. أضيفوا أيضاً `SECRET_KEY` و`ALLOWED_ORIGINS` كمتغيرات بيئة
5. بعد النشر بتحصلون رابط مثل `https://your-project.vercel.app`

**نشر الواجهة على GitHub Pages (مجاني دائماً، بدون بطاقة، عندكم أصلاً على GitHub):**
1. نسخة من الواجهة موجودة أصلاً بمجلد `docs/index.html` بالمشروع (GitHub Pages يدعم هذا المجلد مباشرة)
2. بصفحة الـ repo على GitHub: **Settings → Pages → Source: Deploy from a branch → Branch: main، المجلد: /docs**
3. بعد دقيقة أو دقيقتين بتحصلون رابط مثل `https://username.github.io/repo-name/`
4. افتحوه وغيّروا حقل "الخادم" أعلى الصفحة لرابط Vercel اللي حصلتوا عليه بالخطوة السابقة

---

## 🛠️ التقنيات

- **Python 3.10+** · **FastAPI** · **SQLAlchemy (Async)**
- **Scikit-learn** · **HuggingFace Transformers** · **AraBERT**
- **SQLite** (تطوير) · **PostgreSQL** (إنتاج)
- **pytest** للاختبارات

---

*مشروع تخرج · 2025*
