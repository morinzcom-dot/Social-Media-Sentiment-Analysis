"""
SentimentAI API v2.0 — نقطة الدخول الرئيسية
8 جداول · 7 routers · Async SQLAlchemy
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import sentiment, posts, stats
from app.routers import platforms, topics, models_router, alerts, users
from app.database import init_db
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 SentimentAI v2.0 — بدء التشغيل...")
    await init_db()
    await _seed_defaults()
    print("✅ الخادم جاهز")
    yield
    print("🛑 إيقاف الخادم")


async def _seed_defaults():
    """إدخال بيانات افتراضية عند أول تشغيل"""
    from app.database import AsyncSessionLocal
    from app.models.db_models import Platform, Topic, ModelVersion
    from sqlalchemy import select
    from pathlib import Path
    import pickle
    import json

    async with AsyncSessionLocal() as db:
        # منصات افتراضية
        for name, display, icon in [
            ("facebook",  "Facebook",  "📘"),
            ("instagram", "Instagram", "📸"),
            ("manual",    "يدوي",      "✍️"),
        ]:
            ex = await db.execute(select(Platform).where(Platform.name == name))
            if not ex.scalar_one_or_none():
                db.add(Platform(name=name, display_name=display, icon=icon))

        # موضوعات افتراضية
        for name, name_ar, color in [
            ("products",  "منتجات وخدمات", "#10b981"),
            ("politics",  "السياسة",        "#ef4444"),
            ("sports",    "الرياضة",         "#3b82f6"),
            ("health",    "الصحة",           "#f59e0b"),
            ("general",   "عام",             "#8b5cf6"),
        ]:
            ex = await db.execute(select(Topic).where(Topic.name == name))
            if not ex.scalar_one_or_none():
                db.add(Topic(name=name, name_ar=name_ar, color=color))

        # تسجيل النموذج الحالي
        ex = await db.execute(select(ModelVersion).where(ModelVersion.name == "logistic-v1"))
        if not ex.scalar_one_or_none():
            # مسار مطلق مرتبط بموضع هذا الملف، بدل مسار نسبي يعتمد على working directory
            # للعملية — هذا الأخير غير موثوق على بيئات serverless مثل Vercel وكان يسبب
            # خطأ FileNotFoundError يوقف تشغيل التطبيق بالكامل عند أول طلب (cold start).
            base_dir    = Path(__file__).resolve().parent.parent
            model_path  = base_dir / "data" / "models" / "logistic_model.pkl"
            report_path = base_dir / "data" / "models" / "evaluation_report.json"
            size_kb = round(model_path.stat().st_size / 1024, 1) if model_path.exists() else None
            vocab = None
            if model_path.exists():
                with open(model_path, "rb") as f:
                    m = pickle.load(f)
                v = m["vectorizer"]
                if hasattr(v, "vocabulary_"):
                    vocab = len(v.vocabulary_)
                elif hasattr(v, "transformer_list"):
                    vocab = sum(len(t.vocabulary_) for _, t in v.transformer_list if hasattr(t, "vocabulary_"))

            # قراءة مقاييس الأداء الحقيقية من تقرير التقييم بدلاً من قيم ثابتة وهمية
            accuracy = f1 = None
            train_size = None
            if report_path.exists():
                with open(report_path, encoding="utf-8") as f:
                    rep = json.load(f)
                accuracy   = (rep.get("accuracy") or 0) / 100
                f1         = rep.get("classification", {}).get("weighted avg", {}).get("f1-score")
                train_size = rep.get("train_size")

            db.add(ModelVersion(
                name       = "logistic-v1",
                model_type = "logistic",
                version    = "1.0.0",
                description= "Logistic Regression + TF-IDF char n-gram (2-5)",
                is_active  = True,
                accuracy   = accuracy,
                f1_score   = f1,
                train_size = train_size,
                vocab_size = vocab,
                file_path  = str(model_path),
                file_size_kb = size_kb,
            ))

        await db.commit()
        print("  ✅ بيانات افتراضية مُهيَّأة")


app = FastAPI(
    title       = "SentimentAI API v2.0",
    description = "نظام تحليل مشاعر وسائل التواصل — 8 جداول · عربي + إنجليزي",
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ──
app.include_router(sentiment.router,      prefix="/api/v1/sentiment", tags=["تحليل المشاعر"])
app.include_router(posts.router,          prefix="/api/v1/posts",     tags=["المنشورات"])
app.include_router(stats.router,          prefix="/api/v1/stats",     tags=["الإحصاءات"])
app.include_router(platforms.router,      prefix="/api/v1/platforms", tags=["المنصات"])
app.include_router(topics.router,         prefix="/api/v1/topics",    tags=["الموضوعات"])
app.include_router(models_router.router,  prefix="/api/v1/models",    tags=["النماذج"])
app.include_router(alerts.router,         prefix="/api/v1/alerts",    tags=["التنبيهات"])
app.include_router(users.router,          prefix="/api/v1/users",     tags=["المستخدمون"])


@app.get("/", tags=["الحالة"])
async def root():
    return {
        "message": "SentimentAI API v2.0 🧠",
        "tables":  8,
        "routers": 8,
        "docs":    "/docs",
        "status":  "يعمل ✅",
    }

@app.get("/health", tags=["الحالة"])
async def health():
    return {"status": "healthy", "version": "2.0.0"}
