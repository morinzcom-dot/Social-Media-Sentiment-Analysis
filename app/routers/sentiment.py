"""
/api/v1/sentiment — تحليل المشاعر (موسّع)
"""
import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List, Optional

from app.database import get_db
from app.models.schemas import (
    AnalyzeRequest, BatchAnalyzeRequest,
    SentimentResult, BatchResult, PostUpdate, PostOut,
)
from app.models.db_models import Post, Platform, Topic, ModelVersion, AnalysisSession, PostKeyword
from app.services.sentiment_service import SentimentService

router = APIRouter()

def get_service() -> SentimentService:
    return SentimentService.get_instance()

# ── helper: استخرج كلمات مفتاحية بسيطة ──
def extract_keywords(text: str, language: str) -> List[dict]:
    import re
    AR_STOP = {"في","من","على","إلى","عن","مع","هذا","هذه","التي","الذي","كان","هو","هي","و","أو","لكن","لأن","إن","إذا","بعد","قبل","كل","لم","لن","قد","ليس"}
    EN_STOP = {"the","a","an","is","it","in","on","at","to","for","of","and","or","but","not","this","that","was","are"}
    words = re.findall(r'[\u0600-\u06FF]{3,}|[a-zA-Z]{3,}', text.lower())
    stop  = AR_STOP if language == "ar" else EN_STOP
    freq  = {}
    for w in words:
        if w not in stop:
            freq[w] = freq.get(w, 0) + 1
    return [{"keyword": k, "frequency": v} for k, v in sorted(freq.items(), key=lambda x: -x[1])[:10]]


async def _resolve_platform_id(db, body) -> Optional[int]:
    """يحوّل اسم المنصة (platform) إلى platform_id الفعلي، مع إعطاء الأولوية
    لـ platform_id إذا أُرسل صراحةً. بدون هذا، كان اسم المنصة يُتجاهل بصمت
    وتُحفظ كل المنشورات بدون منصة محددة."""
    explicit_id = getattr(body, "platform_id", None)
    if explicit_id:
        return explicit_id
    name = getattr(body, "platform", None)
    if not name:
        return None
    res = await db.execute(select(Platform).where(Platform.name == name))
    platform = res.scalar_one_or_none()
    return platform.id if platform else None


async def _save_post(db, result: dict, body, model_ver_id: Optional[int]) -> int:
    text = result["text_original"]
    kw_data = extract_keywords(result["text_clean"] or text, result["language"])
    platform_id = await _resolve_platform_id(db, body)
    post = Post(
        text             = text,
        text_clean       = result["text_clean"],
        language         = result["language"],
        char_count       = len(text),
        word_count       = len(text.split()),
        platform_id      = platform_id,
        topic_id         = getattr(body, "topic_id", None),
        author_id        = getattr(body, "author_id", None),
        post_url         = getattr(body, "post_url", None),
        posted_at        = getattr(body, "posted_at", None),
        sentiment        = result["sentiment"],
        confidence       = result["confidence"],
        score_positive   = result["scores"]["positive"],
        score_negative   = result["scores"]["negative"],
        score_neutral    = result["scores"]["neutral"],
        model_version_id = model_ver_id,
        status           = "analyzed",
    )
    db.add(post)
    await db.flush()
    for kw in kw_data:
        db.add(PostKeyword(
            post_id   = post.id,
            keyword   = kw["keyword"],
            frequency = kw["frequency"],
            language  = result["language"],
        ))
    # تحديث عداد الموضوع
    if post.topic_id:
        await db.execute(
            update(Topic).where(Topic.id == post.topic_id)
            .values(post_count=Topic.post_count + 1)
        )
    return post.id


@router.post("/analyze", response_model=SentimentResult, summary="تحليل نص واحد")
async def analyze_single(
    body: AnalyzeRequest,
    db:   AsyncSession = Depends(get_db),
    svc:  SentimentService = Depends(get_service),
):
    try:
        result = svc.analyze(body.text)
    except Exception as e:
        raise HTTPException(500, f"خطأ في التحليل: {e}")

    # جلب النموذج النشط
    mv = await db.execute(select(ModelVersion).where(ModelVersion.is_active == True))
    model_ver = mv.scalar_one_or_none()

    post_id = None
    if body.save:
        post_id = await _save_post(db, result, body, model_ver.id if model_ver else None)

    return SentimentResult(**result, post_id=post_id)


@router.post("/analyze/batch", response_model=BatchResult, summary="تحليل دفعة نصوص")
async def analyze_batch(
    body: BatchAnalyzeRequest,
    db:   AsyncSession = Depends(get_db),
    svc:  SentimentService = Depends(get_service),
):
    if len(body.texts) > 100:
        raise HTTPException(400, "الحد الأقصى 100 نص في الدفعة")

    t0 = time.time()
    mv = await db.execute(select(ModelVersion).where(ModelVersion.is_active == True))
    model_ver = mv.scalar_one_or_none()

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    total_conf = 0.0
    schema_results = []

    for text in body.texts:
        r = svc.analyze(text)
        counts[r["sentiment"]] += 1
        total_conf += r["confidence"]
        schema_results.append(SentimentResult(**r, post_id=None))

    if body.save:
        for r in schema_results:
            await _save_post(db, {
                "text_original": r.text_original,
                "text_clean":    r.text_clean,
                "language":      r.language,
                "sentiment":     r.sentiment,
                "confidence":    r.confidence,
                "scores":        r.scores,
            }, body, model_ver.id if model_ver else None)

        total = len(body.texts)
        dur   = time.time() - t0
        sess  = AnalysisSession(
            total_posts      = total,
            positive         = counts["positive"],
            negative         = counts["negative"],
            neutral          = counts["neutral"],
            avg_confidence   = round(total_conf / total, 4),
            platform_id      = body.platform_id,
            topic_id         = body.topic_id,
            model_version_id = model_ver.id if model_ver else None,
            duration_seconds = round(dur, 3),
        )
        db.add(sess)

    total = len(body.texts)
    dur   = time.time() - t0
    return BatchResult(
        total        = total,
        positive     = counts["positive"],
        negative     = counts["negative"],
        neutral      = counts["neutral"],
        positive_pct = round(counts["positive"] / total * 100, 1),
        negative_pct = round(counts["negative"] / total * 100, 1),
        neutral_pct  = round(counts["neutral"]  / total * 100, 1),
        avg_confidence = round(total_conf / total, 4),
        duration_ms  = round(dur * 1000, 1),
        results      = schema_results,
    )


@router.patch("/posts/{post_id}", response_model=PostOut, summary="تحديث تصنيف منشور")
async def update_post(
    post_id: int,
    body:    PostUpdate,
    db:      AsyncSession = Depends(get_db),
):
    res  = await db.execute(select(Post).where(Post.id == post_id))
    post = res.scalar_one_or_none()
    if not post:
        raise HTTPException(404, "المنشور غير موجود")
    for field, val in body.model_dump(exclude_none=True).items():
        setattr(post, field, val)
    if body.human_label and post.sentiment:
        post.is_correct = (body.human_label == post.sentiment)
    return post


@router.post("/train", summary="تدريب النموذج")
async def train_model(
    texts:  List[str],
    labels: List[str],
    db:     AsyncSession = Depends(get_db),
    svc:    SentimentService = Depends(get_service),
):
    if len(texts) != len(labels):
        raise HTTPException(400, "عدد النصوص والتصنيفات يجب أن يتطابق")
    if len(texts) < 30:
        raise HTTPException(400, "يلزم 30 مثالاً على الأقل")
    valid = {"positive", "negative", "neutral"}
    if any(l not in valid for l in labels):
        raise HTTPException(400, f"التصنيفات الصالحة: {valid}")

    report = svc.train(texts, labels)

    mv = ModelVersion(
        name       = f"logistic-v{int(time.time())}",
        model_type = "logistic",
        version    = "auto",
        accuracy   = report.get("accuracy", 0) / 100,
        train_size = len(texts),
        is_active  = False,
    )
    db.add(mv)
    return {"message": "✅ تم التدريب", **report}
