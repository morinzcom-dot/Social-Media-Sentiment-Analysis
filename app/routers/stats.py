"""
/api/v1/stats — إحصاءات لوحة التحكم (موسّع)
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.database import get_db
from app.models.db_models import Post, Platform, Topic, ModelVersion, Alert, PostKeyword
from app.models.schemas import StatsOut

router = APIRouter()

POS = lambda: case((Post.sentiment == "positive", 1), else_=0)
NEG = lambda: case((Post.sentiment == "negative", 1), else_=0)
NEU = lambda: case((Post.sentiment == "neutral",  1), else_=0)


@router.get("/", response_model=StatsOut, summary="إحصاءات شاملة")
async def get_stats(
    days:        int  = Query(30, ge=1, le=365),
    platform_id: Optional[int] = None,
    topic_id:    Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    since = datetime.now(timezone.utc) - timedelta(days=days)

    # ── base query ──
    def base(q):
        q = q.where(Post.created_at >= since)
        if platform_id: q = q.where(Post.platform_id == platform_id)
        if topic_id:    q = q.where(Post.topic_id    == topic_id)
        return q

    # ── الإجماليات ──
    row = (await db.execute(base(select(
        func.count().label("total"),
        func.sum(POS()).label("pos"),
        func.sum(NEG().label("neg") if False else NEG()).label("neg"),
        func.sum(NEU()).label("neu"),
        func.avg(Post.confidence).label("avg_conf"),
        func.sum(case((Post.status == "flagged", 1), else_=0)).label("flagged"),
        func.sum(case((Post.is_spam == True, 1), else_=0)).label("spam"),
    )))).one()

    total = row.total or 0
    pos   = row.pos   or 0
    neg   = row.neg   or 0
    neu   = row.neu   or 0
    safe  = total or 1

    # ── حسب المنصة ──
    plat_rows = (await db.execute(
        base(select(
            Post.platform_id,
            Platform.display_name,
            func.count().label("cnt"),
            func.sum(POS()).label("pos"),
            func.sum(NEG()).label("neg"),
        ).join(Platform, Post.platform_id == Platform.id, isouter=True))
        .group_by(Post.platform_id, Platform.display_name)
    )).all()

    by_platform = [{
        "platform_id":   r.platform_id,
        "name":          r.display_name or "manual",
        "count":         r.cnt,
        "positive":      r.pos or 0,
        "negative":      r.neg or 0,
    } for r in plat_rows]

    # ── حسب اللغة ──
    lang_rows = (await db.execute(
        base(select(Post.language, func.count().label("cnt")))
        .group_by(Post.language)
    )).all()
    by_language = [{"language": r.language, "count": r.cnt} for r in lang_rows]

    # ── حسب الموضوع ──
    topic_rows = (await db.execute(
        base(select(
            Topic.name_ar, Topic.name, Topic.color,
            func.count().label("cnt"),
            func.sum(POS()).label("pos"),
            func.sum(NEG()).label("neg"),
        ).join(Topic, Post.topic_id == Topic.id, isouter=True))
        .group_by(Topic.id, Topic.name_ar, Topic.name, Topic.color)
        .order_by(func.count().desc())
    )).all()
    by_topic = [{
        "name":     r.name_ar or r.name or "غير محدد",
        "color":    r.color or "#6366f1",
        "count":    r.cnt,
        "positive": r.pos or 0,
        "negative": r.neg or 0,
    } for r in topic_rows]

    # ── اتجاه 30 يوماً ──
    trend = []
    for i in range(min(days, 30)-1, -1, -1):
        d_start = datetime.now(timezone.utc) - timedelta(days=i)
        d_end   = d_start + timedelta(days=1)
        dr = (await db.execute(base(select(
            func.count().label("cnt"),
            func.sum(POS()).label("p"),
            func.sum(NEG()).label("n"),
        )).where(Post.created_at >= d_start, Post.created_at < d_end))).one()
        trend.append({
            "date":     d_start.strftime("%Y-%m-%d"),
            "total":    dr.cnt or 0,
            "positive": dr.p or 0,
            "negative": dr.n or 0,
        })

    # ── أبرز الكلمات ──
    kw_rows = (await db.execute(
        select(PostKeyword.keyword, func.sum(PostKeyword.frequency).label("total"))
        .join(Post).where(Post.created_at >= since)
        .group_by(PostKeyword.keyword)
        .order_by(func.sum(PostKeyword.frequency).desc())
        .limit(15)
    )).all()
    top_keywords = [{"keyword": r.keyword, "total": r.total} for r in kw_rows]

    # ── دقة النموذج النشط ──
    mv = (await db.execute(
        select(ModelVersion.accuracy).where(ModelVersion.is_active == True)
    )).scalar_one_or_none()

    # ── تنبيهات غير مقروءة ──
    unread = (await db.execute(
        select(func.count()).select_from(Alert).where(Alert.is_read == False)
    )).scalar() or 0

    return StatsOut(
        total_analyzed = total,
        positive       = pos,
        negative       = neg,
        neutral        = neu,
        positive_pct   = round(pos / safe * 100, 1),
        negative_pct   = round(neg / safe * 100, 1),
        neutral_pct    = round(neu / safe * 100, 1),
        avg_confidence = round((row.avg_conf or 0) * 100, 1),
        flagged_count  = row.flagged or 0,
        spam_count     = row.spam    or 0,
        by_platform    = by_platform,
        by_language    = by_language,
        by_topic       = by_topic,
        recent_trend   = trend,
        top_keywords   = top_keywords,
        model_accuracy = round(mv * 100, 1) if mv else None,
        unread_alerts  = unread,
    )


@router.get("/summary", summary="ملخص سريع")
async def summary(db: AsyncSession = Depends(get_db)):
    total = (await db.execute(select(func.count()).select_from(Post))).scalar() or 0
    pos   = (await db.execute(select(func.count()).where(Post.sentiment=="positive"))).scalar() or 0
    mv    = (await db.execute(select(ModelVersion).where(ModelVersion.is_active==True))).scalar_one_or_none()
    unread= (await db.execute(select(func.count()).select_from(Alert).where(Alert.is_read==False))).scalar() or 0
    return {
        "total":          total,
        "positive":       pos,
        "positive_pct":   round(pos / total * 100, 1) if total else 0,
        "model":          mv.name if mv else "logistic",
        "model_accuracy": round((mv.accuracy or 0)*100, 1) if mv else None,
        "unread_alerts":  unread,
        "status":         "يعمل ✅",
    }
