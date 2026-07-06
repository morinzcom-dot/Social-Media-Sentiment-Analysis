"""
/api/v1/alerts — نظام التنبيهات
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func
from typing import List, Optional
from datetime import datetime, timezone

from app.database import get_db
from app.models.db_models import Alert, Post
from app.models.schemas import AlertCreate, AlertOut

router = APIRouter()

@router.get("/", response_model=List[AlertOut], summary="جميع التنبيهات")
async def list_alerts(
    unread_only: bool  = False,
    severity:    Optional[str] = None,
    limit:       int   = Query(50, ge=1, le=200),
    db: AsyncSession   = Depends(get_db),
):
    q = select(Alert).order_by(Alert.created_at.desc()).limit(limit)
    if unread_only: q = q.where(Alert.is_read == False)
    if severity:    q = q.where(Alert.severity == severity)
    res = await db.execute(q)
    return res.scalars().all()

@router.post("/", response_model=AlertOut, summary="إنشاء تنبيه")
async def create_alert(body: AlertCreate, db: AsyncSession = Depends(get_db)):
    a = Alert(**body.model_dump())
    db.add(a)
    await db.flush()
    return a

@router.patch("/{aid}/read", summary="تعليم كمقروء")
async def mark_read(aid: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Alert).where(Alert.id == aid))
    a   = res.scalar_one_or_none()
    if not a: raise HTTPException(404, "التنبيه غير موجود")
    a.is_read = True
    return {"message": "✅ تم التعليم كمقروء"}

@router.patch("/read-all", summary="تعليم الكل كمقروء")
async def mark_all_read(db: AsyncSession = Depends(get_db)):
    await db.execute(update(Alert).values(is_read=True))
    return {"message": "✅ تم تعليم الكل"}

@router.patch("/{aid}/resolve", summary="إغلاق التنبيه")
async def resolve_alert(aid: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Alert).where(Alert.id == aid))
    a   = res.scalar_one_or_none()
    if not a: raise HTTPException(404, "التنبيه غير موجود")
    a.is_resolved = True
    a.resolved_at = datetime.now(timezone.utc)
    return {"message": "✅ تم الإغلاق"}

@router.post("/auto-check", summary="فحص تلقائي للسمعة")
async def auto_check(
    threshold_neg: float = 0.60,
    db: AsyncSession = Depends(get_db),
):
    """يفحص نسبة السلبية في آخر 100 منشور وينشئ تنبيهاً إذا تجاوزت العتبة"""
    from sqlalchemy import case as sa_case
    row = (await db.execute(
        select(
            func.count().label("total"),
            func.sum(sa_case((Post.sentiment=="negative",1),else_=0)).label("neg"),
        ).order_by(Post.created_at.desc()).limit(100)
    )).one()

    total = row.total or 1
    neg_ratio = (row.neg or 0) / total

    if neg_ratio >= threshold_neg:
        severity = "critical" if neg_ratio >= 0.80 else "high"
        a = Alert(
            title        = f"ارتفاع حاد في السلبية: {neg_ratio*100:.1f}%",
            description  = f"نسبة السلبية تجاوزت {threshold_neg*100:.0f}% في آخر {total} منشور",
            alert_type   = "sentiment_spike",
            severity     = severity,
            threshold    = threshold_neg,
            actual_value = round(neg_ratio, 4),
        )
        db.add(a)
        await db.flush()
        return {"triggered": True, "severity": severity, "neg_ratio": round(neg_ratio*100,1)}

    return {"triggered": False, "neg_ratio": round(neg_ratio*100,1), "message": "السمعة طبيعية ✅"}
