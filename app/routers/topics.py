"""
/api/v1/topics — إدارة الموضوعات
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List

from app.database import get_db
from app.models.db_models import Topic, Post
from app.models.schemas import TopicCreate, TopicOut

router = APIRouter()

@router.get("/", response_model=List[TopicOut], summary="جميع الموضوعات")
async def list_topics(active_only: bool = True, db: AsyncSession = Depends(get_db)):
    q = select(Topic).order_by(Topic.post_count.desc())
    if active_only:
        q = q.where(Topic.is_active == True)
    res = await db.execute(q)
    return res.scalars().all()

@router.post("/", response_model=TopicOut, summary="إضافة موضوع")
async def create_topic(body: TopicCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(Topic).where(Topic.name == body.name))
    if existing.scalar_one_or_none():
        raise HTTPException(400, f"الموضوع '{body.name}' موجود مسبقاً")
    t = Topic(**body.model_dump())
    db.add(t)
    await db.flush()
    return t

@router.get("/{tid}", response_model=TopicOut)
async def get_topic(tid: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Topic).where(Topic.id == tid))
    t   = res.scalar_one_or_none()
    if not t:
        raise HTTPException(404, "الموضوع غير موجود")
    return t

@router.get("/{tid}/stats", summary="إحصاءات موضوع")
async def topic_stats(tid: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Topic).where(Topic.id == tid))
    t   = res.scalar_one_or_none()
    if not t:
        raise HTTPException(404, "الموضوع غير موجود")
    rows = (await db.execute(
        select(Post.sentiment, func.count().label("cnt"))
        .where(Post.topic_id == tid)
        .group_by(Post.sentiment)
    )).all()
    dist = {r.sentiment: r.cnt for r in rows}
    total = sum(dist.values()) or 1
    return {
        "topic":    t.name_ar or t.name,
        "total":    sum(dist.values()),
        "positive": dist.get("positive", 0),
        "negative": dist.get("negative", 0),
        "neutral":  dist.get("neutral",  0),
        "positive_pct": round(dist.get("positive",0)/total*100, 1),
    }

@router.delete("/{tid}", summary="حذف موضوع")
async def delete_topic(tid: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Topic).where(Topic.id == tid))
    t   = res.scalar_one_or_none()
    if not t:
        raise HTTPException(404, "الموضوع غير موجود")
    await db.delete(t)
    return {"message": f"✅ حُذف الموضوع {t.name}"}
