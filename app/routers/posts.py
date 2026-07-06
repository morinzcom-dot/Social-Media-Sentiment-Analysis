"""
/api/v1/posts — CRUD المنشورات (موسّع)
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func
from typing import Optional, List

from app.database import get_db
from app.models.db_models import Post, PostKeyword
from app.models.schemas import PostOut, KeywordOut, KeywordFreq

router = APIRouter()


@router.get("/", response_model=List[PostOut], summary="جلب المنشورات مع فلترة متقدمة")
async def get_posts(
    skip:        int  = Query(0, ge=0),
    limit:       int  = Query(20, ge=1, le=500),
    platform_id: Optional[int] = None,
    topic_id:    Optional[int] = None,
    sentiment:   Optional[str] = None,
    language:    Optional[str] = None,
    status:      Optional[str] = None,
    is_spam:     Optional[bool] = None,
    is_correct:  Optional[bool] = None,
    min_confidence: Optional[float] = None,
    search:      Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    q = select(Post).order_by(Post.created_at.desc())
    if platform_id:  q = q.where(Post.platform_id  == platform_id)
    if topic_id:     q = q.where(Post.topic_id      == topic_id)
    if sentiment:    q = q.where(Post.sentiment      == sentiment)
    if language:     q = q.where(Post.language       == language)
    if status:       q = q.where(Post.status         == status)
    if is_spam is not None:    q = q.where(Post.is_spam    == is_spam)
    if is_correct is not None: q = q.where(Post.is_correct == is_correct)
    if min_confidence:         q = q.where(Post.confidence >= min_confidence)
    if search:                 q = q.where(Post.text.ilike(f"%{search}%"))
    res = await db.execute(q.offset(skip).limit(limit))
    return res.scalars().all()


@router.get("/count", summary="عدد المنشورات")
async def count_posts(
    sentiment: Optional[str] = None,
    platform_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    q = select(func.count()).select_from(Post)
    if sentiment:   q = q.where(Post.sentiment   == sentiment)
    if platform_id: q = q.where(Post.platform_id == platform_id)
    res = await db.execute(q)
    return {"count": res.scalar()}


@router.get("/{post_id}", response_model=PostOut, summary="تفاصيل منشور")
async def get_post(post_id: int, db: AsyncSession = Depends(get_db)):
    res  = await db.execute(select(Post).where(Post.id == post_id))
    post = res.scalar_one_or_none()
    if not post:
        raise HTTPException(404, "المنشور غير موجود")
    return post


@router.get("/{post_id}/keywords", response_model=List[KeywordOut], summary="كلمات منشور")
async def get_post_keywords(post_id: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(
        select(PostKeyword).where(PostKeyword.post_id == post_id)
        .order_by(PostKeyword.frequency.desc())
    )
    return res.scalars().all()


@router.delete("/{post_id}", summary="حذف منشور")
async def delete_post(post_id: int, db: AsyncSession = Depends(get_db)):
    res  = await db.execute(select(Post).where(Post.id == post_id))
    post = res.scalar_one_or_none()
    if not post:
        raise HTTPException(404, "المنشور غير موجود")
    await db.delete(post)
    return {"message": f"✅ حُذف المنشور {post_id}"}


@router.get("/keywords/top", response_model=List[KeywordFreq], summary="أكثر الكلمات تكراراً")
async def top_keywords(
    limit:     int  = Query(20, ge=1, le=100),
    language:  Optional[str] = None,
    sentiment: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    q = (
        select(PostKeyword.keyword, func.sum(PostKeyword.frequency).label("total"))
        .group_by(PostKeyword.keyword)
        .order_by(func.sum(PostKeyword.frequency).desc())
        .limit(limit)
    )
    if language:
        q = q.where(PostKeyword.language == language)
    if sentiment:
        q = q.join(Post).where(Post.sentiment == sentiment)
    res = await db.execute(q)
    return [{"keyword": r[0], "total": r[1], "sentiment": sentiment} for r in res.all()]
