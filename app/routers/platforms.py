"""
/api/v1/platforms — إدارة المنصات
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List
from datetime import datetime, timezone

from app.database import get_db
from app.models.db_models import Platform
from app.models.schemas import PlatformCreate, PlatformOut

router = APIRouter()

@router.get("/", response_model=List[PlatformOut], summary="جميع المنصات")
async def list_platforms(db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Platform).order_by(Platform.name))
    return res.scalars().all()

@router.post("/", response_model=PlatformOut, summary="إضافة منصة")
async def create_platform(body: PlatformCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(Platform).where(Platform.name == body.name))
    if existing.scalar_one_or_none():
        raise HTTPException(400, f"المنصة '{body.name}' موجودة مسبقاً")
    p = Platform(**body.model_dump())
    db.add(p)
    await db.flush()
    return p

@router.get("/{pid}", response_model=PlatformOut, summary="تفاصيل منصة")
async def get_platform(pid: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Platform).where(Platform.id == pid))
    p   = res.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "المنصة غير موجودة")
    return p

@router.patch("/{pid}/sync", summary="تحديث وقت المزامنة")
async def mark_synced(pid: int, fetched: int = 0, db: AsyncSession = Depends(get_db)):
    await db.execute(
        update(Platform).where(Platform.id == pid)
        .values(
            last_sync_at  = datetime.now(timezone.utc),
            total_fetched = Platform.total_fetched + fetched,
        )
    )
    return {"message": "✅ تم التحديث"}

@router.delete("/{pid}", summary="حذف منصة")
async def delete_platform(pid: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(Platform).where(Platform.id == pid))
    p   = res.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "المنصة غير موجودة")
    await db.delete(p)
    return {"message": f"✅ حُذفت المنصة {p.name}"}
