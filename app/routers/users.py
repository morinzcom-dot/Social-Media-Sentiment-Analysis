"""
/api/v1/users — إدارة المستخدمين
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List
import hashlib

from app.database import get_db
from app.models.db_models import User
from app.models.schemas import UserCreate, UserOut, UserUpdate

router = APIRouter()

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

@router.get("/", response_model=List[UserOut], summary="جميع المستخدمين")
async def list_users(db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).order_by(User.created_at.desc()))
    return res.scalars().all()

@router.post("/", response_model=UserOut, summary="إضافة مستخدم")
async def create_user(body: UserCreate, db: AsyncSession = Depends(get_db)):
    ex = await db.execute(select(User).where(User.username == body.username))
    if ex.scalar_one_or_none():
        raise HTTPException(400, f"المستخدم '{body.username}' موجود مسبقاً")
    u = User(
        username      = body.username,
        email         = body.email,
        password_hash = _hash(body.password),
        full_name     = body.full_name,
        role          = body.role,
    )
    db.add(u)
    await db.flush()
    return u

@router.get("/{uid}", response_model=UserOut, summary="تفاصيل مستخدم")
async def get_user(uid: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.id == uid))
    u   = res.scalar_one_or_none()
    if not u: raise HTTPException(404, "المستخدم غير موجود")
    return u

@router.patch("/{uid}", response_model=UserOut, summary="تعديل مستخدم")
async def update_user(uid: int, body: UserUpdate, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.id == uid))
    u   = res.scalar_one_or_none()
    if not u: raise HTTPException(404, "المستخدم غير موجود")
    for f, v in body.model_dump(exclude_none=True).items():
        setattr(u, f, v)
    return u

@router.delete("/{uid}", summary="حذف مستخدم")
async def delete_user(uid: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.id == uid))
    u   = res.scalar_one_or_none()
    if not u: raise HTTPException(404, "المستخدم غير موجود")
    await db.delete(u)
    return {"message": f"✅ حُذف المستخدم {u.username}"}
