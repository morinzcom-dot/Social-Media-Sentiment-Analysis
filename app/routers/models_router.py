"""
/api/v1/models — إدارة إصدارات النماذج
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List
from pathlib import Path

from app.database import get_db
from app.models.db_models import ModelVersion
from app.models.schemas import ModelVersionCreate, ModelVersionOut

router = APIRouter()

@router.get("/", response_model=List[ModelVersionOut], summary="جميع النماذج")
async def list_models(db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(ModelVersion).order_by(ModelVersion.trained_at.desc()))
    return res.scalars().all()

@router.get("/active", response_model=ModelVersionOut, summary="النموذج النشط")
async def get_active(db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(ModelVersion).where(ModelVersion.is_active == True))
    m   = res.scalar_one_or_none()
    if not m:
        raise HTTPException(404, "لا يوجد نموذج نشط")
    return m

@router.post("/", response_model=ModelVersionOut, summary="تسجيل نموذج جديد")
async def register_model(body: ModelVersionCreate, db: AsyncSession = Depends(get_db)):
    mv = ModelVersion(**body.model_dump())
    # حجم ملف النموذج — مسار مطلق مرتبط بموضع هذا الملف بدل مسار نسبي غير موثوق
    base_dir   = Path(__file__).resolve().parent.parent.parent
    model_file = base_dir / "data" / "models" / "logistic_model.pkl"
    if model_file.exists():
        mv.file_path    = str(model_file)
        mv.file_size_kb = round(model_file.stat().st_size / 1024, 1)
    db.add(mv)
    await db.flush()
    return mv

@router.patch("/{mid}/activate", summary="تفعيل نموذج")
async def activate_model(mid: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(ModelVersion).where(ModelVersion.id == mid))
    m   = res.scalar_one_or_none()
    if not m:
        raise HTTPException(404, "النموذج غير موجود")
    # تعطيل الآخرين
    await db.execute(update(ModelVersion).values(is_active=False))
    m.is_active = True
    return {"message": f"✅ تم تفعيل '{m.name} v{m.version}'"}

@router.get("/{mid}/compare", summary="مقارنة نموذجَين")
async def compare_two(mid1: int, mid2: int, db: AsyncSession = Depends(get_db)):
    r1 = (await db.execute(select(ModelVersion).where(ModelVersion.id == mid1))).scalar_one_or_none()
    r2 = (await db.execute(select(ModelVersion).where(ModelVersion.id == mid2))).scalar_one_or_none()
    if not r1 or not r2:
        raise HTTPException(404, "أحد النماذج غير موجود")
    def fmt(m):
        return {
            "id": m.id, "name": m.name, "version": m.version,
            "accuracy":   round((m.accuracy   or 0)*100, 1),
            "f1_score":   round((m.f1_score   or 0)*100, 1),
            "train_size": m.train_size,
        }
    winner = r1.name if (r1.accuracy or 0) >= (r2.accuracy or 0) else r2.name
    return {"model_1": fmt(r1), "model_2": fmt(r2), "winner": winner}

@router.delete("/{mid}", summary="حذف نموذج")
async def delete_model(mid: int, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(ModelVersion).where(ModelVersion.id == mid))
    m   = res.scalar_one_or_none()
    if not m:
        raise HTTPException(404, "النموذج غير موجود")
    if m.is_active:
        raise HTTPException(400, "لا يمكن حذف النموذج النشط")
    await db.delete(m)
    return {"message": f"✅ حُذف النموذج {m.name}"}
