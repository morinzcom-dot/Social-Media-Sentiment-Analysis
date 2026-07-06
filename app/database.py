"""
إعداد قاعدة البيانات — SQLAlchemy (Async)
يدعم SQLite للتطوير و PostgreSQL للإنتاج
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from app.config import settings


def _normalize_db_url(url: str) -> tuple[str, dict]:
    """
    منصات الاستضافة المُدارة (Render, Railway, Neon, Supabase...) توفر رابط اتصال
    بصيغة postgres:// أو postgresql:// بدون تحديد سائق (driver)، وغالباً مع
    ?sslmode=require بنهاية الرابط. لكن:
    1) SQLAlchemy غير المتزامن (async) يتطلب صراحةً postgresql+asyncpg://.
    2) عكس psycopg2، مكتبة asyncpg لا تقبل sslmode كمعامل اتصال مباشر —
       ترمي TypeError: connect() got an unexpected keyword argument 'sslmode'.
       الحل: نحذف sslmode من الرابط ونمرر خيار SSL المكافئ عبر connect_args.
    """
    connect_args: dict = {}

    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    if "+asyncpg" in url:
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query))
        sslmode = query.pop("sslmode", None)
        if sslmode and sslmode != "disable":
            connect_args["ssl"] = "require"
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))

    return url, connect_args


_db_url, _connect_args = _normalize_db_url(settings.DATABASE_URL)

# ── محرك قاعدة البيانات ──
engine = create_async_engine(
    _db_url,
    connect_args=_connect_args,
    echo=False,          # اجعلها True لرؤية استعلامات SQL
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """الفئة الأساسية لجميع نماذج قاعدة البيانات"""
    pass


async def init_db():
    """إنشاء الجداول إذا لم تكن موجودة"""
    from app.models import db_models   # noqa: F401 — تسجيل النماذج
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency: جلسة قاعدة بيانات لكل طلب"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
