"""
╔══════════════════════════════════════════════════════════════╗
║   Alembic — إدارة تغييرات قاعدة البيانات (Migrations)       ║
║                                                              ║
║   الاستخدام:                                                 ║
║     alembic init migrations                                  ║
║     alembic revision --autogenerate -m "وصف التغيير"        ║
║     alembic upgrade head                                     ║
║     alembic downgrade -1    (تراجع خطوة واحدة)              ║
╚══════════════════════════════════════════════════════════════╝
"""

# alembic.ini — يُنشأ بأمر: alembic init migrations
# ثم عدّل sqlalchemy.url في الملف
#
# env.py الذي يُنشأ تلقائياً — أضف هذا:
#
# from app.database import Base
# from app.models import db_models  # تسجيل النماذج
# target_metadata = Base.metadata

ALEMBIC_CONFIG = """
# alembic.ini
[alembic]
script_location = migrations
sqlalchemy.url = sqlite:///./sentiment_ai.db

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = Streamlit
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""

# حفظ الإعدادات
with open("alembic.ini", "w", encoding="utf-8") as f:
    f.write(ALEMBIC_CONFIG)

print("✅ alembic.ini جاهز — شغّل: alembic init migrations")
