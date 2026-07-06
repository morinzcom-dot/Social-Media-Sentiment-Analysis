"""
نقطة دخول لمنصة Vercel — يبحث Vercel تلقائياً عن متغيّر app
بملف main.py/app.py بجذر المشروع. تطبيقنا الفعلي داخل app/main.py،
فهذا الملف يعيد تصديره فقط ليكتشفه Vercel دون أي تعديل على الكود الأصلي.
"""
from app.main import app
