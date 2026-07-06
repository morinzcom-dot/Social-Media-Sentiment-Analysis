"""
╔══════════════════════════════════════════════════╗
║   جمع البيانات من Facebook Graph API            ║
╚══════════════════════════════════════════════════╝

المتطلبات:
  - Facebook App ID + App Secret
  - Access Token (Page Token للصفحات)
  - تفعيل صلاحيات: pages_read_engagement, pages_show_list

الاستخدام:
  collector = FacebookCollector()
  posts = await collector.fetch_page_posts(page_id="your_page_id", limit=50)
"""

import httpx
import asyncio
from typing import List, Dict, Optional
from datetime import datetime

from app.config import settings


BASE_URL = "https://graph.facebook.com/v19.0"


class FacebookCollector:
    """جامع منشورات Facebook عبر Graph API"""

    def __init__(self):
        self.token   = settings.FACEBOOK_ACCESS_TOKEN
        self.app_id  = settings.FACEBOOK_APP_ID
        self.app_secret = settings.FACEBOOK_APP_SECRET

        if not self.token:
            print("⚠️  FACEBOOK_ACCESS_TOKEN غير مضبوط في .env")

    async def fetch_page_posts(
        self,
        page_id: str,
        limit:   int = 50,
        fields:  str = "id,message,created_time,reactions.summary(true),comments.summary(true)",
    ) -> List[Dict]:
        """
        جلب منشورات صفحة Facebook

        Args:
            page_id : معرّف الصفحة (رقم أو اسم)
            limit   : عدد المنشورات (الحد الأقصى 100 لكل طلب)
            fields  : الحقول المطلوبة

        Returns:
            قائمة المنشورات مع نصوصها
        """
        url    = f"{BASE_URL}/{page_id}/posts"
        params = {
            "access_token": self.token,
            "fields":        fields,
            "limit":         min(limit, 100),
        }

        posts = []
        async with httpx.AsyncClient(timeout=30) as client:
            while len(posts) < limit:
                try:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                except httpx.HTTPStatusError as e:
                    print(f"❌ خطأ HTTP {e.response.status_code}: {e.response.text}")
                    break
                except Exception as e:
                    print(f"❌ خطأ في الاتصال: {e}")
                    break

                for item in data.get("data", []):
                    if "message" in item and item["message"].strip():
                        posts.append({
                            "external_id":  item.get("id"),
                            "text":         item["message"],
                            "platform":     "facebook",
                            "created_time": item.get("created_time"),
                            "reactions":    item.get("reactions", {}).get("summary", {}).get("total_count", 0),
                            "comments":     item.get("comments",  {}).get("summary", {}).get("total_count", 0),
                        })

                # Pagination — جلب الصفحة التالية
                next_url = data.get("paging", {}).get("next")
                if not next_url or len(posts) >= limit:
                    break
                url    = next_url
                params = {}  # الرابط التالي يحتوي كل الباراميترات

        print(f"✅ تم جلب {len(posts)} منشور من {page_id}")
        return posts[:limit]

    async def fetch_post_comments(self, post_id: str, limit: int = 100) -> List[Dict]:
        """جلب تعليقات منشور محدد"""
        url    = f"{BASE_URL}/{post_id}/comments"
        params = {
            "access_token": self.token,
            "fields": "id,message,created_time,like_count",
            "limit":  min(limit, 100),
        }
        comments = []
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, params=params)
            if response.status_code == 200:
                for item in response.json().get("data", []):
                    if "message" in item:
                        comments.append({
                            "external_id": item["id"],
                            "text":        item["message"],
                            "platform":    "facebook",
                            "created_time": item.get("created_time"),
                        })
        return comments

    async def verify_token(self) -> bool:
        """التحقق من صلاحية الـ Access Token"""
        url    = f"{BASE_URL}/me"
        params = {"access_token": self.token, "fields": "id,name"}
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params)
            if r.status_code == 200:
                data = r.json()
                print(f"✅ Token صالح — الصفحة: {data.get('name')} (ID: {data.get('id')})")
                return True
            print(f"❌ Token غير صالح: {r.json().get('error', {}).get('message')}")
            return False


# ── واجهة CLI للاختبار ──
async def demo():
    collector = FacebookCollector()
    is_valid  = await collector.verify_token()
    if not is_valid:
        print("⚠️  يرجى إضافة FACEBOOK_ACCESS_TOKEN في ملف .env")
        return

    posts = await collector.fetch_page_posts("me", limit=10)
    for p in posts:
        print(f"📌 {p['text'][:80]}...")


if __name__ == "__main__":
    asyncio.run(demo())
