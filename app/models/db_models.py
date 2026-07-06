"""
╔══════════════════════════════════════════════════════════════════╗
║   قاعدة البيانات الموسّعة — SentimentAI v2.0                   ║
║                                                                  ║
║   الجداول (8):                                                   ║
║   1. users            — المستخدمون وصلاحياتهم                   ║
║   2. platforms        — المنصات المدعومة وإعداداتها              ║
║   3. topics           — الموضوعات والتصنيفات                     ║
║   4. posts            — المنشورات المُحلَّلة (موسّع)             ║
║   5. post_keywords    — الكلمات المفتاحية لكل منشور              ║
║   6. analysis_sessions— جلسات التحليل الجماعي (موسّع)           ║
║   7. model_versions   — إصدارات النماذج ومقاييس أدائها          ║
║   8. alerts           — تنبيهات السمعة والأحداث الحرجة          ║
╚══════════════════════════════════════════════════════════════════╝
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Boolean,
    ForeignKey, Index, UniqueConstraint, CheckConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


# ══════════════════════════════════════════════
#  1. users — المستخدمون
# ══════════════════════════════════════════════
class User(Base):
    """
    المستخدمون الذين يستخدمون النظام
    - admin   : صلاحيات كاملة
    - analyst : تحليل وعرض
    - viewer  : عرض فقط
    """
    __tablename__ = "users"

    id           = Column(Integer, primary_key=True, index=True)
    username     = Column(String(100), unique=True, nullable=False, index=True)
    email        = Column(String(255), unique=True, nullable=False)
    password_hash= Column(String(255), nullable=False)           # bcrypt hash
    full_name    = Column(String(200), nullable=True)
    role         = Column(String(20),  nullable=False, default="viewer")  # admin|analyst|viewer
    is_active    = Column(Boolean, default=True)
    last_login   = Column(DateTime(timezone=True), nullable=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())
    updated_at   = Column(DateTime(timezone=True), onupdate=func.now())

    # العلاقات
    sessions     = relationship("AnalysisSession", back_populates="user",
                                 foreign_keys="AnalysisSession.created_by")
    alerts       = relationship("Alert", back_populates="user",
                                 foreign_keys="[Alert.user_id]")

    __table_args__ = (
        CheckConstraint("role IN ('admin','analyst','viewer')", name="ck_user_role"),
    )

    def __repr__(self):
        return f"<User {self.username} [{self.role}]>"


# ══════════════════════════════════════════════
#  2. platforms — المنصات
# ══════════════════════════════════════════════
class Platform(Base):
    """
    منصات التواصل الاجتماعي المدعومة
    تُخزَّن إعدادات API وحالة الاتصال لكل منصة
    """
    __tablename__ = "platforms"

    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String(50),  unique=True, nullable=False)   # facebook | instagram | twitter | manual
    display_name  = Column(String(100), nullable=False)                # Facebook | Instagram
    icon          = Column(String(10),  nullable=True)                 # 📘 | 📸
    api_endpoint  = Column(String(500), nullable=True)                 # رابط API
    is_active     = Column(Boolean, default=True)
    last_sync_at  = Column(DateTime(timezone=True), nullable=True)     # آخر مزامنة
    total_fetched = Column(Integer, default=0)                         # إجمالي المنشورات المجلوبة
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    # العلاقات
    posts         = relationship("Post", back_populates="platform_rel")

    def __repr__(self):
        return f"<Platform {self.name}>"


# ══════════════════════════════════════════════
#  3. topics — الموضوعات
# ══════════════════════════════════════════════
class Topic(Base):
    """
    تصنيف المنشورات حسب الموضوع
    مثال: منتجات، خدمات، سياسة، رياضة، صحة
    """
    __tablename__ = "topics"

    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String(100), unique=True, nullable=False)   # products | politics | sports
    name_ar       = Column(String(100), nullable=True)                 # منتجات | سياسة | رياضة
    description   = Column(Text,        nullable=True)
    color         = Column(String(10),  nullable=True, default="#6366f1")  # لون العرض في الواجهة
    is_active     = Column(Boolean, default=True)
    post_count    = Column(Integer, default=0)                         # عداد تحديثي
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    # العلاقات
    posts         = relationship("Post", back_populates="topic")

    def __repr__(self):
        return f"<Topic {self.name_ar or self.name}>"


# ══════════════════════════════════════════════
#  4. posts — المنشورات (موسّع)
# ══════════════════════════════════════════════
class Post(Base):
    """
    المنشورات المُحلَّلة — الجدول الرئيسي (موسّع من 9 → 22 عمود)

    تحسينات v2:
    + ربط بالمستخدم والمنصة والموضوع (FK)
    + درجات المشاعر الثلاث بشكل صريح
    + عدد التفاعلات (likes, comments, shares)
    + علامة التحقق اليدوي (human_label)
    + حالة المعالجة (status)
    + معرّف المؤلف في المنصة
    """
    __tablename__ = "posts"

    # ── تعريف ──
    id             = Column(Integer, primary_key=True, index=True)
    external_id    = Column(String(200), nullable=True, index=True)    # ID في المنصة الأصلية

    # ── المحتوى ──
    text           = Column(Text,    nullable=False)                   # النص الأصلي
    text_clean     = Column(Text,    nullable=True)                    # النص بعد التنظيف
    language       = Column(String(10), nullable=False, default="ar")  # ar | en | mixed
    char_count     = Column(Integer, nullable=True)                    # عدد الأحرف
    word_count     = Column(Integer, nullable=True)                    # عدد الكلمات

    # ── المصدر ──
    platform_id    = Column(Integer, ForeignKey("platforms.id"), nullable=True, index=True)
    author_id      = Column(String(200), nullable=True)                # معرّف المؤلف في المنصة
    post_url       = Column(String(1000), nullable=True)               # رابط المنشور الأصلي
    posted_at      = Column(DateTime(timezone=True), nullable=True)    # وقت النشر الأصلي

    # ── التصنيف ──
    topic_id       = Column(Integer, ForeignKey("topics.id"), nullable=True, index=True)

    # ── نتائج التحليل ──
    sentiment      = Column(String(20), nullable=True, index=True)     # positive|negative|neutral
    confidence     = Column(Float,      nullable=True)                 # 0.0 – 1.0
    score_positive = Column(Float,      nullable=True)                 # احتمال الإيجابية
    score_negative = Column(Float,      nullable=True)                 # احتمال السلبية
    score_neutral  = Column(Float,      nullable=True)                 # احتمال الحياد
    model_version_id = Column(Integer,  ForeignKey("model_versions.id"), nullable=True)

    # ── التحقق اليدوي ──
    human_label    = Column(String(20), nullable=True)                 # التصنيف البشري (للمقارنة)
    is_correct     = Column(Boolean,    nullable=True)                 # هل توقع النموذج صحيح؟

    # ── التفاعل ──
    likes_count    = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    shares_count   = Column(Integer, default=0)

    # ── الحالة ──
    status         = Column(String(20), nullable=False, default="analyzed")  # pending|analyzed|flagged|ignored
    is_spam        = Column(Boolean, default=False)
    created_at     = Column(DateTime(timezone=True), server_default=func.now())
    updated_at     = Column(DateTime(timezone=True), onupdate=func.now())

    # ── العلاقات ──
    platform_rel   = relationship("Platform",     back_populates="posts")
    topic          = relationship("Topic",         back_populates="posts")
    model_version  = relationship("ModelVersion",  back_populates="posts")
    keywords       = relationship("PostKeyword",   back_populates="post",
                                   cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_posts_sentiment_platform", "sentiment", "platform_id"),
        Index("ix_posts_created_sentiment",  "created_at", "sentiment"),
        CheckConstraint(
            "sentiment IN ('positive','negative','neutral') OR sentiment IS NULL",
            name="ck_post_sentiment",
        ),
        CheckConstraint(
            "status IN ('pending','analyzed','flagged','ignored')",
            name="ck_post_status",
        ),
    )

    def __repr__(self):
        return f"<Post id={self.id} sentiment={self.sentiment}>"


# ══════════════════════════════════════════════
#  5. post_keywords — الكلمات المفتاحية
# ══════════════════════════════════════════════
class PostKeyword(Base):
    """
    الكلمات المفتاحية المستخرجة من كل منشور
    تُستخدم لبناء سحابة الكلمات وتحليل الاتجاهات
    """
    __tablename__ = "post_keywords"

    id         = Column(Integer, primary_key=True, index=True)
    post_id    = Column(Integer, ForeignKey("posts.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    keyword    = Column(String(200), nullable=False, index=True)
    frequency  = Column(Integer, default=1)                      # تكرار الكلمة في المنشور
    weight     = Column(Float,   default=1.0)                    # وزن TF-IDF
    language   = Column(String(10), default="ar")

    # العلاقات
    post       = relationship("Post", back_populates="keywords")

    __table_args__ = (
        UniqueConstraint("post_id", "keyword", name="uq_post_keyword"),
        Index("ix_keyword_lookup", "keyword", "language"),
    )

    def __repr__(self):
        return f"<Keyword '{self.keyword}' post={self.post_id}>"


# ══════════════════════════════════════════════
#  6. analysis_sessions — جلسات التحليل (موسّع)
# ══════════════════════════════════════════════
class AnalysisSession(Base):
    """
    جلسات التحليل الجماعي — موسّع
    تُسجَّل كل عملية batch مع تفاصيلها الكاملة
    """
    __tablename__ = "analysis_sessions"

    id               = Column(Integer, primary_key=True, index=True)
    name             = Column(String(200), nullable=True)              # اسم الجلسة (اختياري)

    # ── المحتوى ──
    total_posts      = Column(Integer, default=0)
    positive         = Column(Integer, default=0)
    negative         = Column(Integer, default=0)
    neutral          = Column(Integer, default=0)
    pending          = Column(Integer, default=0)                      # لم يُحلَّل بعد
    flagged          = Column(Integer, default=0)                      # منشورات مُعلَّمة

    # ── المقاييس ──
    avg_confidence   = Column(Float,   default=0.0)
    min_confidence   = Column(Float,   nullable=True)
    max_confidence   = Column(Float,   nullable=True)
    avg_word_count   = Column(Float,   nullable=True)

    # ── المصدر ──
    platform_id      = Column(Integer, ForeignKey("platforms.id"), nullable=True)
    topic_id         = Column(Integer, ForeignKey("topics.id"),    nullable=True)
    created_by       = Column(Integer, ForeignKey("users.id"),     nullable=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=True)

    # ── الوقت ──
    started_at       = Column(DateTime(timezone=True), server_default=func.now())
    completed_at     = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)                    # مدة التحليل

    # ── العلاقات ──
    platform         = relationship("Platform")
    topic            = relationship("Topic")
    user             = relationship("User", back_populates="sessions",
                                    foreign_keys=[created_by])
    model_version    = relationship("ModelVersion", back_populates="sessions")

    @property
    def positive_pct(self):
        return round(self.positive / self.total_posts * 100, 1) if self.total_posts else 0

    @property
    def negative_pct(self):
        return round(self.negative / self.total_posts * 100, 1) if self.total_posts else 0

    def __repr__(self):
        return f"<Session id={self.id} total={self.total_posts}>"


# ══════════════════════════════════════════════
#  7. model_versions — إصدارات النماذج
# ══════════════════════════════════════════════
class ModelVersion(Base):
    """
    سجل إصدارات نماذج التحليل وأداءها
    يتيح مقارنة نماذج مختلفة وتتبع التحسن عبر الزمن
    """
    __tablename__ = "model_versions"

    id               = Column(Integer, primary_key=True, index=True)
    name             = Column(String(100), nullable=False)             # logistic-v1 | arabert-v2
    model_type       = Column(String(50),  nullable=False)             # logistic | arabert | svm
    version          = Column(String(20),  nullable=False, default="1.0.0")
    description      = Column(Text, nullable=True)
    is_active        = Column(Boolean, default=False)                  # النموذج المستخدم حالياً

    # ── مقاييس الأداء ──
    accuracy         = Column(Float, nullable=True)                    # الدقة الكلية
    f1_score         = Column(Float, nullable=True)                    # F1 المتوسط
    precision_score  = Column(Float, nullable=True)
    recall_score     = Column(Float, nullable=True)
    cv_mean          = Column(Float, nullable=True)                    # Cross-Validation Mean
    cv_std           = Column(Float, nullable=True)

    # ── بيانات التدريب ──
    train_size       = Column(Integer, nullable=True)                  # عدد أمثلة التدريب
    test_size        = Column(Integer, nullable=True)
    vocab_size       = Column(Integer, nullable=True)                  # حجم المفردات

    # ── الملف ──
    file_path        = Column(String(500), nullable=True)              # مسار ملف النموذج
    file_size_kb     = Column(Float, nullable=True)

    trained_at       = Column(DateTime(timezone=True), server_default=func.now())
    trained_by       = Column(Integer, ForeignKey("users.id"), nullable=True)

    # ── العلاقات ──
    posts            = relationship("Post",            back_populates="model_version")
    sessions         = relationship("AnalysisSession", back_populates="model_version")

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_name_version"),
    )

    def __repr__(self):
        return f"<ModelVersion {self.name} v{self.version} acc={self.accuracy}>"


# ══════════════════════════════════════════════
#  8. alerts — التنبيهات
# ══════════════════════════════════════════════
class Alert(Base):
    """
    تنبيهات النظام عند تجاوز عتبات معينة
    مثال: ارتفاع مفاجئ في السلبية > 60%، كلمة مفتاحية حساسة
    """
    __tablename__ = "alerts"

    id           = Column(Integer, primary_key=True, index=True)
    title        = Column(String(300), nullable=False)
    description  = Column(Text, nullable=True)
    alert_type   = Column(String(50),  nullable=False)                 # sentiment_spike | keyword | volume
    severity     = Column(String(20),  nullable=False, default="medium")  # low|medium|high|critical
    is_read      = Column(Boolean, default=False)
    is_resolved  = Column(Boolean, default=False)

    # ── السياق ──
    platform_id  = Column(Integer, ForeignKey("platforms.id"), nullable=True)
    topic_id     = Column(Integer, ForeignKey("topics.id"),    nullable=True)
    threshold    = Column(Float,   nullable=True)                      # العتبة التي تجاوزتها
    actual_value = Column(Float,   nullable=True)                      # القيمة الفعلية

    # ── المستخدم ──
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolved_by  = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolved_at  = Column(DateTime(timezone=True), nullable=True)

    created_at   = Column(DateTime(timezone=True), server_default=func.now())

    # ── العلاقات ──
    platform     = relationship("Platform")
    topic        = relationship("Topic")
    user         = relationship("User", back_populates="alerts",
                                 foreign_keys=[user_id])

    __table_args__ = (
        CheckConstraint(
            "severity IN ('low','medium','high','critical')",
            name="ck_alert_severity",
        ),
        CheckConstraint(
            "alert_type IN ('sentiment_spike','keyword','volume','accuracy_drop','new_platform')",
            name="ck_alert_type",
        ),
        Index("ix_alerts_unread", "is_read", "severity"),
    )

    def __repr__(self):
        return f"<Alert [{self.severity}] {self.title[:40]}>"
