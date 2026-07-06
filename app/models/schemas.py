"""
مخططات Pydantic v2 — التحقق من صحة البيانات
يغطي جميع الجداول الثمانية
"""

from pydantic import BaseModel, Field, field_validator, EmailStr
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


# ══ Enums ══
class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"

class RoleEnum(str, Enum):
    ADMIN    = "admin"
    ANALYST  = "analyst"
    VIEWER   = "viewer"

class SeverityEnum(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"

class StatusEnum(str, Enum):
    PENDING  = "pending"
    ANALYZED = "analyzed"
    FLAGGED  = "flagged"
    IGNORED  = "ignored"


# ══════════════════════════════════════════════
#  USER schemas
# ══════════════════════════════════════════════
class UserCreate(BaseModel):
    username:  str  = Field(..., min_length=3, max_length=100)
    email:     str  = Field(..., min_length=5, max_length=255)
    password:  str  = Field(..., min_length=6)
    full_name: Optional[str] = None
    role:      RoleEnum = RoleEnum.VIEWER

class UserOut(BaseModel):
    id:         int
    username:   str
    email:      str
    full_name:  Optional[str]
    role:       str
    is_active:  bool
    last_login: Optional[datetime]
    created_at: datetime
    model_config = {"from_attributes": True}

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    role:      Optional[RoleEnum] = None
    is_active: Optional[bool] = None


# ══════════════════════════════════════════════
#  PLATFORM schemas
# ══════════════════════════════════════════════
class PlatformCreate(BaseModel):
    name:         str = Field(..., min_length=2, max_length=50)
    display_name: str = Field(..., max_length=100)
    icon:         Optional[str] = None
    api_endpoint: Optional[str] = None

class PlatformOut(BaseModel):
    id:            int
    name:          str
    display_name:  str
    icon:          Optional[str]
    is_active:     bool
    total_fetched: int
    last_sync_at:  Optional[datetime]
    created_at:    datetime
    model_config = {"from_attributes": True}


# ══════════════════════════════════════════════
#  TOPIC schemas
# ══════════════════════════════════════════════
class TopicCreate(BaseModel):
    name:        str = Field(..., min_length=2, max_length=100)
    name_ar:     Optional[str] = None
    description: Optional[str] = None
    color:       Optional[str] = "#6366f1"

class TopicOut(BaseModel):
    id:          int
    name:        str
    name_ar:     Optional[str]
    description: Optional[str]
    color:       Optional[str]
    is_active:   bool
    post_count:  int
    created_at:  datetime
    model_config = {"from_attributes": True}


# ══════════════════════════════════════════════
#  POST schemas
# ══════════════════════════════════════════════
class AnalyzeRequest(BaseModel):
    text:        str  = Field(..., min_length=2, max_length=3000)
    platform:    Optional[str] = "manual"
    platform_id: Optional[int] = None
    topic_id:    Optional[int] = None
    author_id:   Optional[str] = None
    post_url:    Optional[str] = None
    posted_at:   Optional[datetime] = None
    save:        bool = True

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("النص لا يمكن أن يكون فارغاً")
        return v.strip()

    model_config = {"json_schema_extra": {"example": {
        "text": "المنتج رائع جداً وأنصح به!",
        "platform": "facebook",
        "topic_id": 1,
        "save": True,
    }}}

class BatchAnalyzeRequest(BaseModel):
    texts:       List[str]       = Field(..., min_length=1, max_length=200)
    platform:    Optional[str]   = "manual"
    platform_id: Optional[int]   = None
    topic_id:    Optional[int]   = None
    save:        bool = True

class SentimentResult(BaseModel):
    text_original:  str
    text_clean:     str
    sentiment:      SentimentEnum
    sentiment_ar:   str
    confidence:     float = Field(..., ge=0.0, le=1.0)
    confidence_pct: str
    language:       str
    model_used:     str
    scores:         Dict[str, float]
    post_id:        Optional[int] = None
    model_config = {"from_attributes": True}

class BatchResult(BaseModel):
    total:        int
    positive:     int
    negative:     int
    neutral:      int
    positive_pct: float
    negative_pct: float
    neutral_pct:  float
    avg_confidence: float
    duration_ms:  float
    results:      List[SentimentResult]

class PostOut(BaseModel):
    id:             int
    text:           str
    text_clean:     Optional[str]
    language:       str
    sentiment:      Optional[str]
    confidence:     Optional[float]
    score_positive: Optional[float]
    score_negative: Optional[float]
    score_neutral:  Optional[float]
    platform_id:    Optional[int]
    topic_id:       Optional[int]
    author_id:      Optional[str]
    post_url:       Optional[str]
    posted_at:      Optional[datetime]
    likes_count:    int
    comments_count: int
    shares_count:   int
    status:         str
    is_spam:        bool
    human_label:    Optional[str]
    is_correct:     Optional[bool]
    word_count:     Optional[int]
    char_count:     Optional[int]
    created_at:     datetime
    model_config = {"from_attributes": True}

class PostUpdate(BaseModel):
    human_label: Optional[str]  = None
    topic_id:    Optional[int]  = None
    status:      Optional[StatusEnum] = None
    is_spam:     Optional[bool] = None


# ══════════════════════════════════════════════
#  KEYWORD schemas
# ══════════════════════════════════════════════
class KeywordOut(BaseModel):
    id:        int
    keyword:   str
    frequency: int
    weight:    float
    language:  str
    model_config = {"from_attributes": True}

class KeywordFreq(BaseModel):
    keyword:   str
    total:     int
    sentiment: Optional[str] = None


# ══════════════════════════════════════════════
#  MODEL VERSION schemas
# ══════════════════════════════════════════════
class ModelVersionCreate(BaseModel):
    name:        str = Field(..., min_length=2, max_length=100)
    model_type:  str
    version:     str = "1.0.0"
    description: Optional[str] = None
    accuracy:    Optional[float] = None
    f1_score:    Optional[float] = None
    train_size:  Optional[int]   = None

class ModelVersionOut(BaseModel):
    id:              int
    name:            str
    model_type:      str
    version:         str
    description:     Optional[str]
    is_active:       bool
    accuracy:        Optional[float]
    f1_score:        Optional[float]
    precision_score: Optional[float]
    recall_score:    Optional[float]
    cv_mean:         Optional[float]
    train_size:      Optional[int]
    vocab_size:      Optional[int]
    file_size_kb:    Optional[float]
    trained_at:      datetime
    model_config = {"from_attributes": True}


# ══════════════════════════════════════════════
#  ALERT schemas
# ══════════════════════════════════════════════
class AlertCreate(BaseModel):
    title:        str = Field(..., min_length=3, max_length=300)
    description:  Optional[str] = None
    alert_type:   str
    severity:     SeverityEnum = SeverityEnum.MEDIUM
    platform_id:  Optional[int] = None
    topic_id:     Optional[int] = None
    threshold:    Optional[float] = None
    actual_value: Optional[float] = None

class AlertOut(BaseModel):
    id:           int
    title:        str
    description:  Optional[str]
    alert_type:   str
    severity:     str
    is_read:      bool
    is_resolved:  bool
    threshold:    Optional[float]
    actual_value: Optional[float]
    platform_id:  Optional[int]
    topic_id:     Optional[int]
    created_at:   datetime
    resolved_at:  Optional[datetime]
    model_config = {"from_attributes": True}


# ══════════════════════════════════════════════
#  SESSION schemas
# ══════════════════════════════════════════════
class SessionOut(BaseModel):
    id:               int
    name:             Optional[str]
    total_posts:      int
    positive:         int
    negative:         int
    neutral:          int
    pending:          int
    flagged:          int
    avg_confidence:   float
    min_confidence:   Optional[float]
    max_confidence:   Optional[float]
    platform_id:      Optional[int]
    topic_id:         Optional[int]
    duration_seconds: Optional[float]
    started_at:       datetime
    completed_at:     Optional[datetime]
    model_config = {"from_attributes": True}


# ══════════════════════════════════════════════
#  STATS schemas
# ══════════════════════════════════════════════
class StatsOut(BaseModel):
    total_analyzed:  int
    positive:        int
    negative:        int
    neutral:         int
    positive_pct:    float
    negative_pct:    float
    neutral_pct:     float
    avg_confidence:  float
    flagged_count:   int
    spam_count:      int
    by_platform:     List[Dict]
    by_language:     List[Dict]
    by_topic:        List[Dict]
    recent_trend:    List[Dict]
    top_keywords:    List[Dict]
    model_accuracy:  Optional[float]
    unread_alerts:   int
