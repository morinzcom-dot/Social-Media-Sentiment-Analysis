"""
╔══════════════════════════════════════════════════════════════╗
║   run.py — تشغيل المشروع بأمر واحد                         ║
║                                                              ║
║   python run.py              → تشغيل الخادم                 ║
║   python run.py --setup      → إعداد أول مرة                ║
║   python run.py --test       → تشغيل الاختبارات             ║
║   python run.py --demo       → عرض تجريبي سريع              ║
║   python run.py --evaluate   → تقييم النموذج                ║
╚══════════════════════════════════════════════════════════════╝
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


BANNER = """
╔══════════════════════════════════════════════════╗
║    ███████╗███████╗███╗   ██╗████████╗          ║
║    ██╔════╝██╔════╝████╗  ██║╚══██╔══╝          ║
║    ███████╗█████╗  ██╔██╗ ██║   ██║             ║
║    ╚════██║██╔══╝  ██║╚██╗██║   ██║             ║
║    ███████║███████╗██║ ╚████║   ██║             ║
║    ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝             ║
║                                                  ║
║    SentimentAI — نظام تحليل المشاعر             ║
║    مشروع تخرج | Python 3.10+ | FastAPI           ║
╚══════════════════════════════════════════════════╝
"""


def check_python():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        print(f"❌ Python {v.major}.{v.minor} — يلزم Python 3.9+")
        sys.exit(1)
    print(f"✅ Python {v.major}.{v.minor}.{v.micro}")


def check_dependencies():
    """التحقق من تثبيت المكتبات الأساسية"""
    required = {
        "fastapi":      "fastapi",
        "uvicorn":      "uvicorn",
        "sqlalchemy":   "sqlalchemy",
        "aiosqlite":    "aiosqlite",
        "pydantic":     "pydantic",
        "sklearn":      "scikit-learn",
        "numpy":        "numpy",
        "httpx":        "httpx",
    }
    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} — مفقود")
            missing.append(pkg)

    if missing:
        print(f"\n⚠️  مكتبات مفقودة: {', '.join(missing)}")
        install = input("تثبيتها الآن؟ [y/n]: ").strip().lower()
        if install == "y":
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + missing,
                check=True
            )
        else:
            sys.exit(1)


def check_model():
    """التحقق من وجود النموذج المدرَّب"""
    model_path = Path("data/models/logistic_model.pkl")
    if not model_path.exists():
        print("⚠️  النموذج غير موجود — تدريب تلقائي...")
        subprocess.run([sys.executable, "seeder.py", "--source", "builtin"], check=True)
    else:
        print("✅ النموذج المدرَّب موجود")


def check_env():
    """التحقق من ملف .env"""
    env_path = Path(".env")
    if not env_path.exists():
        example = Path(".env.example")
        if example.exists():
            import shutil
            shutil.copy(example, env_path)
            print("✅ تم إنشاء .env من .env.example")
        else:
            # إنشاء .env أساسي
            with open(env_path, "w") as f:
                f.write("DATABASE_URL=sqlite+aiosqlite:///./sentiment_ai.db\n")
                f.write("MODEL_TYPE=logistic\n")
                f.write("SECRET_KEY=dev-secret-key-change-in-production\n")
                f.write('ALLOWED_ORIGINS=["http://localhost:3000"]\n')
            print("✅ تم إنشاء .env افتراضي")
    else:
        print("✅ ملف .env موجود")


def setup():
    """إعداد شامل للمشروع — أول مرة"""
    print("\n🔧 إعداد المشروع...\n")
    check_python()
    print("\nالتحقق من المكتبات:")
    check_dependencies()
    print("\nالتحقق من الملفات:")
    check_env()
    Path("data/models").mkdir(parents=True, exist_ok=True)
    print("✅ مجلد data/models موجود")
    print("\nتدريب النموذج:")
    check_model()
    print("\n✅ الإعداد مكتمل! شغّل: python run.py")


def run_server(host="0.0.0.0", port=8000, reload=True):
    """تشغيل خادم FastAPI"""
    print(f"\n🚀 تشغيل الخادم على http://localhost:{port}")
    print(f"   📚 التوثيق: http://localhost:{port}/docs")
    print(f"   🔁 ReDoc:   http://localhost:{port}/redoc")
    print(f"\n   اضغط Ctrl+C للإيقاف\n")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", host,
        "--port", str(port),
        "--reload" if reload else "",
    ])


def run_tests():
    """تشغيل الاختبارات"""
    print("\n🧪 تشغيل الاختبارات...\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    )
    if result.returncode == 0:
        print("\n✅ جميع الاختبارات نجحت!")
    else:
        print("\n❌ بعض الاختبارات فشلت — راجع الأخطاء أعلاه")
    return result.returncode


def run_demo():
    """عرض تجريبي سريع"""
    import sys
    sys.path.insert(0, ".")

    from app.services.sentiment_service import SentimentService

    service = SentimentService()

    samples = [
        ("المنتج رائع جداً وأنصح به بشدة! جودة ممتازة وتوصيل سريع", "إيجابي"),
        ("خدمة سيئة جداً لن أتعامل معهم مرة أخرى أبداً", "سلبي"),
        ("المنتج عادي يؤدي الغرض لا ناقص ولا زايد", "محايد"),
        ("This product is absolutely amazing! Best purchase ever made.", "إيجابي"),
        ("Terrible experience very disappointed with quality", "سلبي"),
        ("Okay product nothing special does the job fine", "محايد"),
    ]

    print("\n" + "═"*60)
    print("  🎯 عرض تجريبي — SentimentAI Live Demo")
    print("═"*60)

    correct = 0
    for text, expected_ar in samples:
        r = service.analyze(text)
        ar_map = {"positive": "إيجابي 😊", "negative": "سلبي 😠", "neutral": "محايد 😐"}
        result_ar = ar_map[r["sentiment"]]
        ok = "✅" if r["sentiment_ar"].startswith(expected_ar[:2]) else "❌"
        if ok == "✅":
            correct += 1

        lang_flag = "🇸🇦" if r["language"] == "ar" else "🇺🇸" if r["language"] == "en" else "🌍"
        print(f"\n  {lang_flag} النص     : {text[:55]}")
        print(f"     النتيجة  : {result_ar}")
        print(f"     الثقة    : {r['confidence_pct']}")
        print(f"     النموذج  : {r['model_used']}")
        print(f"     {ok} {'صحيح' if ok == '✅' else 'خطأ'}")

    print(f"\n{'═'*60}")
    print(f"  الدقة الإجمالية: {correct}/{len(samples)} = {correct/len(samples)*100:.0f}%")
    print(f"{'═'*60}\n")


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        description="SentimentAI — أداة تشغيل المشروع",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--setup",    action="store_true", help="إعداد المشروع أول مرة")
    parser.add_argument("--test",     action="store_true", help="تشغيل الاختبارات")
    parser.add_argument("--demo",     action="store_true", help="عرض تجريبي سريع")
    parser.add_argument("--evaluate", action="store_true", help="تقييم النموذج")
    parser.add_argument("--compare",  action="store_true", help="مقارنة النماذج")
    parser.add_argument("--host",     default="0.0.0.0",   help="عنوان الاستماع")
    parser.add_argument("--port",     type=int, default=8000, help="رقم المنفذ")
    parser.add_argument("--no-reload",action="store_true", help="إيقاف auto-reload")
    args = parser.parse_args()

    if args.setup:
        setup()
    elif args.test:
        sys.exit(run_tests())
    elif args.demo:
        run_demo()
    elif args.evaluate:
        from evaluate import evaluate_saved_model
        evaluate_saved_model()
    elif args.compare:
        from evaluate import compare_models
        compare_models()
    else:
        # التشغيل الافتراضي
        check_env()
        check_model()
        run_server(args.host, args.port, not args.no_reload)


if __name__ == "__main__":
    main()
