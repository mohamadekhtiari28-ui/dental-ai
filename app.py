import base64
import os
import re

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ═════════════════════════════════════════════
# تنظیمات صفحه
# ═════════════════════════════════════════════
st.set_page_config(
    page_title="ماکان | دستیار هوشمند تشخیص دندان عقل",
    page_icon="🦷",
    layout="wide",
)

LOGO_PATH = "macan_logo.png"

# ═════════════════════════════════════════════
# استایل (تم تیره پزشکی)
# ═════════════════════════════════════════════
CSS = """
@import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font@v30.1.0/dist/font-face.css');

:root{
  --bg:#050b1e; --card:#0f1a3a;
  --line:rgba(56,189,248,.18);
  --accent:#22d3ee; --accent2:#0ea5e9;
  --text:#e6f1ff; --muted:#8fa6c8;
  --warn:#f59e0b;
}

html, body, [class*="css"], .stApp, button, input, textarea, label{
  font-family:'Vazir', Tahoma, sans-serif !important;
}

.stApp{
  background: radial-gradient(1200px 600px at 50% -10%, #12306b 0%, var(--bg) 60%) fixed;
  color:var(--text);
}

#MainMenu, footer {visibility:hidden;}

.block-container{max-width:1180px; padding-top:2rem; padding-bottom:3rem;}

[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *{
  direction:rtl; text-align:right;
}

/* ─── تب‌ها ─── */
.stTabs [data-baseweb="tab-list"]{
  direction:rtl; gap:8px;
  background:rgba(255,255,255,.03);
  padding:6px; border-radius:12px;
  border:1px solid var(--line);
}
.stTabs [data-baseweb="tab"]{
  border-radius:10px !important;
  padding:8px 14px !important;
  color:var(--muted) !important;
}
.stTabs [aria-selected="true"]{
  color:var(--accent) !important;
  background:rgba(34,211,238,.1);
}
.stTabs [data-baseweb="tab-highlight"]{
  background-color:var(--accent) !important;
}

/* ─── اکسپندر ─── */
[data-testid="stExpander"]{
  border:1px solid var(--line) !important;
  border-radius:16px !important;
  background:rgba(15,26,58,.6);
  overflow:hidden;
  margin-bottom:14px;
}
[data-testid="stExpander"] p{direction:rtl; text-align:right;}

/* ─── آپلودر ─── */
[data-testid="stFileUploader"] section{
  background:rgba(14,165,233,.05) !important;
  border:2px dashed rgba(34,211,238,.5) !important;
  border-radius:16px !important;
  padding:22px !important;
}
[data-testid="stFileUploader"] section:hover{
  border-color:var(--accent) !important;
  background:rgba(34,211,238,.09) !important;
}
[data-testid="stFileUploader"] button{
  background:linear-gradient(90deg, var(--accent2), var(--accent)) !important;
  color:#031022 !important;
  border:none !important;
  border-radius:10px !important;
  font-weight:700 !important;
}

/* ─── هدر ─── */
.hero{
  background:linear-gradient(135deg, rgba(14,165,233,.18), rgba(99,102,241,.12));
  border:1px solid var(--line);
  border-radius:18px;
  padding:22px 28px;
  margin-bottom:16px;
  box-shadow:0 10px 40px rgba(0,0,0,.35);
}
.hero-inner{
  display:flex;
  align-items:center;
  gap:22px;
}
.hero-text{flex:1;}
.logo-wrap{
  background:#ffffff;
  border-radius:16px;
  padding:8px 12px;
  box-shadow:0 6px 20px rgba(0,0,0,.35);
  flex-shrink:0;
}
.logo-wrap img{
  height:84px;
  width:auto;
  display:block;
}
.logo-fallback{
  font-size:3.2rem;
  flex-shrink:0;
}
.hero h1{margin:0 0 4px; font-size:1.9rem; color:#7dd3fc; font-weight:800;}
.hero h2{margin:0 0 6px; font-size:1.1rem; color:#e0f2fe; font-weight:700;}
.hero p{margin:0; color:var(--muted);}

/* ─── هشدار و اطلاع ─── */
.notice{
  background:rgba(245,158,11,.1);
  border:1px solid rgba(245,158,11,.4);
  border-radius:12px;
  padding:12px 16px;
  color:#fde68a;
  line-height:1.9;
  font-size:.92rem;
  margin-bottom:16px;
}
.notice.info{
  background:rgba(56,189,248,.1);
  border-color:rgba(56,189,248,.35);
  color:#bae6fd;
}

/* ─── عنوان بخش ─── */
.section-title{
  font-size:1.15rem; font-weight:800; color:#bae6fd;
  margin:22px 0 12px; padding-right:12px;
  border-right:4px solid var(--accent);
}

/* ─── آمار کلی ─── */
.stats{display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:8px 0 18px;}
.stat{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px; text-align:center;}
.sv{font-size:1.6rem; font-weight:800; color:var(--accent);}
.sl{font-size:.82rem; color:var(--muted); margin-top:4px;}

/* ─── کارت دندان ─── */
.tooth-card{
  background:linear-gradient(160deg, #122250, #0b1530);
  border:1px solid var(--line);
  border-top:4px solid var(--accent);
  border-radius:16px;
  padding:16px;
  box-shadow:0 8px 24px rgba(0,0,0,.35);
  margin-bottom:12px;
}
.tc-head{display:flex; justify-content:space-between; align-items:flex-start; gap:8px; margin-bottom:12px;}
.tc-num{font-size:1.25rem; font-weight:800; color:#e0f2fe;}
.tc-jaw{font-size:.82rem; color:var(--muted); margin-top:2px;}
.badge{padding:4px 10px; border-radius:999px; font-size:.78rem; font-weight:700; white-space:nowrap;}
.tc-grid{display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px;}
.metric{background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.05); border-radius:10px; padding:8px 10px;}
.metric.wide{grid-column:span 2;}
.metric .lbl{font-size:.75rem; color:var(--muted);}
.metric .val{font-size:1rem; font-weight:700; color:var(--text); margin-top:2px;}
.conf-row{display:flex; justify-content:space-between; font-size:.8rem; color:var(--muted); margin-bottom:6px;}
.bar{height:8px; background:rgba(255,255,255,.08); border-radius:999px; overflow:hidden;}
.bar span{display:block; height:100%; background:linear-gradient(90deg, var(--accent2), var(--accent)); border-radius:999px;}

/* ─── کارت‌های جزئیات ─── */
.dentist-card, .patient-card, .size-box{
  border-radius:12px; padding:14px 16px; margin-bottom:12px; line-height:1.9;
}
.dentist-card{background:rgba(14,165,233,.08); border-right:4px solid #0ea5e9;}
.patient-card{background:rgba(34,197,94,.08); border-right:4px solid #22c55e;}
.size-box{background:rgba(245,158,11,.08); border-right:4px solid var(--warn);}
.dentist-card h4, .patient-card h4{margin:0 0 6px; color:#e0f2fe;}
.dentist-card ul, .patient-card ul{padding-right:20px; margin:4px 0;}
.dentist-card p, .patient-card p{margin:4px 0;}
.sb-row{margin:3px 0;}
.sb-row.small{font-size:.82rem; color:var(--muted);}

/* ─── فوتر ─── */
.footer-note{text-align:center; color:var(--muted); font-size:.8rem; margin-top:24px;}
"""
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)

# ═════════════════════════════════════════════
# توابع کمکی نمایش
# ═════════════════════════════════════════════
FA_DIGITS = str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫")


def fa(value):
    """تبدیل ارقام به فارسی"""
    return str(value).translate(FA_DIGITS)


def clean(html):
    """حذف فاصله‌های اضافی از HTML تا Markdown آن را کد-بلاک نکند"""
    return " ".join(line.strip() for line in html.splitlines() if line.strip())


def inline_md(text):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def text_to_html(text):
    """تبدیل متن با خط تیره به لیست HTML"""
    out, items = [], []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("- "):
            items.append(f"<li>{inline_md(line[2:])}</li>")
        else:
            if items:
                out.append("<ul>" + "".join(items) + "</ul>")
                items = []
            out.append(f"<p>{inline_md(line)}</p>")
    if items:
        out.append("<ul>" + "".join(items) + "</ul>")
    return "".join(out)


def logo_html():
    """لوگوی ماکان؛ در صورت نبود فایل، ایموجی دندان نمایش داده می‌شود"""
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<div class="logo-wrap"><img src="data:image/png;base64,{b64}" alt="لوگوی ماکان"></div>'
    return '<div class="logo-fallback">🦷</div>'


# ═════════════════════════════════════════════
# بارگذاری مدل
# ═════════════════════════════════════════════
@st.cache_resource
def load_model():
    if not os.path.exists("best_model.pt"):
        return None
    return YOLO("best_model.pt")


model = load_model()

if model is None:
    st.error("❌ فایل مدل (best_model.pt) پیدا نشد. لطفاً در کنار app.py در گیت‌هاب آن را آپلود کنید.")
    st.stop()

names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
target_teeth = ["18", "28", "38", "48"]
wisdom_indices = [idx for idx, name in names.items() if name in target_teeth]

if not wisdom_indices:
    st.error("هیچ‌کدام از کلاس‌های ۱۸، ۲۸، ۳۸، ۴۸ در مدل پیدا نشد.")
    st.stop()

# ═════════════════════════════════════════════
# اطلاعات پایه هر دندان
# ═════════════════════════════════════════════
TOOTH_DETAILS = {
    "18": {"title": "دندان عقل بالا - سمت راست", "jaw": "maxilla", "side": "right"},
    "28": {"title": "دندان عقل بالا - سمت چپ", "jaw": "maxilla", "side": "left"},
    "38": {"title": "دندان عقل پایین - سمت چپ", "jaw": "mandible", "side": "left"},
    "48": {"title": "دندان عقل پایین - سمت راست", "jaw": "mandible", "side": "right"},
}


# ═════════════════════════════════════════════
# توابع تولید توصیه‌ها (منطق دست‌نخورده)
# ═════════════════════════════════════════════
def generate_insights(tooth_name, w_mm, h_mm, aspect_ratio):
    info = TOOTH_DETAILS.get(tooth_name, {})
    jaw = info.get("jaw", "mandible")

    # تخمین جهت تقریبی از روی تناسب ابعاد باکس (عرض به ارتفاع)
    # اگر عرض پهن‌تر از حد معمول باشد یعنی دندان خوابیده/کج شده
    if jaw == "mandible":
        if aspect_ratio >= 0.88:
            impaction_type = "احتمال نهفتگی افقی یا زاویه‌دار به سمت جلو (Mesioangular / Horizontal)"
            dentist_tips = """
            - **وضعیت احتمالی:** دندان دارای زاویه خوابیده یا مزیالی است و احتمالاً به ریشه دندان ۷ فشار می‌آورد.
            - **ریسک بالینی:** احتمال تحلیل ریشه دندان مجاور (External Root Resorption) و ایجاد پاکت پریودنتال عمیق.
            - **ارزیابی عصب فک:** با توجه به عمق قرارگیری، بررسی دقیق فاصله آپکس ریشه‌ها با کانال عصب مندیبولار (IAN) در رادیوگرافی الزامی است؛ در صورت سوپرایمپوز شدن آپکس با کانال، درخواست **CBCT** توصیه می‌شود.
            - **طرح درمان:** معمولاً نیازمند جراحی با خارج‌سازی بافت نرم، فلپ و ترجیحاً سکشن کردن (Odontotomy) تاج و ریشه است.
            """
            patient_tips = f"""
            سلام دوست من! دندان عقل پایینت (شماره {tooth_name}) ظاهراً صاف درنیامده و مثل این می‌ماند که کمی **خوابیده یا کج به دندان کناری‌اش تکیه داده است**.
            - این حالت می‌تواند باعث گیر کردن غذا و ایجاد پوسیدگی روی دندان سالم کناری‌اش شود.
            - حتماً به دندان‌پزشک سر بزن تا با یک عکس سه‌بعدی یا دقیق‌تر چک کند و در صورت نیاز با یک جراحی ساده و بی‌درد خیالت را راحت کند!
            """
        else:
            impaction_type = "احتمال وضعیت عمودی یا زاویه کم (Vertical)"
            dentist_tips = """
            - **وضعیت احتمالی:** دندان مسیر رویش عمودی‌تری دارد.
            - **ریسک بالینی:** احتمال پریکورونیت (عفونت لثه روی دندان نیمه‌نهفته) و گیر غذایی در ناحیه رترومولار.
            - **طرح درمان:** در صورت داشتن فضای کافی در قوس و اکلوژن مناسب می‌تواند نگهداری شود، در غیر این صورت خارج‌سازی ساده یا جراحی جزیی لثه.
            """
            patient_tips = f"""
            دندان عقلت (شماره {tooth_name}) نسبتاً صاف و سرپاست!
            - مهم‌ترین موضوع درباره‌اش این است که انتهایی‌ترین جای دهان است و مسواک زدنش سخته.
            - اگر احساس درد یا بوی بد دهان نداری عالیه، ولی با دندان‌پزشکت چک کن تا مطمئن بشی لثه روش متورم نمی‌شود یا تمیز کردنش اذیتت نمی‌کند.
            """
    else:  # فک بالا (18, 28)
        impaction_type = "دندان عقل فک بالا (Maxillary Molar)"
        dentist_tips = """
        - **وضعیت کف سینوس:** ارتباط و نزدیکی آپکس با کف سینوس ماگزیلاری (Maxillary Sinus) ارزیابی شود (ریسک جابجایی دندان یا ریشه به داخل سینوس در صورت وارد کردن فشار نامناسب).
        - **توبریوزیته:** دقت در حین الویشن جهت پیشگیری از شکستگی استخوان توبریوزیته فک بالا.
        - **طرح درمان:** به دلیل تراکم استخوانی کمتر فک بالا (کنسلوس بیشتر)، معمولاً خارج‌سازی روتین آسان‌تری نسبت به فک پایین دارد مگر آنکه دیستوانگولار و های-امپکت باشد.
        """
        patient_tips = f"""
        این دندان در فک بالای شماست (شماره {tooth_name}). دندان‌های عقل بالا معمولاً کشیدنشان نسبت به فک پایین خیلی راحت‌تر و با درد و نقاهت خیلی کمتری همراه است.
        دندان‌پزشک موقعیتش را نسبت به فضای هوایی بالای دندانت (سینوس) چک می‌کند تا بدون هیچ نگرانی درمانت انجام شود.
        """

    return impaction_type, dentist_tips, patient_tips


def get_status(jaw, aspect_ratio):
    """وضعیت نمایشی هر دندان بر اساس همان قاعده تناسب باکس"""
    if jaw == "mandible" and aspect_ratio >= 0.88:
        return {"label": "نیاز به بررسی دقیق", "color": "#ef4444",
                "bg": "rgba(239,68,68,.15)", "short": "افقی / مزیالی (احتمالی)"}
    if jaw == "mandible":
        return {"label": "وضعیت نسبتاً عمودی", "color": "#22c55e",
                "bg": "rgba(34,197,94,.15)", "short": "عمودی (احتمالی)"}
    return {"label": "بررسی سینوس", "color": "#f59e0b",
            "bg": "rgba(245,158,11,.15)", "short": "فک بالا"}


def tooth_card_html(e):
    s = e["status"]
    return clean(f"""
    <div class="tooth-card" style="border-top-color:{s['color']}">
      <div class="tc-head">
        <div>
          <div class="tc-num">دندان {e['tooth']}</div>
          <div class="tc-jaw">{e['jaw_fa']}</div>
        </div>
        <span class="badge" style="color:{s['color']}; background:{s['bg']}">{s['label']}</span>
      </div>
      <div class="tc-grid">
        <div class="metric">
          <div class="lbl">عرض تقریبی</div>
          <div class="val">{e['w_txt']} mm</div>
        </div>
        <div class="metric">
          <div class="lbl">طول تقریبی</div>
          <div class="val">{e['h_txt']} mm</div>
        </div>
        <div class="metric wide">
          <div class="lbl">شکل قرارگیری (تقریبی)</div>
          <div class="val">{s['short']}</div>
        </div>
      </div>
      <div class="conf-row"><span>اطمینان مدل</span><span>{e['conf_txt']}٪</span></div>
      <div class="bar"><span style="width:{e['conf']*100:.1f}%"></span></div>
    </div>
    """)


# ═════════════════════════════════════════════
# هدر و اطلاعیه
# ═════════════════════════════════════════════
st.markdown(clean(f"""
<div class="hero">
  <div class="hero-inner">
    {logo_html()}
    <div class="hero-text">
      <h1>خوش آمدید به ماکان</h1>
      <h2>دستیار هوشمند و بالینی دندان عقل (OPG)</h2>
      <p>تشخیص، تخمین ابعاد و ارائه توصیه‌های تخصصی بر پایه هوش مصنوعی YOLO</p>
    </div>
  </div>
</div>
"""), unsafe_allow_html=True)

st.markdown(clean("""
<div class="notice">
  ⚠️ <b>توجه:</b> ابعاد و تحلیل‌های این برنامه تخمین هوش مصنوعی هستند و جایگزین
  محاسبه کالیبره رادیولوژیست یا ارزیابی بالینی دندان‌پزشک نیستند.
</div>
"""), unsafe_allow_html=True)

# ═════════════════════════════════════════════
# آپلود تصویر
# ═════════════════════════════════════════════
st.markdown('<div class="section-title">📤 بارگذاری تصویر</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "تصویر رادیوگرافی پانورامیک (OPG) خود را بارگذاری کنید:",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(clean("""
    <div class="notice info">
      تصویر رادیوگرافی پانورامیک (OPG) را بارگذاری کنید تا دندان‌های عقل
      (۱۸، ۲۸، ۳۸، ۴۸) شناسایی و تحلیل شوند.
    </div>
    """), unsafe_allow_html=True)

# ═════════════════════════════════════════════
# پردازش
# ═════════════════════════════════════════════
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]

    # تخمین مقیاس میلی‌متر بر حسب رزولوشن تصویر (تقریبی و غیرکالیبره)
    mm_per_pixel = 155.0 / img_h if img_h > 0 else 0.12

    with st.spinner("⏳ در حال پردازش تصویر و استخراج ویژگی‌های آناتومیک..."):
        results = model.predict(
            source=img_array,
            classes=wisdom_indices,
            conf=0.35,
            show_conf=False,
            line_width=3,
            verbose=False,
        )

    result = results[0]
    result_image_bgr = result.plot(conf=False)
    result_image_rgb = cv2.cvtColor(result_image_bgr, cv2.COLOR_BGR2RGB)

    # ─── نمایش تصویر ───
    st.markdown('<div class="section-title">🖼️ نتیجه تشخیص</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.image(
            result_image_rgb,
            caption="تصویر OPG همراه با محدوده‌های تشخیص داده شده",
            use_container_width=True,
        )

    # ─── ساخت اطلاعات دندان‌ها ───
    entries = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        tooth = names[cls_id]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        w_px = float(x2 - x1)
        h_px = float(y2 - y1)

        w_mm = round(w_px * mm_per_pixel, 1)
        h_mm = round(h_px * mm_per_pixel, 1)
        aspect = w_px / h_px if h_px > 0 else 1.0

        impaction, dentist_tips, patient_tips = generate_insights(tooth, w_mm, h_mm, aspect)

        info = TOOTH_DETAILS.get(tooth, {"title": f"دندان {tooth}", "jaw": "mandible"})
        jaw = info.get("jaw", "mandible")

        entries.append({
            "tooth": tooth,
            "title": info.get("title", f"دندان {tooth}"),
            "jaw_fa": "فک پایین (Mandible)" if jaw == "mandible" else "فک بالا (Maxilla)",
            "w_txt": fa(f"{w_mm:.1f}"),
            "h_txt": fa(f"{h_mm:.1f}"),
            "conf": conf,
            "conf_txt": fa(f"{conf * 100:.1f}"),
            "impaction": impaction,
            "dentist": dentist_tips,
            "patient": patient_tips,
            "status": get_status(jaw, aspect),
        })

    entries.sort(key=lambda e: target_teeth.index(e["tooth"]) if e["tooth"] in target_teeth else 99)

    # ─── حالت بدون تشخیص ───
    if len(entries) == 0:
        st.markdown(clean("""
        <div class="notice info">
          ℹ️ در این تصویر هیچ دندان عقلی (۱۸، ۲۸، ۳۸، ۴۸) یافت نشد
          (یا ممکن است قبلاً کشیده شده باشند).
        </div>
        """), unsafe_allow_html=True)

    else:
        # ─── آمار کلی ───
        review_count = sum(1 for e in entries if e["status"]["label"] == "نیاز به بررسی دقیق")
        avg_conf = sum(e["conf"] for e in entries) / len(entries)
        count_txt = fa(len(entries))
        review_txt = fa(review_count)
        avg_txt = fa(f"{avg_conf * 100:.1f}") + "٪"

        st.markdown('<div class="section-title">📊 خلاصه نتایج</div>', unsafe_allow_html=True)
        st.markdown(clean(f"""
        <div class="stats">
          <div class="stat">
            <div class="sv">{count_txt}</div>
            <div class="sl">دندان شناسایی‌شده</div>
          </div>
          <div class="stat">
            <div class="sv" style="color:#f87171">{review_txt}</div>
            <div class="sl">نیازمند بررسی دقیق</div>
          </div>
          <div class="stat">
            <div class="sv">{avg_txt}</div>
            <div class="sl">میانگین اطمینان مدل</div>
          </div>
        </div>
        """), unsafe_allow_html=True)

        # ─── کارت‌های دندان ───
        st.markdown('<div class="section-title">🦷 وضعیت دندان‌های شناسایی‌شده</div>', unsafe_allow_html=True)
        cols = st.columns(len(entries))
        for col, e in zip(cols, entries):
            with col:
                st.markdown(tooth_card_html(e), unsafe_allow_html=True)

        # ─── جزئیات هر دندان ───
        st.markdown('<div class="section-title">📋 تحلیل تفصیلی</div>', unsafe_allow_html=True)

        for e in entries:
            s = e["status"]
            with st.expander(f"📍 دندان {e['tooth']} — {e['title']}", expanded=True):

                st.markdown(clean(f"""
                <div class="size-box" style="border-right-color:{s['color']}">
                  <div class="sb-row"><b>📏 تخمین ابعاد تاج و ریشه:</b> عرض تقریبی {e['w_txt']} میلی‌متر | طول کلی {e['h_txt']} میلی‌متر</div>
                  <div class="sb-row"><b>🧭 وضعیت تقریبی:</b> {e['impaction']}</div>
                  <div class="sb-row small">🎯 میزان اطمینان مدل در این موقعیت: {e['conf_txt']}٪</div>
                </div>
                """), unsafe_allow_html=True)

                tab_dentist, tab_patient = st.tabs([
                    "👨‍⚕️ ملاحظات بالینی (دندان‌پزشک)",
                    "💬 پیام ساده و دوستانه (بیمار)",
                ])

                with tab_dentist:
                    st.markdown(clean(f"""
                    <div class="dentist-card">
                      <h4>📋 توصیه‌ها و هشدارهای تشخیصی</h4>
                      {text_to_html(e['dentist'])}
                    </div>
                    """), unsafe_allow_html=True)

                with tab_patient:
                    st.markdown(clean(f"""
                    <div class="patient-card">
                      <h4>💡 اوضاع دندانت چطوره؟</h4>
                      {text_to_html(e['patient'])}
                    </div>
                    """), unsafe_allow_html=True)

    st.markdown(clean("""
    <div class="footer-note">
      ⚠️ خروجی این برنامه صرفاً کمک‌تشخیصی است و جایگزین ارزیابی دندان‌پزشک یا گزارش رادیولوژیست نیست.
    </div>
    """), unsafe_allow_html=True)
    
