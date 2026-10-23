import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Dental AI | تشخیص هوشمند دندان عقل", page_icon="🦷", layout="wide")

# ═══════════════════════════════════════════════════════════════
#  🎨  تم تیره نیلی — الهام‌گرفته از Human Teeth Anatomy
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700;900&display=swap');

/* ── پس‌زمینه ── */
.stApp {
    background: linear-gradient(180deg, #060b1e 0%, #0b1538 40%, #0d1a45 60%, #060b1e 100%);
    color: #d4dff7;
    font-family: 'Vazirmatn', Tahoma, sans-serif;
}
#MainMenu, header, footer {visibility: hidden;}
.main .block-container {max-width: 1150px; padding-top: 1.5rem;}

/* ── هیرو ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 900;
    color: #fff;
    text-shadow: 0 0 40px rgba(0,180,255,.6), 0 0 80px rgba(0,180,255,.15);
    direction: rtl;
    margin: 0 0 .5rem;
}
.hero .sub {
    font-size: 1.15rem;
    color: #6db3f2;
    direction: rtl;
    margin-bottom: .8rem;
}
.hero .pill-row {
    display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; direction: rtl;
}
.hero .pill {
    background: rgba(0,120,255,.18);
    border: 1px solid rgba(0,180,255,.25);
    border-radius: 30px;
    padding: .35rem 1rem;
    font-size: .85rem;
    color: #7cc4ff;
}

/* ── کارت شیشه‌ای ── */
.glass {
    background: rgba(12,22,58,.65);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(0,170,255,.12);
    border-radius: 20px;
    padding: 1.6rem;
    margin: 1rem 0;
    box-shadow: 0 10px 40px rgba(0,0,0,.35);
}
.glass:hover {border-color: rgba(0,170,255,.28);}

/* ── بج دندان ── */
.tooth-head {
    display: flex;
    align-items: center;
    gap: 14px;
    background: linear-gradient(135deg, rgba(0,100,255,.25), rgba(0,60,180,.08));
    border: 1px solid rgba(0,180,255,.22);
    border-radius: 16px;
    padding: 1rem 1.4rem;
    margin-bottom: 1.2rem;
    direction: rtl;
}
.tooth-head .num {
    font-size: 2.2rem;
    font-weight: 900;
    color: #00d4ff;
    text-shadow: 0 0 20px rgba(0,200,255,.5);
    line-height: 1;
}
.tooth-head .info h3 {margin:0; color:#fff; font-size:1.15rem;}
.tooth-head .info p {margin:.15rem 0 0; color:#6db3f2; font-size:.9rem;}

/* ── استت‌باکس ── */
.stats-row {display:flex; gap:10px; flex-wrap:wrap; margin:1rem 0;}
.s-box {
    flex:1; min-width:130px;
    background: rgba(0,25,70,.55);
    border:1px solid rgba(0,170,255,.1);
    border-radius:14px;
    padding:.9rem .8rem;
    text-align:center;
    direction:rtl;
}
.s-box .lbl {color:#5da0e8; font-size:.78rem; margin-bottom:.25rem;}
.s-box .val {color:#fff; font-size:1.35rem; font-weight:700;}

/* ── نوار ریسک ── */
.r-high {background:linear-gradient(135deg,rgba(244,67,54,.13),rgba(244,67,54,.04)); border:1px solid rgba(244,67,54,.22); border-right:4px solid #f44336; border-radius:12px; padding:.9rem 1.1rem; margin:.8rem 0; direction:rtl; text-align:right;}
.r-med  {background:linear-gradient(135deg,rgba(255,152,0,.13),rgba(255,152,0,.04)); border:1px solid rgba(255,152,0,.22); border-right:4px solid #ff9800; border-radius:12px; padding:.9rem 1.1rem; margin:.8rem 0; direction:rtl; text-align:right;}
.r-low  {background:linear-gradient(135deg,rgba(76,175,80,.13),rgba(76,175,80,.04)); border:1px solid rgba(76,175,80,.22); border-right:4px solid #4CAF50; border-radius:12px; padding:.9rem 1.1rem; margin:.8rem 0; direction:rtl; text-align:right;}

/* ── تب‌ها ── */
.stTabs [data-baseweb="tab-list"] {
    gap:6px;
    background: rgba(0,15,55,.55);
    border-radius:14px;
    padding:5px;
    direction:rtl;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(0,35,95,.3);
    border-radius:10px;
    padding:.55rem 1.4rem;
    color:#6db3f2;
    font-family:'Vazirmatn',sans-serif;
    font-size:.95rem;
    border:1px solid transparent;
    transition:all .25s;
}
.stTabs [data-baseweb="tab"]:hover {background:rgba(0,75,180,.28); color:#fff;}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background:linear-gradient(135deg,rgba(0,110,255,.5),rgba(0,65,195,.35));
    color:#fff; border-color:rgba(0,180,255,.35);
    box-shadow:0 4px 18px rgba(0,120,255,.25);
}
.stTabs [data-baseweb="tab-panel"] {
    padding:1.4rem;
    background:rgba(8,14,42,.5);
    border-radius:14px;
    border:1px solid rgba(0,170,255,.07);
}

/* ── متریک ── */
.stMetric {background:rgba(0,25,70,.45); border:1px solid rgba(0,170,255,.1); border-radius:12px; padding:.85rem;}
.stMetric label {color:#5da0e8 !important; font-family:'Vazirmatn' !important;}
.stMetric [data-testid="stMetricValue"] {color:#fff !important; font-family:'Vazirmatn' !important;}

/* ── متن بیمار ── */
.pat-box {
    background:linear-gradient(135deg,rgba(76,175,80,.1),rgba(0,180,255,.04));
    border:1px solid rgba(76,175,80,.18);
    border-radius:16px;
    padding:1.5rem 1.6rem;
    direction:rtl; text-align:right;
    color:#d4e8d4; line-height:2.3; font-size:1.02rem;
}
.pat-box h3 {color:#81c784; margin-top:0;}
.pat-box strong {color:#a5d6a7;}
.pat-box ul {padding-right:1.2rem;}
.pat-box li {margin-bottom:.3rem;}

/* ── متن دکتر ── */
.doc-box {
    background:linear-gradient(135deg,rgba(33,150,243,.1),rgba(0,80,255,.04));
    border:1px solid rgba(33,150,243,.18);
    border-radius:16px;
    padding:1.5rem 1.6rem;
    direction:rtl; text-align:right;
    color:#c4d8f0; line-height:2; font-size:.95rem;
}
.doc-box h3 {color:#64b5f6; margin-top:0;}
.doc-box strong {color:#90caf9;}
.doc-box ul {padding-right:1.2rem;}
.doc-box li {margin-bottom:.35rem;}
.doc-box table {width:100%; border-collapse:collapse; margin:.8rem 0;}
.doc-box th {background:rgba(0,80,180,.25); color:#fff; padding:.5rem .8rem; border:1px solid rgba(0,170,255,.1); font-size:.88rem;}
.doc-box td {padding:.5rem .8rem; border:1px solid rgba(0,170,255,.08); color:#b0c8e8;}

/* ── جمع‌بندی ── */
.sum-hero {
    background:linear-gradient(135deg,rgba(0,40,120,.55),rgba(0,18,70,.35));
    border:1px solid rgba(0,180,255,.18);
    border-radius:22px;
    padding:2rem;
    direction:rtl; text-align:right;
    box-shadow:0 10px 35px rgba(0,0,0,.3);
}

/* ── سلب مسئولیت ── */
.disclaimer {
    background:rgba(255,193,7,.06);
    border:1px solid rgba(255,193,7,.18);
    border-radius:14px;
    padding:1.2rem 1.4rem;
    direction:rtl; text-align:right;
    color:#ffe082; line-height:2; font-size:.92rem;
}

/* ── دیوایدر ── */
.gradient-div {
    height:2px;
    background:linear-gradient(90deg,transparent,rgba(0,180,255,.35),transparent);
    margin:2rem 0; border:none;
}

/* ── آپلود ── */
.stFileUploader {border:2px dashed rgba(0,180,255,.25) !important; border-radius:16px !important; padding:1rem !important; background:rgba(0,18,55,.3) !important;}

/* ── اسپینر ── */
.stSpinner > div {color:#6db3f2 !important;}

/* ── اسکرولبار ── */
::-webkit-scrollbar {width:8px;}
::-webkit-scrollbar-track {background:#0a0e27;}
::-webkit-scrollbar-thumb {background:rgba(0,120,255,.35); border-radius:4px;}
::-webkit-scrollbar-thumb:hover {background:rgba(0,120,255,.55);}

/* ── انیمیشن ── */
@keyframes pulse-glow {
    0%,100%{box-shadow:0 0 15px rgba(0,180,255,.08);}
    50%{box-shadow:0 0 35px rgba(0,180,255,.22);}
}
.glow {animation:pulse-glow 4s ease-in-out infinite;}

/* ── متن فارسی ── */
.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
    direction:rtl !important;
    text-align:right !important;
    font-family:'Vazirmatn',Tahoma,sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  📦  بارگذاری مدل
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return YOLO("best_model.pt") if os.path.exists("best_model.pt") else None

model = load_model()
if model is None:
    st.error("❌ فایل best_model.pt یافت نشد.")
    st.stop()

names = model.names
target_teeth = ["18", "28", "38", "48"]
wisdom_indices = [idx for idx, n in names.items() if n in target_teeth]

# ═══════════════════════════════════════════════════════════════
#  🦷  اطلاعات هر دندان
# ═══════════════════════════════════════════════════════════════
TOOTH = {
    "18": {
        "label": "بالا-راست", "region": "upper", "emoji": "↖️",
        "adjacent": "17", "structure": "سینوس فک بالا (Maxillary Sinus)",
        "std": "تاج: ۷–۱۱ مم  |  عرض: ۸–۱۱ مم"
    },
    "28": {
        "label": "بالا-چپ", "region": "upper", "emoji": "↗️",
        "adjacent": "27", "structure": "سینوس فک بالا (Maxillary Sinus)",
        "std": "تاج: ۷–۱۱ مم  |  عرض: ۸–۱۱ مم"
    },
    "38": {
        "label": "پایین-چپ", "region": "lower", "emoji": "↙️",
        "adjacent": "37", "structure": "عصب زیرفکی (Inferior Alveolar Nerve)",
        "std": "تاج: ۸–۱۲ مم  |  عرض: ۹–۱۲ مم"
    },
    "48": {
        "label": "پایین-راست", "region": "lower", "emoji": "↘️",
        "adjacent": "47", "structure": "عصب زیرفکی (Inferior Alveolar Nerve)",
        "std": "تاج: ۸–۱۲ مم  |  عرض: ۹–۱۲ مم"
    },
}

# ═══════════════════════════════════════════════════════════════
#  🔧  توابع کمکی
# ═══════════════════════════════════════════════════════════════
def estimate_size(w_px, h_px, img_w):
    mm_px = 206.25 / max(img_w, 1) * 0.82
    return w_px * mm_px, h_px * mm_px

def classify(w, h):
    r = w / max(h, 1)
    if r > 1.35:
        return "افقی (Horizontal)", "↔️", "high", "ریسک بالا — احتمال فشار بر دندان مجاور و کیست"
    elif r > 1.05:
        return "زاویه‌دار (Angled)", "📐", "med", "ریسک متوسط — نیاز به پایش و بررسی دوره‌ای"
    elif r > 0.75:
        return "نیمه‌عمودی (Semi-Vertical)", "↕️", "low", "ریسک پایین — وضعیت نسبتاً مناسب"
    else:
        return "عمودی (Vertical)", "⬆️", "low", "ریسک پایین — بهترین حالت ممکن"

def patient_text(tn, imp, w_mm, h_mm):
    t = TOOTH[tn]
    region_fa = "فک بالا" if t["region"] == "upper" else "فک پایین"
    if "افقی" in imp:
        body = (
            "دندان عقل شما **افقی** قرار گرفته — یعنی به‌جای اینکه راست بالا بیاید، به پهلو در استخوان گیر کرده. "
            "این حالت شایع است و معمولاً نیاز به جراحی سرپایی دارد. نگران نباشید، روش‌های مدرن امروزی این جراحی را بسیار ساده و کم‌درد کرده‌اند."
        )
        icon = "🔴"
    elif "زاویه‌دار" in imp:
        body = (
            "دندان شما **زاویه‌دار** است — یعنی کمی کج رشد کرده. "
            "اگر درد یا علامتی ندارید، نیازی به عجله نیست. ولی هر ۶ ماه وضعیتش را با دندان‌پزشک بررسی کنید."
        )
        icon = "🟠"
    else:
        body = (
            "دندان شما **تقریباً صاف و عمودی** است — بهترین وضعیت! "
            "در معاینات دوره‌ای بررسی کنید و نگران نباشید."
        )
        icon = "🟢"

    return f"""
<div class="pat-box">
<h3>{t['emoji']} دندان {tn} — {t['label']} ({region_fa})</h3>

<p>📏 <strong>اندازه تخمینی:</strong> {w_mm:.1f} × {h_mm:.1f} میلی‌متر</p>
<p>{icon} <strong>وضعیت:</strong> {imp}</p>
<p>{body}</p>

<p><strong>✅ نشانه‌هایی که باید مراقبشان باشید:</strong></p>
<ul>
<li>درد یا فشار در ناحیه پشت دهان</li>
<li>قرمزی، تورم یا بوی بد لثه</li>
<li>گیر کردن غذا بین دندان‌ها</li>
<li>مشکل در باز کردن دهان</li>
</ul>

<p>💡 <strong>قدم بعدی:</strong> نتیجه را به دندان‌پزشک نشان دهید و در مورد وضعیت {tn} مشورت کنید.</p>
</div>
"""

def doctor_text(tn, imp, w_mm, h_mm):
    t = TOOTH[tn]
    adj = t["adjacent"]
    struct = t["structure"]

    extra = ""
    if t["region"] == "lower":
        extra = """
<li>ارزیابی دقیق رابطه ریشه با <strong>کانال عصب زیرفکی (IAC)</strong> با CBCT</li>
<li>در صورت نزدیکی شدید → بررسی <strong>coronectomy</strong> به‌جای extraction</li>
<li>هشدار به بیمار درباره paresthesia / dysesthesia post-op</li>
"""
    else:
        extra = """
<li>ارزیابی نزدیکی ریشه به <strong>کف سینوس ماگزیلاری</strong></li>
<li>در صورت نزدیکی شدید → بررسی ریسک <strong>oro-antral communication</strong></li>
<li>CBCT برای ارزیابی سه‌بعدی رابطه ریشه-سینوس</li>
"""

    risk_line = ""
    if "افقی" in imp:
        risk_line = "<li>⚠️ <strong>ریسک بالا:</strong> کیست فولیکولار، رزورپشن دندان {adj}، آسیب عصبی → <strong>جراحی ارجح</strong></li>".format(adj=adj)
    elif "زاویه‌دار" in imp:
        risk_line = f"<li>⚠️ تعیین دقیق جهت زاویه (مزیو/دیستو) و نزدیکی به دندان {adj}</li>"

    return f"""
<div class="doc-box">
<h3>👨‍⚕️ گزارش بالینی — دندان {tn} ({t['label']})</h3>

<table>
<tr><th>پارامتر</th><th>مقدار</th></tr>
<tr><td>شماره FDI</td><td>{tn}</td></tr>
<tr><td>موقعیت</td><td>{t['label']}</td></tr>
<tr><td>ابعاد تخمینی</td><td>{w_mm:.1f} × {h_mm:.1f} mm</td></tr>
<tr><td>ابعاد استاندارد</td><td>{t['std']}</td></tr>
<tr><td>نوع نهفتگی (تخمین)</td><td>{imp}</td></tr>
<tr><td>دندان مجاور</td><td>{adj}</td></tr>
<tr><td>ساختار حیاتی نزدیک</td><td>{struct}</td></tr>
</table>

<p><strong>📋 توصیه‌های بالینی:</strong></p>
<ul>
<li>ارزیابی فضای رویش نسبت به قوس دندانی</li>
<li>بررسی وضعیت ریشه‌ها (تعداد، طول، انحنا)</li>
<li>بررسی پیرامون تاج (follicle) و هرگونه radiolucency</li>
<li>بررسی وضعیت دندان {adj} (پوسیدگی / رزورپشن)</li>
{extra}
</ul>

<p><strong>⚠️ هشدارهای ویژه:</strong></p>
<ul>
{risk_line}
<li>در صورت نهفتگی عمیق → جراحی فک و صورت توصیه می‌شود</li>
</ul>
</div>
"""

# ═══════════════════════════════════════════════════════════════
#  🌟  هدر
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>🦷 تشخیص هوشمند دندان عقل</h1>
    <p class="sub">با هوش مصنوعی، موقعیت، اندازه و وضعیت نهفتگی دندان‌های عقل خود را بررسی کنید</p>
    <div class="pill-row">
        <span class="pill">📍 تشخیص موقعیت</span>
        <span class="pill">📏 تخمین اندازه</span>
        <span class="pill">🔍 بررسی نهفتگی</span>
        <span class="pill">💬 توصیه به بیمار</span>
        <span class="pill">👨‍⚕️ گزارش پزشکی</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  📤  آپلود
# ═══════════════════════════════════════════════════════════════
uploaded = st.file_uploader("📎 تصویر OPG خود را آپلود کنید", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        img_arr = np.array(image)
        img_h, img_w = img_arr.shape[:2]

        with st.spinner("⏳ در حال تحلیل تصویر با هوش مصنوعی..."):
            results = model.predict(
                source=img_arr, classes=wisdom_indices,
                conf=0.4, show_conf=False, line_width=3, verbose=False
            )

        result = results[0]
        result_rgb = cv2.cvtColor(result.plot(conf=False), cv2.COLOR_BGR2RGB)
        boxes = result.boxes
        count = len(boxes)

        # ── تصویر نتیجه ──
        st.image(result_rgb, caption="✅ نتیجه تشخیص دندان‌های عقل", use_container_width=True)

        st.markdown('<hr class="gradient-div">', unsafe_allow_html=True)

        if count == 0:
            st.info("ℹ️ هیچ دندان عقلی در این تصویر شناسایی نشد. تصویر دیگری امتحان کنید.")
        else:
            # ── جمع‌بندی آماری ──
            st.markdown("### 📊 خلاصه تشخیص")
            c1, c2, c3 = st.columns(3)
            c1.metric("🦷 تعداد شناسایی‌شده", f"{count} عدد")
            upper_n = sum(1 for b in boxes if names[int(b.cls[0])] in ["18","28"])
            lower_n = sum(1 for b in boxes if names[int(b.cls[0])] in ["38","48"])
            c2.metric("⬆️ فک بالا", f"{upper_n} عدد")
            c3.metric("⬇️ فک پایین", f"{lower_n} عدد")

            if count == 4:
                st.success("🎉 هر چهار دندان عقل شناسایی شد!")
            st.markdown('<hr class="gradient-div">', unsafe_allow_html=True)

            # ── تحلیل هر دندان ──
            for box in boxes:
                cls_id = int(box.cls[0])
                tn = names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bw, bh = x2 - x1, y2 - y1
                w_mm, h_mm = estimate_size(bw, bh, img_w)
                imp, imp_icon, imp_level, imp_desc = classify(bw, bh)
                info = TOOTH.get(tn, {"label":"نامشخص","region":"upper","emoji":"🦷","adjacent":"—","structure":"—","std":"—"})

                # هدر دندان
                st.markdown(f"""
<div class="tooth-head glow">
    <div class="num">{tn}</div>
    <div class="info">
        <h3>دندان عقل {tn} — {info['label']}</h3>
        <p>{info['emoji']}  {'فک بالا' if info['region']=='upper' else 'فک پایین'}  |  {imp_icon} {imp}</p>
    </div>
</div>
""", unsafe_allow_html=True)

                # آمار
                sc = st.columns(4)
                sc[0].metric("📏 عرض", f"{w_mm:.1f} mm")
                sc[1].metric("📐 ارتفاع", f"{h_mm:.1f} mm")
                sc[2].metric("🔍 نهفتگی", f"{imp_icon} {imp.split('(')[0].strip()}")
                sc[3].metric("📊 اطمینان", f"{conf*100:.0f}%")

                # نوار ریسک
                rc = {"high":"r-high","med":"r-med","low":"r-low"}
                st.markdown(f'<div class="{rc[imp_level]}"><strong>{imp_icon} {imp}</strong> — {imp_desc}</div>', unsafe_allow_html=True)

                # تب‌ها
                t1, t2, t3 = st.tabs(["💬 توصیه بیمار", "👨‍⚕️ گزارش پزشک", "📊 جزئیات فنی"])

                with t1:
                    st.markdown(patient_text(tn, imp, w_mm, h_mm), unsafe_allow_html=True)

                with t2:
                    st.markdown(doctor_text(tn, imp, w_mm, h_mm), unsafe_allow_html=True)

                with t3:
                    st.markdown(f"""
<div class="doc-box">
<h3>📊 اطلاعات فنی — دندان {tn}</h3>
<table>
<tr><th>پارامتر</th><th>مقدار</th></tr>
<tr><td>ابعاد پیکسلی</td><td>{bw:.0f} × {bh:.0f} px</td></tr>
<tr><td>عرض تخمینی</td><td>{w_mm:.1f} mm</td></tr>
<tr><td>ارتفاع تخمینی</td><td>{h_mm:.1f} mm</td></tr>
<tr><td>نسبت W/H</td><td>{bw/max(bh,1):.2f}</td></tr>
<tr><td>ابعاد استاندارد</td><td>{info['std']}</td></tr>
<tr><td>نوع نهفتگی</td><td>{imp}</td></tr>
<tr><td>میزان اطمینان مدل</td><td>{conf*100:.1f}%</td></tr>
</table>
<p><strong>🔧 روش تخمین:</strong> مقیاس‌گذاری بر اساس میدان دید ~۲۰۶ مم OPG | نسبت W/H بودینگ‌باکس برای طبقه‌بندی نهفتگی</p>
<p><strong>⚠️ محدودیت:</strong> تخمین تقریبی است و جایگزین CBCT یا معاینه بالینی نیست.</p>
</div>
""", unsafe_allow_html=True)

                st.markdown('<hr class="gradient-div">', unsafe_allow_html=True)

            # ── جمع‌بندی نهایی ──
            tooth_list = ", ".join(names[int(b.cls[0])] for b in boxes)
            st.markdown("### 📋 جمع‌بندی نهایی")

            s1, s2 = st.tabs(["💬 جمع‌بندی بیمار", "👨‍⚕️ جمع‌بندی پزشک"])

            with s1:
                st.markdown(f"""
<div class="sum-hero">
<h3>💚 عزیز، خلاصه نتیجه شما:</h3>
<p style="font-size:1.1rem; line-height:2.3;">
دندان‌های عقل <strong>{tooth_list}</strong> در تصویر شما شناسایی شدند.
</p>
<p style="font-size:1.05rem; line-height:2.3;">
☑️ نتیجه را به دندان‌پزشک نشان دهید<br>
☑️ درباره وضعیت هر دندان مشورت کنید<br>
☑️ اگر درد یا علامتی ندارید، عجله‌ای نیست<br>
☑️ نگران نباشید — قابل مدیریت است 💚
</p>
</div>
""", unsafe_allow_html=True)

            with s2:
                st.markdown(f"""
<div class="sum-hero">
<h3>👨‍⚕️ جمع‌بندی بالینی:</h3>
<p>دندان‌های شناسایی‌شده: <strong>{tooth_list}</strong></p>
<p style="line-height:2.3;">
☑️ بررسی OPG از نظر رابطه ریشه با ساختارهای حیاتی<br>
☑️ ارزیابی فضای رویش و رابطه با دندان مجاور<br>
☑️ CBCT در صورت نزدیکی به عصب / سینوس<br>
☑️ طرح درمان بر اساس معاینه حضوری و اولویت‌بندی<br>
☑️ مستندسازی و پیگیری دوره‌ای
</p>
</div>
""", unsafe_allow_html=True)

            # ── سلب مسئولیت ──
            st.markdown("""
<div class="disclaimer">
⚠️ <strong>توجه مهم:</strong>
این تحلیل توسط هوش مصنوعی انجام شده و جنبه اطلاع‌رسانی دارد.
تشخیص نهایی نهفتگی، طرح درمان و نیاز به جراحی، حتماً باید توسط
<strong>دندان‌پزشک متخصص</strong> و با معاینه حضوری انجام شود.
این سامانه جایگزین مشاوره پزشکی نیست.
</div>
""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ خطا در پردازش: {str(e)}")
else:
    st.markdown("""
<div class="glass" style="text-align:center; padding:3rem 2rem;">
<p style="font-size:4rem; margin:0;">👆</p>
<h3 style="color:#7cc4ff; direction:rtl;">تصویر رادیوگرافی OPG خود را در کادر بالا آپلود کنید</h3>
<p style="color:#6db3f2; direction:rtl;">فرمت‌های مجاز: JPG, JPEG, PNG</p>
</div>

<div class="glass" style="direction:rtl; text-align:right;">
<h3 style="color:#7cc4ff;">📌 راهنمای استفاده</h3>
<ol style="line-height:2.5;">
<li>یک تصویر <strong>OPG (رادیوگرافی پانورامیک)</strong> آپلود کنید</li>
<li>هوش مصنوعی دندان‌های عقل (۱۸، ۲۸، ۳۸، ۴۸) را شناسایی می‌کند</li>
<li>برای هر دندان: <strong>توصیه بیمار</strong>، <strong>گزارش پزشک</strong> و <strong>جزئیات فنی</strong> نمایش داده می‌شود</li>
<li>نتیجه را به دندان‌پزشک خود نشان دهید</li>
</ol>
</div>
""", unsafe_allow_html=True)
