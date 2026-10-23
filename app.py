import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="MACAN | دستیار هوشمند دندان عقل",
    page_icon="🦷",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://v1.fontapi.ir/css/Vazir');

.stApp {
    background: radial-gradient(circle at 50% 0%, #0b1b3f 0%, #030814 55%, #02040c 100%) !important;
    font-family: 'Vazir', Tahoma, sans-serif !important;
    color: #e0f2fe !important;
    direction: rtl;
}
html, body, [class*="css"], p, div, h1, h2, h3, h4, span, label {
    font-family: 'Vazir', Tahoma, sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }

.logo-wrap {
    text-align: center;
    margin: 4px auto 0 auto;
    background: transparent !important;
}
.logo-wrap svg, .logo-wrap img {
    background: transparent !important;
    filter: drop-shadow(0 0 18px rgba(0,240,255,.45));
}
.brand-sub {
    text-align: center;
    color: #7dd3fc;
    font-size: 14px;
    margin: 2px 0 10px 0;
}

.hero-banner {
    max-width: 1100px;
    margin: 0 auto 12px auto;
    text-align: center;
    padding: 16px 18px;
    border: 1px solid rgba(0,240,255,.28);
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(0,240,255,.08), rgba(3,8,26,.18));
    box-shadow: 0 0 28px rgba(0,180,255,.16);
}
.hero-banner h1 {
    color: #00f0ff !important;
    font-size: 28px !important;
    letter-spacing: .6px;
    text-shadow: 0 0 16px rgba(0,240,255,.7);
    margin: 0 0 6px 0 !important;
}
.hero-banner p { color: #9ad8ff !important; margin: 0 !important; font-size: 14px !important; }

.scan-caption { text-align: center; margin: 6px 0 10px 0; }
.scan-caption h3 {
    color: #00f0ff !important;
    font-size: 22px !important;
    margin: 0 0 6px 0 !important;
    text-shadow: 0 0 12px rgba(0,240,255,.5);
}
.scan-caption p { color: #bfe9ff !important; font-size: 14px !important; margin: 0 !important; }

.tooth-stage {
    width: 180px;
    height: 180px;
    margin: 6px auto 8px auto;
    perspective: 900px;
}
.tooth-spin {
    width: 180px;
    height: 180px;
    animation: spinY 7s linear infinite;
    transform-style: preserve-3d;
    filter: drop-shadow(0 0 16px rgba(0,240,255,.55));
}
@keyframes spinY {
    from { transform: rotateY(0deg); }
    to { transform: rotateY(360deg); }
}

[data-testid="stFileUploader"] {
    max-width: 520px !important;
    margin: 0 auto 14px auto !important;
    direction: ltr !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: radial-gradient(circle at 50% 50%, rgba(0,240,255,.08), rgba(4,16,45,.95)) !important;
    border: 2px dashed rgba(0,240,255,.7) !important;
    border-radius: 18px !important;
    min-height: 92px !important;
    padding: 14px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 0 24px rgba(0,240,255,.16), inset 0 0 18px rgba(0,240,255,.06) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }

[data-testid="stFileUploaderDropzone"] button {
    position: relative !important;
    display: block !important;
    margin: 0 auto !important;
    min-width: 210px !important;
    height: 48px !important;
    overflow: hidden !important;
    text-indent: -9999px !important;
    font-size: 0 !important;
    color: transparent !important;
    background: linear-gradient(135deg, #00b4d8, #00f0ff) !important;
    border: none !important;
    border-radius: 14px !important;
    box-shadow: 0 0 22px rgba(0,240,255,.65) !important;
}
[data-testid="stFileUploaderDropzone"] button * {
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stFileUploaderDropzone"] button::after {
    content: "آپلود تصویر OPG";
    position: absolute;
    inset: 0;
    text-indent: 0 !important;
    font-size: 16px !important;
    font-weight: 900 !important;
    font-family: 'Vazir', Tahoma, sans-serif !important;
    color: #03101f !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    visibility: visible !important;
}
[data-testid="stFileUploaderFile"] {
    background: rgba(0,240,255,.12) !important;
    border: 1px solid #00f0ff !important;
    border-radius: 10px !important;
    color: #fff !important;
}

.opg-wrap [data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
}
.opg-wrap img {
    max-height: 320px !important;
    width: auto !important;
    object-fit: contain !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0,240,255,.35) !important;
    box-shadow: 0 0 22px rgba(0,240,255,.18) !important;
}

.hud-card {
    background: rgba(8,24,58,.72);
    border: 1px solid rgba(0,240,255,.32);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,.28);
}
.standby {
    max-width: 980px;
    margin: 10px auto 20px auto;
    text-align: center;
    padding: 28px 18px;
}
.standby h3 { color: #00f0ff !important; margin-bottom: 8px !important; }
.standby p { color: #9fb7d3 !important; max-width: 720px; margin: 0 auto !important; }

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(4,15,38,.85);
    padding: 8px 12px;
    border-radius: 12px;
    border: 1px solid rgba(0,240,255,.2);
}
.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
    font-weight: 800 !important;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    color: #00f0ff !important;
    background: rgba(0,240,255,.14) !important;
    border: 1px solid #00f0ff !important;
}
.badge-urgent { background: rgba(239,68,68,.18); color:#f87171; border:1px solid #ef4444; padding:3px 10px; border-radius:20px; font-size:12px; }
.badge-safe { background: rgba(34,197,94,.18); color:#4ade80; border:1px solid #22c55e; padding:3px 10px; border-radius:20px; font-size:12px; }
.badge-warning { background: rgba(245,158,11,.18); color:#fbbf24; border:1px solid #f59e0b; padding:3px 10px; border-radius:20px; font-size:12px; }
</style>
""", unsafe_allow_html=True)

TOOTH_SVG = """
<svg viewBox="0 0 300 320" width="180" height="180" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="tg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#e7fdff"/>
      <stop offset="100%" stop-color="#67e8f9"/>
    </linearGradient>
    <filter id="tgw"><feGaussianBlur stdDeviation="3.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <ellipse cx="150" cy="292" rx="58" ry="10" fill="#00f0ff" opacity=".25"/>
  <path d="M150 28
           C206 28 238 62 236 108
           C234 138 224 168 214 196
           C206 220 200 252 196 278
           C193 296 186 308 174 308
           C164 308 158 292 150 268
           C142 292 136 308 126 308
           C114 308 107 296 104 278
           C100 252 94 220 86 196
           C76 168 66 138 64 108
           C62 62 94 28 150 28 Z"
        fill="url(#tg)" stroke="#00f0ff" stroke-width="5" filter="url(#tgw)"/>
  <path d="M186 72 A70 70 0 0 1 214 128" fill="none" stroke="#9af7ff" stroke-width="3" opacity=".9"/>
  <path d="M180 84 A56 56 0 0 1 204 128" fill="none" stroke="#9af7ff" stroke-width="2.4" opacity=".7"/>
  <path d="M174 96 A42 42 0 0 1 194 128" fill="none" stroke="#9af7ff" stroke-width="2" opacity=".5"/>
  <line x1="48" y1="148" x2="252" y2="148" stroke="#c4ffff" stroke-width="2">
    <animate attributeName="opacity" values=".25;1;.25" dur="2.2s" repeatCount="indefinite"/>
  </line>
</svg>
"""

MACAN_LOGO = """
<div class="logo-wrap">
<svg viewBox="0 0 420 430" width="210" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#7af6ff"/>
      <stop offset="100%" stop-color="#1d7dff"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3.4" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <path d="M210 36
           C272 36 318 78 314 136
           C311 172 298 208 286 240
           C276 264 270 300 264 338
           C260 364 254 392 238 404
           C226 414 216 398 210 368
           C204 398 194 414 182 404
           C166 392 160 364 156 338
           C150 300 144 264 134 240
           C122 208 109 172 106 136
           C102 78 148 36 210 36 Z"
        fill="none" stroke="url(#g)" stroke-width="9" stroke-linejoin="round" filter="url(#glow)"/>
  <path d="M250 88 A72 72 0 0 1 292 164" fill="none" stroke="#5cf0ff" stroke-width="3" opacity=".95"/>
  <path d="M244 102 A58 58 0 0 1 278 164" fill="none" stroke="#5cf0ff" stroke-width="2.4" opacity=".72"/>
  <path d="M238 116 A44 44 0 0 1 264 164" fill="none" stroke="#5cf0ff" stroke-width="2" opacity=".52"/>
  <path d="M232 130 A32 32 0 0 1 250 164" fill="none" stroke="#5cf0ff" stroke-width="1.6" opacity=".38"/>
  <line x1="64" y1="186" x2="356" y2="186" stroke="#9af7ff" stroke-width="2.4" opacity=".95">
    <animate attributeName="opacity" values=".35;1;.35" dur="2.4s" repeatCount="indefinite"/>
  </line>
  <text x="210" y="422" text-anchor="middle" font-family="Arial, Helvetica, sans-serif"
        font-size="44" font-weight="700" letter-spacing="12" fill="url(#g)" filter="url(#glow)">MACAN</text>
</svg>
</div>
"""


@st.cache_resource
def load_model():
    if not os.path.exists("best_model.pt"):
        return None
    return YOLO("best_model.pt")


def fit_image(img, max_w=860, max_h=320):
    h, w = img.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def analyze_tooth(name, w_mm, h_mm, ratio, conf):
    if name in ["38", "48"]:
        arch = "فک پایین (Mandible)"
        if ratio >= 1.08:
            return {
                "arch": arch,
                "winter": "تخمین: افقی (Horizontal)",
                "difficulty": "بالا",
                "badge": '<span class="badge-urgent">احتمال نهفتگی خوابیده</span>',
                "doc": "کادر تشخیص پهن است؛ احتمال نهفتگی افقی و تماس با مولر دوم. بررسی تحلیل ریشه دندان ۷، فاصله با کانال IAN و در صورت ابهام CBCT. خروج معمولاً با فلپ و سکشن انجام می‌شود.",
                "patient": "این دندان احتمالاً خوابیده رشد کرده و ممکن است به دندان کناری فشار بیاورد. جای نگرانی نیست، ولی حتماً عکس را به دندان‌پزشک نشان بده.",
            }
        if ratio >= 0.85:
            return {
                "arch": arch,
                "winter": "تخمین: مزیوانگولار",
                "difficulty": "متوسط تا بالا",
                "badge": '<span class="badge-warning">احتمال زاویه به جلو</span>',
                "doc": "احتمال mesioangular؛ ریسک پاکت دیستال دندان ۷ و پریکورونیت. فضای رترومولار و طرح جراحی را ارزیابی کنید.",
                "patient": "احتمالاً کج و متمایل به دندان جلویی است. غذا ممکن است بین دو دندان گیر کند. دندان‌پزشک می‌گوید نگه داشتن بهتر است یا کشیدن.",
            }
        return {
            "arch": arch,
            "winter": "تخمین: عمودی (Vertical)",
            "difficulty": "ساده تا متوسط",
            "badge": '<span class="badge-safe">رویش عمودی</span>',
            "doc": "مسیر نزدیک به عمودی. کلاس Pell & Gregory، پوشش لثه و بهداشت ناحیه را بسنجید. اگر فضا کافی باشد ممکن است نگهداری شود.",
            "patient": "جهت دندان نسبتاً صاف به نظر می‌رسد. اگر جا کافی باشد شاید اصلاً نیازی به کشیدن نباشد؛ مسواک زدن ته دهان خیلی مهم است.",
        }

    return {
        "arch": "فک بالا (Maxilla)",
        "winter": "ماگزیلاری / تخمین اولیه",
        "difficulty": "متوسط",
        "badge": '<span class="badge-safe">فک بالا</span>',
        "doc": "نزدیکی ریشه به کف سینوس ماگزیلاری و ریسک شکستگی توبروزیته را در پلان خروج ببینید. خروج فک بالا معمولاً ساده‌تر از فک پایین است.",
        "patient": "دندان عقل فک بالا معمولاً کشیدنش راحت‌تر و نقاهتش کوتاه‌تر است. دندان‌پزشک فاصله‌اش با سینوس را چک می‌کند.",
    }


model = load_model()
if model is None:
    st.error("❌ فایل مدل (best_model.pt) پیدا نشد.")
    st.stop()

names = model.names
wisdom_indices = [idx for idx, n in names.items() if n in ["18", "28", "38", "48"]]

# لوگو بدون پس‌زمینه
l1, l2, l3 = st.columns([1.45, 1, 1.45])
with l2:
    if os.path.exists("logo.png"):
        st.image(Image.open("logo.png").convert("RGBA"), use_container_width=True)
    else:
        st.markdown(MACAN_LOGO, unsafe_allow_html=True)

st.markdown('<div class="brand-sub">دستیار هوشمند تشخیص و تحلیل دندان عقل از روی OPG</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "upload_opg",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(f'<div class="tooth-stage"><div class="tooth-spin">{TOOTH_SVG}</div></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="scan-caption">
        <h3>درگاه ورودی اسکن رادیوگرافی فک (OPG)</h3>
        <p>تصویر پانورامیک را اینجا بارگذاری کنید • JPG / PNG • حداکثر ۲۰۰ مگابایت</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="hud-card standby">
        <h3>⚡ سیستم در حالت آماده‌باش (Standby)</h3>
        <p>با بارگذاری OPG، هوش مصنوعی دندان‌های عقل ۱۸، ۲۸، ۳۸ و ۴۸ را شناسایی می‌کند و گزارش جداگانه برای دندان‌پزشک و بیمار می‌سازد.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_h = img_array.shape[0]
    mm_per_pixel = 152.0 / img_h if img_h else 0.12

    with st.spinner("⚡ MACAN در حال تحلیل OPG..."):
        results = model.predict(
            source=img_array,
            classes=wisdom_indices,
            conf=0.35,
            show_conf=False,
            verbose=False,
        )

    res = results[0]
    annotated = img_array.copy()
    teeth_data = []

    for box in res.boxes:
        name = names[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        w_px, h_px = x2 - x1, y2 - y1
        w_mm = round(w_px * mm_per_pixel, 1)
        h_mm = round(h_px * mm_per_pixel, 1)
        ratio = w_px / h_px if h_px else 1.0

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 240, 255), 2)
        cv2.rectangle(annotated, (x1, y1 - 24), (x1 + 54, y1), (0, 114, 255), -1)
        cv2.putText(annotated, f"#{name}", (x1 + 4, y1 - 7), cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 2)

        info = analyze_tooth(name, w_mm, h_mm, ratio, conf)
        teeth_data.append({
            "name": name,
            "w": w_mm,
            "h": h_mm,
            "conf": conf,
            **info,
        })

    left, mid, right = st.columns([1.15, 1.7, 1.15])
    with mid:
        st.markdown('<div class="opg-wrap">', unsafe_allow_html=True)
        st.image(fit_image(annotated), caption="اسکن OPG")
        st.markdown("</div>", unsafe_allow_html=True)

    if not teeth_data:
        st.info("دندان عقلی با اطمینان کافی پیدا نشد.")
    else:
        tab1, tab2, tab3 = st.tabs(["🌐 اطلس دندان‌ها", "👨‍⚕️ کنسول جراحی", "💬 گزارش بیمار"])

        with tab1:
            st.markdown("### وضعیت دندان‌های عقل در قوس دندانی")
            cols = st.columns(len(teeth_data))
            for i, item in enumerate(sorted(teeth_data, key=lambda x: x["name"])):
                with cols[i]:
                    st.markdown(f"""
                    <div class="hud-card">
                        <div style="font-size:20px;font-weight:800;color:#00f0ff;">🦷 دندان {item['name']}</div>
                        <div style="color:#94a3b8;font-size:13px;margin:6px 0;">{item['arch']}</div>
                        {item['badge']}
                        <hr style="border-color:rgba(0,240,255,.2);">
                        <div style="font-size:14px;line-height:1.95;">
                            عرض تقریبی: <b>{item['w']} mm</b><br>
                            طول تقریبی: <b>{item['h']} mm</b><br>
                            زاویه: <b>{item['winter']}</b><br>
                            اطمینان مدل: <b>{item['conf']*100:.1f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab2:
            for item in sorted(teeth_data, key=lambda x: x["name"]):
                st.markdown(f"""
                <div class="hud-card" style="border-right:5px solid #00f0ff;">
                    <b style="color:#38bdf8;font-size:17px;">دندان #{item['name']}</b>
                    <span style="float:left;color:#cbd5e1;">سختی تخمینی: {item['difficulty']}</span>
                    <div style="margin-top:8px;line-height:1.85;">{item['doc']}</div>
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            for item in sorted(teeth_data, key=lambda x: x["name"]):
                st.markdown(f"""
                <div class="hud-card" style="border-right:5px solid #22c55e;">
                    <div style="color:#4ade80;font-weight:800;margin-bottom:6px;">دندان شماره {item['name']}</div>
                    {item['patient']}
                </div>
                """, unsafe_allow_html=True)
            st.caption("گزارش MACAN تخمین اولیه است و جایگزین معاینه دندان‌پزشک نیست. زاویه و اندازه بدون کالیبراسیون رادیوگرافی محاسبه شده‌اند.")
