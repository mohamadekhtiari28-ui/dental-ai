import os
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
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
    background: radial-gradient(circle at 50% 0%, #0a1738 0%, #03081a 100%) !important;
    font-family: 'Vazir', Tahoma, sans-serif !important;
    color: #e0f2fe !important;
    direction: rtl;
}
html, body, [class*="css"], p, div, h1, h2, h3, h4, span {
    font-family: 'Vazir', Tahoma, sans-serif !important;
}
#MainMenu, footer, header {visibility: hidden;}

.brand-wrap { text-align: center; margin: 6px 0 4px 0; }
.brand-name {
    color: #00f0ff;
    font-size: 34px;
    font-weight: 900;
    letter-spacing: 10px;
    text-shadow: 0 0 18px rgba(0,240,255,.75);
    margin: 0;
}
.brand-sub { color: #7dd3fc; font-size: 14px; margin-top: 4px; }

.scan-caption { text-align: center; margin: 4px 0 10px 0; }
.scan-caption h3 { color: #00f0ff !important; font-size: 20px !important; margin: 0 0 4px 0; }
.scan-caption p { color: #7dd3fc !important; font-size: 13px !important; margin: 0; }

[data-testid="stFileUploader"] {
    max-width: 640px !important;
    margin: 0 auto 14px auto !important;
    direction: ltr !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: radial-gradient(circle at 50% 50%, rgba(0,240,255,.08) 0%, rgba(4,16,45,.95) 100%) !important;
    border: 2px dashed rgba(0,240,255,.65) !important;
    border-radius: 18px !important;
    min-height: 110px !important;
    padding: 16px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
[data-testid="stFileUploaderDropzone"] button {
    position: relative !important;
    display: block !important;
    margin: 0 auto !important;
    min-width: 220px !important;
    height: 48px !important;
    background: linear-gradient(135deg, #00b4d8 0%, #00f0ff 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    box-shadow: 0 0 22px rgba(0,240,255,.65) !important;
    color: transparent !important;
    font-size: 0 !important;
}
[data-testid="stFileUploaderDropzone"] button * {
    display: none !important;
    visibility: hidden !important;
}
[data-testid="stFileUploaderDropzone"] button::after {
    content: "انتخاب تصویر OPG";
    display: block !important;
    visibility: visible !important;
    color: #03101f !important;
    font-family: 'Vazir', Tahoma, sans-serif !important;
    font-size: 17px !important;
    font-weight: 900 !important;
    line-height: 48px !important;
}
[data-testid="stFileUploaderFile"] {
    background: rgba(0,240,255,.15) !important;
    border: 1px solid #00f0ff !important;
    border-radius: 10px !important;
    color: #fff !important;
}

.hud-card {
    background: rgba(8,24,58,.65);
    border: 1px solid rgba(0,240,255,.3);
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: rgba(4,15,38,.8);
    padding: 8px 12px;
    border-radius: 12px;
}
.stTabs [aria-selected="true"] {
    color: #00f0ff !important;
    background: rgba(0,240,255,.15) !important;
    border: 1px solid #00f0ff !important;
}
.badge-urgent { background: rgba(239,68,68,.2); color:#f87171; border:1px solid #ef4444; padding:4px 10px; border-radius:20px; font-size:13px; }
.badge-safe { background: rgba(34,197,94,.2); color:#4ade80; border:1px solid #22c55e; padding:4px 10px; border-radius:20px; font-size:13px; }
.badge-warning { background: rgba(245,158,11,.2); color:#fbbf24; border:1px solid #f59e0b; padding:4px 10px; border-radius:20px; font-size:13px; }
</style>
""", unsafe_allow_html=True)

MACAN_SVG = """
<div style="text-align:center;filter:drop-shadow(0 0 16px rgba(0,240,255,.55));">
<svg viewBox="0 0 420 430" width="190">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#7af6ff"/>
      <stop offset="100%" stop-color="#1d7dff"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3.5" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <path d="M210 42
           C268 42 312 82 308 138
           C305 172 294 206 282 238
           C274 258 268 292 262 328
           C258 354 254 382 238 396
           C226 406 216 392 210 362
           C204 392 194 406 182 396
           C166 382 162 354 158 328
           C152 292 146 258 138 238
           C126 206 115 172 112 138
           C108 82 152 42 210 42 Z"
        fill="none" stroke="url(#g)" stroke-width="9"
        stroke-linejoin="round" filter="url(#glow)"/>
  <path d="M248 95 A70 70 0 0 1 286 168" fill="none" stroke="#5cf0ff" stroke-width="3" opacity=".9"/>
  <path d="M242 108 A58 58 0 0 1 274 168" fill="none" stroke="#5cf0ff" stroke-width="2.4" opacity=".7"/>
  <path d="M236 120 A46 46 0 0 1 262 168" fill="none" stroke="#5cf0ff" stroke-width="2" opacity=".55"/>
  <path d="M230 132 A34 34 0 0 1 250 168" fill="none" stroke="#5cf0ff" stroke-width="1.6" opacity=".4"/>
  <line x1="70" y1="188" x2="350" y2="188" stroke="#9af7ff" stroke-width="2.2" opacity=".95">
    <animate attributeName="opacity" values=".35;1;.35" dur="2.4s" repeatCount="indefinite"/>
  </line>
  <text x="210" y="418" text-anchor="middle"
        font-family="Arial, Helvetica, sans-serif"
        font-size="46" font-weight="700" letter-spacing="10"
        fill="url(#g)" filter="url(#glow)">MACAN</text>
</svg>
</div>
"""


@st.cache_resource
def load_model():
    if not os.path.exists("best_model.pt"):
        return None
    return YOLO("best_model.pt")


def fit_image(img, max_w=860, max_h=340):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


model = load_model()
if model is None:
    st.error("❌ فایل مدل (best_model.pt) پیدا نشد.")
    st.stop()

names = model.names
wisdom_indices = [idx for idx, name in names.items() if name in ["18", "28", "38", "48"]]

# لوگو
l1, l2, l3 = st.columns([1.35, 1, 1.35])
with l2:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.markdown(MACAN_SVG, unsafe_allow_html=True)

st.markdown("""
<div class="brand-wrap">
    <div class="brand-sub">دستیار هوشمند تشخیص و تحلیل دندان عقل از روی OPG</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("upload_opg", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is None:
    components.html("""
    <!DOCTYPE html><html><head><meta charset="utf-8"/>
    <style>html,body{margin:0;overflow:hidden;background:transparent}canvas{display:block;margin:0 auto}</style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script></head>
    <body><script>
    const H=190, scene=new THREE.Scene();
    const camera=new THREE.PerspectiveCamera(40, window.innerWidth/H, 0.1, 100);
    camera.position.set(0,0.15,7.3);
    const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
    renderer.setSize(window.innerWidth,H); renderer.setClearColor(0x000000,0);
    document.body.appendChild(renderer.domElement);
    const mat=new THREE.MeshPhongMaterial({color:0xeefbff,shininess:130,specular:0x9befff,emissive:0x0b3d4d});
    const tooth=new THREE.Group();
    const crown=new THREE.Mesh(new THREE.SphereGeometry(1.05,48,32),mat);
    crown.scale.set(1.22,0.8,1.02); crown.position.y=0.72; tooth.add(crown);
    function addRoot(x,z,tx,tz,len){const m=new THREE.Mesh(new THREE.CylinderGeometry(0.21,0.07,len,16),mat); m.position.set(x,-0.52,z); m.rotation.x=tx; m.rotation.z=tz; tooth.add(m);}
    addRoot(-0.38,0.12,0.22,0.18,2.15); addRoot(0.40,0.10,0.18,-0.2,2.05); addRoot(0.02,-0.28,0.34,0,1.65);
    scene.add(tooth);
    const ring=new THREE.Mesh(new THREE.TorusGeometry(2.15,0.018,8,90), new THREE.MeshBasicMaterial({color:0x00f0ff,transparent:true,opacity:0.5}));
    ring.rotation.x=Math.PI/2; ring.position.y=-1.75; scene.add(ring);
    scene.add(new THREE.AmbientLight(0x6eb6d0,0.75));
    const l1=new THREE.PointLight(0x00f0ff,2.3,30); l1.position.set(4,5,6); scene.add(l1);
    function tick(t){requestAnimationFrame(tick); tooth.rotation.y+=0.013; tooth.position.y=Math.sin(t*0.0015)*0.12; ring.rotation.z+=0.01; renderer.render(scene,camera);}
    tick(0);
    </script></body></html>
    """, height=190, scrolling=False)

    st.markdown("""
    <div class="scan-caption">
        <h3>درگاه ورودی اسکن رادیوگرافی فک (OPG)</h3>
        <p>تصویر پانورامیک را انتخاب کنید • JPG / PNG</p>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_h = img_array.shape[0]
    mm_per_pixel = 152.0 / img_h if img_h > 0 else 0.12

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
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        w_px, h_px = x2 - x1, y2 - y1
        w_mm = round(w_px * mm_per_pixel, 1)
        h_mm = round(h_px * mm_per_pixel, 1)
        ratio = w_px / h_px if h_px else 1.0

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 240, 255), 2)
        cv2.rectangle(annotated, (x1, y1 - 26), (x1 + 58, y1), (0, 114, 255), -1)
        cv2.putText(annotated, f"#{name}", (x1 + 4, y1 - 7), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

        if name in ["38", "48"]:
            arch = "فک پایین"
            if ratio >= 1.08:
                winter, badge, difficulty = "تخمین: افقی", '<span class="badge-urgent">احتمال خوابیده</span>', "بالا"
                doc = "احتمال نهفتگی افقی و فشار به مولر دوم. بررسی تحلیل ریشه دندان ۷ و فاصله با IAN؛ در ابهام CBCT."
                patient = "این دندان احتمالاً خوابیده است و ممکن است به دندان کناری فشار بیاورد. تصمیم نهایی فقط با دندان‌پزشک است."
            elif ratio >= 0.85:
                winter, badge, difficulty = "تخمین: مزیوانگولار", '<span class="badge-warning">احتمال کج به جلو</span>', "متوسط"
                doc = "احتمال mesioangular؛ ریسک پاکت دیستال دندان ۷ و پریکورونیت."
                patient = "احتمالاً کج و متمایل به دندان جلویی است. فضای بین دو دندان غذا گیر می‌اندازد."
            else:
                winter, badge, difficulty = "تخمین: عمودی", '<span class="badge-safe">نسبتاً صاف</span>', "ساده تا متوسط"
                doc = "مسیر نزدیک به عمودی. ارزیابی فضای قوس و پوشش لثه."
                patient = "جهت دندان نسبتاً صاف به نظر می‌رسد. اگر جا کافی باشد شاید کشیدن لازم نباشد."
        else:
            arch = "فک بالا"
            winter, badge, difficulty = "ماگزیلاری (تخمین)", '<span class="badge-safe">فک بالا</span>', "متوسط"
            doc = "نزدیکی ریشه به کف سینوس و ریسک توبروزیته را در پلان خروج ببینید."
            patient = "دندان عقل فک بالا معمولاً خروج راحت‌تری دارد. فاصله با سینوس را دندان‌پزشک چک می‌کند."

        teeth_data.append({
            "name": name, "arch": arch, "w": w_mm, "h": h_mm,
            "winter": winter, "difficulty": difficulty, "badge": badge,
            "doc": doc, "patient": patient, "conf": conf_score,
        })

    left, mid, right = st.columns([1.1, 1.8, 1.1])
    with mid:
        st.image(fit_image(annotated), caption="اسکن OPG")

    if not teeth_data:
        st.info("دندان عقلی با اطمینان کافی پیدا نشد.")
    else:
        tab1, tab2, tab3 = st.tabs(["🌐 اطلس دندان‌ها", "👨‍⚕️ کنسول جراحی", "💬 گزارش بیمار"])

        with tab1:
            cols = st.columns(len(teeth_data))
            for i, item in enumerate(sorted(teeth_data, key=lambda x: x["name"])):
                with cols[i]:
                    st.markdown(f"""
                    <div class="hud-card">
                        <div style="font-size:20px;font-weight:bold;color:#00f0ff;">🦷 دندان {item['name']}</div>
                        <div style="color:#94a3b8;font-size:13px;margin:6px 0;">{item['arch']}</div>
                        {item['badge']}
                        <hr style="border-color:rgba(0,240,255,0.2);">
                        <div style="font-size:14px;line-height:1.9;">
                            عرض تقریبی: <b>{item['w']} mm</b><br>
                            طول تقریبی: <b>{item['h']} mm</b><br>
                            زاویه: <b>{item['winter']}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab2:
            for item in sorted(teeth_data, key=lambda x: x["name"]):
                st.markdown(f"""
                <div class="hud-card" style="border-right:5px solid #00f0ff;">
                    <b style="color:#38bdf8;font-size:17px;">دندان #{item['name']}</b>
                    <span style="float:left;color:#cbd5e1;">سختی تخمینی: {item['difficulty']}</span>
                    <div style="margin-top:8px;line-height:1.8;">{item['doc']}</div>
                </div>
                """, unsafe_allow_html=True)

        with tab3:
            for item in sorted(teeth_data, key=lambda x: x["name"]):
                st.markdown(f"""
                <div class="hud-card" style="border-right:5px solid #22c55e;">
                    <div style="color:#4ade80;font-weight:bold;margin-bottom:6px;">دندان شماره {item['name']}</div>
                    {item['patient']}
                </div>
                """, unsafe_allow_html=True)
            st.caption("این گزارش اولیه MACAN است و جایگزین معاینه دندان‌پزشک نیست.")
