import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# ─── تنظیمات صفحه با چیدمان باز ───
st.set_page_config(
    page_title="AI Dental Studio | تحلیل تخصصی OPG", 
    page_icon="⚡", 
    layout="wide"
)

# ─── استایل نئونی و تم گرافیکی سایبرنتیک با بازطراحی کادر آپلود ───
st.markdown("""
    <style>
    @import url('https://v1.fontapi.ir/css/Vazir');
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0a1738 0%, #03081a 100%) !important;
        font-family: 'Vazir', Tahoma, sans-serif !important;
        color: #e0f2fe !important;
        direction: rtl;
        text-align: right;
    }
    
    html, body, [class*="css"], p, div, h1, h2, h3, h4, span {
        font-family: 'Vazir', Tahoma, sans-serif !important;
        direction: rtl;
    }

    /* بنر اصلی و عنوان درخشان */
    .glow-header {
        text-align: center;
        padding: 25px 10px;
        background: linear-gradient(180deg, rgba(0, 240, 255, 0.08) 0%, rgba(3, 8, 26, 0.2) 100%);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 16px;
        box-shadow: 0 0 35px rgba(0, 150, 255, 0.2);
        margin-bottom: 25px;
    }
    .glow-title {
        color: #00f0ff;
        font-size: 32px;
        font-weight: 900;
        letter-spacing: 1px;
        text-shadow: 0 0 15px rgba(0, 240, 255, 0.7);
        margin-bottom: 6px;
    }
    .glow-subtitle {
        color: #7dd3fc;
        font-size: 15px;
    }

    /* ─── کادر آپلود نئونی وسط‌چین ─── */
    [data-testid="stFileUploader"] {
        max-width: 860px;
        margin: 0 auto 40px auto;
        direction: rtl !important;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: radial-gradient(circle at 50% 50%, rgba(0, 240, 255, 0.09) 0%, rgba(4, 16, 45, 0.98) 100%) !important;
        border: 2px dashed rgba(0, 234, 255, 0.65) !important;
        border-radius: 20px !important;
        padding: 40px 20px 38px 20px !important;
        min-height: 190px !important;
        box-shadow: 0 0 40px rgba(0, 234, 255, 0.20), inset 0 0 25px rgba(0, 234, 255, 0.10) !important;
        transition: all 0.3s ease-in-out !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        gap: 14px !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #00eaff !important;
        box-shadow: 0 0 55px rgba(0, 234, 255, 0.50), inset 0 0 40px rgba(0, 234, 255, 0.15) !important;
        transform: scale(1.01);
    }

    /* محتوای داخلی دراپ‌زون */
    [data-testid="stFileUploaderDropzone"] > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        width: 100% !important;
        gap: 10px !important;
    }

    /* راهنمای متن داخل کادر آپلود */
    [data-testid="stFileUploaderDropzoneInstructions"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        margin: 0 auto !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
    }

    /* متن اصلی (Drag and drop file here) */
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        font-size: 17px !important;
        font-weight: 800 !important;
        color: #f1f9ff !important;
        text-align: center !important;
        line-height: 2.0 !important;
        letter-spacing: 0.2px !important;
        text-shadow: 0 0 12px rgba(0, 234, 255, 0.35) !important;
    }

    /* متن فرعی (Limit 200MB per file • JPG, PNG) */
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        font-size: 14.5px !important;
        font-weight: 600 !important;
        color: #7dd3fc !important;
        text-align: center !important;
        margin-top: 6px !important;
        opacity: 0.95 !important;
    }

    /* دکمه Browse files */
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #0399d6 0%, #00eaff 100%) !important;
        color: #03081a !important;
        font-weight: 900 !important;
        font-size: 15px !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 9px 32px !important;
        margin-top: 6px !important;
        box-shadow: 0 0 24px rgba(0, 234, 255, 0.65) !important;
        transition: all 0.25s ease !important;
        letter-spacing: 0.3px !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        background: linear-gradient(135deg, #04b5f5 0%, #33f3ff 100%) !important;
        box-shadow: 0 0 40px rgba(0, 234, 255, 0.95) !important;
        transform: scale(1.05);
    }

    [data-testid="stFileUploaderDropzone"] button:active {
        transform: scale(1.02);
    }

    /* فایل انتخاب‌شده */
    [data-testid="stFileUploaderFile"] {
        background: rgba(0, 234, 255, 0.18) !important;
        border: 1px solid #00eaff !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        padding: 7px 16px !important;
        margin-top: 10px !important;
        box-shadow: 0 0 18px rgba(0, 234, 255, 0.35) !important;
    }

    /* کارت‌های شیشه‌ای مدرن */
    .hud-card {
        background: rgba(8, 24, 58, 0.65);
        border: 1px solid rgba(0, 240, 255, 0.3);
        border-radius: 14px;
        padding: 18px 22px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 18px;
        transition: all 0.3s ease;
    }
    .hud-card:hover {
        border-color: #00f0ff;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.35);
    }

    /* تب‌های اختصاصی */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: rgba(4, 15, 38, 0.8);
        padding: 8px 14px;
        border-radius: 12px;
        border: 1px solid rgba(0, 240, 255, 0.2);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 17px !important;
        font-weight: bold;
        color: #94a3b8 !important;
        border-radius: 8px;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        color: #00f0ff !important;
        background: rgba(0, 240, 255, 0.15) !important;
        border: 1px solid #00f0ff !important;
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
    }

    /* برچسب‌های وضعیت */
    .badge-urgent {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 13px;
    }
    .badge-safe {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 13px;
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

# ─── بارگذاری مدل هوش مصنوعی ───
@st.cache_resource
def load_model():
    if not os.path.exists("best_model.pt"):
        return None
    return YOLO("best_model.pt")

model = load_model()

if model is None:
    st.error("❌ فایل مدل هوش مصنوعی (best_model.pt) در مخزن یافت نشد.")
    st.stop()

names = model.names
target_teeth = ["18", "28", "38", "48"]
wisdom_indices = [idx for idx, name in names.items() if name in target_teeth]

# ─── بنر بالایی هدر ───
st.markdown("""
<div class="glow-header">
    <div class="glow-title">HUMAN TEETH ANATOMY & AI DIAGNOSTIC STUDIO</div>
    <div class="glow-subtitle">سامانه فوق‌پیشرفته تحلیل رادیوگرافی پانورامیک (OPG) و ره‌گیری دندان‌های عقل</div>
</div>
""", unsafe_allow_html=True)

# ─── آیکون و عنوان بخش آپلود ───
st.markdown("""
<div style="text-align: center; margin-bottom: 10px;">
    <div style="font-size: 46px; line-height: 1; filter: drop-shadow(0 0 18px rgba(0, 234, 255, 0.85));">🩻</div>
    <div style="font-size: 21px; font-weight: 900; color: #00eaff; text-shadow: 0 0 14px rgba(0, 234, 255, 0.65); margin-top: 8px;">
        درگاه ورودی اسکن رادیوگرافی فک (OPG)
    </div>
    <div style="font-size: 14.5px; font-weight: 600; color: #7dd3fc; margin-top: 5px; line-height: 1.9;">
        تصویر رادیوگرافی پانورامیک خود را اینجا بارگذاری کنید
    </div>
</div>
""", unsafe_allow_html=True)

# ─── کادر آپلود فایل ───
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]
    
    mm_per_pixel = 152.0 / img_h if img_h > 0 else 0.12

    with st.spinner("⚡ در حال اسکن لایه‌های رادیولوژی و پردازش آناتومیک فک..."):
        results = model.predict(
            source=img_array, 
            classes=wisdom_indices, 
            conf=0.35, 
            show_conf=False, 
            line_width=3,
            verbose=False
        )

    res = results[0]
    
    annotated = img_array.copy()
    detected_boxes = res.boxes
    teeth_data = []

    for box in detected_boxes:
        cls_id = int(box.cls[0])
        name = names[cls_id]
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
        w_px = x2 - x1
        h_px = y2 - y1
        w_mm = round(w_px * mm_per_pixel, 1)
        h_mm = round(h_px * mm_per_pixel, 1)
        aspect_ratio = w_px / h_px if h_px > 0 else 1.0
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 240, 255), 2)
        cv2.rectangle(annotated, (x1-2, y1-2), (x2+2, y2+2), (0, 114, 255), 1)
        corner_len = 12
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 3)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 3)
        cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 3)
        cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 3)
        
        cv2.rectangle(annotated, (x1, y1 - 28), (x1 + 65, y1), (0, 114, 255), -1)
        cv2.putText(annotated, f"#{name}", (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

        if name in ["38", "48"]:
            arch_type = "فک پایین (Mandible)"
            if aspect_ratio >= 1.08:
                winter = "کاملاً افقی (Horizontal)"
                difficulty = "بالا (جراحی تخصصی با سکشن)"
                badge = '<span class="badge-urgent">ریسک بالا - نیازمند خارج‌سازی</span>'
                doc_analysis = "نهفتگی افقی با انطباق بالا بر دندان مولر دوم (#7)؛ احتمال بالای تحلیل ریشه (Root Resorption) و گیر پاتولوژیک؛ بررسی دقیق نزدیکی به کانال مندیبولار (IAN) با CBCT الزامی است؛ ضرورت انجام همی‌سکشن و استئوتومی کورتیکال."
                patient_analysis = "این دندان کاملاً به صورت افقی (خوابیده) رشد کرده و سرش به ریشه دندان مجاور تکیه داده است. با گذشت زمان می‌تواند به دندان کناری آسیب بزند یا سبب دردهای نامشخص فک شود. بهتر است در اولین فرصت با یک جراحی کوتاه و دقیق خارج شود."
            elif aspect_ratio >= 0.85:
                winter = "زاویه‌دار رو به جلو (Mesioangular)"
                difficulty = "متوسط رو به بالا (جراحی استخوانی)"
                badge = '<span class="badge-warning">زاویه‌دار به جلو - نیازمند پایش</span>'
                doc_analysis = "دندان مزیوانگولار است؛ تشکیل پاکت پریودنتال عمیق در مزیال دندان عقل و دیستال دندان ۷ محتمل است؛ پریکورونیت عودکننده در بافت نرم رترومولار پیش‌بینی می‌شود."
                patient_analysis = "دندانت کج درآمده و مثل این می‌ماند که به دندان سالمت تکیه زده باشد! فضای بین این دو دندان مستعد گیر کردن غذاست و تمیز کردنش سخته. برای پیشگیری از خرابی دندان کناری، خارج کردنش منطقی‌ترین راه حل است."
            else:
                winter = "عمودی (Vertical)"
                difficulty = "ساده تا متوسط"
                badge = '<span class="badge-safe">رویش عمودی</span>'
                doc_analysis = "مسیر رویش نرمال عمودی؛ در صورت کفایت طول قوس مندیبل، پتانسیل رویش کامل دارد؛ ارزیابی فضا بین دندان و راموس مندیبل (Pell & Gregory Class I/II)."
                patient_analysis = "جهت دندانت کاملاً صاف و سربالاست! اگر جای کافی در فکت داشته باشد ممکن است اصلاً نیازی به کشیدن نداشته باشد. فقط حواست باشد ته دهان است و باید موقع مسواک زدن خوب به آن برسی."
        else:
            arch_type = "فک بالا (Maxilla)"
            winter = "ماگزیلاری عمودی / مایل"
            difficulty = "متوسط (وابسته به مجاورت سینوس)"
            badge = '<span class="badge-safe">فک بالا</span>'
            doc_analysis = "ارزیابی فاصله ریشه‌ها با کف سینوس ماگزیلاری (Maxillary Sinus)؛ دقت در فشار لوکساتور برای جلوگیری از جابجایی ریشه (Displacement) به فضای سینوسی؛ مراقبت از شکستگی استخوان توبروزیته."
            patient_analysis = "این دندان در فک بالایت است. خبر خوش اینه که دندان‌های عقل فک بالا معمولاً استخوان نرم‌تری در اطرافشان دارند و کشیدنشان خیلی سریع‌تر و با کمترین ناراحتی بعد از عمل انجام می‌شود."

        teeth_data.append({
            "name": name,
            "arch": arch_type,
            "w": w_mm,
            "h": h_mm,
            "winter": winter,
            "difficulty": difficulty,
            "badge": badge,
            "doc": doc_analysis,
            "patient": patient_analysis,
            "conf": conf_score
        })

    st.image(annotated, caption="✨ اسکن هوشمند OPG با مرزبندی تشخیصی HUD", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_overview, tab_doctor, tab_patient = st.tabs([
        "🌐 اطلس دندان‌ها (Teeth HUD Map)",
        "👨‍⚕️ کنسول جراحی (Clinical Console)",
        "💬 وضعیت برای بیمار (Patient Hub)"
    ])

    with tab_overview:
        st.markdown("### 📊 وضعیت دندان‌های عقل در قوس دندانی (Dental Arch):")
        cols = st.columns(len(teeth_data) if len(teeth_data) > 0 else 1)
        for i, item in enumerate(sorted(teeth_data, key=lambda x: x["name"])):
            with cols[i]:
                st.markdown(f"""
                <div class="hud-card">
                    <div style="font-size: 24px; font-weight: bold; color: #00f0ff;">🦷 دندان شماره {item['name']}</div>
                    <div style="color: #94a3b8; font-size: 13px; margin-bottom: 8px;">{item['arch']}</div>
                    {item['badge']}
                    <hr style="border-color: rgba(0,240,255,0.2);">
                    <div style="font-size: 14px; line-height: 2;">
                        📐 عرض تقریبی: <b>{item['w']} mm</b><br>
                        📏 طول تخمینی: <b>{item['h']} mm</b><br>
                        🧭 زاویه وینتر: <b>{item['winter']}</b><br>
                        🎯 اطمینان AI: <b>{item['conf']*100:.1f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_doctor:
        st.markdown("### 📋 ارزیابی رادیولوژیک و پلان درمانی پیشنهادی (مختص جراح / دندان‌پزشک):")
        for item in sorted(teeth_data, key=lambda x: x["name"]):
            st.markdown(f"""
            <div class="hud-card" style="border-right: 5px solid #00f0ff;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 20px; font-weight: bold; color: #38bdf8;">گزارش بالینی دندان عقل #{item['name']} ({item['arch']})</span>
                    <span style="font-size: 14px; color: #cbd5e1;">درجه سختی خروج: <b>{item['difficulty']}</b></span>
                </div>
                <div style="margin-top: 12px; font-size: 14.5px; line-height: 1.8; color: #e2e8f0;">
                    <b>🔹 طبقه‌بندی زاویه (Winter's Classification):</b> {item['winter']}<br>
                    <b>🔹 ملاحظات بیومکانیک و رادیوگرافیک:</b> {item['doc']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_patient:
        st.markdown("### 💡 اوضاع دندان‌های عقل شما چطوره؟ (به زبان خیلی ساده و خودمانی):")
        for item in sorted(teeth_data, key=lambda x: x["name"]):
            st.markdown(f"""
            <div class="hud-card" style="border-right: 5px solid #22c55e;">
                <div style="font-size: 19px; font-weight: bold; color: #4ade80; margin-bottom: 8px;">
                    🌱 وضعیت دندان شماره {item['name']} شما:
                </div>
                <div style="font-size: 15px; line-height: 1.9; color: #f8fafc;">
                    {item['patient']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-top: 20px; padding: 15px; background: rgba(0, 240, 255, 0.05); border-radius: 12px; border: 1px dashed rgba(0, 240, 255, 0.3);">
            💙 <b>یادآوری دوستانه:</b> این گزارش کامپیوتر هوشمنده و اصلاً جای نگرانی نداره! در اولین فرصت رادیوگرافی‌ات رو به دندان‌پزشکت نشون بده تا بر اساس وضعیت لثه و دهانت بهترین راهکار رو بهت پیشنهاد بده.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="hud-card" style="text-align: center; padding: 40px 20px;">
        <h3 style="color: #00f0ff;">⚡ سیستم در حالت آماده‌باش (Standby)</h3>
        <p style="color: #94a3b8; max-width: 650px; margin: 15px auto; line-height: 1.9;">
            با بارگذاری تصویر رادیوگرافی پانورامیک (OPG) در درگاه بالا، هوش مصنوعی به‌سرعت دندان‌های عقل (۱۸، ۲۸، ۳۸، ۴۸) را شناسایی، تحلیل و گزارش تخصصی دوگانه را ارائه می‌دهد.
        </p>
    </div>
    """, unsafe_allow_html=True)
