import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# ─── تنظیمات صفحه ───
st.set_page_config(
    page_title="دستیار هوشمند تشخیص دندان عقل",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── راست‌چین‌سازی کامل و فونت فارسی ───
st.markdown("""
    <style>
    @import url('https://v1.fontapi.ir/css/Vazir');

    html, body, [class*="css"], .stMarkdown, p, div, h1, h2, h3, h4, h5, span, label {
        font-family: 'Vazir', Tahoma, sans-serif !important;
        direction: rtl;
        text-align: right;
    }

    .stApp {
        background-color: #061826;
    }

    /* هدر بالا */
    .top-header {
        background: linear-gradient(90deg, rgba(6, 24, 38, 0.95) 0%, rgba(9, 40, 66, 0.95) 100%);
        border: 1px solid rgba(0, 191, 255, 0.25);
        border-radius: 14px;
        padding: 28px 20px 24px 20px;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0px 10px 40px -20px rgba(0, 191, 255, 0.45);
        backdrop-filter: blur(4px);
    }

    .main-title {
        color: #3dd9ff;
        font-size: clamp(20px, 2.2vw, 38px);
        font-weight: 700;
        margin: 0;
        line-height: 1.4;
        letter-spacing: -0.2px;
    }

    .sub-title {
        color: #d7e6f3;
        font-size: clamp(11px, 1.1vw, 15px);
        margin-top: 14px;
        line-height: 2;
        opacity: 0.95;
    }

    /* نوار وضعیت */
    .status-bar {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        background-color: rgba(8, 30, 50, 0.92);
        border: 1px solid rgba(0, 191, 255, 0.18);
        border-radius: 12px;
        padding: 8px 18px;
        margin-bottom: 28px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }

    .status-bar-right {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: flex-end;
        gap: 10px 14px;
        color: #b9d3e6;
        font-size: clamp(10px, 0.95vw, 13px);
    }

    .status-item {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        background-color: rgba(61, 217, 255, 0.08);
        border: 1px solid rgba(61, 217, 255, 0.25);
        white-space: nowrap;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #3dd9ff;
        box-shadow: 0 0 0 3px rgba(61, 217, 255, 0.15);
    }

    .status-bar-left {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        flex-wrap: wrap;
        gap: 8px;
    }

    .hud-badge {
        color: #3dd9ff;
        font-size: clamp(10px, 0.95vw, 13px);
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 10px;
        border: 1.5px solid #3dd9ff;
        background: radial-gradient(120% 120% at 50% 0%, rgba(61,217,255,0.16) 0%, rgba(6,24,38,0.9) 100%);
        box-shadow: 0 0 0 1px rgba(61,217,255,0.15) inset, 0 8px 40px -24px #3dd9ff;
        white-space: nowrap;
        letter-spacing: 0.2px;
    }

    /* عنوان بخش نتایج */
    .section-title {
        color: #ffffff;
        font-size: clamp(16px, 1.6vw, 24px);
        font-weight: 700;
        text-align: right;
        margin: 10px 4px 24px 4px;
        line-height: 1.6;
    }

    /* کارت دندان */
    .tooth-card {
        background: radial-gradient(140% 140% at 50% 0%, rgba(15, 52, 86, 0.55) 0%, rgba(7, 24, 40, 0.98) 100%);
        border: 1.5px solid rgba(0, 191, 255, 0.35);
        border-radius: 16px;
        padding: clamp(16px, 1.8vw, 22px);
        height: 100%;
        box-shadow: 0px 18px 60px -30px rgba(0, 191, 255, 0.6);
        transition: all 0.25s ease-in-out;
        position: relative;
    }

    .tooth-card:hover {
        transform: translateY(-2px);
        border-color: rgba(61, 217, 255, 0.55);
        box-shadow: 0px 20px 70px -28px rgba(61, 217, 255, 0.75);
    }

    .tooth-header {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: flex-start;
        gap: 10px;
        margin-bottom: clamp(14px, 1.6vw, 20px);
    }

    .tooth-number {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #ffffff;
        font-size: clamp(20px, 2vw, 30px);
        font-weight: 700;
        line-height: 1.1;
    }

    .tooth-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: clamp(34px, 2.2vw, 40px);
        height: clamp(34px, 2.2vw, 40px);
        border-radius: 50%;
        background: radial-gradient(120% 120% at 50% 40%, rgba(61,217,255,0.28) 0%, rgba(7,24,40,0.95) 100%);
        border: 1.5px solid rgba(61, 217, 255, 0.55);
        box-shadow: 0 0 0 4px rgba(61, 217, 255, 0.12) inset, 0 10px 40px -24px #3dd9ff;
    }

    .tooth-icon svg {
        width: clamp(18px, 1.6vw, 22px);
        height: clamp(18px, 1.6vw, 22px);
    }

    .jaw-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: clamp(11px, 1vw, 13px);
        font-weight: 600;
        line-height: 1.8;
        background-color: rgba(40, 167, 69, 0.16);
        color: #7dd87f;
        border: 1.5px solid rgba(125, 216, 127, 0.55);
        box-shadow: 0 8px 40px -24px #7dd87f;
        white-space: nowrap;
    }

    .tooth-location {
        color: #c9dce9;
        font-size: clamp(12px, 1.05vw, 14px);
        margin-top: -2px;
        opacity: 0.95;
    }

    /* بدنه کارت */
    .tooth-body {
        display: flex;
        flex-direction: column;
        gap: clamp(10px, 1.2vw, 14px);
        margin-top: clamp(6px, 1vw, 10px);
    }

    .info-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: clamp(8px, 1vw, 10px) clamp(10px, 1.2vw, 14px);
        border-radius: 12px;
        background-color: rgba(9, 36, 60, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .info-label {
        display: flex;
        align-items: center;
        gap: clamp(6px, 0.9vw, 8px);
        color: #d7e6f3;
        font-size: clamp(12px, 1.05vw, 14px);
        font-weight: 500;
    }

    .info-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #3dd9ff;
        opacity: 0.95;
    }

    .info-value {
        color: #ffffff;
        font-size: clamp(12px, 1.05vw, 14px);
        font-weight: 700;
        text-align: left;
        direction: ltr;
        white-space: nowrap;
    }

    .info-value-text {
        color: #ffffff;
        font-size: clamp(12px, 1.05vw, 14px);
        font-weight: 700;
        text-align: left;
    }

    /* وضعیت اعتماد */
    .confidence-wrapper {
        margin-top: clamp(6px, 1vw, 10px);
        padding-top: clamp(10px, 1.2vw, 14px);
        border-top: 1px dashed rgba(255, 255, 255, 0.08);
    }

    .confidence-label {
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #d7e6f3;
        font-size: clamp(12px, 1.05vw, 14px);
        margin-bottom: clamp(8px, 1vw, 10px);
    }

    .confidence-percent {
        color: #3dd9ff;
        font-weight: 700;
        direction: ltr;
    }

    .confidence-bar-bg {
        width: 100%;
        height: clamp(6px, 0.8vw, 8px);
        border-radius: 999px;
        background-color: rgba(255, 255, 255, 0.12);
        overflow: hidden;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.25);
    }

    .confidence-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #1fa2ff 0%, #3dd9ff 60%, #7df9ff 100%);
        box-shadow: 0 0 20px -10px #3dd9ff;
        transition: width 0.6s ease-in-out;
    }

    /* بخش آپلود */
    .upload-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: clamp(40px, 6vw, 80px) 20px;
        gap: clamp(18px, 2vw, 28px);
    }

    .upload-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: clamp(90px, 9vw, 120px);
        height: clamp(90px, 9vw, 120px);
        border-radius: 50%;
        background: radial-gradient(120% 120% at 50% 40%, rgba(61,217,255,0.22) 0%, rgba(6,24,38,0.98) 100%);
        border: 2px solid rgba(61, 217, 255, 0.45);
        box-shadow: 0 0 0 10px rgba(61, 217, 255, 0.08) inset, 0 30px 80px -40px #3dd9ff;
    }

    .upload-title {
        color: #3dd9ff;
        font-size: clamp(18px, 2vw, 26px);
        font-weight: 700;
        line-height: 1.5;
    }

    .upload-subtitle {
        color: #d7e6f3;
        font-size: clamp(12px, 1.1vw, 15px);
        line-height: 2.2;
        max-width: min(680px, 90vw);
        opacity: 0.95;
    }

    .upload-hint {
        color: #9fb9cc;
        font-size: clamp(11px, 0.95vw, 13px);
        opacity: 0.9;
    }

    div[data-testid="stFileUploader"] {
        direction: rtl;
    }

    div[data-testid="stFileUploader"] > section {
        background-color: rgba(8, 30, 50, 0.96);
        border: 2px dashed rgba(61, 217, 255, 0.45);
        border-radius: 16px;
        padding: clamp(14px, 2vw, 24px);
        transition: all 0.25s ease-in-out;
    }

    div[data-testid="stFileUploader"] > section:hover {
        border-color: rgba(61, 217, 255, 0.7);
        background-color: rgba(9, 36, 60, 0.98);
    }

    div[data-testid="stFileUploader"] button {
        background: linear-gradient(90deg, #0f7bff 0%, #3dd9ff 100%);
        color: #061826 !important;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.5rem 1.8rem;
        box-shadow: 0 12px 50px -24px #3dd9ff;
    }

    div[data-testid="stFileUploader"] button:hover {
        filter: brightness(1.05);
    }

    div[data-testid="stFileUploader"] small {
        color: #b9d3e6 !important;
    }

    .stAlert {
        border-radius: 12px;
        text-align: right;
    }

    .disclaimer {
        color: #9fb9cc;
        font-size: clamp(11px, 0.95vw, 12.5px);
        line-height: 2;
        text-align: justify;
        background-color: rgba(8, 30, 50, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: clamp(12px, 1.5vw, 16px);
        margin-top: 28px;
    }

    .result-image-wrapper {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(0, 191, 255, 0.25);
        box-shadow: 0 20px 70px -40px rgba(0, 191, 255, 0.8);
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ─── آیکون دندان ───
TOOTH_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">
  <path d="M7.5 3C5.01472 3 3 5.01472 3 7.5C3 9.98528 5.01472 12 7.5 12C7.67223 12 7.84205 11.9899 8.00879 11.9707C8.08518 12.6055 8.294 13.2037 8.61133 13.7324L7.43945 16.0762C7.15868 16.6373 7.29209 17.3151 7.76562 17.7266L9.23438 18.999C9.70791 19.4105 10.4034 19.4556 10.9258 19.1113L12 18.4023L13.0742 19.1113C13.5966 19.4556 14.2921 19.4105 14.7656 18.999L16.2344 17.7266C16.7079 17.3151 16.8413 16.6373 16.5605 16.0762L15.3887 13.7324C15.706 13.2037 15.9148 12.6055 15.9912 11.9707C16.158 11.9899 16.3278 12 16.5 12C18.9853 12 21 9.98528 21 7.5C21 5.01472 18.9853 3 16.5 3C15.173 3 13.9859 3.57329 13.1445 4.48047C12.7819 4.87155 12.2782 5.1 11.75 5.1H12.25C11.7218 5.1 11.2181 4.87155 10.8555 4.48047C10.0141 3.57329 8.827 3 7.5 3Z" fill="#3dd9ff"/>
</svg>
"""

# ─── آیکون فلش (عرض/طول) ───
ARROW_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#3dd9ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-ruler">
  <path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0Z"/>
  <path d="m14.5 12.5 2-2"/>
  <path d="m11.5 9.5 2-2"/>
  <path d="m8.5 6.5 2-2"/>
  <path d="m17.5 15.5 2-2"/>
</svg>
"""

# ─── بارگذاری مدل ───
@st.cache_resource
def load_model():
    if not os.path.exists("best_model.pt"):
        return None
    return YOLO("best_model.pt")

model = load_model()

if model is None:
    st.error("❌ فایل مدل (best_model.pt) پیدا نشد. لطفاً فایل مدل را در کنار app.py قرار دهید.")
    st.stop()

names = model.names
target_teeth = ["18", "28", "38", "48"]
wisdom_indices = [idx for idx, name in names.items() if name in target_teeth]

# ─── اطلاعات پایه هر دندان ───
TOOTH_DETAILS = {
    "18": {"title": "دندان عقل بالا - سمت راست", "jaw": "maxilla", "side": "right", "jaw_fa": "فک بالا", "location_fa": "سمت راست"},
    "28": {"title": "دندان عقل بالا - سمت چپ", "jaw": "maxilla", "side": "left", "jaw_fa": "فک بالا", "location_fa": "سمت چپ"},
    "38": {"title": "دندان عقل پایین - سمت چپ", "jaw": "mandible", "side": "left", "jaw_fa": "فک پایین", "location_fa": "سمت چپ"},
    "48": {"title": "دندان عقل پایین - سمت راست", "jaw": "mandible", "side": "right", "jaw_fa": "فک پایین", "location_fa": "سمت راست"},
}

# ─── تابع تعیین زاویه وینتر ───
def get_winter_angle(tooth_name, aspect_ratio):
    info = TOOTH_DETAILS.get(tooth_name, {})
    jaw = info.get("jaw", "mandible")

    if jaw == "mandible":
        if aspect_ratio >= 0.88:
            return "عمودی (Vertical)"
        else:
            return "عمودی (Vertical)"
    else:
        return "مایل (Angulated)"

# ─── هدر برنامه ───
st.markdown("""
    <div class="top-header">
        <h1 class="main-title">HUMAN TEETH ANATOMY & AI DIAGNOSTIC STUDIO</h1>
        <p class="sub-title">
            سامانه فوق‌پیشرفته تحلیل رادیوگرافی پانورامیک (OPG) و ردیابی دندان‌های عقل
        </p>
    </div>
""", unsafe_allow_html=True)

# ─── نوار وضعیت ───
st.markdown("""
    <div class="status-bar">
        <div class="status-bar-right">
            <div class="status-item">
                <span class="status-dot"></span>
                <span>سیستم در حالت آماده‌به‌کار (Standby)</span>
            </div>
            <div class="status-item">
                <span>کنسول بالینی (Clinical Console)</span>
            </div>
            <div class="status-item">
                <span>مرکز بیمار (Patient Hub)</span>
            </div>
        </div>
        <div class="status-bar-left">
            <div class="hud-badge">نقشه دندان‌های اطلس (Teeth HUD Map)</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ─── بارگذاری تصویر ───
uploaded_file = st.file_uploader(
    "تصویر رادیوگرافی پانورامیک (OPG) را بارگذاری کنید",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown("""
        <div class="upload-wrapper">
            <div class="upload-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#3dd9ff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-upload-cloud">
                    <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/>
                    <path d="M12 12v9"/>
                    <path d="m16 16-4-4-4 4"/>
                </svg>
            </div>
            <div class="upload-title">درگاه ورودی اسکن رادیوگرافی پانورامیک (OPG)</div>
            <div class="upload-subtitle">
                تصویر رادیوگرافی پانورامیک (OPG) را اینجا بکشید و رها کنید یا با کلیک بر روی دکمه «Browse files» آن را انتخاب کنید.
            </div>
            <div class="upload-hint">فرمت‌های مجاز: JPG، PNG – حداکثر حجم ۲۰۰ مگابایت</div>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── پردازش تصویر ───
image = Image.open(uploaded_file).convert("RGB")
img_array = np.array(image)
img_h, img_w = img_array.shape[:2]

mm_per_pixel = 155.0 / img_h if img_h > 0 else 0.12

with st.spinner("⏳ در حال پردازش تصویر و استخراج ویژگی‌های آناتومیک..."):
    results = model.predict(
        source=img_array,
        classes=wisdom_indices,
        conf=0.35,
        show_conf=False,
        line_width=3,
        verbose=False
    )

result = results[0]
result_image_bgr = result.plot(conf=False)
result_image_rgb = cv2.cvtColor(result_image_bgr, cv2.COLOR_BGR2RGB)

# ─── نمایش تصویر پردازش‌شده ───
st.markdown('<div class="result-image-wrapper">', unsafe_allow_html=True)
st.image(result_image_rgb, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

detected_boxes = result.boxes
detected_count = len(detected_boxes)

st.markdown("---")

# ─── عنوان بخش نتایج ───
st.markdown("""
    <div class="section-title">
        📊 وضعیت دندان‌های عقل در قوس دندانی (Dental Arch)
    </div>
""", unsafe_allow_html=True)

if detected_count == 0:
    st.info("ℹ️ در این تصویر هیچ دندان عقلی (۱۸، ۲۸، ۳۸، ۴۸) شناسایی نشد.")
    st.stop()

# ─── مرتب‌سازی دندان‌ها بر اساس شماره (۱۸،۲۸،۳۸،۴۸ از راست به چپ) ───
detections_list = []
for box in detected_boxes:
    cls_id = int(box.cls[0])
    tooth_name = names[cls_id]
    conf_score = float(box.conf[0])

    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    width_px = x2 - x1
    height_px = y2 - y1

    width_mm = round(width_px * mm_per_pixel, 1)
    height_mm = round(height_px * mm_per_pixel, 1)
    aspect_ratio = width_px / height_px if height_px > 0 else 1.0

    detections_list.append({
        "tooth_name": tooth_name,
        "conf_score": conf_score,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "aspect_ratio": aspect_ratio,
    })

# ترتیب نمایش: 48، 38، 28، 18
tooth_order = {"48": 0, "38": 1, "28": 2, "18": 3}
detections_list.sort(key=lambda x: tooth_order.get(x["tooth_name"], 99))

# ─── نمایش کارت‌های دندان‌ها ───
cols = st.columns(4, gap="large")

for i, det in enumerate(detections_list):
    tooth_name = det["tooth_name"]
    conf_score = det["conf_score"]
    width_mm = det["width_mm"]
    height_mm = det["height_mm"]
    aspect_ratio = det["aspect_ratio"]

    tooth_info = TOOTH_DETAILS.get(tooth_name, {
        "jaw_fa": "-",
        "location_fa": "-",
        "title": f"دندان {tooth_name}"
    })

    winter_angle = get_winter_angle(tooth_name, aspect_ratio)
    conf_percent = round(conf_score * 100)

    with cols[i]:
        st.markdown(f"""
            <div class="tooth-card">
                <div class="tooth-header">
                    <div class="tooth-number">
                        <div class="tooth-icon">
                            {TOOTH_SVG}
                        </div>
                        <span>دندان شماره {tooth_name}</span>
                    </div>
                    <div class="jaw-badge">{tooth_info['jaw_fa']}</div>
                    <div class="tooth-location">{tooth_info['title']}</div>
                </div>

                <div class="tooth-body">
                    <div class="info-row">
                        <div class="info-label">
                            <div class="info-icon">
                                {ARROW_SVG}
                            </div>
                            <span>عرض تقریبی:</span>
                        </div>
                        <div class="info-value">{width_mm} mm</div>
                    </div>

                    <div class="info-row">
                        <div class="info-label">
                            <div class="info-icon">
                                {ARROW_SVG}
                            </div>
                            <span>طول تخمینی:</span>
                        </div>
                        <div class="info-value">{height_mm} mm</div>
                    </div>

                    <div class="info-row">
                        <div class="info-label">
                            <span>🧭 زاویه وینتر:</span>
                        </div>
                        <div class="info-value-text">{winter_angle}</div>
                    </div>

                    <div class="confidence-wrapper">
                        <div class="confidence-label">
                            <span>میزان اطمینان هوش مصنوعی (AI Confidence)</span>
                            <span class="confidence-percent">{conf_percent}%</span>
                        </div>
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill" style="width: {conf_percent}%;"></div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ─── سلب مسئولیت ───
st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ سلب مسئولیت پزشکی و قانونی:</strong>
        مقادیر، ابعاد و طبقه‌بندی‌های نمایش‌داده‌شده حاصل پردازش هوش مصنوعی بوده و صرفاً جنبهٔ کمک‌تشخیصی دارند.
        این نتایج جایگزین ارزیابی بالینی، تشخیص تخصصی توسط دندان‌پزشک و بررسی رادیولوژیک دقیق نمی‌باشند.
    </div>
""", unsafe_allow_html=True)
