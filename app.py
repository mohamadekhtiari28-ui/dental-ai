import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="دستیار هوشمند دندان عقل", page_icon="🦷", layout="wide")

st.markdown("""
<style>
.tooth-header{background:linear-gradient(135deg,#667eea15,#764ba215);border-radius:12px;padding:1rem 1.2rem;border-right:4px solid #667eea;margin:.8rem 0}
.imp-high{background:#ffebee;border-radius:10px;padding:.8rem;border-right:4px solid #f44336}
.imp-med{background:#fff8e1;border-radius:10px;padding:.8rem;border-right:4px solid #ff9800}
.imp-low{background:#e8f5e9;border-radius:10px;padding:.8rem;border-right:4px solid #4CAF50}
.summary-patient{background:#e8f5e9;border-radius:16px;padding:1.5rem;border-right:4px solid #4CAF50}
.summary-doctor{background:#e3f2fd;border-radius:16px;padding:1.5rem;border-right:4px solid #2196F3}
</style>
""", unsafe_allow_html=True)

st.title("🦷 دستیار هوشمند تشخیص دندان عقل")
st.caption("موقعیت  |  اندازه تقریبی  |  وضعیت نهفتگی  |  توصیه‌ها")

# --- بارگذاری مدل ---
@st.cache_resource
def load_model():
    return YOLO("best_model.pt") if os.path.exists("best_model.pt") else None

model = load_model()
if model is None:
    st.error("❌ فایل best_model.pt یافت نشد.")
    st.stop()

names = model.names
target_teeth = ["18", "28", "38", "48"]
wisdom_indices = [idx for idx, name in names.items() if name in target_teeth]

# --- اطلاعات دندان ---
tooth_info = {
    "18": {"label": "بالا-راست", "region": "upper", "emoji": "↖️"},
    "28": {"label": "بالا-چپ",  "region": "upper", "emoji": "↗️"},
    "38": {"label": "پایین-چپ", "region": "lower", "emoji": "↙️"},
    "48": {"label": "پایین-راست", "region": "lower", "emoji": "↘️"},
}

# --- توابع ---
def estimate_mm(w, h, img_w):
    mm_per_px = (165.0 * 1.25) / max(img_w, 1)
    return w * mm_per_px * 0.85, h * mm_per_px * 0.85

def classify_impaction(w, h):
    r = w / max(h, 1)
    if r > 1.5:
        return "افقی", "↔️", "high", "نیازمند بررسی جدی"
    elif r > 1.15:
        return "زاویه‌دار", "↗️", "med", "نیازمند پایش"
    elif r > 0.85:
        return "نیمه‌عمودی", "↕️", "low", "وضعیت نسبتاً خوب"
    else:
        return "عمودی", "↕️", "low", "وضعیت مطلوب"

# --- توصیه بیمار (خلاصه) ---
def patient_advice(tooth, imp_type, w_mm, h_mm):
    t = tooth_info[tooth]
    if "افقی" in imp_type:
        body = "دندان شما افقی شده. رایج و قابل مدیریت است. با دندان‌پزشک مشورت کنید؛ معمولاً جراحی سرپایی ساده مشکل را حل می‌کند."
    elif "زاویه‌دار" in imp_type:
        body = "دندان شما زاویه‌دار است — شایع‌ترین حالت. اگر درد ندارید عجله‌ای نیست، ولی بررسی کنید."
    elif "نیمه‌عمودی" in imp_type:
        body = "دندان تقریباً صاف است. در معاینات دوره‌ای بررسی کنید."
    else:
        body = "دندان صاف و عمودی است — بهترین وضعیت! فقط در معاینات بررسی کنید."

    return f"""
**{t['emoji']} دندان {tooth} ({t['label']})**

📏 **اندازه:** {w_mm:.1f} × {h_mm:.1f} میلی‌متر
🔍 **وضعیت:** {imp_type}

{body}

> 💚 **حرف آخر:** نتیجه را به دندان‌پزشک نشان دهید. نگران نباشید — در بیشتر موارد قابل مدیریت است.
"""

# --- توصیه دندان‌پزشک (خلاصه) ---
def doctor_advice(tooth, imp_type, w_mm, h_mm):
    t = tooth_info[tooth]
    region = t["region"]

    base = f"""
**دندان {tooth} ({t['label']})**

| پارامتر | مقدار |
|---------|-------|
| اندازه تخمینی | {w_mm:.1f} × {h_mm:.1f} mm |
| نوع نهفتگی | {imp_type} |
"""

    if region == "lower":
        base += """
**پیشنهادات:**
- بررسی ارتباط ریشه با کانال IAN → در صورت نزدیکی، CBCT
- بررسی دندان ۷ مجاور (پوسیدگی / رزورپشن)
- ارزیابی عمق نهفتگی نسبت به لبه آلوئولر
"""
    else:
        base += """
**پیشنهادات:**
- بررسی نزدیکی ریشه به سینوس ماگزیلاری
- بررسی دندان ۷ مجاور
- ارزیابی فضای رویش و عمق نهفتگی
"""

    if "افقی" in imp_type:
        base += "\n⚠️ **ریسک بالا:** کیست/تومور، آسیب به دندان مجاور → **جراحی ارجح است**\n"
    elif "زاویه‌دار" in imp_type:
        base += "\n⚠️ تعیین جهت زاویه (مزیو/دیستو) و نزدیکی به دندان ۷\n"

    return base

# ============================================================
# آپلود
# ============================================================
uploaded_file = st.file_uploader("📎 انتخاب تصویر OPG", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_h, img_w = img_array.shape[:2]

        with st.spinner("⏳ در حال تحلیل..."):
            results = model.predict(
                source=img_array, classes=wisdom_indices,
                conf=0.4, show_conf=False, line_width=3, verbose=False
            )

        result = results[0]
        result_rgb = cv2.cvtColor(result.plot(conf=False), cv2.COLOR_BGR2RGB)
        boxes = result.boxes
        count = len(boxes)

        # --- تصویر ---
        st.image(result_rgb, caption="✅ نتیجه تشخیص", use_container_width=True)
        st.divider()

        if count == 0:
            st.info("ℹ️ هیچ دندان عقلی شناسایی نشد. تصویر دیگری امتحان کنید.")
        else:
            st.success(f"✨ **{count}** دندان عقل شناسایی شد!")

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                tn = names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bw, bh = x2 - x1, y2 - y1
                w_mm, h_mm = estimate_mm(bw, bh, img_w)
                imp_type, imp_icon, imp_level, imp_text = classify_impaction(bw, bh)
                info = tooth_info.get(tn, {"label": "", "region": "upper", "emoji": "🦷"})

                # هدر
                st.markdown(f"""
                <div class="tooth-header">
                    <h3>{info['emoji']} دندان {tn} — {info['label']}</h3>
                </div>
                """, unsafe_allow_html=True)

                # آمار
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📏 عرض", f"{w_mm:.1f} mm")
                c2.metric("📐 ارتفاع", f"{h_mm:.1f} mm")
                c3.metric("🔍 نهفتگی", f"{imp_icon} {imp_type}")
                c4.metric("📊 اطمینان", f"{conf*100:.0f}%")

                # نوار ریسک
                cls_map = {"high": "imp-high", "med": "imp-med", "low": "imp-low"}
                st.markdown(f'<div class="{cls_map[imp_level]}"><strong>🔍 {imp_icon} {imp_type}</strong> — {imp_text}</div>', unsafe_allow_html=True)

                # تب‌های جداگانه
                tab_patient, tab_doctor = st.tabs(["💬 توصیه بیمار", "👨‍⚕️ توصیه دندان‌پزشک"])

                with tab_patient:
                    st.markdown(patient_advice(tn, imp_type, w_mm, h_mm))

                with tab_doctor:
                    st.markdown(doctor_advice(tn, imp_type, w_mm, h_mm))

                st.divider()

            # --- جمع‌بندی ---
            st.markdown("## 📋 جمع‌بندی")
            tooth_list = ", ".join([names[int(b.cls[0])] for b in boxes])

            tab_sum_p, tab_sum_d = st.tabs(["💬 جمع‌بندی بیمار", "👨‍⚕️ جمع‌بندی دندان‌پزشک"])

            with tab_sum_p:
                st.markdown(f"""
                <div class="summary-patient">
                <h3>💚 عزیز، خلاصه:</h3>
                <p>تعداد <b>{count}</b> دندان عقل شناسایی شد ({tooth_list}).</p>
                <p>☑️ نتیجه را به دندان‌پزشک نشان دهید<br>
                ☑️ درباره هر دندان مشورت کنید<br>
                ☑️ نگران نباشید — قابل مدیریت است 💚</p>
                </div>
                """, unsafe_allow_html=True)

            with tab_sum_d:
                st.markdown(f"""
                <div class="summary-doctor">
                <h3>👨‍⚕️ جمع‌بندی:</h3>
                <p>دندان‌های شناسایی‌شده: <b>{tooth_list}</b></p>
                <p>☑️ بررسی OPG از نظر ارتباط ریشه با ساختارهای حیاتی<br>
                ☑️ بررسی دندان‌های مجاور و فضای رویش<br>
                ☑️ CBCT در صورت نیاز<br>
                ☑️ برنامه درمانی بر اساس معاینه حضوری</p>
                </div>
                """, unsafe_allow_html=True)

            st.warning("⚠️ تحلیل توسط هوش مصنوعی است و جایگزین معاینه دندان‌پزشک نیست.")

    except Exception as e:
        st.error(f"❌ خطا: {str(e)}")
else:
    st.info("👆 تصویر OPG خود را آپلود کنید.")
