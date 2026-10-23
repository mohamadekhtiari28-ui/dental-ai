import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# ============================================================
# تنظیمات صفحه
# ============================================================
st.set_page_config(
    page_title="دستیار هوشمند دندان عقل",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# استایل سفارشی
st.markdown("""
<style>
    .tooth-header {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border-radius: 15px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        border-right: 5px solid #667eea;
    }
    .patient-card {
        background-color: #e8f5e9;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border-right: 5px solid #4CAF50;
    }
    .doctor-card {
        background-color: #e3f2fd;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border-right: 5px solid #2196F3;
    }
    .impaction-high {
        background-color: #ffebee;
        border-radius: 12px;
        padding: 1rem;
        border-right: 4px solid #f44336;
    }
    .impaction-medium {
        background-color: #fff8e1;
        border-radius: 12px;
        padding: 1rem;
        border-right: 4px solid #ff9800;
    }
    .impaction-low {
        background-color: #e8f5e9;
        border-radius: 12px;
        padding: 1rem;
        border-right: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# عنوان
# ============================================================
st.title("🦷 دستیار هوشمند تشخیص دندان عقل")
st.markdown("""
<div style="text-align: center; font-size: 1.05em; color: #555; margin-bottom: 1.5rem;">
تصویر رادیوگرافی پانورامیک (OPG) خود را بارگذاری کنید تا:<br>
📍 موقعیت &nbsp;|&nbsp; 📏 اندازه تقریبی &nbsp;|&nbsp; 🔍 وضعیت نهفتگی &nbsp;|&nbsp; 💬 توصیه‌های شخصی‌سازی‌شده
</div>
""", unsafe_allow_html=True)

# ============================================================
# بارگذاری مدل
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.exists("best_model.pt"):
        return None
    return YOLO("best_model.pt")

model = load_model()
if model is None:
    st.error("❌ فایل مدل (best_model.pt) یافت نشد. مطمئن شوید فایل مدل در کنار app.py قرار دارد.")
    st.stop()

names = model.names
target_teeth = ["18", "28", "38", "48"]
wisdom_indices = [idx for idx, name in names.items() if name in target_teeth]

# ============================================================
# اطلاعات دندان‌ها
# ============================================================
tooth_info = {
    "18": {"label": "دندان عقل بالا-راست", "en": "Upper Right Third Molar", "region": "upper", "emoji": "↖️"},
    "28": {"label": "دندان عقل بالا-چپ", "en": "Upper Left Third Molar", "region": "upper", "emoji": "↗️"},
    "38": {"label": "دندان عقل پایین-چپ", "en": "Lower Left Third Molar", "region": "lower", "emoji": "↙️"},
    "48": {"label": "دندان عقل پایین-راست", "en": "Lower Right Third Molar", "region": "lower", "emoji": "↘️"},
}

# ============================================================
# توابع تحلیل
# ============================================================

def estimate_size_mm(bbox_w_px, bbox_h_px, img_width):
    """تخمین اندازه تقریبی بر حسب میلی‌متر"""
    # OPG استاندارد ≈ 165mm آناتومی واقعی، ضریب بزرگ‌نمایی ≈ 1.25
    # کمی کوچک‌تر کردن باکس (YOLO معمولاً کمی بزرگ‌تر برمی‌گرداند)
    factor = 0.85
    mm_per_pixel = (165.0 * 1.25) / max(img_width, 1)
    w_mm = bbox_w_px * mm_per_pixel * factor
    h_mm = bbox_h_px * mm_per_pixel * factor
    return w_mm, h_mm


def classify_impaction(bbox_w, bbox_h, region):
    """تخمین نوع نهفتگی از روی نسبت عرض/ارتفاع باکس (بسیار تقریبی)"""
    ratio = bbox_w / max(bbox_h, 1)

    if ratio > 1.5:
        return {
            "type": "افقی (Horizontal)", "icon": "↔️",
            "level": "high", "level_text": "نیازمند بررسی جدی", "color": "#f44336"
        }
    elif ratio > 1.15:
        return {
            "type": "زاویه‌دار (Angulated)", "icon": "↗️",
            "level": "medium", "level_text": "نیازمند پایش", "color": "#ff9800"
        }
    elif ratio > 0.85:
        return {
            "type": "نیمه‌عمودی (Semi-vertical)", "icon": "↕️",
            "level": "low", "level_text": "وضعیت نسبتاً خوب", "color": "#4CAF50"
        }
    else:
        return {
            "type": "عمودی (Vertical)", "icon": "↕️",
            "level": "low", "level_text": "وضعیت مطلوب", "color": "#4CAF50"
        }


# ============================================================
# توصیه محاوره‌ای برای بیمار
# ============================================================
def get_patient_advice(tooth_name, impaction, w_mm, h_mm):
    info = tooth_info[tooth_name]
    imp = impaction["type"]

    advice = f"""
### 💬 گفتگوی من با شما

سلام! 😊 بیایید با هم نگاهی به نتایج بیندازیم.

دندان **{tooth_name}** ({info['label']}) در تصویر شما شناسایی شد.

📏 **اندازه تقریبی:** حدود {w_mm:.1f} میلی‌متر × {h_mm:.1f} میلی‌متر
🔍 **وضعیت تخمینی:** {impaction['icon']} {imp}

---
"""

    if "افقی" in imp:
        advice += """
**راستش را بخواهید،** دندان عقل شما کمی به حالت افقی درآمده. این حالت رایج است و بگذارید آرامتان کنم:

🤗 **جای نگرانی نیست!** این وضعیت به‌خوبی قابل مدیریت است.

📌 **چه کار باید بکنید؟**
- در اولین فرصت با دندان‌پزشک خود مشورت کنید
- اگر درد یا تورم ندارید، عجله‌ای نیست ولی بررسی را فراموش نکنید
- معمولاً جراحی کشیدن این دندان، عملی ساده و سرپایی است

😊 **نکته امیدوارکننده:** اکثر بیماران پس از جراحی، طی یک هفته کاملاً بهبود می‌یابند.
"""

    elif "زاویه‌دار" in imp:
        advice += """
**خب،** دندان عقل شما کمی زاویه گرفته. یعنی دقیقاً صاف نیست و شاید مقداری متمایل شده.

🤗 **نگران نباشید!** این رایج‌ترین حالت دندان عقل است.

📌 **چه کار باید بکنید؟**
- با دندان‌پزشک خود درباره وضعیت این دندان صحبت کنید
- اگر درد یا گیر غذایی ندارید، شاید فقط نیاز به پایش داشته باشید
- در صورت نیاز، جراحی ساده‌ای می‌تواند مشکل را حل کند

😊 **خبر خوب:** در بسیاری از موارد، دندان زاویه‌دار بدون مشکل سال‌ها کنار می‌ماند!
"""

    elif "نیمه‌عمودی" in imp:
        advice += """
**آفرین!** به نظر می‌رسد دندان عقل شما تقریباً صاف و نیمه‌عمودی است. وضعیتش خوب است! 👍

🤗 **بسیار عالی!**

📌 **چه کار باید بکنید؟**
- در معاینات دوره‌ای دندان‌پزشکی، وضعیت آن بررسی شود
- اگر مشکلی ندارید، احتمالاً نیازی به اقدام خاصی نیست
- بهداشت دهان را رعایت کنید (مخصوصاً اطراف این دندان)

😊 **تبریک!** دندان عقل شما در وضعیت خوبی قرار دارد.
"""

    else:  # عمودی
        advice += """
**عالیه!** 🎉 دندان عقل شما به حالت عمودی و صاف قرار دارد. این بهترین وضعیت ممکن است!

🤗 **دیگر نگران هیچ چیز نباشید!**

📌 **چه کار باید بکنید؟**
- فقط در معاینات دوره‌ای (هر ۶ ماه) وضعیت آن بررسی شود
- بهداشت دهان را خوب رعایت کنید
- اگر فضای کافی برای رویش دارد، شاید اصلاً نیازی به کشیدن نداشته باشد

😊 **تبریک!** دندان عقل شما در بهترین وضعیت قرار دارد.
"""

    advice += """

---

✨ **حرف آخر:** هر تصمیمی درباره دندان‌های عقل، با مشورت دندان‌پزشک شما گرفته می‌شود.
این تحلیل هوشمند فقط یک نگاه اولیه است تا شما با اطلاعات بیشتری با پزشک خود صحبت کنید. 💚
"""
    return advice


# ============================================================
# توصیه تخصصی برای دندان‌پزشک
# ============================================================
def get_dentist_advice(tooth_name, impaction, w_mm, h_mm):
    info = tooth_info[tooth_name]
    region = info["region"]

    advice = f"""
### 👨‍⚕️ توصیه تخصصی — دندان {tooth_name}

| پارامتر | مقدار |
|---------|-------|
| موقعیت | {info['label']} ({info['en']}) |
| اندازه تخمینی (تاج-ریشه) | {w_mm:.1f} × {h_mm:.1f} mm |
| نوع نهفتگی تخمینی | {impaction['icon']} {impaction['type']} |
| سطح ریسک | {impaction['level_text']} |

#### ⚕️ پیشنهادات بالینی:
"""

    if region == "lower":
        advice += """
- **بررسی ارتباط ریشه با کانال IAN:** در صورت نزدیکی ریشه به کانال، CBCT توصیه می‌شود
- **ارزیابی فضای رویش:** مقایسه فضای موجود در قوس با عرض مزیودیستال دندان
- **بررسی دندان ۷ مجاور:** ریسک پوسیدگی یا رزورپشن ریشه
- **ارزیابی عمق نهفتگی:** نسبت به لبه آلوئولر و خط اکلوزال
"""
    else:
        advice += """
- **بررسی نزدیکی ریشه به سینوس ماگزیلاری:** ریسک ارتباط سینوسی پس از کشیدن
- **بررسی دندان ۷ مجاور:** ارزیابی پوسیدگی یا رزورپشن احتمالی
- **ارزیابی فضای رویش و عمق نهفتگی** نسبت به توبروزیته ماگزیلا
"""

    if "افقی" in impaction["type"]:
        advice += """
#### ⚠️ توجه ویژه — نهفتگی افقی:
- ریسک بالاتر تشکیل کیست یا تومور
- آسیب احتمالی به دندان مجاور
- عفونت مکرر پریکورونیت
- **جراحی ارجح است**
"""
    elif "زاویه‌دار" in impaction["type"]:
        advice += """
#### ⚠️ توجه — نهفتگی زاویه‌دار:
- تعیین جهت زاویه (مزیوانگولار vs دیستوانگولار)
- نزدیکی ریشه به دندان ۷
- احتمال عفونت مکرر
"""

    advice += """
- **پیشنهاد ارجاع:** در صورت نیاز به جراحی، ارجاع به جراح فک و صورت
- **مستندسازی:** مقایسه OPG فعلی با قبل در پرونده بیمار
"""
    return advice


# ============================================================
# آپلود تصویر
# ============================================================
uploaded_file = st.file_upader(
    "📎 انتخاب تصویر OPG (رادیوگرافی پانورامیک)",
    type=["jpg", "jpeg", "png"],
    help="فرمت‌های مجاز: JPG, PNG"
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_h, img_w = img_array.shape[:2]

        with st.spinner("⏳ در حال تحلیل تصویر توسط هوش مصنوعی..."):
            results = model.predict(
                source=img_array,
                classes=wisdom_indices,
                conf=0.4,
                show_conf=False,
                line_width=3,
                verbose=False
            )

        result = results[0]
        plotted = result.plot(conf=False)
        result_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        # --- نمایش تصویر + خلاصه سریع ---
        col_img, col_side = st.columns([3, 1])

        with col_img:
            st.image(result_rgb, caption="✅ نتیجه تشخیص دندان‌های عقل", use_container_width=True)

        boxes = result.boxes
        count = len(boxes)

        with col_side:
            if count == 0:
                st.info("ℹ️ هیچ دندان عقلی شناسایی نشد.")
            else:
                st.metric("🦷 دندان‌های یافت‌شده", count)
                st.markdown("---")
                for box in boxes:
                    cls_id = int(box.cls[0])
                    tn = names[cls_id]
                    ti = tooth_info.get(tn, {"emoji": "🦷", "label": ""})
                    st.write(f"{ti['emoji']} **{tn}** — {ti['label']}")

        st.divider()

        # --- نتیجه ---
        if count == 0:
            st.markdown("""
            ### 🤷 چیزی شناسایی نشد
            
            ممکن است:
            - دندان عقلی وجود نداشته باشد
            - تصویر کیفیت کافی نداشته باشد
            - دندان‌ها خارج از محدوده تصویر باشند
            
            💡 **پیشنهاد:** تصویر دیگری امتحان کنید یا با دندان‌پزشک مشورت کنید.
            """)
        else:
            st.success(f"✨ تعداد **{count}** دندان عقل در تصویر شما شناسایی شد!")
            st.markdown("---")

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                tooth_name = names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox_w = x2 - x1
                bbox_h = y2 - y1

                w_mm, h_mm = estimate_size_mm(bbox_w, bbox_h, img_w)
                info = tooth_info.get(tooth_name, {"label": "نامشخص", "region": "upper", "emoji": "🦷"})
                impaction = classify_impaction(bbox_w, bbox_h, info.get("region", "upper"))

                # هدر دندان
                st.markdown(f"""
                <div class="tooth-header">
                    <h3>{info['emoji']} دندان {tooth_name} — {info['label']}</h3>
                </div>
                """, unsafe_allow_html=True)

                # آمار
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("📏 عرض تقریبی", f"{w_mm:.1f} mm")
                with c2:
                    st.metric("📐 ارتفاع تقریبی", f"{h_mm:.1f} mm")
                with c3:
                    st.metric("🔍 نوع نهفتگی", impaction["icon"] + " " + impaction["type"])
                with c4:
                    st.metric("📊 اطمینان", f"{conf*100:.1f}%")

                st.progress(min(conf, 1.0))

                # کارت نوع نهفتگی
                level_class = f"impaction-{impaction['level']}"
                st.markdown(f"""
                <div class="{level_class}">
                    <strong>🔍 وضعیت تخمینی:</strong> {impaction['icon']} {impaction['type']} — {impaction['level_text']}
                </div>
                """, unsafe_allow_html=True)

                # توصیه بیمار
                with st.expander("💬 توصیه دوستانه برای شما (بیمار)", expanded=True):
                    st.markdown(get_patient_advice(tooth_name, impaction, w_mm, h_mm))

                # توصیه دندان‌پزشک
                with st.expander("👨‍⚕️ توصیه تخصصی برای دندان‌پزشک", expanded=False):
                    st.markdown(get_dentist_advice(tooth_name, impaction, w_mm, h_mm))

                st.markdown("---")

            # ========================================
            # جمع‌بندی نهایی
            # ========================================
            st.markdown("## 📋 جمع‌بندی نهایی")

            tooth_list = ", ".join([names[int(b.cls[0])] for b in boxes])

            st.markdown(f"""
            <div class="patient-card">
                <h3>💚 عزیز، خلاصه اینکه:</h3>
                <p>
                تعداد **{count}** دندان عقل در تصویر شما شناسایی شد.
                تحلیل هوشمند ما وضعیت تقریبی هر کدام را بررسی کرده.
                </p>
                <p>
                <strong>مهم‌ترین کاری که باید بکنید:</strong><br>
                ☑️ نتایج را به دندان‌پزشک خود نشان دهید<br>
                ☑️ درباره وضعیت هر دندان با او مشورت کنید<br>
                ☑️ نگران نباشید — در بیشتر موارد، همه چیز به‌خوبی قابل مدیریت است 💚
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="doctor-card">
                <h3>👨‍⚕️ خلاصه برای دندان‌پزشک:</h3>
                <p>
                تعداد **{count}** دندان عقل ({tooth_list}) شناسایی شد.
                جزئیات تخمینی اندازه و نوع نهفتگی در بخش بالا آمده است.
                </p>
                <p>
                <strong>پیشنهادات کلی:</strong><br>
                ☑️ بررسی OPG از نظر ارتباط ریشه با ساختارهای حیاتی<br>
                ☑️ بررسی فضای رویش و وضعیت دندان‌های مجاور<br>
                ☑️ در نظر گرفتن CBCT در صورت نیاز<br>
                ☑️ برنامه‌ریزی درمانی فردی بر اساس معاینه حضوری
                </p>
            </div>
            """, unsafe_allow_html=True)

            # سلب مسئولیت
            st.warning("""
            ⚠️ **توجه مهم پزشکی:**
            این تحلیل توسط هوش مصنوعی انجام شده و صرفاً جنبه اطلاع‌رسانی دارد.
            تخمین اندازه‌ها تقریبی است و بسته به دستگاه رادیوگرافی متفاوت است.
            نوع نهفتگی نیز تخمینی است و تشخیص نهایی نیاز به بررسی تخصصی دارد.
            **این ابزار جایگزین معاینه حضوری دندان‌پزشک نیست.**
            """)

    except Exception as e:
        st.error(f"❌ خطایی در پردازش تصویر رخ داد: {str(e)}")
        st.caption("لطفاً تصویر دیگری را امتحان کنید.")
else:
    st.info("👆 لطفاً تصویر رادیوگرافی OPG خود را در کادر بالا آپلود کنید.")
