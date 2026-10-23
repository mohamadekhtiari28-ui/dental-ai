import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# تنظیمات صفحه
st.set_page_config(
    page_title="دستیار هوشمند دندان عقل", 
    page_icon="🦷", 
    layout="centered"
)

# عنوان و توضیحات
st.title("🦷 دستیار هوشمند تشخیص دندان عقل")
st.markdown("""
تصویر رادیوگرافی پانورامیک (OPG) خود را بارگذاری کنید تا موقعیت دندان‌های عقل شناسایی شود.
این سیستم فقط دندان‌های شماره ۱۸، ۲۸، ۳۸ و ۴۸ را بررسی می‌کند.
""")

# بارگذاری مدل با کش (برای سرعت بیشتر)
@st.cache_resource
def load_model():
    # اطمینان از وجود فایل مدل
    if not os.path.exists("best_model.pt"):
        return None
    return YOLO("best_model.pt")

model = load_model()

if model is None:
    st.error("❌ خطا: فایل مدل (best_model.pt) پیدا نشد. لطفاً مطمئن شوید فایل مدل در کنار app.py آپلود شده است.")
    st.stop()

# تعریف نام‌ها و توضیحات
names = model.names
target_teeth = ["18", "28", "38", "48"]
# پیدا کردن ایندکس کلاس‌های مورد نظر در مدل
wisdom_indices = [idx for idx, name in names.items() if name in target_teeth]

explanations = {
    "18": "📍 دندان عقل بالا سمت راست (Upper Right)",
    "28": "📍 دندان عقل بالا سمت چپ (Upper Left)",
    "38": "📍 دندان عقل پایین سمت چپ (Lower Left)",
    "48": "📍 دندان عقل پایین سمت راست (Lower Right)"
}

# آپلود فایل
uploaded_file = st.file_uploader("انتخاب تصویر OPG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # باز کردن تصویر
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        
        with st.spinner("⏳ در حال تحلیل تصویر توسط هوش مصنوعی..."):
            # اجرای پیش‌بینی
            # classes=wisdom_indices: فقط دندان‌های عقل را جستجو کن
            # conf=0.4: حداقل اطمینان ۴ درصد
            # show_conf=False: درصد اطمینان را روی تصویر ننویس (تمیزتر شود)
            results = model.predict(
                source=img_array, 
                classes=wisdom_indices, 
                conf=0.4, 
                show_conf=False, 
                line_width=3,
                verbose=False # جلوگیری از چاپ لاگ‌های اضافی در کنسول
            )
        
        result = results[0]
        
        # رسم باکس‌ها روی تصویر (خروجی به فرمت BGR است)
        plotted_image = result.plot(conf=False)
        # تبدیل BGR به RGB برای نمایش صحیح در Streamlit
        result_image_rgb = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
        
        # نمایش تصویر نهایی
        st.image(result_image_rgb, caption="✅ نتیجه تشخیص دندان‌های عقل", use_container_width=True)
        
        # تحلیل نتایج
        detected_boxes = result.boxes
        detected_count = len(detected_boxes)
        
        st.divider()
        
        if detected_count == 0:
            st.info("ℹ️ هیچ دندان عقلی (۱۸، ۲۸، ۳۸، ۴۸) در این تصویر با اطمینان کافی شناسایی نشد.")
            st.caption("ممکن است دندان‌ها هنوز رویش نکرده باشند، کشیده شده باشند، یا زاویه تصویر مناسب نباشد.")
        else:
            st.success(f"✨ تعداد {detected_count} دندان عقل شناسایی شد!")
            
            # نمایش جزئیات هر دندان پیدا شده
            for i, box in enumerate(detected_boxes):
                cls_id = int(box.cls[0])
                tooth_name = names[cls_id]
                conf_score = float(box.conf[0])
                
                with st.expander(f"جزئیات دندان {tooth_name} ({i+1})", expanded=True):
                    st.write(explanations.get(tooth_name, "موقعیت نامشخص"))
                    
                    # نوار پیشرفت اطمینان
                    st.write(f"**میزان اطمینان تشخیص:**")
                    st.progress(min(conf_score, 1.0)) # محدود کردن به ۱ برای جلوگیری از خطای گرافیکی
                    st.caption(f"{conf_score*100:.1f}% اطمینان")
                    
                    # مختصات تقریبی (اختیاری - برای دیباگ یا اطلاعات فنی)
                    # x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # st.text(f"مختصات: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

        # سلب مسئولیت پزشکی
        st.warning("""
        ⚠️ **توجه مهم پزشکی:**
        این تحلیل صرفاً توسط هوش مصنوعی انجام شده و جنبه اطلاع‌رسانی دارد. 
        تشخیص نهایی وضعیت نهفتگی، زاویه ریشه و نیاز به جراحی، حتماً باید توسط **دندان‌پزشک متخصص** و با معاینه حضوری انجام شود.
        """)

    except Exception as e:
        st.error(f"❌ خطایی در پردازش تصویر رخ داد: {str(e)}")
        st.caption("لطفاً تصویر دیگری را امتحان کنید یا با پشتیبانی تماس بگیرید.")
else:
    st.info("👆 لطفاً تصویر رادیوگرافی خود را در کادر بالا آپلود کنید.")
