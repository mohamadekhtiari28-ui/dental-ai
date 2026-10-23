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
    layout="wide"
)

# ─── راست‌چین‌سازی کامل و فونت فارسی ───
st.markdown("""
    <style>
    @import url('https://v1.fontapi.ir/css/Vazir');
    html, body, [class*="css"], .stMarkdown, p, div, h1, h2, h3, h4, span {
        font-family: 'Vazir', Tahoma, sans-serif !important;
        direction: rtl;
        text-align: right;
    }
    .stTabs [data-baseweb="tab-list"] {
        direction: rtl;
        gap: 12px;
    }
    .dentist-card {
        background-color: #f0f7ff;
        border-right: 5px solid #0066cc;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .patient-card {
        background-color: #f6fff5;
        border-right: 5px solid #28a745;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 12px;
        line-height: 1.8;
    }
    .size-box {
        background-color: #fff9db;
        border-right: 5px solid #f59f00;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ─── بارگذاری مدل ───
@st.cache_resource
def load_model():
    if not os.path.exists("best_model.pt"):
        return None
    return YOLO("best_model.pt")

model = load_model()

if model is None:
    st.error("❌ فایل مدل (best_model.pt) پیدا نشد. لطفاً در کنار app.py در گیت‌هاب آن را آپلود کنید.")
    st.stop()

names = model.names
target_teeth = ["18", "28", "38", "48"]
wisdom_indices = [idx for idx, name in names.items() if name in target_teeth]

# ─── اطلاعات پایه هر دندان ───
TOOTH_DETAILS = {
    "18": {"title": "دندان عقل بالا - سمت راست", "jaw": "maxilla", "side": "right"},
    "28": {"title": "دندان عقل بالا - سمت چپ", "jaw": "maxilla", "side": "left"},
    "38": {"title": "دندان عقل پایین - سمت چپ", "jaw": "mandible", "side": "left"},
    "48": {"title": "دندان عقل پایین - سمت راست", "jaw": "mandible", "side": "right"},
}

# ─── توابع تولید توصیه‌های تخصصی و عامیانه ───
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

# ─── UI اصلی ───
st.title("🦷 دستیار هوشمند و بالینی دندان عقل (OPG)")
st.caption("سیستم تخصصی تشخیص، تخمین اندازه و ارائه توصیه‌های دندان‌پزشکی و عمومی بر پایه هوش مصنوعی")

uploaded_file = st.file_uploader("تصویر رادیوگرافی پانورامیک (OPG) خود را بارگذاری کنید:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]
    
    # تخمین مقیاس میلی‌متر بر حسب رزولوشن تصویر
    # میانگین ارتفاع عمودی عکس OPG استاندارد حدود ۱۵۰ تا ۱۶۰ میلی‌متر میدان دید دارد
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

    # نمایش تصویر
    st.image(result_image_rgb, caption="تصویر OPG همراه با محدوده‌های تشخیص داده شده", use_container_width=True)

    detected_boxes = result.boxes
    detected_count = len(detected_boxes)

    st.markdown("---")

    if detected_count == 0:
        st.info("ℹ️ در این تصویر هیچ دندان عقلی (۱۸، ۲۸، ۳۸، ۴۸) یافت نشد (یا ممکن است قبلاً کشیده شده باشند).")
    else:
        st.success(f"✨ تعداد {detected_count} دندان عقل با موفقیت شناسایی و تحلیل شدند:")

        for i, box in enumerate(detected_boxes):
            cls_id = int(box.cls[0])
            tooth_name = names[cls_id]
            conf_score = float(box.conf[0])

            # محاسبه مختصات و ابعاد
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            width_px = x2 - x1
            height_px = y2 - y1
            
            # تبدیل به میلی‌متر تقریبی
            width_mm = round(width_px * mm_per_pixel, 1)
            height_mm = round(height_px * mm_per_pixel, 1)
            aspect_ratio = width_px / height_px if height_px > 0 else 1.0

            impaction_type, dentist_tips, patient_tips = generate_insights(tooth_name, width_mm, height_mm, aspect_ratio)

            tooth_info = TOOTH_DETAILS.get(tooth_name, {"title": f"دندان {tooth_name}"})

            # کارد کشویی برای هر دندان
            with st.expander(f"📍 دندان شماره {tooth_name} — {tooth_info['title']}", expanded=True):
                
                # مشخصات ابعاد و نهفتگی
                st.markdown(f"""
                <div class="size-box">
                    <b>📏 تخمین ابعاد تاج و ریشه:</b> عرض تقریبی: <b>{width_mm} میلی‌متر</b> | طول کلی: <b>{height_mm} میلی‌متر</b><br>
                    <b>🧭 وضعیت تقریبی:</b> {impaction_type}<br>
                    <small>🎯 میزان دقت مدل در این موقعیت: {conf_score*100:.1f} درصد</small>
                </div>
                """, unsafe_allow_html=True)

                # تب‌های تفکیک شده دندان‌پزشک و بیمار
                tab_dentist, tab_patient = st.tabs(["👨‍⚕️ ملاحظات بالینی (مخصوص دندان‌پزشک)", "💬 پیام ساده و دوستانه (برای بیمار)"])

                with tab_dentist:
                    st.markdown(f"""
                    <div class="dentist-card">
                        <h4>📋 توصیه‌ها و هشدارهای تشخیصی:</h4>
                        {dentist_tips}
                    </div>
                    """, unsafe_allow_html=True)

                with tab_patient:
                    st.markdown(f"""
                    <div class="patient-card">
                        <h4>💡 اوضاع دندانت چطوره؟</h4>
                        {patient_tips}
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")
    st.warning("""
    ⚠️ **سلب مسئولیت قانونی و پزشکی:**
    ابعاد و تحلیل‌های ذکر شده تخمین هوش مصنوعی هستند و جایگزین محاسبه کالیبره رادیولوژیست یا ارزیابی کلینیکی دندان‌پزشک نیستند.
    """)
