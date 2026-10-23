
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="دستیار هوشمند دندان عقل", page_icon="🦷", layout="centered")

st.title("🦷 دستیار هوشمند تشخیص دندان عقل")
st.markdown("تصویر رادیوگرافی پانورامیک (OPG) خود را بارگذاری کنید تا موقعیت دندان‌های عقل شناسایی شود.")

@st.cache_resource
def load_model():
    return YOLO("best_model.pt")

model = load_model()
names = model.names
target_teeth = ["18", "28", "38", "48"]
wisdom_indices = [idx for idx, name in names.items() if name in target_teeth]

explanations = {
    "18": "دندان عقل بالا سمت راست",
    "28": "دندان عقل بالا سمت چپ",
    "38": "دندان عقل پایین سمت چپ",
    "48": "دندان عقل پایین سمت راست"
}

uploaded_file = st.file_uploader("انتخاب تصویر OPG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    with st.spinner("در حال تحلیل تصویر..."):
        results = model.predict(source=img_array, classes=wisdom_indices, conf=0.4, show_conf=False, line_width=3)
    
    result_image = results[0].plot(conf=False)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    st.image(result_image, caption="نتیجه تشخیص", use_container_width=True)
    
    detected_count = len(results[0].boxes)
    
    if detected_count == 0:
        st.info("هیچ دندان عقلی در این تصویر شناسایی نشد.")
    else:
        st.success(f"تعداد {detected_count} دندان عقل شناسایی شد.")
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            tooth_name = names[cls_id]
            conf = float(box.conf[0])
            st.markdown(f"### 🦷 دندان شماره {tooth_name}")
            st.write(explanations.get(tooth_name, ""))
            st.progress(conf)
            st.caption(f"میزان اطمینان مدل: {conf*100:.0f} درصد")
    
    st.warning("⚠️ این تحلیل اولیه است و جایگزین معاینه حضوری و نظر دندان‌پزشک نیست. برای تشخیص قطعی حتماً به دندان‌پزشک مراجعه کنید.")
