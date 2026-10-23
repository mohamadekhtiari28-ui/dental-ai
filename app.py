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

# ─── استایل
