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
        font-size: clamp(18px, 1.9vw, 28px);
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
        font-size: clamp(11px, 1vw, 13px);
        margin-top: -2px;
        opacity: 0.95;
        line-height: 1.8;
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
        margin-top: clamp(4px, 0.8vw, 8px);
        padding-top: clamp(10px, 1.2vw, 14px);
        border-top: 1px dashed rgba(255, 255, 255, 0.08);
    }

    .confidence-label {
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #d7e6f3;
        font-size: clamp(11px, 1vw, 13px);
        margin-bottom: clamp(8px, 1vw, 10px);
        line-height: 1.8;
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

    /* تصویر نتیجه */
    .result-image-wrapper {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(0, 191, 255, 0.25);
        box-shadow: 0 20px 70px -40px rgba(0, 191, 255, 0.8);
        margin-bottom: 10px;
    }

    /* Expander سفارشی */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(0, 191, 255, 0.25) !important;
        border-radius: 14px !important;
        background-color: rgba(8, 30, 50, 0.92) !important;
        box-shadow: 0px 14px 60px -32px rgba(0, 191, 255, 0.6);
        overflow: hidden;
    }

    div[data-testid="stExpander"] details {
        background-color: transparent !important;
    }

    div[data-testid="stExpander"] summary {
        background-color: rgba(9, 36, 60, 0.6) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: clamp(13px, 1.05vw, 15px) !important;
        padding: clamp(10px, 1.2vw, 14px) clamp(14px, 1.6vw, 18px) !important;
        border-radius: 14px !important;
    }

    div[data-testid="stExpander"] summary:hover {
        background-color: rgba(10, 42, 70, 0.8) !important;
    }

    div[data-testid="stExpander"] svg {
        fill: #3dd9ff !important;
        color: #3dd9ff !important;
    }

    div[data-testid="stExpander"] > div {
        padding: clamp(14px, 1.6vw, 18px) !important;
    }

    /* تب‌ها */
    .stTabs [data-baseweb="tab-list"] {
        direction: rtl;
        gap: 10px;
        background-color: rgba(9, 36, 60, 0.5);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        color: #d7e6f3 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 6px 16px !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(61, 217, 255, 0.18) !important;
        color: #ffffff !important;
        border: 1px solid rgba(61, 217, 255, 0.45) !important;
    }

    /* کارت‌های تب */
    .dentist-card {
        background: radial-gradient(140% 140% at 50% 0%, rgba(15, 52, 86, 0.45) 0%, rgba(7, 24, 40, 0.98) 100%);
        border-right: 4px solid #3dd9ff;
        padding: clamp(14px, 1.6vw, 18px);
        border-radius: 12px;
        margin-bottom: 8px;
        line-height: 2;
        color: #e6f3ff;
        box-shadow: 0px 14px 60px -32px rgba(61, 217, 255, 0.5);
    }

    .patient-card {
        background: radial-gradient(140% 140% at 50% 0%, rgba(20, 70, 40, 0.40) 0%, rgba(7, 24, 40, 0.98) 100%);
        border-right: 4px solid #7dd87f;
        padding: clamp(14px, 1.6vw, 18px);
        border-radius: 12px;
        margin-bottom: 8px;
        line-height: 2;
        color: #f1fff2;
        box-shadow: 0px 14px 60px -32px rgba(125, 216, 127, 0.5);
    }

    .dentist-card h4,
    .patient-card h4 {
        color: #ffffff;
        margin-top: 0;
        margin-bottom: clamp(10px, 1.2vw, 14px);
        font-size: clamp(14px, 1.1vw, 16px);
    }

    .dentist-card ul,
    .patient-card ul {
        padding-right: 20px;
        margin: 10px 0;
    }

    .dentist-card li,
    .patient-card li {
        margin-bottom: 8px;
    }

    /* هشدار */
    .stAlert {
        border-radius: 12px;
        text-align: right;
        background-color: rgba(255, 193, 7, 0.12) !important;
        color: #ffe9a6 !important;
        border: 1px solid rgba(255, 193, 7, 0.4) !important;
    }

    .disclaimer {
        color: #9fb9cc;
        font-size: clamp(11px, 0.95vw, 12.5px);
        line-height: 2.2;
        text-align: justify;
        background-color: rgba(8, 30, 50, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: clamp(12px, 1.5vw, 16px);
        margin-top: 28px;
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
    </style>
""", unsafe_allow_html=True)

# ─── آیکون دندان ───
TOOTH_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">
  <path d="M7.5 3C5.01472 3 3 5.01472 3 7.5C3 9.98528 5.01472 12 7.5 12C7.67223 12 7.84205 11.9899 8.00879 11.9707C8.08518 
