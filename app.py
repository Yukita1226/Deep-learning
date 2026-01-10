import streamlit as st
import os
from PIL import Image
from ThothPaddle import ocr_engine
from ai import analyzis  # มั่นใจว่าใน ai/__init__.py มีฟังก์ชันนี้

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AI Handwriting Grader | Project V1.0",
    page_icon="🎓",
    layout="wide"
)

# --- 2. Custom Style ---
# แก้ไขจาก unsafe_allow_value เป็น unsafe_allow_html เพื่อแก้ TypeError
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Sidebar (Settings & Status) ---
with st.sidebar:
    st.title("⚙️ System Control")
    st.info("Graduation Project v1.0\nStatus: Online")
    mode = st.selectbox("Grading Mode", ["Standard", "Strict (Academic)", "Feedback Only"])
    if st.button("🔄 Reset System"):
        st.cache_resource.clear()
        st.rerun()

# --- 4. Main UI ---
st.title("🎓 AI-Powered Handwriting Grading System")
st.write("Upload a handwritten exam paper to extract text and evaluate with RAG-based AI.")

# Layout: 2 Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Input: Student Paper")
    uploaded_file = st.file_uploader("Drop image here...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        # Save temp file เพื่อส่งต่อให้ OCR Engine
        with open("temp_upload.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display image
        st.image(uploaded_file, caption="Handwritten Source", use_container_width=True)

with col2:
    st.subheader("🤖 AI Processing & Results")
    
    if uploaded_file:
        # Step 1: Processing Stage
        # การใช้ Orchestration ใน app.py แทนการเรียกวนลูป (Circular) ใน OCR Engine
        with st.status("🔍 Processing Pipeline...", expanded=True) as status:
            st.write("Running Ensemble OCR (ThothPaddle)...")
            # อ่านลายมือจากภาพ
            extracted_text = ocr_engine("temp_upload.jpg")
            
            st.write("Analyzing and Grading with Llama 3.3 & Tavily...")
            # ส่งข้อความไปให้ AI ตรวจสอบความถูกต้องและให้คะแนน
            final_result = analyzis(extracted_text)
            
            status.update(label="Evaluation Complete!", state="complete", expanded=False)

        # Step 2: Display Results
        with st.expander("📄 View Extracted Text (OCR Result)"):
            st.text_area("Original Content Detected:", value=extracted_text, height=150)

        # แสดงผลคะแนนและเหตุผล
        st.markdown("### 🏆 Final Evaluation")
        st.success(final_result)
        
        # ปุ่มสำหรับ Download Report (สำหรับเก็บหลักฐานการตรวจ)
        st.download_button(
            label="📥 Download Report",
            data=final_result,
            file_name="grading_report.txt",
            mime="text/plain"
        )
    else:
        st.warning("Please upload an image to start the evaluation.")

# --- 5. Footer ---
st.divider()
st.caption("Developed for University Graduation Project | Technology: Llama 3.3, Tavily RAG, ThothPaddle OCR")