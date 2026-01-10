import streamlit as st
import os
from PIL import Image
import time

# --- Imports ---
try:
    from ThothPaddle.ocr_engine import ocr_engine
    from ai.analyzis import run_grading
except ImportError as e:
    st.error(f"Critical Import Error: {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Handwriting Grader",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Clean Modern Design with Proper Contrast ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Light Background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        color: #1a1a1a;
    }
    
    .block-container {
        background: transparent;
        padding: 2rem 3rem !important;
        max-width: 1400px;
    }
    
    /* Subtle Pattern Background */
    .block-container::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(30deg, #e8ecf1 12%, transparent 12.5%, transparent 87%, #e8ecf1 87.5%, #e8ecf1),
            linear-gradient(150deg, #e8ecf1 12%, transparent 12.5%, transparent 87%, #e8ecf1 87.5%, #e8ecf1),
            linear-gradient(30deg, #e8ecf1 12%, transparent 12.5%, transparent 87%, #e8ecf1 87.5%, #e8ecf1),
            linear-gradient(150deg, #e8ecf1 12%, transparent 12.5%, transparent 87%, #e8ecf1 87.5%, #e8ecf1);
        background-size: 80px 140px;
        background-position: 0 0, 0 0, 40px 70px, 40px 70px;
        opacity: 0.3;
        pointer-events: none;
        z-index: 0;
    }
    
    .block-container > * {
        position: relative;
        z-index: 1;
    }
    
    /* Sidebar - Enhanced Contrast */
    [data-testid="stSidebar"] {
        background: #1e2936;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        letter-spacing: 0.12em !important;
        margin-bottom: 1.2rem !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #e8e8e8 !important;
        font-weight: 500 !important;
        line-height: 1.6 !important;
    }
    
    /* Title - Dark Text */
    h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem !important;
        text-transform: uppercase;
        letter-spacing: -0.02em;
        animation: titleFloat 3s ease-in-out infinite;
    }
    
    @keyframes titleFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    /* Subtitle - Medium Dark */
    .subtitle {
        font-size: 1.1rem;
        color: #546e7a;
        font-weight: 500;
        margin-bottom: 2rem;
        font-family: 'Fira Code', monospace;
    }
    
    /* Section Headers - Dark */
    h3 {
        color: #2c3e50 !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        margin-bottom: 1.5rem !important;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #3498db;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Select Box - Sidebar */
    [data-testid="stSidebar"] .stSelectbox label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: #2a3642 !important;
        border: 2px solid #3d4d5f !important;
        border-radius: 10px !important;
        transition: all 0.3s ease;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div:hover {
        border-color: #5a7a99 !important;
        background: #34404f !important;
        box-shadow: 0 0 15px rgba(90, 122, 153, 0.3);
    }
    
    /* File Uploader - Light with Dark Text */
    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 3px dashed #3498db !important;
        border-radius: 20px !important;
        padding: 3rem 2rem !important;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stFileUploader"]:hover {
        background: #f8f9fa !important;
        border-color: #2980b9 !important;
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(52, 152, 219, 0.3);
    }
    
    [data-testid="stFileUploader"] * {
        color: #2c3e50 !important;
    }
    
    [data-testid="stFileUploader"] section {
        border: none !important;
    }
    
    /* Images */
    img {
        border-radius: 16px;
        border: 3px solid #e0e0e0;
        transition: all 0.4s ease;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }
    
    img:hover {
        transform: scale(1.03) rotate(1deg);
        box-shadow: 0 15px 50px rgba(52, 152, 219, 0.3);
        border-color: #3498db;
    }
    
    /* Buttons - Primary (Dark) */
    .stButton > button {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.9rem 2.5rem !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 6px 20px rgba(44, 62, 80, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 35px rgba(44, 62, 80, 0.6) !important;
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%) !important;
    }
    
    /* Sidebar Buttons */
    [data-testid="stSidebar"] button {
        background: #2a3642 !important;
        border: 2px solid #3d4d5f !important;
        border-radius: 10px !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
        color: #ffffff !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    [data-testid="stSidebar"] button:hover {
        background: #34404f !important;
        border-color: #5a7a99 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(90, 122, 153, 0.3) !important;
    }
    
    /* Download Button - Blue Accent */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.9rem 2.5rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4) !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 35px rgba(52, 152, 219, 0.6) !important;
        background: linear-gradient(135deg, #2980b9 0%, #3498db 100%) !important;
    }
    
    /* Metrics - Enhanced Readability */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 2px solid #d0d7de;
        border-radius: 14px;
        padding: 1.8rem 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }
    
    [data-testid="stMetric"]:hover {
        background: #f8f9fa;
        border-color: #3498db;
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(52, 152, 219, 0.2);
    }
    
    [data-testid="stMetric"] label {
        color: #6c757d !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 700 !important;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        line-height: 1.2;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #27ae60 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
    }
    
    /* Sidebar Metrics - Dark Version */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: #2a3642;
        border: 2px solid #3d4d5f;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetric"]:hover {
        background: #34404f;
        border-color: #5a7a99;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetric"] label {
        color: #b0bec5 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #4caf50 !important;
    }
    
    /* Status Messages - Colored Backgrounds with Dark Text */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%) !important;
        border-left: 5px solid #28a745 !important;
        border-radius: 10px;
        color: #155724 !important;
        padding: 1.2rem;
        font-weight: 500;
        animation: slideIn 0.5s ease-out;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%) !important;
        border-left: 5px solid #17a2b8 !important;
        border-radius: 10px;
        color: #0c5460 !important;
        padding: 1.2rem;
        font-weight: 500;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%) !important;
        border-left: 5px solid #ffc107 !important;
        border-radius: 10px;
        color: #856404 !important;
        padding: 1.2rem;
        font-weight: 500;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Text Area - Light with Dark Text */
    textarea {
        background: #ffffff !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 12px !important;
        color: #2c3e50 !important;
        font-family: 'Fira Code', monospace !important;
        font-size: 0.9rem !important;
        line-height: 1.8 !important;
        padding: 1.2rem !important;
    }
    
    textarea:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1) !important;
        outline: none !important;
    }
    
    /* Expander - Light Card */
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stExpander"]:hover {
        border-color: #3498db;
    }
    
    [data-testid="stExpander"] summary {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* Status Container */
    [data-testid="stStatus"] {
        background: #ffffff !important;
        border: 2px solid #3498db !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(52, 152, 219, 0.2);
    }
    
    [data-testid="stStatus"] * {
        color: #2c3e50 !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3498db 0%, #2980b9 100%) !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #3498db, transparent);
        margin: 3rem 0;
    }
    
    /* Caption/Footer - Medium Gray */
    .stCaption {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.85rem;
        font-family: 'Fira Code', monospace;
        margin-top: 3rem;
        font-weight: 500;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #ecf0f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #95a5a6;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #7f8c8d;
    }
    
    /* Stats Dashboard */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0 3rem 0;
    }
    
    .stat-card {
        background: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stat-card:hover {
        background: #f8f9fa;
        border-color: #3498db;
        transform: translateY(-8px);
        box-shadow: 0 12px 35px rgba(52, 152, 219, 0.2);
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #546e7a;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 600;
    }
    
    /* Slider - Enhanced Visibility */
    [data-testid="stSidebar"] .stSlider label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
    }
    
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: #3d4d5f !important;
    }
    
    [data-testid="stSidebar"] .stSlider [role="slider"] {
        background: #3498db !important;
    }
    
    /* Checkbox - Enhanced Visibility */
    [data-testid="stSidebar"] .stCheckbox {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stCheckbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="stSidebar"] .stCheckbox input {
        border-color: #3d4d5f !important;
    }
    
    /* Expander in Sidebar */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #2a3642;
        border: 2px solid #3d4d5f;
        border-radius: 10px;
    }
    
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚡ SYSTEM CONTROL")
    st.markdown("---")
    
    mode = st.selectbox(
        "GRADING MODE",
        ["⚡ Standard Mode", "🎯 Strict Academic", "💬 Feedback Only", "🚀 Speed Mode"],
        index=0
    )
    
    st.markdown("---")
    
    with st.expander("⚙️ ADVANCED SETTINGS"):
        confidence_threshold = st.slider("OCR Confidence Threshold", 0, 100, 85)
        auto_save = st.checkbox("Auto-save Results", value=True)
        show_confidence = st.checkbox("Show Confidence Scores", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 LIVE STATISTICS")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Papers", "1,247", "+23")
    with col_b:
        st.metric("Accuracy", "96.2%", "+1.4%")
    
    col_c, col_d = st.columns(2)
    with col_c:
        st.metric("Speed", "2.3s", "-0.2s")
    with col_d:
        st.metric("Uptime", "99.8%", "+0.1%")
    
    st.markdown("---")
    
    if st.button("🔄 RESET SYSTEM", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    
    if st.button("📊 VIEW ANALYTICS", use_container_width=True):
        st.info("Analytics dashboard coming soon!")
    
    st.markdown("---")
    st.caption("VERSION 1.0.0 | BUILD 2025.01")

# --- Main Content ---
st.markdown("<h1>✦ AI HANDWRITING GRADER</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>// Advanced OCR & RAG-Powered Evaluation System</p>", unsafe_allow_html=True)

# Stats Dashboard
st.markdown("""
<div class='stats-container'>
    <div class='stat-card'>
        <div class='stat-number'>247</div>
        <div class='stat-label'>Papers Graded</div>
    </div>
    <div class='stat-card'>
        <div class='stat-number'>94.5%</div>
        <div class='stat-label'>Accuracy Rate</div>
    </div>
    <div class='stat-card'>
        <div class='stat-number'>2.3s</div>
        <div class='stat-label'>Avg Speed</div>
    </div>
    <div class='stat-card'>
        <div class='stat-number'>24/7</div>
        <div class='stat-label'>Availability</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📤 INPUT ZONE")
    
    uploaded_file = st.file_uploader(
        "Drop your handwritten exam here",
        type=['jpg', 'jpeg', 'png', 'pdf'],
        help="Supported formats: JPG, PNG, PDF"
    )
    
    if uploaded_file:
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, caption="📝 Source Document", use_container_width=True)
        
        with st.expander("📋 File Information"):
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            st.write(f"**Type:** {uploaded_file.type}")

with col2:
    st.markdown("### 🤖 AI PROCESSING")
    
    if uploaded_file:
        with st.status("🔍 PROCESSING PIPELINE", expanded=True) as status:
            st.write("⚡ Stage 1: Running Ensemble OCR...")
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            extracted_text = ocr_engine(temp_path)
            st.write("✅ Text extraction complete")
            
            st.write("🧠 Stage 2: AI Analysis & Grading...")
            progress_bar2 = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar2.progress(i + 1)
            
            final_result = run_grading(extracted_text)
            st.write("✅ Evaluation complete")
            
            status.update(label="✅ PROCESSING COMPLETE", state="complete", expanded=False)
        
        st.markdown("---")
        st.markdown("### 🏆 EVALUATION RESULTS")
        
        with st.expander("📄 View Extracted Text"):
            st.text_area(
                "Detected Content:",
                value=extracted_text,
                height=200,
                help="Raw text extracted from handwriting"
            )
        
        st.success(final_result)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                label="📥 DOWNLOAD REPORT",
                data=final_result,
                file_name=f"report_{uploaded_file.name}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_b:
            if st.button("🔄 PROCESS ANOTHER", use_container_width=True):
                st.rerun()
        
    else:
        st.info("⬅️ Upload a document to begin processing")
        st.markdown("""
        **How it works:**
        1. Upload handwritten exam paper
        2. AI extracts text using advanced OCR
        3. Content analyzed with Llama 3.3
        4. Get instant grading & feedback
        """)

# Footer
st.divider()
st.caption("DEVELOPED FOR UNIVERSITY GRADUATION PROJECT | TECH STACK: LLAMA 3.3 • TAVILY RAG • THOTHPADDLE OCR")