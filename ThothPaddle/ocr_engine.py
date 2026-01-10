import os
import re
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import pytesseract
import easyocr
from groq import Groq


load_dotenv()

# --- OCR Setup ---
def force_tesseract():
    candidates = [r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]
    for p in candidates:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return True
    return False

HAS_TESSERACT = force_tesseract()
_EASY_READER = easyocr.Reader(['th', 'en'], gpu=False)

def post_ocr_correct(text: str) -> str:
    """ใช้ LLM ในการซ่อมแซมข้อความที่อ่านผิดจาก OCR"""
    if not text or len(text) < 10: return text
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"Correct OCR errors, spelling, and grammar in this text while preserving the original meaning:\n{text}"
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except: return text

def ocr_engine(image_path: str) -> str:
    """แปลงภาพลายมือเป็นข้อความที่สะอาดแล้ว"""
    if not os.path.exists(image_path): return ""
    
    # 1. อ่านด้วย EasyOCR (เหมาะสำหรับลายมือ)
    results = _EASY_READER.readtext(image_path, detail=0, paragraph=True)
    raw_text = "\n".join(results)
    
    # 2. ปรับปรุงข้อความด้วย AI
    return post_ocr_correct(raw_text)

# ==================================================
# MAIN: จุดรันโปรเจกต์
# ==================================================
if __name__ == "__main__":
    # ใส่ path ของรูปภาพลายมือที่นี่
    IMAGE_TO_TEST = "../scribes/S__6619144.jpg" 
    
    print(f"--- Step 1: Extracting Handwriting ---")
    extracted_text = ocr_engine(IMAGE_TO_TEST)
    
    if extracted_text:
        print(f"Text Extracted: {extracted_text}\n")
        
        print(f"--- Step 2: Evaluating Answer ---")
        final_result = run_grading(extracted_text)
        
        print("\n" + "="*30)
        print("FINAL EVALUATION RESULT")
        print("="*30)
        print(final_result)
        print("="*30)
    else:
        print("Error: Could not extract text from image.")