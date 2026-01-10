import os
import re
import cv2
import numpy as np
from PIL import Image

from dotenv import load_dotenv

# -------- OCR libs --------
import pytesseract
import easyocr

# -------- LLM (Groq) --------
from groq import Groq

# -------- TrOCR (optional) --------
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    HAS_TROCR = True
except Exception:
    HAS_TROCR = False


# ==================================================
# 0) Load .env
# ==================================================
load_dotenv()

# ==================================================
# 1) Auto-detect Tesseract (ไม่พึ่ง PATH)
# ==================================================
def force_tesseract() -> bool:
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for p in candidates:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            print(f"✅ Using tesseract at: {p}")
            return True
    print("⚠️ Tesseract not found → Tesseract fallback disabled")
    return False


HAS_TESSERACT = force_tesseract()

# ==================================================
# 2) EasyOCR cache
# ==================================================
_EASY_READERS = {}

def get_easy_reader(langs=("en",), gpu=False):
    key = (tuple(langs), bool(gpu))
    if key not in _EASY_READERS:
        _EASY_READERS[key] = easyocr.Reader(list(langs), gpu=gpu)
    return _EASY_READERS[key]


# ==================================================
# 3) Text utils + quality
# ==================================================
def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def text_quality_score(text: str) -> float:
    """score 0..1 (ยิ่งสูงยิ่งดี)"""
    if not text:
        return 0.0
    t = text.strip()
    if not t:
        return 0.0

    n = len(t)
    alnum = sum(1 for c in t if c.isalnum())
    alpha = sum(1 for c in t if c.isalpha())
    spaces = sum(1 for c in t if c.isspace())
    weird = sum(1 for c in t if c in "�□▯")

    alnum_ratio = alnum / n
    alpha_ratio = alpha / n
    space_ratio = spaces / n
    weird_penalty = min(1.0, weird / 2.0)

    tokens = re.findall(r"[A-Za-z]+", t)
    if tokens:
        avg_len = sum(len(x) for x in tokens) / len(tokens)
        token_score = 1.0 - min(1.0, abs(avg_len - 4.5) / 6.0)
    else:
        token_score = 0.0

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        bad_lines = 0
        for ln in lines:
            ln_alpha = sum(1 for c in ln if c.isalpha())
            if ln_alpha < 2:
                bad_lines += 1
        line_score = 1.0 - (bad_lines / max(1, len(lines)))
    else:
        line_score = 0.0

    score = (
        0.35 * alnum_ratio +
        0.25 * alpha_ratio +
        0.20 * token_score +
        0.20 * line_score
    )
    score *= (1.0 - 0.35 * weird_penalty)
    if space_ratio > 0.35:
        score *= 0.85

    return max(0.0, min(1.0, score))

def is_bad_text(text: str, min_chars: int = 40, min_score: float = 0.55) -> bool:
    if not text or len(text.strip()) < min_chars:
        return True
    return text_quality_score(text) < min_score


# ==================================================
# 4) Preprocess helpers (handwriting-friendly)
# ==================================================
def preprocess_for_tesseract(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if max(h, w) < 1400:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.convertScaleAbs(gray, alpha=1.35, beta=0)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th

def red_ink_mask(bgr):
    """mask หมึกแดง ช่วยตัดเส้นตาราง/พื้นหลัง"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 70, 50], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 70, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def segment_lines_from_mask(mask, min_line_height=12):
    proj = (mask > 0).sum(axis=1)
    thresh = max(5, int(0.01 * mask.shape[1]))
    on = proj > thresh

    bands = []
    y, H = 0, mask.shape[0]
    while y < H:
        if on[y]:
            y1 = y
            while y < H and on[y]:
                y += 1
            y2 = y
            if (y2 - y1) >= min_line_height:
                bands.append((y1, y2))
        else:
            y += 1

    merged = []
    for y1, y2 in bands:
        if not merged:
            merged.append([y1, y2])
        else:
            if y1 - merged[-1][1] <= 8:
                merged[-1][1] = y2
            else:
                merged.append([y1, y2])
    return [(a, b) for a, b in merged]

def crop_tight_x(mask, y1, y2):
    roi = mask[y1:y2, :]
    xs = np.where(roi > 0)[1]
    if xs.size == 0:
        return 0, mask.shape[1] - 1
    x1 = max(0, int(xs.min()) - 10)
    x2 = min(mask.shape[1] - 1, int(xs.max()) + 10)
    return x1, x2


# ==================================================
# 5) OCR engines
# ==================================================
def ocr_tesseract_best(image_path: str, lang="eng", psms=(6, 4, 11, 3)) -> str:
    if not HAS_TESSERACT:
        return ""
    img = cv2.imread(image_path)
    if img is None:
        return ""
    proc = preprocess_for_tesseract(img)

    best_text, best_score = "", -1.0
    for psm in psms:
        config = f"--oem 3 --psm {psm}"
        try:
            text = pytesseract.image_to_string(proc, lang=lang, config=config)
            text = normalize_text(text)
        except Exception:
            continue
        sc = text_quality_score(text)
        if sc > best_score:
            best_score, best_text = sc, text
    return best_text

def ocr_easyocr(image_path: str, langs=("en",), gpu=False) -> str:
    reader = get_easy_reader(langs=langs, gpu=gpu)
    lines = reader.readtext(image_path, detail=0, paragraph=True)
    return normalize_text("\n".join(lines))

def ocr_azure_read(image_path: str) -> str:
    endpoint = os.getenv("AZURE_VISION_ENDPOINT")
    key = os.getenv("AZURE_VISION_KEY")
    if not endpoint or not key:
        return ""
    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.ai.vision.imageanalysis import ImageAnalysisClient
        from azure.ai.vision.imageanalysis.models import VisualFeatures

        client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        with open(image_path, "rb") as f:
            data = f.read()

        result = client.analyze(image_data=data, visual_features=[VisualFeatures.READ])
        out = []
        if result.read and result.read.blocks:
            for block in result.read.blocks:
                for line in block.lines:
                    out.append(line.text)
        return normalize_text("\n".join(out))
    except Exception:
        return ""

# ---- TrOCR handwritten (cache) ----
_TROCR = {"model": None, "processor": None, "device": None, "name": None}

def get_trocr(model_name: str):
    if not HAS_TROCR:
        return None, None, None
    if _TROCR["model"] is None or _TROCR["name"] != model_name:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
        model.eval()
        _TROCR.update({"model": model, "processor": processor, "device": device, "name": model_name})
        print(f"✅ TrOCR loaded: {model_name} on {device.upper()}")
    return _TROCR["model"], _TROCR["processor"], _TROCR["device"]

def trocr_decode(pil_img: Image.Image, model, processor, device) -> str:
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=64)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return normalize_text(text)

def ocr_trocr_handwritten(image_path: str, model_name: str) -> str:
    model, processor, device = get_trocr(model_name)
    if model is None:
        return ""

    bgr = cv2.imread(image_path)
    if bgr is None:
        return ""

    mask = red_ink_mask(bgr)
    bands = segment_lines_from_mask(mask)

    # ถ้าแยกบรรทัดไม่ได้ ให้ OCR ทั้งภาพ
    if not bands:
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return trocr_decode(pil, model, processor, device)

    lines_out = []
    for (y1, y2) in bands:
        x1, x2 = crop_tight_x(mask, y1, y2)
        crop = bgr[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # เน้นเฉพาะหมึก (ลดตาราง)
        out = 255 - cv2.bitwise_and(255 - gray, crop_mask)
        out = cv2.convertScaleAbs(out, alpha=1.2, beta=0)

        pil = Image.fromarray(out).convert("RGB")
        txt = trocr_decode(pil, model, processor, device)
        if txt:
            lines_out.append(txt)

    return normalize_text("\n".join(lines_out))


# ==================================================
# 6) Post-OCR correction (Groq Llama 3)
# ==================================================
_GROQ_CLIENT = None

def get_groq_client():
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        _GROQ_CLIENT = Groq(api_key=api_key)
    return _GROQ_CLIENT

def post_ocr_correct(text: str) -> str:
    """
    ใช้ LLM แก้คำ/เรียงประโยค/grammar
    เปิด-ปิดได้ด้วย OCR_POST_CORRECT=true/false
    """
    enabled = os.getenv("OCR_POST_CORRECT", "true").lower() == "true"
    if not enabled:
        return text

    if not text or len(text.strip()) < 10:
        return text

    client = get_groq_client()
    if client is None:
        return text

    model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    temperature = float(os.getenv("GROQ_TEMPERATURE", "0.2"))

    prompt = f"""
You are a professional English proofreader.

Task:
- Correct spelling mistakes
- Fix grammar
- Reorder words if OCR output is disordered
- Preserve the original meaning
- Do NOT add new information

Return only the corrected text.

OCR text:
\"\"\"
{text}
\"\"\"
""".strip()

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        out = resp.choices[0].message.content.strip()
        return out if out else text
    except Exception:
        return text


# ==================================================
# 7) Unified engine (TrOCR → Azure → Tesseract → EasyOCR → LLM correction)
# ==================================================
def ocr_engine(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    min_chars = int(os.getenv("OCR_MIN_CHARS", "20"))
    min_score = float(os.getenv("OCR_MIN_SCORE", "0.50"))

    prefer_hand = os.getenv("OCR_PREFER_HANDWRITING", "true").lower() == "true"
    use_azure = os.getenv("OCR_USE_AZURE", "false").lower() == "true"

    trocr_model = os.getenv("TROCR_MODEL", "microsoft/trocr-base-handwritten")
    tesseract_lang = os.getenv("OCR_TESS_LANG", "eng")
    easy_lang = os.getenv("OCR_EASYOCR_LANG", "en")
    easy_gpu = os.getenv("OCR_EASYOCR_GPU", "false").lower() == "true"

    # 1) TrOCR (handwriting)
    if prefer_hand and HAS_TROCR:
        t3 = ocr_trocr_handwritten(image_path, model_name=trocr_model)
        sc3 = text_quality_score(t3)
        if not is_bad_text(t3, min_chars=min_chars, min_score=min_score):
            print(f"🧠 OCR by TrOCR (score={sc3:.2f})")
            return post_ocr_correct(t3)
        print(f"🧠 TrOCR low quality (score={sc3:.2f}) → fallback")

    # 2) Azure
    if use_azure:
        az = ocr_azure_read(image_path)
        sca = text_quality_score(az)
        if az and not is_bad_text(az, min_chars=min_chars, min_score=min_score):
            print(f"🧠 OCR by Azure (score={sca:.2f})")
            return post_ocr_correct(az)
        if az:
            print(f"🧠 Azure low quality (score={sca:.2f}) → fallback")

    # 3) Tesseract
    ts = ocr_tesseract_best(image_path, lang=tesseract_lang)
    sct = text_quality_score(ts)
    if ts and not is_bad_text(ts, min_chars=min_chars, min_score=min_score):
        print(f"🧠 OCR by Tesseract (score={sct:.2f})")
        return post_ocr_correct(ts)
    if ts:
        print(f"🧠 Tesseract low quality (score={sct:.2f}) → fallback")

    # 4) EasyOCR
    eo = ocr_easyocr(image_path, langs=(easy_lang,), gpu=easy_gpu)
    sce = text_quality_score(eo) if eo else 0.0
    if eo:
        print(f"🧠 OCR by EasyOCR (score={sce:.2f})")
        return post_ocr_correct(eo)

    return post_ocr_correct(ts or "")


# ==================================================
# 8) Run
# ==================================================
if __name__ == "__main__":
    img_path = "../scribes/S__6619144.jpg"
    result = ocr_engine(img_path)

    print("\n===== FINAL RESULT =====")
    print(result)
    print("========================")
