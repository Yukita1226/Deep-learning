import os
import re
import cv2
import pytesseract
import easyocr

# ==================================================
# 0) Auto-detect Tesseract (ไม่พึ่ง PATH)
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
# 1) EasyOCR (cache reader) — cache แยกตาม (langs,gpu)
# ==================================================
_EASY_READERS = {}


def get_easy_reader(langs=("en",), gpu=False):
    key = (tuple(langs), bool(gpu))
    if key not in _EASY_READERS:
        _EASY_READERS[key] = easyocr.Reader(list(langs), gpu=gpu)
    return _EASY_READERS[key]


# ==================================================
# 2) Text utils
# ==================================================
def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def is_bad_text(text: str, min_chars: int = 40) -> bool:
    if not text:
        return True

    t = text.strip()
    if len(t) < min_chars:
        return True

    bad = sum(1 for c in t if c in "�□▯")
    if bad >= 2:
        return True

    alnum = sum(1 for c in t if c.isalnum())
    if alnum < max(5, int(0.12 * len(t))):
        return True

    if re.search(r"(.)\1{8,}", t):
        return True

    return False


# ==================================================
# 3) Preprocess (เหมาะกับ screenshot / doc)
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


# ==================================================
# 4) OCR engines
# ==================================================
def ocr_tesseract(image_path: str, lang="eng", psm: int = 6) -> str:
    if not HAS_TESSERACT:
        return ""

    img = cv2.imread(image_path)
    if img is None:
        return ""

    proc = preprocess_for_tesseract(img)
    config = f"--oem 3 --psm {psm}"

    try:
        text = pytesseract.image_to_string(proc, lang=lang, config=config)
        return normalize_text(text)
    except Exception:
        return ""


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

        client = ImageAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

        with open(image_path, "rb") as f:
            image_data = f.read()

        result = client.analyze(
            image_data=image_data,
            visual_features=[VisualFeatures.READ],
        )

        lines = []
        if result.read and result.read.blocks:
            for block in result.read.blocks:
                for line in block.lines:
                    lines.append(line.text)

        return normalize_text("\n".join(lines))
    except Exception:
        return ""


# ==================================================
# 5) Hybrid OCR (Azure → Tesseract → EasyOCR)
# ==================================================
def hybrid_ocr(
    image_path: str,
    min_chars: int = 40,
    azure_first: bool = True,
    tesseract_lang: str = "eng",
    tesseract_psm: int = 6,
    easyocr_langs=("en",),
    easyocr_gpu: bool = False,
) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 1) Azure เป็นหลัก
    if azure_first:
        a = ocr_azure_read(image_path)
        if not is_bad_text(a, min_chars=min_chars):
            print("🧠 OCR by Azure (READ)")
            return a
        if a:
            print("🧠 Azure returned text but considered low quality → fallback")

    # 2) Tesseract รอง
    t = ocr_tesseract(image_path, lang=tesseract_lang, psm=tesseract_psm)
    if not is_bad_text(t, min_chars=min_chars):
        print("🧠 OCR by Tesseract")
        return t

    # 3) EasyOCR ปิดท้าย
    e = ocr_easyocr(image_path, langs=easyocr_langs, gpu=easyocr_gpu)
    if e:
        print("🧠 OCR by EasyOCR (final fallback)")
        return e

    return a or t or e


# ==================================================
# 6) Run
# ==================================================
if __name__ == "__main__":
    img_path = "../scribes/S__6619144.jpg"

    result = hybrid_ocr(
        img_path,
        min_chars=40,
        azure_first=True,
        tesseract_lang="eng",
        tesseract_psm=6,
        easyocr_langs=("en",),
        easyocr_gpu=False,
    )

    print("\n===== OCR RESULT =====")
    print(result)
    print("======================")
