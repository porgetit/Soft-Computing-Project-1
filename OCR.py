# -*- coding: utf-8 -*-
"""ocr

OCR sencillo para dígitos individuales mediante Tesseract, con preprocesado
orientado a imágenes pequeñas (celdas de Sudoku).
"""

import re
from dataclasses import dataclass
from typing import Optional
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import numpy as np
from collections import Counter


@dataclass
class DigitOCR:
    """
    OCR simple para un solo dígito usando PyTesseract.
    - En Windows, configure `tesseract_cmd` si Tesseract no está en PATH.
    - `psm=10` indica a Tesseract tratar la imagen como un único carácter.
    """
    tesseract_cmd: Optional[str] = None
    psm: int = 10   # 10: single char, 13: raw line (pruebe 13 si los trazos son finos)
    oem: int = 3    # LSTM engine

    def __post_init__(self) -> None:
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def _preprocess(self, img: Image.Image) -> Image.Image:
        # 1) Gris + normalizado
        img = img.convert("L")
        img = ImageOps.autocontrast(img)

        # 2) Upscale
        target_h = 120  # un poco más alto ayuda con trazos finos
        w, h = img.size
        scale = target_h / max(h, 1)
        img = img.resize((max(int(w * scale), 1), target_h), Image.LANCZOS)

        # 3) Suavizado + sharpen ligero
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

        # 4) Binarización
        img = ImageOps.posterize(img, bits=3)
        img = img.point(lambda p: 255 if p > 110 else 0)

        # 5) Recorte de marco: quita bordes oscuros que tocan el límite
        arr = np.array(img)  # 0 = negro, 255 = blanco
        fg = (arr == 0)      # foreground
        ys, xs = np.where(fg)
        if ys.size and xs.size:
            y0, y1 = max(0, ys.min() - 2), min(arr.shape[0], ys.max() + 2)
            x0, x1 = max(0, xs.min() - 2), min(arr.shape[1], xs.max() + 2)
            # Contracción suave solo si toca el borde (para remover rejilla)
            pad = 2
            if y0 <= 0 or x0 <= 0 or y1 >= arr.shape[0] or x1 >= arr.shape[1]:
                y0 = min(max(y0 + pad, 0), arr.shape[0] - 1)
                x0 = min(max(x0 + pad, 0), arr.shape[1] - 1)
                y1 = max(min(y1 - pad, arr.shape[0]), 0)
                x1 = max(min(x1 - pad, arr.shape[1]), 0)
            if y1 > y0 and x1 > x0:
                arr = arr[y0:y1, x0:x1]
                img = Image.fromarray(arr)

        # 6) Borde blanco de silencio
        img = ImageOps.expand(img, border=8, fill=255)
        return img

    def recognize(self, image_path: str) -> int:
        img = Image.open(image_path)
        img = self._preprocess(img)

        config = f"--oem {self.oem} --psm {self.psm} -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(img, config=config)

        digits = re.findall(r"\d", text)
        if not digits:
            return 0  # vacío

        counts = Counter(digits)
        digit, freq = counts.most_common(1)[0]
        if freq / len(digits) < 0.6:
            return 0
        return int(digit)

