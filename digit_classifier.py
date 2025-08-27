# digit_classifier.py
# Clasificador robusto con OpenCV (encapsulado):
# - Preprocesado: Otsu+INV, recorte por contorno, centrado 20x20
# - Features: HOG (cv2.HOGDescriptor)
# - Modelo: k-NN (cv2.ml.KNearest)
# - Entrenamiento: intenta usar samples 'digits.png' de OpenCV; si no están, genera dataset sintético con cv2.putText
# - Inferencia: consenso sobre pequeñas variantes
#
# Interfaz compatible con main.py:
#   DigitClassifierConfig, DigitClassifier.run() -> np.ndarray (9,9)

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os, re
import numpy as np
import cv2
from tqdm import tqdm

@dataclass
class DigitClassifierConfig:
    cells_dir: str = "extracted_cells"              # recortes de la rectificada original
    cells_dir_bin: Optional[str] = "extracted_cells_bin"  # recortes binarios espejo (opcional)
    grid_size: int = 9
    out_npy: str = "sudoku_grid.npy"

    target_size: int = 28
    empty_ink_ratio: float = 0.013

    model_path: str = "digit_knn_opencv.npz"
    k_neighbors: int = 5
    verbose: bool = False
    seed: int = 42

    test_augs: int = 9
    min_votes: int = 4

class DigitClassifier:
    def __init__(self, cfg: DigitClassifierConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.knn = cv2.ml.KNearest_create()
        self.hog = self._build_hog(cfg.target_size)

    # ----------------- HOG de OpenCV -----------------
    def _build_hog(self, ts: int) -> cv2.HOGDescriptor:
        return cv2.HOGDescriptor((ts, ts), (14, 14), (7, 7), (7, 7), 9)

    def _hog_feats(self, img: np.ndarray) -> np.ndarray:
        f = self.hog.compute(img)
        return f.reshape(1, -1).astype(np.float32)
        
    def _bbox_from_bin(self, bw: np.ndarray) -> Tuple[np.ndarray, float, Optional[Tuple[int,int,int,int]]]:
        # bw esperado: 0/255 (blanco = dígito, negro = fondo)
        bw = (bw > 127).astype(np.uint8) * 255

        # eliminar componentes que tocan el borde (rejilla)
        h, w = bw.shape
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        for lab in range(1, num):
            x, y, ww, hh, _area = stats[lab]
            if (x == 0) or (y == 0) or (x + ww >= w) or (y + hh >= h):
                bw[labels == lab] = 0

        ink_ratio = float(np.count_nonzero(bw)) / (bw.size + 1e-6)

        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return np.zeros_like(bw), ink_ratio, None

        c = max(cnts, key=cv2.contourArea)
        x, y, ww, hh = cv2.boundingRect(c)
        pad = max(1, int(0.09 * max(ww, hh)))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(w, x + ww + pad); y1 = min(h, y + hh + pad)

        # rellenar el dígito (contornos a sólido)
        filled = np.zeros_like(bw)
        cv2.drawContours(filled, cnts, -1, 255, thickness=-1)

        crop = filled[y0:y1, x0:x1]
        return crop, ink_ratio, (x0, y0, x1, y1)

    def _normalize_to_canvas(self, crop: np.ndarray) -> np.ndarray:
        ts = self.cfg.target_size
        ch, cw = crop.shape
        if ch == 0 or cw == 0:
            return np.zeros((ts, ts), np.uint8)
        scale = min(ts / cw, ts / ch)
        nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
        rs = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((ts, ts), np.uint8)
        ox = (ts - nw) // 2; oy = (ts - nh) // 2
        canvas[oy:oy+nh, ox:ox+nw] = rs
        return canvas

    def _discover(self, root: str) -> List[Tuple[int,int,str]]:
        pat = re.compile(r"cell_r(\d+)_c(\d+)\.(?:png|jpg|jpeg)$")
        files = []
        if not root or not os.path.isdir(root):
            return files
        for fn in os.listdir(root):
            m = pat.match(fn)
            if m:
                r, c = int(m.group(1)), int(m.group(2))
                files.append((r, c, os.path.join(root, fn)))
        files.sort(key=lambda x: (x[0], x[1]))
        return files

    # --------------- Preprocesado de una celda ---------------
    def _preprocess_cell(self, bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Devuelve (img20, ink_ratio) con:
          - Otsu invertido (dígito blanco=255, fondo negro=0)
          - Limpieza de componentes que toquen el borde (remueve grilla)
          - Recorte por contorno mayor restante
          - Centrando a 20x20
        """
        gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1) Binarización robusta
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # 2) Eliminar TODO componente conectado que toque el borde (la grilla vive en los bordes)
        h, w = bw.shape
        num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        for lab in range(1, num):
            x, y, ww, hh, area = stats[lab]
            touches_border = (x == 0) or (y == 0) or (x + ww >= w) or (y + hh >= h)
            if touches_border:
                bw[labels == lab] = 0

        # Recalcular "tinta" tras limpiar bordes (para decidir celda vacía)
        ink_ratio = float(np.count_nonzero(bw)) / (bw.size + 1e-6)

        # 3) Si no queda nada significativo, devolver lienzo vacío
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return np.zeros((self.cfg.target_size, self.cfg.target_size), np.uint8), ink_ratio

        # 4) Tomar el contorno mayor (dígito) y expandir un pelín
        c = max(cnts, key=cv2.contourArea)
        x, y, ww, hh = cv2.boundingRect(c)
        pad = max(1, int(0.09 * max(ww, hh)))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(w, x + ww + pad); y1 = min(h, y + hh + pad)
        crop = bw[y0:y1, x0:x1]

        # 5) Centrar y redimensionar a 20x20 sin deformar aspecto
        ts = self.cfg.target_size
        ch, cw = crop.shape
        scale = min(ts / cw, ts / ch)
        nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
        rs = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((ts, ts), np.uint8)
        ox = (ts - nw) // 2; oy = (ts - nh) // 2
        canvas[oy:oy+nh, ox:ox+nw] = rs

        return canvas, ink_ratio

    # --------------- Variantes para consenso ---------------
    def _variants(self, img20: np.ndarray) -> List[np.ndarray]:
        """Pequeñas rotaciones y traslaciones (sin salir del lienzo)."""
        ts = img20.shape[0]
        out = [img20]
        # rotaciones ±3°
        for ang in (-3, 3):
            M = cv2.getRotationMatrix2D((ts/2, ts/2), ang, 1.0)
            rot = cv2.warpAffine(img20, M, (ts, ts), flags=cv2.INTER_NEAREST, borderValue=0)
            out.append(rot)
        # rotaciones ±3°
        for ang in (-5, 5):
            M = cv2.getRotationMatrix2D((ts/2, ts/2), ang, 1.0)
            rot = cv2.warpAffine(img20, M, (ts, ts), flags=cv2.INTER_NEAREST, borderValue=0)
            out.append(rot)
        # desplazamientos ±1 px
        shifts = [(-1,0),(1,0),(0,-1),(0,1)]
        for dx, dy in shifts:
            M = np.float32([[1,0,dx],[0,1,dy]])
            sh = cv2.warpAffine(img20, M, (ts, ts), flags=cv2.INTER_NEAREST, borderValue=0)
            out.append(sh)
        return out[:self.cfg.test_augs]

    # --------------- Entrenamiento / carga del modelo ---------------
    def _load_or_train(self):
        # cargar features/labels si existen
        if os.path.exists(self.cfg.model_path):
            data = np.load(self.cfg.model_path)
            feats = data["feats"].astype(np.float32)
            labels = data["labels"].astype(np.int32)
            self.knn.train(feats, cv2.ml.ROW_SAMPLE, labels)
            return

        # 1) intentar usar el sample oficial de OpenCV (5000 dígitos)
        feats, labels = self._try_opencv_sample()
        if feats is None:
            # 2) fallback: dataset sintético con fuentes Hershey
            feats, labels = self._build_synthetic()

        # entrenar y guardar
        self.knn.train(feats, cv2.ml.ROW_SAMPLE, labels)
        np.savez(self.cfg.model_path, feats=feats, labels=labels)

    def _try_opencv_sample(self):
        try:
            path = cv2.samples.findFile("digits.png", required=False)
            if not path or not os.path.exists(path):
                return None, None
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # la imagen es 100 filas x 50 cols de dígitos 20x20 (total 5000, 0..9 cada 500)
            cells = [np.hsplit(r, 50) for r in np.vsplit(img, 50)]  # 50x50 bloques de 20px -> (50 filas)
            cells = np.array(cells)  # shape (50,50,20,20)
            cells = cells.reshape(-1, 20, 20)  # 2500? (depende de versión), algunos tienen 5000 con 50x100
            # Si son 100x50:
            if cells.shape[0] == 2500:
                # intentar 100x50
                cells = [np.hsplit(r, 50) for r in np.vsplit(img, 100)]
                cells = np.array(cells).reshape(-1, 20, 20)  # 5000
            n = cells.shape[0]
            # etiquetas 0..9 repetidas 500 veces
            labels = np.repeat(np.arange(10), n // 10)
            # Nos quedamos con 1..9 (Sudoku)
            mask = labels > 0
            X = (255 - cells[mask]).astype(np.uint8)  # invertimos para que el dígito sea blanco
            feats = np.vstack([self._hog_feats(x) for x in X]).astype(np.float32)
            y = labels[mask].astype(np.int32).reshape(-1, 1)
            return feats, y
        except Exception:
            return None, None

    def _build_synthetic(self):
        """Genera dataset 1..9 con fuentes Hershey (varía grosor, escala, rotación, shift)."""
        ts = self.cfg.target_size
        fonts = [
            cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_COMPLEX_SMALL,
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        ]
        samples_per_digit = 600  # robusto
        feats, labels = [], []
        for d in range(1, 10):
            for _ in range(samples_per_digit):
                img = np.zeros((ts, ts), np.uint8)
                font = fonts[self.rng.integers(0, len(fonts))]
                scale = float(self.rng.uniform(0.7, 1.2))
                thick = int(self.rng.integers(1, 3))
                text = str(d)
                (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
                x = (ts - tw) // 2 + int(self.rng.integers(-2, 3))
                y = (ts + th) // 2 + int(self.rng.integers(-2, 3))
                cv2.putText(img, text, (x, y), font, scale, 255, thick, cv2.LINE_AA)

                # pequeñas rotaciones
                ang = float(self.rng.uniform(-8, 8))
                M = cv2.getRotationMatrix2D((ts/2, ts/2), ang, 1.0)
                img = cv2.warpAffine(img, M, (ts, ts), flags=cv2.INTER_NEAREST, borderValue=0)

                feats.append(self._hog_feats(img))
                labels.append([d])
        feats = np.vstack(feats).astype(np.float32)
        labels = np.array(labels, dtype=np.int32)
        return feats, labels

    # --------------- Celdas del directorio ---------------
    def _discover_cells(self) -> List[Tuple[int, int, str]]:
        pat = re.compile(r"cell_r(\d+)_c(\d+)\.(?:png|jpg|jpeg)$")
        files = []
        for fn in os.listdir(self.cfg.cells_dir):
            m = pat.match(fn)
            if m:
                r, c = int(m.group(1)), int(m.group(2))
                files.append((r, c, os.path.join(self.cfg.cells_dir, fn)))
        files.sort(key=lambda x: (x[0], x[1]))
        return files

    # --------------- API principal ---------------
    def run(self) -> np.ndarray:
        self._load_or_train()

        files_gray = self._discover(self.cfg.cells_dir)
        files_bin  = self._discover(self.cfg.cells_dir_bin) if self.cfg.cells_dir_bin else []
        use_bin = len(files_bin) == len(files_gray) == self.cfg.grid_size * self.cfg.grid_size

        g = self.cfg.grid_size
        grid = np.zeros((g, g), dtype=np.int32)

        it = files_gray
        if self.cfg.verbose:
            try:
                from tqdm import tqdm
                it = tqdm(files_gray, desc="Celdas", ncols=80)
            except Exception:
                pass

        for idx, (r, c, path_gray) in enumerate(it):
            gray = cv2.imread(path_gray, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                grid[r, c] = 0
                continue

            # si tenemos la binaria espejo, úsala para limpieza/bbox/ink
            if use_bin:
                _, _, path_bin = files_bin[idx]
                bw = cv2.imread(path_bin, cv2.IMREAD_GRAYSCALE)
                crop_bin, ink, bbox = self._bbox_from_bin(bw)
                if ink < self.cfg.empty_ink_ratio or bbox is None:
                    grid[r, c] = 0
                    continue
                norm = self._normalize_to_canvas(crop_bin)
            else:
                # fallback: el prepro antiguo por Otsu+INV
                _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                crop_bin, ink, _ = self._bbox_from_bin(bw)
                if ink < self.cfg.empty_ink_ratio:
                    grid[r, c] = 0
                    continue
                norm = self._normalize_to_canvas(crop_bin)

            # consenso con pequeñas variantes
            variants = [norm]
            ts = self.cfg.target_size
            # rot ±3°, ±5°
            for ang in (-5, -3, 3, 5):
                M = cv2.getRotationMatrix2D((ts/2, ts/2), ang, 1.0)
                variants.append(cv2.warpAffine(norm, M, (ts, ts), flags=cv2.INTER_NEAREST, borderValue=0))
            # shifts ±1 px
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                M = np.float32([[1,0,dx],[0,1,dy]])
                variants.append(cv2.warpAffine(norm, M, (ts, ts), flags=cv2.INTER_NEAREST, borderValue=0))
            variants = variants[:self.cfg.test_augs]

            votes = []
            for v in variants:
                feat = self._hog_feats(v)
                _, results, _, _ = self.knn.findNearest(feat, k=self.cfg.k_neighbors)
                votes.append(int(results[0,0]))

            vals, counts = np.unique(votes, return_counts=True)
            pred = int(vals[np.argmax(counts)])
            if counts.max() < self.cfg.min_votes and len(vals) > 1:
                feat = self._hog_feats(norm)
                _, results, _, _ = self.knn.findNearest(feat, k=self.cfg.k_neighbors)
                pred = int(results[0,0])
            grid[r, c] = pred

        if self.cfg.out_npy:
            np.save(self.cfg.out_npy, grid)
        return grid
