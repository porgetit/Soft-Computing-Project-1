# rectify.py
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from PIL import Image
from rich.console import Console


@dataclass
class RectifyConfig:
    mask_path: str
    src_image_path: str
    output_path: str
    out_size: int = 450
    verbose: bool = False


class CornerRectifier:
    def __init__(self, config: RectifyConfig):
        self.cfg = config
        self.console = Console()

    # --------------------------
    # Utilidades de registro
    # --------------------------
    def _log(self, msg: str):
        if self.cfg.verbose:
            self.console.print(msg, style="bold cyan")

    # --------------------------
    # 1) Estimación de esquinas
    # --------------------------
    @staticmethod
    def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
        """Extrae píxeles de borde para estimar esquinas."""
        h, w = mask.shape
        padded = np.pad(mask, 1, mode="constant")
        neigh_sum = (
            padded[0:h,   0:w]   + padded[0:h,   1:w+1] + padded[0:h,   2:w+2] +
            padded[1:h+1, 0:w]   +                         padded[1:h+1, 2:w+2] +
            padded[2:h+2, 0:w] + padded[2:h+2, 1:w+1] + padded[2:h+2, 2:w+2]
        )
        boundary = (mask == 1) & (neigh_sum < 8)
        ys, xs = np.nonzero(boundary)
        return np.stack([xs, ys], axis=1)

    @staticmethod
    def _corners_by_xy_extrema(points_xy: np.ndarray) -> np.ndarray:
        """Heurística clásica para estimar 4 esquinas."""
        x = points_xy[:, 0].astype(np.float64)
        y = points_xy[:, 1].astype(np.float64)
        s = x + y
        d = x - y
        tl = points_xy[np.argmin(s)]
        br = points_xy[np.argmax(s)]
        tr = points_xy[np.argmax(d)]
        bl = points_xy[np.argmin(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def estimate_corners(self, mask: np.ndarray) -> np.ndarray:
        self._log("Estimando esquinas…")
        pts = self._boundary_pixels(mask)
        if len(pts) < 4:
            raise ValueError("No hay suficientes puntos de borde para estimar esquinas.")
        return self._corners_by_xy_extrema(pts)

    # --------------------------
    # 2) Homografía y warp
    # --------------------------
    @staticmethod
    def _get_perspective_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Calcula H (3x3) con DLT simplificado (h_33 = 1)."""
        A = []
        b = []
        for (x, y), (u, v) in zip(src, dst):
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
            b.append(u)
            b.append(v)
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        h_vec, *_ = np.linalg.lstsq(A, b, rcond=None)
        H = np.array([
            [h_vec[0], h_vec[1], h_vec[2]],
            [h_vec[3], h_vec[4], h_vec[5]],
            [h_vec[6], h_vec[7], 1.0]
        ], dtype=np.float64)
        return H

    @staticmethod
    def _warp_perspective(img: np.ndarray, H: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
        """Warp de perspectiva (mapeo inverso + bilinear)."""
        oh, ow = out_size[1], out_size[0]
        is_color = (img.ndim == 3)
        channels = img.shape[2] if is_color else 1

        jj, ii = np.meshgrid(np.arange(ow), np.arange(oh))
        ones = np.ones_like(jj, dtype=np.float64)
        dst_h = np.stack([jj, ii, ones], axis=-1).reshape(-1, 3)

        Hinv = np.linalg.inv(H)
        src_h = (dst_h @ Hinv.T)
        src_x = (src_h[:, 0] / src_h[:, 2]).reshape(oh, ow)
        src_y = (src_h[:, 1] / src_h[:, 2]).reshape(oh, ow)

        def sample_channel(ch):
            ch = ch.astype(np.float64)
            x0 = np.floor(src_x).astype(int)
            y0 = np.floor(src_y).astype(int)
            x1 = x0 + 1
            y1 = y0 + 1
            wx = src_x - x0
            wy = src_y - y0

            def safe_get(y, x):
                mask = (x >= 0) & (x < ch.shape[1]) & (y >= 0) & (y < ch.shape[0])
                out = np.zeros_like(src_x, dtype=np.float64)
                out[mask] = ch[y[mask], x[mask]]
                return out

            Ia = safe_get(y0, x0)
            Ib = safe_get(y0, x1)
            Ic = safe_get(y1, x0)
            Id = safe_get(y1, x1)
            return (Ia*(1-wx)*(1-wy) + Ib*wx*(1-wy) + Ic*(1-wx)*wy + Id*wx*wy)

        if is_color:
            warped = np.stack([sample_channel(img[..., c]) for c in range(channels)], axis=-1)
        else:
            warped = sample_channel(img)
        return np.clip(warped, 0, 255).astype(np.uint8)

    # --------------------------
    # API principal
    # --------------------------
    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Devuelve (corners_src, warped_img, H)."""
        self._log("Cargando máscara...")
        mask = (np.array(Image.open(self.cfg.mask_path).convert("L")) > 127).astype(np.uint8)

        corners_src = self.estimate_corners(mask)

        W = H = int(self.cfg.out_size)
        corners_dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)

        self._log("Calculando homografía…")
        Hmat = self._get_perspective_transform(corners_src, corners_dst)

        self._log("Cargando imagen fuente…")
        src = np.array(Image.open(self.cfg.src_image_path))
        self._log("Aplicando warp de perspectiva…")
        warped = self._warp_perspective(src, Hmat, (W, H))
        Image.fromarray(warped).save(self.cfg.output_path)
        self._log(f"✅ Guardado: {self.cfg.output_path}")
        return corners_src, warped, Hmat
        
    def warp_with_H(self, src_image_path: str, Hmat: np.ndarray, output_path: str) -> np.ndarray:
        src = np.array(Image.open(src_image_path))
        W = H = int(self.cfg.out_size)
        warped = self._warp_perspective(src, Hmat, (W, H))
        Image.fromarray(warped).save(output_path)
        return warped
