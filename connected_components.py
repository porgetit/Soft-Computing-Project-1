# connected_components.py
from dataclasses import dataclass
import numpy as np
from PIL import Image
from rich.console import Console
from tqdm import tqdm
from collections import deque


@dataclass
class ConnectedConfig:
    """Configuración para análisis de componentes conectados."""
    input_path: str
    output_path: str
    verbose: bool = False


class ConnectedComponents:
    def __init__(self, config: ConnectedConfig):
        self.config = config
        self.console = Console()

    def _log(self, msg: str):
        if self.config.verbose:
            self.console.print(msg, style="bold cyan")

    def _get_neighbors(self, i, j, h, w):
        """8-conectividad"""
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    yield ni, nj

    def run(self) -> np.ndarray:
        """Encuentra el mayor componente conectado de la imagen binaria."""
        self._log("Cargando imagen binaria...")
        img = Image.open(self.config.input_path).convert("L")
        arr = (np.array(img, dtype=np.uint8) > 127).astype(np.uint8)  # 0/1

        h, w = arr.shape
        labels = np.zeros((h, w), dtype=np.int32)
        label = 0
        areas = {}

        row_iter = range(h)
        if self.config.verbose:
            row_iter = tqdm(row_iter, desc="Etiquetando", ncols=80, colour="blue")

        # BFS para etiquetar
        for i in row_iter:
            for j in range(w):
                if arr[i, j] == 1 and labels[i, j] == 0:
                    label += 1
                    q = deque([(i, j)])
                    labels[i, j] = label
                    area = 0
                    while q:
                        ci, cj = q.popleft()
                        area += 1
                        for ni, nj in self._get_neighbors(ci, cj, h, w):
                            if arr[ni, nj] == 1 and labels[ni, nj] == 0:
                                labels[ni, nj] = label
                                q.append((ni, nj))
                    areas[label] = area

        self._log(f"Detectados {label} componentes.")
        if not areas:
            self._log("⚠️ No se detectaron componentes.")
            return arr

        # Identificar el mayor
        largest_label = max(areas, key=areas.get)
        self._log(f"Mayor componente: etiqueta {largest_label}, área {areas[largest_label]} píxeles.")

        # Generar máscara del mayor componente
        largest = (labels == largest_label).astype(np.uint8)
        Image.fromarray((largest * 255).astype(np.uint8)).save(self.config.output_path)

        self._log(f"✅ Componente más grande guardado en {self.config.output_path}")
        return largest
