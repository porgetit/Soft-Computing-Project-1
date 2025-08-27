# -*- coding: utf-8 -*-
"""thresholding

Umbralización binaria de imágenes en escala de grises con opción de progreso.
"""

from dataclasses import dataclass
import numpy as np
from PIL import Image
from rich.console import Console
from tqdm import tqdm

@dataclass
class ThresholdConfig:
    """Configuración de umbralización binaria."""
    input_path: str
    output_path: str
    threshold: float = 0.5
    verbose: bool = False
    
class Thresholding:
    def __init__(self, config: ThresholdConfig):
        self.config = config
        self.console = Console()
        
    def _log(self, msg: str) -> None:
        if self.config.verbose:
            try:
                self.console.print(msg, style="bold cyan")
            except Exception:
                print(msg.encode("ascii", "ignore").decode("ascii"))
            
    def run(self) -> np.ndarray:
        """Ejecuta la umbralización sobre la imagen."""
        self._log("Cargando imagen...")
        img = Image.open(self.config.input_path).convert("L")
        arr = np.array(img, dtype=float) / 255.0
        
        self._log("Aplicando umbralización...")
        h, w = arr.shape
        binary = np.zeros_like(arr, dtype=np.uint8)
        
        row_iter = range(h)
        if self.config.verbose:
            row_iter = tqdm(row_iter, desc="Umbralizando filas...", ncols=80, colour="blue")
            
        for i in row_iter:
            for j in range(w):
                binary[i, j] = 1 if arr[i, j] >= self.config.threshold else 0
        
        self._log("Guardando resultado...")
        Image.fromarray((binary*255).astype(np.uint8)).save(self.config.output_path)
        
        self._log(f"✅ Imagen binaria guardada en {self.config.output_path}")
        return binary
