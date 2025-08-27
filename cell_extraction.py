
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import numpy as np
from PIL import Image
from rich.console import Console
from tqdm import tqdm

@dataclass
class Cell:
    """Representa una celda del tablero de Sudoku."""
    row: int
    col: int
    bbox: Tuple[int, int, int, int]
    image: Image.Image

@dataclass
class CellExtractionConfig:
    """Configuración para extraer celdas de un Sudoku rectificado."""
    input_path: str
    output_dir: str = "extracted_cells"
    grid_size: int = 9
    padding_ratio: float = 0.015
    verbose: bool = False
    # Fuente/binaria opcional para recortes espejo
    input_path_bin: Optional[str] = None
    output_dir_bin: Optional[str] = "extracted_cells_bin"


class CellExtractor:
    def __init__(self, config: CellExtractionConfig):
        self.cfg = config
        self.console = Console()

    def _log(self, msg: str):
        if self.cfg.verbose:
            self.console.print(msg, style="bold cyan")

    def _ensure_outdir(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    def _safe_int(self, x: float) -> int:
        return int(round(x))

    def run(self) -> List[List[Cell]]:
        """
        Divide la imagen rectificada en grid_size x grid_size celdas.
        Aplica padding interno para reducir captura de líneas de la cuadrícula.
        Guarda cada recorte etiquetado y devuelve una lista de listas de Cell.
        """
        self._log("Cargando imagen rectificada…")
        img = Image.open(self.cfg.input_path)
        img_bin = Image.open(self.cfg.input_path_bin).convert("L") if self.cfg.input_path_bin else None

        arr = np.array(img)
        H, W = arr.shape[:2]
        side = min(H, W)

        g = self.cfg.grid_size
        cell_size = side / g
        pad = self.cfg.padding_ratio * cell_size

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        if img_bin and self.cfg.output_dir_bin:
            os.makedirs(self.cfg.output_dir_bin, exist_ok=True)

        grid: List[List[Cell]] = []
        rows_iter = tqdm(range(g), desc="Celdas por fila", ncols=80, colour="blue") if self.cfg.verbose else range(g)

        for r in rows_iter:
            row_cells: List[Cell] = []
            for c in range(g):
                x0 = c * cell_size; y0 = r * cell_size
                x1 = (c + 1) * cell_size; y1 = (r + 1) * cell_size

                x0p = int(round(max(0, x0 + pad))); y0p = int(round(max(0, y0 + pad)))
                x1p = int(round(min(side, x1 - pad))); y1p = int(round(min(side, y1 - pad)))
                bbox = (x0p, y0p, x1p, y1p)

                cell_img = img.crop(bbox)
                out_path = os.path.join(self.cfg.output_dir, f"cell_r{r}_c{c}.png")
                cell_img.save(out_path)

                # espejo binario (si existe)
                if img_bin and self.cfg.output_dir_bin:
                    bin_crop = img_bin.crop(bbox)
                    out_bin = os.path.join(self.cfg.output_dir_bin, f"cell_r{r}_c{c}.png")
                    bin_crop.save(out_bin)

                row_cells.append(Cell(row=r, col=c, bbox=bbox, image=cell_img))
            grid.append(row_cells)

        self._log(f"✅ Guardadas 81 celdas en: {self.cfg.output_dir}" + (f" y {self.cfg.output_dir_bin}" if img_bin else ""))
        return grid
