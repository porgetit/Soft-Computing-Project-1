# --- PATCH en main.py ---

from utilities import FuzzyEdgeConfig, FuzzyEdgeDetector
from thresholding import ThresholdConfig, Thresholding
from connected_components import ConnectedConfig, ConnectedComponents
from rectify import RectifyConfig, CornerRectifier
from cell_extraction import CellExtractionConfig, CellExtractor
from digit_classifier import DigitClassifierConfig, DigitClassifier
from rich import print
from OCR import DigitOCR # Solución temporal al problema del reconocimiento de dígitos

FILE_NAME = "f2"
VERBOSE = False

if __name__ == "__main__":
    # 1) FIS -> umbral -> mayor componente (igual)
    detector = FuzzyEdgeDetector(FuzzyEdgeConfig(
        sx=0.1, sy=0.1,
        input_path=f"{FILE_NAME}.jpg",
        output_path=f"{FILE_NAME}_edges.jpg",
        verbose=VERBOSE
    ))
    detector.run()

    thresholder = Thresholding(ThresholdConfig(
        input_path=f"{FILE_NAME}_edges.jpg",
        output_path=f"{FILE_NAME}_thresh.jpg",
        threshold=0.4,
        verbose=VERBOSE
    ))
    thresholder.run()

    cc = ConnectedComponents(ConnectedConfig(
        input_path=f"{FILE_NAME}_thresh.jpg",
        output_path=f"{FILE_NAME}_largest.jpg",
        verbose=VERBOSE
    ))
    cc.run()

    # 2) Rectificar la ORIGINAL y, reusando H, también la BINARIA
    rectifier = CornerRectifier(RectifyConfig(
        mask_path=f"{FILE_NAME}_largest.jpg",
        src_image_path=f"{FILE_NAME}.jpg",                 # usa la original
        output_path=f"{FILE_NAME}_rectified.jpg",
        out_size=810,
        verbose=VERBOSE
    ))
    corners, warped, H = rectifier.run()
    rectifier.warp_with_H(
        src_image_path=f"{FILE_NAME}_thresh.jpg",
        Hmat=H,
        output_path=f"{FILE_NAME}_rectified_bin.jpg"
    )

    # 3) Extraer celdas espejo (gris + bin)
    cell_extractor = CellExtractor(CellExtractionConfig(
        input_path=f"{FILE_NAME}_rectified.jpg",
        input_path_bin=f"{FILE_NAME}_rectified_bin.jpg",
        output_dir="extracted_cells",
        output_dir_bin="extracted_cells_bin",
        grid_size=9,
        padding_ratio=0.015,   # moderado; la rejilla se quita luego por componentes
        verbose=VERBOSE
    ))
    cell_extractor.run()
    
    ocr = DigitOCR(psm=10)
    base_path = r"C:/Users/usuario/Documents/OCR/extracted_cells"

    # Matriz para guardar los dígitos reconocidos
    sudoku_grid = []

    for r in range(9):
        row = []
        for c in range(9):
            path = f"{base_path}/cell_r{r}_c{c}.png"
            digit = ocr.recognize(path)
            row.append(digit)
            print(f"({r},{c}) -> {digit}")
        sudoku_grid.append(row)

    # Mostrar la grilla completa
    print("\nSudoku reconocido:")
    for row in sudoku_grid:
        print(row)
    
