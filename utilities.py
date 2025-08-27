from dataclasses import dataclass
from typing import Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
from rich.console import Console
from tf import LinguisticVariable, FuzzySet, gaussmf, trimf, mamdani, centroid


def convolution(img: np.ndarray, kernel: np.ndarray, verbose: bool = False, desc: str = "Convolución") -> np.ndarray:
    """Aplica convolución 2D con padding cero y barra de progreso opcional."""
    m, n = kernel.shape
    h, w = img.shape
    pad_h, pad_w = m // 2, n // 2
    img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    result = np.zeros_like(img, dtype=float)

    row_iter = range(h)
    if verbose:
        row_iter = tqdm(row_iter, desc=desc, ncols=80, colour="blue")

    for i in row_iter:
        for j in range(w):
            region = img_padded[i:i+m, j:j+n]
            result[i, j] = np.sum(region * kernel)
    return result



@dataclass
class FuzzyEdgeConfig:
    """Configuración del detector de bordes difuso."""
    sx: float = 0.1        # desviación en x
    sy: float = 0.1        # desviación en y
    input_path: str = "f1.png"
    output_path: str = "edges_fuzzy.png"
    normalize: bool = True # normalizar imagen a [0,1]
    verbose: bool = False  # mostrar progreso con rich+tqdm


class FuzzyEdgeDetector:
    def __init__(self, config: FuzzyEdgeConfig):
        self.config = config
        self.console = Console()
        self.image: np.ndarray = None
        self.grad_x: np.ndarray = None
        self.grad_y: np.ndarray = None
        self.output_img: np.ndarray = None

        # Construir FIS al inicializar
        self.variables, self.rules = self._build_fis()
        self._build_lut()

    def _log(self, msg: str):
        """Imprime mensajes solo si verbose=True."""
        if self.config.verbose:
            self.console.print(msg, style="bold cyan")

    def _load_image(self) -> np.ndarray:
        self._log("[2/4] Cargando imagen...")
        img = Image.open(self.config.input_path).convert("L")
        arr = np.array(img, dtype=float)
        if self.config.normalize:
            arr = arr / 255.0
        return arr

    def _compute_gradients(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._log("[3/4] Calculando gradientes...")
        Gx = np.array([[-1, 1]])
        Gy = Gx.T
        grad_x = convolution(img, Gx, verbose=self.config.verbose, desc="Gradiente Ix")
        grad_y = convolution(img, Gy, verbose=self.config.verbose, desc="Gradiente Iy")
        return grad_x, grad_y
        
    # ---------------- LUT para el centroid de (w0,w1) ----------------
    def _build_lut(self, samples: int = 513, levels: int = 256):
        """
        Precalcula una tabla LUT[levels, levels] que, dado (w0, w1),
        devuelve el centroide de max( min(white, w0), min(black, w1) )
        usando las mismas MFs de salida que usas en Iout.
        """
        self._log("[0/4] Precomputando LUT de centroides…")

        # Universo y MFs de salida (idénticas a _build_fis)
        xs = np.linspace(*self.variables["Iout"].universe, samples)
        # white: trimf(x; 0,0,1) -> rampa descendente hasta 0 en x=1
        white = np.maximum(0.0, np.minimum(1.0, (1.0 - xs) / (1.0 - 0.0 + 1e-12)))
        white[xs <= 0.0] = 1.0  # pico en 0
        # black: trimf(x; 0,1,1) -> rampa ascendente desde 0 hasta 1 en x=1
        black = np.maximum(0.0, np.minimum(1.0, (xs - 0.0) / (1.0 - 0.0 + 1e-12)))
        black[xs >= 1.0] = 1.0  # pico en 1

        # traps para integración (constantes, reutilizables)
        dx = xs[1] - xs[0]
        x_col = xs  # (samples,)

        L = levels
        lut = np.empty((L, L), dtype=np.float32)

        # vector de alturas discretas 0..1
        heights = np.linspace(0.0, 1.0, L, dtype=np.float64)

        # Para eficiencia, precomputamos arrays "clipped" por todas las alturas
        white_clip = np.minimum(white[None, :], heights[:, None])   # (L, samples)
        black_clip = np.minimum(black[None, :], heights[:, None])   # (L, samples)

        # Recorremos todas las combinaciones (w0, w1)
        for i in range(L):
            # Agregamos vía 'max' con todas las posibles w1 de una
            mu_i = np.maximum(white_clip[i][None, :], black_clip)   # (L, samples)
            num = np.trapz(mu_i * x_col[None, :], x_col, axis=1)    # (L,)
            den = np.trapz(mu_i, x_col, axis=1) + 1e-12
            lut[i, :] = (num / den).astype(np.float32)

        self._lut = lut  # guardar
        self._lut_levels = L

    def _build_fis(self):
        self._log("[1/4] Construyendo sistema difuso...")
        Ix = LinguisticVariable(
            name="Ix",
            universe=(-1, 1),
            sets={
                "zero": FuzzySet("zero", lambda x: gaussmf(x, 0, self.config.sx)),
                "nonzero": FuzzySet("nonzero", lambda x: 1 - gaussmf(x, 0, self.config.sx)),
            },
        )

        Iy = LinguisticVariable(
            name="Iy",
            universe=(-1, 1),
            sets={
                "zero": FuzzySet("zero", lambda x: gaussmf(x, 0, self.config.sy)),
                "nonzero": FuzzySet("nonzero", lambda x: 1 - gaussmf(x, 0, self.config.sy)),
            },
        )

        Iout = LinguisticVariable(
            name="Iout",
            universe=(0, 1),
            sets={
                "white": FuzzySet("white", lambda x: trimf(x, 0.0, 0.0, 1.0)),
                "black": FuzzySet("black", lambda x: trimf(x, 0.0, 1.0, 1.0)),
            },
        )

        rules = [
            ([("Ix", "zero", "AND"), ("Iy", "zero", "")], ("Iout", "white")),
            ([("Ix", "nonzero", "OR"), ("Iy", "nonzero", "")], ("Iout", "black")),
        ]

        return {"Ix": Ix, "Iy": Iy, "Iout": Iout}, rules

    def _evaluate_fis(self):
        """
        Sustituimos la evaluación por píxel y el centroido por
        una consulta LUT de O(1) por píxel, preservando la semántica Mamdani.
        """
        self._log("[4/4] Evaluando sistema difuso (versión acelerada)…")
        h, w = self.image.shape

        # μ_zero y μ_nonzero (ya vectorizados por gaussmf)
        # Ix ~ N(0, sx), Iy ~ N(0, sy)
        def gauss(val, sigma):
            sigma = max(float(sigma), 1e-12)
            return np.exp(-0.5 * (val / sigma) ** 2)

        mu0_x = gauss(self.grad_x, self.config.sx)          # μ_zero(Ix)
        mu0_y = gauss(self.grad_y, self.config.sy)          # μ_zero(Iy)
        mu1_x = 1.0 - mu0_x                                 # μ_nonzero(Ix)
        mu1_y = 1.0 - mu0_y                                 # μ_nonzero(Iy)

        # Reglas -> pesos
        w0 = np.minimum(mu0_x, mu0_y)                       # min(zero_x, zero_y)
        w1 = np.maximum(mu1_x, mu1_y)                       # max(nonzero_x, nonzero_y)

        # Discretizar a índices de la LUT
        L = self._lut_levels - 1
        i0 = np.clip((w0 * L).astype(np.int32), 0, L)
        i1 = np.clip((w1 * L).astype(np.int32), 0, L)

        # Lookup vectorizado
        self.output_img = self._lut[i0, i1].astype(np.float32)

        return self.output_img

    def run(self):
        """Ejecuta todo el pipeline de detección de bordes difuso."""
        self.image = self._load_image()
        self.grad_x, self.grad_y = self._compute_gradients(self.image)
        self.output_img = self._evaluate_fis()

        Image.fromarray((self.output_img * 255).astype(np.uint8)).save(self.config.output_path)
        self._log(f"✅ Mapa de bordes difuso guardado en {self.config.output_path}")

        return self.output_img
