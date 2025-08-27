# sudoku_solver.py
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
from rich.console import Console


@dataclass
class SudokuSolver:
    """Resuelve un tablero de Sudoku por backtracking con poda.

    Parámetros
    ----------
    grid: List[List[int]]
        Tablero 9x9 con ceros en las casillas vacías.
    verbose: bool, opcional
        Si es ``True`` se muestran mensajes del proceso.
    """
    grid: List[List[int]]
    verbose: bool = False

    def __post_init__(self) -> None:
        if len(self.grid) != 9 or any(len(row) != 9 for row in self.grid):
            raise ValueError("El tablero debe ser 9x9.")
        self.console = Console()

    # --------------------------
    # API pública
    # --------------------------
    def solve(self) -> List[List[int]]:
        """Devuelve una copia del tablero resuelto o lanza ``ValueError`` si no hay solución."""
        board = [row[:] for row in self.grid]
        if self._backtrack(board):
            return board
        raise ValueError("Sudoku sin solución")

    # --------------------------
    # Métodos internos
    # --------------------------
    def _backtrack(self, board: List[List[int]]) -> bool:
        cell = self._select_cell(board)
        if cell is None:
            return True  # sin celdas vacías
        r, c, candidates = cell
        if not candidates:
            return False  # poda: sin opciones
        for num in sorted(candidates):
            self._log(f"Probando {num} en ({r},{c})")
            board[r][c] = num
            if self._backtrack(board):
                return True
            board[r][c] = 0
        self._log(f"Retrocendiendo en ({r},{c})")
        return False

    def _select_cell(self, board: List[List[int]]) -> Optional[Tuple[int, int, Set[int]]]:
        """Selecciona la celda vacía con menos candidatos (heurística MRV)."""
        best: Optional[Tuple[int, int]] = None
        best_cands: Set[int] = set()
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    cands = self._candidates(board, r, c)
                    if not cands:
                        return (r, c, set())  # poda inmediata
                    if best is None or len(cands) < len(best_cands):
                        best, best_cands = (r, c), cands
                        if len(best_cands) == 1:
                            return r, c, best_cands
        return None if best is None else (best[0], best[1], best_cands)

    def _candidates(self, board: List[List[int]], row: int, col: int) -> Set[int]:
        used = set(board[row]) | {board[r][col] for r in range(9)}
        sr, sc = 3 * (row // 3), 3 * (col // 3)
        for r in range(sr, sr + 3):
            used.update(board[r][sc:sc + 3])
        return set(range(1, 10)) - used

    def _log(self, msg: str) -> None:
        if self.verbose:
            self.console.print(msg, style="bold cyan")


if __name__ == "__main__":
    ejemplo = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
    solver = SudokuSolver(ejemplo, verbose=True)
    solucion = solver.solve()
    for fila in solucion:
        print(fila)
