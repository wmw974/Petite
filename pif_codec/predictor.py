
import numpy as np

def left_predictor(row: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], row[:-1]))

def up_predictor(prev_row: np.ndarray) -> np.ndarray:
    return prev_row

def avg_predictor(row: np.ndarray, prev_row: np.ndarray) -> np.ndarray:
    left = np.concatenate(([0], row[:-1]))
    avg = ((left + prev_row) // 2).astype(np.uint8)
    return avg

def paeth_predictor(row: np.ndarray, prev_row: np.ndarray) -> np.ndarray:
    left = np.concatenate(([0], row[:-1]))
    up = prev_row
    upleft = np.concatenate(([0], prev_row[:-1]))
    p = left + up - upleft
    pa = np.abs(p - left)
    pb = np.abs(p - up)
    pc = np.abs(p - upleft)
    pr = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, up, upleft))
    return pr.astype(np.uint8)
