
import numpy as np
import zlib
from .utils import paeth_predictor # ვიყენებთ რელატიურ იმპორტს

def encode_channel(channel_plane, use_all_filters=True):
    height, width = channel_plane.shape
    filtered_data = bytearray()
    prior_scanline = np.zeros(width, dtype=np.uint8)
    prior_prior_scanline = np.zeros(width, dtype=np.uint8)
    filter_range = range(9) if use_all_filters else [4]
    for y in range(height):
        scanline = channel_plane[y]
        if 8 in filter_range and scanline.size > 0 and np.all(scanline == scanline[0]):
            filtered_data.append(8); filtered_data.append(scanline[0])
            prior_prior_scanline, prior_scanline = prior_scanline, scanline.copy()
            continue
        residuals = {}; costs = {}
        for filter_type in filter_range:
            if filter_type >= 7: continue
            filtered_scanline = np.zeros_like(scanline)
            for i in range(width):
                s_val, left, up, upper_left = int(scanline[i]), int(scanline[i-1]) if i > 0 else 0, int(prior_scanline[i]), int(prior_scanline[i-1]) if i > 0 else 0
                pred = 0
                if filter_type == 0: pred = 0
                elif filter_type == 1: pred = left
                elif filter_type == 2: pred = up
                elif filter_type == 3: pred = (left + up) // 2
                elif filter_type == 4: pred = paeth_predictor(left, up, upper_left)
                elif filter_type == 5: pred = left + up - upper_left
                elif filter_type == 6:
                    knight_up_left = int(prior_prior_scanline[i-1]) if y >= 2 and i >= 1 else 0
                    knight_left_up = int(prior_scanline[i-2]) if y >= 1 and i >= 2 else 0
                    pred = (knight_up_left + knight_left_up) // 2
                filtered_scanline[i] = (s_val - pred) & 0xFF
            residuals[filter_type] = filtered_scanline
            costs[filter_type] = np.sum(np.abs(filtered_scanline.astype(np.int8)))
        if 7 in filter_range:
            line_copy_residual = (scanline.astype(np.int16) - prior_scanline.astype(np.int16)).astype(np.int8)
            residuals[7] = line_copy_residual
            costs[7] = np.sum(np.abs(line_copy_residual))
        best_filter_id = min(costs, key=costs.get)
        filtered_data.append(best_filter_id); filtered_data.extend(residuals[best_filter_id].tobytes())
        prior_prior_scanline, prior_scanline = prior_scanline, scanline.copy()
    return bytes(filtered_data)

def decode_channel(compressed_data, height, width):
    filtered_data = zlib.decompress(compressed_data)
    canvas = np.zeros((height, width), dtype=np.uint8)
    prior_scanline = np.zeros(width, dtype=np.uint8)
    prior_prior_scanline = np.zeros(width, dtype=np.uint8)
    cursor = 0
    for y in range(height):
        filter_type = filtered_data[cursor]; cursor += 1
        reconstructed_scanline = np.zeros(width, dtype=np.uint8)
        if filter_type == 8:
            color_val = filtered_data[cursor]; cursor += 1
            reconstructed_scanline.fill(color_val)
        elif filter_type == 7:
            residuals = np.frombuffer(filtered_data[cursor:cursor+width], dtype=np.int8); cursor += width
            reconstructed_scanline = (prior_scanline.astype(np.int16) + residuals).clip(0, 255).astype(np.uint8)
        else:
            residuals = np.frombuffer(filtered_data[cursor:cursor+width], dtype=np.uint8); cursor += width
            for i in range(width):
                left, up, upper_left = int(reconstructed_scanline[i-1]) if i > 0 else 0, int(prior_scanline[i]), int(prior_scanline[i-1]) if i > 0 else 0
                residual = int(residuals[i]);
                if residual > 127: residual -= 256
                pred = 0
                if filter_type == 0: pred = 0
                elif filter_type == 1: pred = left
                elif filter_type == 2: pred = up
                elif filter_type == 3: pred = (left + up) // 2
                elif filter_type == 4: pred = paeth_predictor(left, up, upper_left)
                elif filter_type == 5: pred = left + up - upper_left
                elif filter_type == 6:
                    knight_up_left = int(prior_prior_scanline[i-1]) if y >= 2 and i >= 1 else 0
                    knight_left_up = int(prior_scanline[i-2]) if y >= 1 and i >= 2 else 0
                    pred = (knight_up_left + knight_left_up) // 2
                reconstructed_scanline[i] = (pred + residual) & 0xFF
        canvas[y] = reconstructed_scanline
        prior_prior_scanline, prior_scanline = prior_scanline, reconstructed_scanline.copy()
    return canvas
