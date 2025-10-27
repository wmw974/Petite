
import cv2
import struct
import json
import zlib
from typing import Dict, Optional
import numpy as np
from .utils import create_chunk, quantize_channel
from .codec_core import encode_channel

def _get_best_scan_and_encode(channel_plane, use_all_filters):
    comp_h = zlib.compress(encode_channel(channel_plane, use_all_filters))
    comp_v = zlib.compress(encode_channel(channel_plane.T, use_all_filters))
    if len(comp_h) <= len(comp_v): return comp_h, 0
    else: return comp_v, 1

def encode_pif(image: np.ndarray, profile: Dict, metadata: Optional[Dict] = None) -> bytes:
    height, width, _ = image.shape
    image_payload, scan_flags, color_model_id = b'', 0, 0
    alpha_plane = image[:, :, 3]

    if profile['lossless']:
        b, g, r, _ = cv2.split(image)
        comp_b, scan_b = _get_best_scan_and_encode(b, True)
        comp_g, scan_g = _get_best_scan_and_encode(g, True)
        comp_r, scan_r = _get_best_scan_and_encode(r, True)
        comp_a, scan_a = _get_best_scan_and_encode(alpha_plane, True)
        scan_flags = (scan_a << 3) | (scan_r << 2) | (scan_g << 1) | scan_b
        image_payload = struct.pack('<IIII', len(comp_b), len(comp_g), len(comp_r), len(comp_a)) + comp_b + comp_g + comp_r + comp_a
    else:
        bgr = image[:, :, :3]
        ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycbcr)
        y_q = quantize_channel(y, 8, profile['Y']); cr_q = quantize_channel(cr, 8, profile['Cr'])
        cb_q = quantize_channel(cb, 8, profile['Cb']); a_q = quantize_channel(alpha_plane, 8, profile['A'])
        comp_y, scan_y = _get_best_scan_and_encode(y_q, True); comp_cr, scan_cr = _get_best_scan_and_encode(cr_q, False)
        comp_cb, scan_cb = _get_best_scan_and_encode(cb_q, False); comp_a, scan_a = _get_best_scan_and_encode(a_q, True)
        scan_flags = (scan_a << 3) | (scan_cb << 2) | (scan_cr << 1) | scan_y
        image_payload = struct.pack('<IIII', len(comp_y), len(comp_cr), len(comp_cb), len(comp_a)) + comp_y + comp_cr + comp_cb + comp_a
        color_model_id = 1
    
    header_data = struct.pack('<HBBII', 21, scan_flags, color_model_id, width, height)
    ihdr_chunk = create_chunk('IHDR', header_data)
    idat_chunk = create_chunk('IDAT', image_payload)
    
    meta_chunk = b''
    if metadata:
        meta_json_string = json.dumps(metadata, indent=2)
        meta_chunk = create_chunk('META', meta_json_string.encode('utf-8'))
        
    return b'PIF\x00' + ihdr_chunk + idat_chunk + meta_chunk
