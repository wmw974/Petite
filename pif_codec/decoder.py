
import cv2
import struct
import json
import numpy as np
from typing import Dict, Any, Tuple
from .codec_core import decode_channel

def decode_pif(pif_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    if pif_bytes[:4] != b'PIF\x00':
        raise ValueError("Invalid PIF signature.")
    cursor = 4
    header_info, image_payload = None, None
    metadata = {}
    while cursor < len(pif_bytes):
        chunk_length, chunk_type_bytes = struct.unpack_from('<I4s', pif_bytes, cursor)
        chunk_type = chunk_type_bytes.decode('ascii')
        cursor += 8
        data_end = cursor + chunk_length
        chunk_data = pif_bytes[cursor:data_end]
        if chunk_type == 'IHDR': header_info = chunk_data
        elif chunk_type == 'IDAT': image_payload = chunk_data
        elif chunk_type == 'META':
            try: metadata = json.loads(chunk_data.decode('utf-8'))
            except: pass
        cursor = data_end
    if not header_info or not image_payload:
        raise ValueError("PIF file is missing essential IHDR or IDAT chunks.")
    version, scan_flags, color_model, width, height = struct.unpack('<HBBII', header_info)
    len_1, len_2, len_3, len_4 = struct.unpack('<IIII', image_payload[:16]); ip_cursor = 16
    comp_1 = image_payload[ip_cursor:ip_cursor+len_1]; ip_cursor += len_1
    comp_2 = image_payload[ip_cursor:ip_cursor+len_2]; ip_cursor += len_2
    comp_3 = image_payload[ip_cursor:ip_cursor+len_3]; ip_cursor += len_3
    comp_4 = image_payload[ip_cursor:ip_cursor+len_4]
    scan_1, scan_2, scan_3, scan_4 = scan_flags & 1, (scan_flags >> 1) & 1, (scan_flags >> 2) & 1, (scan_flags >> 3) & 1
    def _decode_and_transpose(comp_data, h, w, scan_mode):
        h_dec, w_dec = (w, h) if scan_mode == 1 else (h, w)
        decoded_plane = decode_channel(comp_data, h_dec, w_dec)
        return decoded_plane.T if scan_mode == 1 else decoded_plane
    if color_model == 0:
        b = _decode_and_transpose(comp_1, height, width, scan_1); g = _decode_and_transpose(comp_2, height, width, scan_2)
        r = _decode_and_transpose(comp_3, height, width, scan_3); a = _decode_and_transpose(comp_4, height, width, scan_4)
        final_image = cv2.merge([b, g, r, a])
    else:
        y = _decode_and_transpose(comp_1, height, width, scan_1); cr = _decode_and_transpose(comp_2, height, width, scan_2)
        cb = _decode_and_transpose(comp_3, height, width, scan_3); a = _decode_and_transpose(comp_4, height, width, scan_4)
        ycbcr = cv2.merge([y, cr, cb]); bgr = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)
        final_image = cv2.merge([bgr, a])
    return final_image, metadata
