
import struct

def paeth_predictor(a, b, c):
    p = a + b - c; pa = abs(p - a); pb = abs(p - b); pc = abs(p - c)
    if pa <= pb and pa <= pc: return a
    elif pb <= pc: return b
    else: return c

def quantize_channel(channel_plane, source_bits, target_bits):
    if target_bits >= source_bits: return channel_plane
    shift = source_bits - target_bits
    return (channel_plane >> shift) << shift

def create_chunk(chunk_type: str, data: bytes) -> bytes:
    if len(chunk_type) != 4:
        raise ValueError("Chunk type must be 4 characters long.")
    length = len(data)
    return struct.pack(f'<I4s', length, chunk_type.encode('ascii')) + data
