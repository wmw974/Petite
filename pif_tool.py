import argparse
import cv2
import json
import time
import numpy as np
from pif_codec.encoder import encode_pif
from pif_codec.decoder import decode_pif

# --- პროფილების განსაზღვრა ---
PROFILES = {
    "lossless":          {'name': 'Lossless', 'lossless': True},
    "visual":            {'name': 'Visually Lossless', 'lossless': False, 'Y': 7, 'Cb': 6, 'Cr': 6, 'A': 8},
    "high":              {'name': 'High Quality', 'lossless': False, 'Y': 6, 'Cb': 5, 'Cr': 5, 'A': 8},
    "compact":           {'name': 'Compact', 'lossless': False, 'Y': 4, 'Cb': 4, 'Cr': 4, 'A': 8},
}

def main():
    parser = argparse.ArgumentParser(description="PIF Image Codec Tool - v21")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- ENCODE ბრძანება ---
    parser_encode = subparsers.add_parser("encode", help="Encode an image into PIF format.")
    parser_encode.add_argument("input", help="Input image file (e.g., PNG, BMP).")
    parser_encode.add_argument("output", help="Output PIF file.")
    parser_encode.add_argument("-p", "--profile", choices=PROFILES.keys(), default="lossless", help="Encoding profile.")

    # --- DECODE ბრძანება ---
    parser_decode = subparsers.add_parser("decode", help="Decode a PIF file into a standard image format.")
    parser_decode.add_argument("input", help="Input PIF file.")
    parser_decode.add_argument("output", help="Output image file (e.g., PNG).")

    # --- INFO ბრძანება ---
    parser_info = subparsers.add_parser("info", help="Display metadata from a PIF file.")
    parser_info.add_argument("input", help="Input PIF file.")
    
    args = parser.parse_args()

    if args.command == "encode":
        print(f"Encoding '{args.input}' to '{args.output}' with profile '{args.profile}'...")
        original_image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
        if original_image is None:
            print(f"Error: Could not read input file '{args.input}'")
            return
            
        if original_image.shape[2] == 3:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)

        metadata = {
            "source_file": args.input,
            "creation_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        
        pif_bytes = encode_pif(original_image, PROFILES[args.profile], metadata)
        
        with open(args.output, "wb") as f:
            f.write(pif_bytes)
        print("✅ Encoding complete.")

    elif args.command == "decode":
        print(f"Decoding '{args.input}' to '{args.output}'...")
        with open(args.input, "rb") as f:
            pif_bytes = f.read()
        
        image, _ = decode_pif(pif_bytes)
        
        cv2.imwrite(args.output, image)
        print("✅ Decoding complete.")
        
    elif args.command == "info":
        print(f"Reading info from '{args.input}'...")
        with open(args.input, "rb") as f:
            pif_bytes = f.read()
            
        _, metadata = decode_pif(pif_bytes)
        
        if metadata:
            print(json.dumps(metadata, indent=2))
        else:
            print("No metadata found in this PIF file.")

if __name__ == "__main__":
    main()

print("ფაილი 'pif_tool.py' შეიქმნა. CLI ხელსაწყო მზად არის გამოსაყენებლად.")
