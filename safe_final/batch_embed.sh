#!/bin/bash
set -e

BIN_DIR="/SAFE/binaries_benign"
WORK_DIR="./workdir"
EMB_DIR="./embeddings"

mkdir -p "$WORK_DIR" "$EMB_DIR"

echo "[+] Starting batch embedding loop..."
for zfile in "$BIN_DIR"/*.z; do
  base=$(basename "$zfile" .z)
  binfile="$WORK_DIR/$base.bin"
  txtfile="$WORK_DIR/$base.txt"
  npyfile="$EMB_DIR/$base"

  echo "[*] Processing $base"

  # 1. Decompress
  ./zdecompress.py "$zfile" "$binfile"

  # 2. Disassemble with radare2
  r2 -e scr.color=false -AA -q -c "afl" -c "pdr@@f" "$binfile" > "$txtfile"

  # 3. Embed via Python
  python3 embed_text.py "$txtfile" "$npyfile"

  echo "[+] Saved embedding to $npyfile"
done

echo "[✓] Batch complete!"
