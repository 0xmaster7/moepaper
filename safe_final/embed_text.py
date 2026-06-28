import re
import json
from safe import SAFE
from asm_embedding.FunctionNormalizer import FunctionNormalizer
from asm_embedding.InstructionsConverter import InstructionsConverter
import sys
import numpy as np
import os
def parse_r2_disasm_text(disasm_text):
    funcs = {}
    cur_addr = None
    cur_insns = []

    # matches headers like: ┌ 13913: entry0 ();
#    header_re = re.compile(r'┌.*:\s*([a-zA-Z0-9_\.]+)\s*\(\);')
    # matches instructions like: │           0x005a99a8      0bf7           or esi, edi
#    insn_re = re.compile(r'0x([0-9a-fA-F]+)\s+[0-9a-fA-F]+\s+([a-z]+)\b(.*)')
    header_re = re.compile(r'^\s*\/.*(fcn\.|sym\.|entry0|int\.)[^\s]+')
    insn_re   = re.compile(r'0x([0-9a-fA-F]+)\s+[0-9a-fA-F]+\s+([a-z]+)\b(.*)')

    for line in disasm_text.splitlines():
        h = header_re.search(line)
        if h:
            # flush previous
            if cur_addr is not None and cur_insns:
                funcs[cur_addr] = {'address': cur_addr, 'instructions': cur_insns}
            cur_insns = []
            cur_addr = None
            continue

        m = insn_re.search(line)
        if m:
            addr = int(m.group(1), 16)
            if cur_addr is None:
                cur_addr = addr
            mnemonic = m.group(2)
            operands = m.group(3).strip()
            cur_insns.append(f"{mnemonic} {operands}".strip())

    # add the last function
    if cur_addr is not None and cur_insns:
        funcs[cur_addr] = {'address': cur_addr, 'instructions': cur_insns}

    return funcs

#outside function so that less RAM is utilized and its not repeatedlty allocating space for SAFE 

conv = InstructionsConverter("data/i2v/word2id.json")
normalizer = FunctionNormalizer(max_instruction=150)
embedder = SAFE("data/safe.pb")

def embed_from_instructions(instructions, model_path="data/safe.pb"):
    """
    instructions: list of instruction strings (e.g., ['mov eax, ebx', 'add eax, 1', ...])
    returns: embedding numpy array (or whatever SAFEEmbedder returns)
    """

    converted = conv.convert_to_ids(instructions)          # list of ids
    insn_batch, length = normalizer.normalize_functions([converted])
    embedding = embedder.embedder.embedd(insn_batch, length)  # use the underlying embedder call
    return embedding

def main():
    if len(sys.argv) < 3:
        print("Usage: python embed_from_text.py <disassembly.txt> <output_dir>")
        sys.exit(1)

    disasm_path, output_root = sys.argv[1], sys.argv[2]

    # use disassembly filename as binary ID
    binary_name = os.path.splitext(os.path.basename(disasm_path))[0]
    binary_out_dir = os.path.join(output_root, binary_name)
    os.makedirs(binary_out_dir, exist_ok=True)

    text = open(disasm_path).read()
    funcs = parse_r2_disasm_text(text)

    if not funcs:
        print("[-] No functions found in disassembly text.")
        sys.exit(0)

    print(f"[i] Parsed {len(funcs)} functions from {disasm_path}")

    binary_embeddings = []   # ✅ COLLECT FUNCTION VECTORS FOR POOLING

    for addr, f in funcs.items():
        ins = f['instructions']
        if len(ins) > 500:
            continue
        try:
            emb = embed_from_instructions(ins)

            fname = f"{hex(addr)}.npy".replace("0x", "")
            np.save(os.path.join(binary_out_dir, fname), emb)

            # ✅ STORE FOR BINARY-LEVEL POOLING
            binary_embeddings.append(emb)

        except Exception as e:
            print(f"[!] Error embedding {hex(addr)}: {e}")

    # ✅ ✅ ✅ BINARY-LEVEL MEAN POOLING
    if binary_embeddings:
        binary_embeddings = np.vstack(binary_embeddings)
        binary_mean = np.mean(binary_embeddings, axis=0)

        np.save(os.path.join(binary_out_dir, "binary_embedding.npy"), binary_mean)
        print(f"[✓] Saved binary-level embedding: {binary_out_dir}/binary_embedding.npy")
    else:
        print("[-] No valid function embeddings to pool for this binary.")
if __name__ == "__main__":
    main()

