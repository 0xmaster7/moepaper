"""One radare2 pass per binary -> SAFE instruction tokens + call graph.

Run as a subprocess (so a hung analysis can be killed):
    python r2_extract.py <file> <level:aaa|aa> <max_functions> <max_insn>
Prints one JSON object on stdout.

Why this replaces safe_final/embed_text.py
------------------------------------------
embed_text.py built instruction strings like "mov ebp, esp" from text
disassembly. SAFE's vocabulary (word2id.json) is keyed by SAFE's own
normalised form, e.g. "X_mov_ebp,_esp". InstructionsConverter therefore
mapped every such string to X_UNK, so each function became a run of identical
UNK ids and the "SAFE" embedding only encoded function length. Here the tokens
are produced with SAFE's original normalisation (filter_memory_references),
from radare2's structured JSON (`aoj`), one call per function.
"""
import json
import sys

import r2pipe

# --------------------------------------------------------------- SAFE normalisation
# Copied from safe_final/asm_embedding/FunctionAnalyzerRadare.py so tokens
# match the vocabulary SAFE was trained with.


def _filter_reg(op):
    return op["value"]


def _filter_imm(op):
    imm = int(op["value"])
    return str(hex(op["value"])) if -5000 <= imm <= 5000 else "HIMM"


def _filter_mem(op):
    if "base" not in op:
        op["base"] = 0
    if op["base"] == 0:
        return "[MEM]"
    return "[" + str(op["base"]) + "*" + str(op.get("scale", 1)) + "+" + str(op.get("disp", 0)) + "]"


def safe_token(ins, arch):
    inst = "" + ins["mnemonic"]
    ops = ins.get("opex", {}).get("operands", [])
    for op in ops:
        t = op.get("type")
        if t == "reg":
            inst += " " + _filter_reg(op)
        elif t == "imm":
            inst += " " + _filter_imm(op)
        elif t == "mem":
            inst += " " + _filter_mem(op)
        if len(ops) > 1:
            inst = inst + ","
    if "," in inst:
        inst = inst[:-1]
    inst = inst.replace(" ", "_")
    return ("A_" if arch == "arm" else "X_") + inst


# --------------------------------------------------------------- ACFG-style node attributes
CATS = {
    "transfer": {"mov", "lea", "load", "store", "xchg", "cmov", "push", "pop", "upush", "rpush", "cpush"},
    "call": {"call", "ucall", "rcall", "icall", "ircall", "ccall", "uccall"},
    "jump": {"jmp", "ujmp", "rjmp", "ijmp", "irjmp", "mjmp"},
    "cond": {"cjmp", "ucjmp", "cret", "ccall"},
    "ret": {"ret", "cret"},
    "arith": {"add", "sub", "mul", "div", "mod", "abs", "sal", "shl", "shr", "sar"},
    "logic": {"and", "or", "xor", "not", "nor", "nand", "rol", "ror"},
    "cmp": {"cmp", "acmp", "test"},
}
CAT_NAMES = list(CATS) + ["other"]


def op_category_counts(ops):
    counts = dict.fromkeys(CAT_NAMES, 0)
    for o in ops:
        t = str(o.get("type", "")).split(",")[0]
        hit = False
        for name, members in CATS.items():
            if t in members:
                counts[name] += 1
                hit = True
        if not hit:
            counts["other"] += 1
    return [counts[n] for n in CAT_NAMES]


# --------------------------------------------------------------- helpers
def _cmdj(r2, cmd):
    try:
        out = r2.cmd(cmd)
        return json.loads(out) if out and out.strip() else None
    except Exception:
        return None


def _faddr(f):
    return f.get("addr", f.get("offset"))


def _fsize(f):
    return f.get("realsz") or f.get("size") or 0


def extract(path, level="aaa", max_functions=2000, max_insn=150):
    r2 = r2pipe.open(path, flags=["-2"])
    try:
        r2.cmd("e scr.color=0")
        r2.cmd("e bin.relocs.apply=true")
        info = _cmdj(r2, "ij") or {}
        binfo = info.get("bin", {})
        arch, bits = binfo.get("arch"), binfo.get("bits")
        r2.cmd(level)
        funcs = _cmdj(r2, "aflj") or []
        funcs = [f for f in funcs if _faddr(f) is not None]
        funcs.sort(key=_faddr)
        funcs = funcs[:max_functions]

        out_funcs = []
        for f in funcs:
            addr, size = _faddr(f), _fsize(f)
            n_req = max(1, min(max_insn, int(f.get("ninstrs") or max_insn)))
            ops = _cmdj(r2, f"aoj {n_req} @ {addr}") or []
            toks, kept = [], []
            for o in ops:
                if size and o.get("addr", addr) >= addr + size:
                    break
                kept.append(o)
                try:
                    toks.append(safe_token(o, arch))
                except Exception:
                    toks.append("X_UNK")
            out_funcs.append({
                "addr": addr, "name": f.get("name", hex(addr)), "size": size,
                "ninstrs": int(f.get("ninstrs") or len(kept)),
                "tokens": toks, "cats": op_category_counts(kept),
            })

        # call graph: agCj (r2 <=5.9: list of {name, imports}; newer: {nodes, edges})
        edges = []
        cg = _cmdj(r2, "agCj")
        if isinstance(cg, list):
            for n in cg:
                for callee in n.get("imports", []) or []:
                    edges.append([n.get("name"), callee])
        elif isinstance(cg, dict) and "nodes" in cg:
            id2name = {n.get("id"): n.get("title", n.get("name")) for n in cg.get("nodes", [])}
            for n in cg["nodes"]:
                for t in n.get("out_nodes", []) or []:
                    edges.append([id2name.get(n.get("id")), id2name.get(t)])
        if not edges:   # fallback: per-function callrefs
            addr2name = {_faddr(f): f.get("name") for f in funcs}
            for f in funcs:
                for c in f.get("callrefs", []) or []:
                    if str(c.get("type", "")).upper().startswith("C"):
                        edges.append([f.get("name"), addr2name.get(c.get("addr"), hex(c.get("addr", 0)))])
        edges = [e for e in edges if e[0] and e[1]]
        return {"ok": True, "arch": arch, "bits": bits, "level": level,
                "functions": out_funcs, "cg_edges": edges}
    finally:
        try:
            r2.quit()
        except Exception:
            pass


if __name__ == "__main__":
    p, lvl, mf, mi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    try:
        res = extract(p, lvl, mf, mi)
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    sys.stdout.write(json.dumps(res))
