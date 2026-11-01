"""Header parity between SOREL malware and benign files.

SOREL "disarmed" every malware binary by zeroing two PE header fields:
FileHeader.Machine and OptionalHeader.Subsystem. Benign files you collect
yourself are intact. If nothing is done, any model that sees raw bytes or
header features (MalConv, EMBER/LightGBM, the gate) can separate the classes
by those two fields alone.

Two views of every file are produced:

  model_view()    -> both fields zeroed for EVERY file (malware and benign).
                     Used for MalConv bytes and EMBER features.
  analysis_view() -> Machine restored from the optional-header magic so
                     radare2 can pick the right architecture. Used only for
                     disassembly (SAFE + call graph). Subsystem stays 0.
"""
import hashlib
import struct

IMAGE_FILE_MACHINE_I386 = 0x014C
IMAGE_FILE_MACHINE_AMD64 = 0x8664
PE32_MAGIC = 0x10B
PE32PLUS_MAGIC = 0x20B


class NotPE(ValueError):
    pass


def _offsets(b: bytes):
    if len(b) < 0x40 or b[:2] != b"MZ":
        raise NotPE("no MZ header")
    e_lfanew = struct.unpack_from("<I", b, 0x3C)[0]
    if e_lfanew + 0x18 + 0x46 > len(b) or b[e_lfanew:e_lfanew + 4] != b"PE\0\0":
        raise NotPE("no PE signature")
    machine_off = e_lfanew + 4                  # IMAGE_FILE_HEADER.Machine (u16)
    opt_off = e_lfanew + 4 + 20                 # IMAGE_OPTIONAL_HEADER
    magic = struct.unpack_from("<H", b, opt_off)[0]
    subsystem_off = opt_off + 0x44              # same offset for PE32 and PE32+
    return machine_off, subsystem_off, magic


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def is_pe(b: bytes) -> bool:
    try:
        _offsets(b)
        return True
    except NotPE:
        return False


def header_fields(b: bytes):
    m_off, s_off, magic = _offsets(b)
    return {"machine": struct.unpack_from("<H", b, m_off)[0],
            "subsystem": struct.unpack_from("<H", b, s_off)[0],
            "magic": magic}


def model_view(b: bytes) -> bytes:
    """Zero Machine and Subsystem (identical treatment for every class)."""
    m_off, s_off, _ = _offsets(b)
    out = bytearray(b)
    struct.pack_into("<H", out, m_off, 0)
    struct.pack_into("<H", out, s_off, 0)
    return bytes(out)


def analysis_view(b: bytes) -> bytes:
    """Restore Machine from the PE32 / PE32+ magic so disassemblers work."""
    m_off, _, magic = _offsets(b)
    out = bytearray(b)
    if struct.unpack_from("<H", out, m_off)[0] == 0:
        if magic == PE32PLUS_MAGIC:
            struct.pack_into("<H", out, m_off, IMAGE_FILE_MACHINE_AMD64)
        elif magic == PE32_MAGIC:
            struct.pack_into("<H", out, m_off, IMAGE_FILE_MACHINE_I386)
    return bytes(out)
