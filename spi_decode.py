import sys
import struct
import argparse
from pathlib import Path

try:
    from unicorn import Uc, UC_ARCH_ARM, UC_MODE_THUMB, UC_HOOK_CODE, UC_MODE_ARM
    from unicorn.arm_const import *
except ImportError:
    sys.exit("Missing dependency: pip install unicorn")

# ---------------------------------------------------------------------------
# Memory map (arbitrary but non-overlapping; each lib gets 256 MB of room)
# ---------------------------------------------------------------------------
BASE_SKIA  = 0x10000000
BASE_MAET  = 0x20000000
BASE_SXQK  = 0x30000000
HEAP_BASE  = 0x40000000
HEAP_SIZE  = 0x00A00000          # 10 MB bump-allocated heap
STUB_BASE  = 0x50000000          # trampolines for stubbed/unresolved symbols
STUB_SLOT  = 0x40                # bytes reserved per stub
STUB_PAGES = 2048                # 2048*4KB = 8 MB of trampoline space (~32k stubs) -
                                  # libskia.so alone drags in ~450+ unique unresolved
                                  # externs (freetype/jpeg/other codecs we never call),
                                  # so this needs real headroom, not 64 pages.
STACK_BASE = 0x60000000
STACK_SIZE = 0x00100000
RETURN_ADDR = 0x6FFFFFF0         # magic "stop here" address for top-level calls

PAGE = 0x1000


def align_up(x, a=PAGE):
    return (x + a - 1) & ~(a - 1)


class ElfImage:
    def __init__(self, path, base):
        self.path = path
        self.base = base
        self.data = Path(path).read_bytes()
        self.symbols = {}          # name -> absolute addr (defined here)
        self.relocs = []           # list of (abs_reloc_addr, type, sym_name, addend0)
        self._parse()

    def _parse(self):
        d = self.data
        if d[:4] != b'\x7fELF':
            raise ValueError(f"{self.path}: not an ELF file")
        ei_class = d[4]
        if ei_class != 1:
            raise ValueError(f"{self.path}: only ELF32 supported")
        (e_type, e_machine, e_version, e_entry, e_phoff, e_shoff, e_flags,
         e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum,
         e_shstrndx) = struct.unpack_from('<HHIIIIIHHHHHH', d, 16)

        # ---- program headers: just need PT_LOAD extents (for info only) ----
        self.loads = []
        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, p_align = \
                struct.unpack_from('<IIIIIIII', d, off)
            if p_type == 1:  # PT_LOAD
                self.loads.append((p_offset, p_vaddr, p_filesz, p_memsz))

        # ---- section headers: need .dynsym/.dynstr/.rel.dyn/.rel.plt ----
        sh = []
        for i in range(e_shnum):
            off = e_shoff + i * e_shentsize
            sh.append(struct.unpack_from('<IIIIIIIIII', d, off))
        shstrtab_off = sh[e_shstrndx][4]

        def sh_name(entry):
            noff = entry[0]
            end = d.index(b'\x00', shstrtab_off + noff)
            return d[shstrtab_off + noff:end].decode()

        dynsym = dynstr = None
        rel_sections = []
        for entry in sh:
            name = sh_name(entry)
            sh_type = entry[1]
            sh_offset = entry[4]
            sh_size = entry[5]
            if name == '.dynsym':
                dynsym = (sh_offset, sh_size)
            elif name == '.dynstr':
                dynstr = (sh_offset, sh_size)
            elif name in ('.rel.dyn', '.rel.plt'):
                rel_sections.append((sh_offset, sh_size))

        if not dynsym or not dynstr:
            raise ValueError(f"{self.path}: no .dynsym/.dynstr")

        dynstr_off, dynstr_size = dynstr
        dynsym_off, dynsym_size = dynsym
        n_syms = dynsym_size // 16

        self._symtab = []  # index -> (name, value, shndx)
        for i in range(n_syms):
            off = dynsym_off + i * 16
            st_name, st_value, st_size, st_info, st_other, st_shndx = \
                struct.unpack_from('<IIIBBH', d, off)
            end = d.index(b'\x00', dynstr_off + st_name)
            name = d[dynstr_off + st_name:end].decode()
            self._symtab.append((name, st_value, st_shndx))
            if name and st_shndx != 0:  # defined (not UND)
                self.symbols[name] = self.base + st_value

        for roff, rsize in rel_sections:
            n = rsize // 8
            for i in range(n):
                r_offset, r_info = struct.unpack_from('<II', d, roff + i * 8)
                r_sym = r_info >> 8
                r_type = r_info & 0xff
                sym_name = self._symtab[r_sym][0] if r_sym < len(self._symtab) else ''
                self.relocs.append((self.base + r_offset, r_type, sym_name))

    def map_into(self, mu):
        total = 0
        for _, vaddr, filesz, memsz in self.loads:
            total = max(total, vaddr + memsz)
        size = align_up(total)
        mu.mem_map(self.base, size)
        for offset, vaddr, filesz, memsz in self.loads:
            if filesz:
                mu.mem_write(self.base + vaddr, self.data[offset:offset + filesz])
        return size


# R_ARM relocation types we care about
R_ARM_ABS32 = 2
R_ARM_GLOB_DAT = 21
R_ARM_JUMP_SLOT = 22
R_ARM_RELATIVE = 23


class Emulator:
    def __init__(self, libskia, libmaet, libsxqk):
        self.mu = Uc(UC_ARCH_ARM, UC_MODE_ARM)
        self._enable_vfp_neon()
        self.mu.mem_map(STACK_BASE, STACK_SIZE)
        self.mu.mem_map(HEAP_BASE, HEAP_SIZE)
        self.mu.mem_map(STUB_BASE, PAGE * STUB_PAGES)
        self.mu.mem_map(RETURN_ADDR & ~0xFFF, PAGE)  # page holding the "stop" address
        self.heap_cursor = HEAP_BASE
        self.stub_cursor = STUB_BASE
        self.stub_names = {}   # addr -> name (for error messages)
        self.py_stubs = {}     # addr -> python callable(mu) -> None (must set r0/pc itself or just return, we auto-return)
        self._named_stub_cache = {}  # name -> addr, so repeated unresolved refs share one trampoline

        self.images = {
            'skia': ElfImage(libskia, BASE_SKIA),
            'maet': ElfImage(libmaet, BASE_MAET),
            'sxqk': ElfImage(libsxqk, BASE_SXQK),
        }
        for img in self.images.values():
            img.map_into(self.mu)

        self.global_syms = {}
        for img in self.images.values():
            self.global_syms.update(img.symbols)

        self._install_libc_stubs()
        self._apply_all_relocations()

        self.mu.hook_add(UC_HOOK_CODE, self._trap_hook, begin=STUB_BASE, end=STUB_BASE + PAGE * STUB_PAGES)
        self.mu.hook_add(UC_HOOK_CODE, self._stop_hook, begin=RETURN_ADDR, end=RETURN_ADDR + 4)
        self._stop = False

    def _enable_vfp_neon(self):
        # Unicorn/QEMU boots the ARM core with the VFP/NEON coprocessor access
        # disabled, so any VFP/NEON instruction (very likely used by hand-tuned
        # routines like sxqk_mset_x64a / sxqk_mcpy_blk / the SAD/diff functions)
        # faults as UC_ERR_INSN_INVALID even though the opcode is perfectly
        # valid. Standard fix: grant full access to coprocessors 10/11 via
        # CPACR (here exposed as the c1_c0_2 pseudo-register), then set the
        # EN bit in FPEXC so the FPU itself is enabled.
        try:
            cpacr = self.mu.reg_read(UC_ARM_REG_C1_C0_2)
            cpacr |= (0xF << 20)  # full access to cp10 + cp11
            self.mu.reg_write(UC_ARM_REG_C1_C0_2, cpacr)
            self.mu.reg_write(UC_ARM_REG_FPEXC, 0x40000000)  # FPEXC.EN
        except Exception as e:
            print(f"[!] warning: could not enable VFP/NEON ({e}) — "
                  f"any float/SIMD instruction will crash as 'invalid instruction'")

    # -- bump allocator -----------------------------------------------------
    def malloc(self, size):
        size = max(8, (size + 7) & ~7)
        addr = self.heap_cursor
        if addr + size > HEAP_BASE + HEAP_SIZE:
            raise MemoryError("emulated heap exhausted, raise HEAP_SIZE")
        self.heap_cursor += size
        return addr

    # -- stub trampolines -----------------------------------------------------
    def _new_stub(self, name, pyfunc=None):
        # De-dupe by name: dozens of relocations across 3 huge libs point at the
        # same missing symbol (e.g. every unresolved FreeType/jpeg call in
        # libskia.so) - giving each its own trampoline is what blew the old
        # fixed-size stub region. Python-backed stubs (libc etc.) are always
        # unique per name anyway, so this is safe to share.
        if pyfunc is None and name in self._named_stub_cache:
            return self._named_stub_cache[name]
        addr = self.stub_cursor
        self.stub_cursor += STUB_SLOT
        if self.stub_cursor > STUB_BASE + PAGE * STUB_PAGES:
            raise MemoryError(
                f"stub trampoline area exhausted after {len(self.stub_names)} "
                f"stubs — raise STUB_PAGES")
        # bx lr  (E12FFF1E)  -- ARM mode
        self.mu.mem_write(addr, struct.pack('<I', 0xE12FFF1E))
        self.stub_names[addr] = name
        if pyfunc:
            self.py_stubs[addr] = pyfunc
        else:
            self._named_stub_cache[name] = addr
        return addr

    def _trap_hook(self, mu, address, size, user_data):
        if address in self.py_stubs:
            self.py_stubs[address](mu, self)
            return
        name = self.stub_names.get(address, '?')
        raise RuntimeError(f"unresolved symbol: {name} was actually called (addr=0x{address:x})")

    def _stop_hook(self, mu, address, size, user_data):
        self._stop = True
        mu.emu_stop()

    # -- libc-ish stubs -------------------------------------------------------
    def _install_libc_stubs(self):
        def s_malloc(mu, emu):
            size = mu.reg_read(UC_ARM_REG_R0)
            mu.reg_write(UC_ARM_REG_R0, emu.malloc(size))

        def s_free(mu, emu):
            pass  # bump allocator: no-op

        def s_memset(mu, emu):
            ptr = mu.reg_read(UC_ARM_REG_R0)
            val = mu.reg_read(UC_ARM_REG_R1) & 0xff
            n = mu.reg_read(UC_ARM_REG_R2)
            if n:
                mu.mem_write(ptr, bytes([val]) * n)
            mu.reg_write(UC_ARM_REG_R0, ptr)

        def s_memcpy(mu, emu):
            dst = mu.reg_read(UC_ARM_REG_R0)
            src = mu.reg_read(UC_ARM_REG_R1)
            n = mu.reg_read(UC_ARM_REG_R2)
            if n:
                mu.mem_write(dst, bytes(mu.mem_read(src, n)))
            mu.reg_write(UC_ARM_REG_R0, dst)

        def s_memcmp(mu, emu):
            a = mu.reg_read(UC_ARM_REG_R0)
            b = mu.reg_read(UC_ARM_REG_R1)
            n = mu.reg_read(UC_ARM_REG_R2)
            ba = bytes(mu.mem_read(a, n)) if n else b''
            bb = bytes(mu.mem_read(b, n)) if n else b''
            if ba < bb:
                res = -1
            elif ba > bb:
                res = 1
            else:
                res = 0
            mu.reg_write(UC_ARM_REG_R0, res & 0xffffffff)

        def s_strlen(mu, emu):
            ptr = mu.reg_read(UC_ARM_REG_R0)
            n = 0
            while bytes(mu.mem_read(ptr + n, 1)) != b'\x00':
                n += 1
                if n > 1 << 20:  # safety valve against runaway reads
                    break
            mu.reg_write(UC_ARM_REG_R0, n)

        def s_strcmp(mu, emu):
            a = mu.reg_read(UC_ARM_REG_R0)
            b = mu.reg_read(UC_ARM_REG_R1)
            sa, sb = bytearray(), bytearray()
            for ptr, buf in ((a, sa), (b, sb)):
                i = 0
                while True:
                    c = bytes(mu.mem_read(ptr + i, 1))
                    if c == b'\x00' or i > (1 << 20):
                        break
                    buf += c
                    i += 1
            res = -1 if bytes(sa) < bytes(sb) else (1 if bytes(sa) > bytes(sb) else 0)
            mu.reg_write(UC_ARM_REG_R0, res & 0xffffffff)

        def s_puts(mu, emu):
            ptr = mu.reg_read(UC_ARM_REG_R0)
            s = b''
            while True:
                b = mu.mem_read(ptr, 1)
                if b == b'\x00':
                    break
                s += bytes(b)
                ptr += 1
            print("[guest puts]", s.decode(errors='replace'))
            mu.reg_write(UC_ARM_REG_R0, 0)

        def s_idiv(mu, emu):
            a = struct.unpack('<i', struct.pack('<I', mu.reg_read(UC_ARM_REG_R0)))[0]
            b = struct.unpack('<i', struct.pack('<I', mu.reg_read(UC_ARM_REG_R1)))[0]
            q = int(a / b) if b else 0  # C-style truncation toward zero, not Python //
            mu.reg_write(UC_ARM_REG_R0, q & 0xffffffff)

        def s_idivmod(mu, emu):
            a = struct.unpack('<i', struct.pack('<I', mu.reg_read(UC_ARM_REG_R0)))[0]
            b = struct.unpack('<i', struct.pack('<I', mu.reg_read(UC_ARM_REG_R1)))[0]
            q = int(a / b) if b else 0
            r = (a - q * b) if b else 0
            mu.reg_write(UC_ARM_REG_R0, q & 0xffffffff)
            mu.reg_write(UC_ARM_REG_R1, r & 0xffffffff)

        def s_sqrt(mu, emu):
            import math
            # NOTE: real ABI passes double in r0:r1 (softfp) or d0 (hardfp).
            # This is a best-effort stub; if colors/sizes look wrong this is
            # the first place to fix (dump both calling conventions).
            lo = mu.reg_read(UC_ARM_REG_R0)
            hi = mu.reg_read(UC_ARM_REG_R1)
            val = struct.unpack('<d', struct.pack('<II', lo, hi))[0]
            r = math.sqrt(max(0.0, val))
            packed = struct.unpack('<II', struct.pack('<d', r))
            mu.reg_write(UC_ARM_REG_R0, packed[0])
            mu.reg_write(UC_ARM_REG_R1, packed[1])

        def s_noop_ok(mu, emu):
            mu.reg_write(UC_ARM_REG_R0, 0)

        def s_noop_one(mu, emu):
            mu.reg_write(UC_ARM_REG_R0, 1)

        def s_op_new(mu, emu):
            # operator new(unsigned int) -- r0 = size -> r0 = ptr (never NULL, ok
            # for our bump allocator; real one would throw std::bad_alloc on failure)
            size = mu.reg_read(UC_ARM_REG_R0)
            mu.reg_write(UC_ARM_REG_R0, emu.malloc(size if size else 1))

        def s_op_delete(mu, emu):
            pass  # bump allocator: no-op, same as free()

        def s_pure_virtual(mu, emu):
            raise RuntimeError("__cxa_pure_virtual called - a vtable slot we "
                                "left pointing at a trap got invoked for real; "
                                "this means some virtual method IS used and "
                                "needs a real implementation, not a stub")

        def u64(lo, hi):
            return (hi << 32) | lo

        def split64(v):
            v &= 0xFFFFFFFFFFFFFFFF
            return v & 0xffffffff, (v >> 32) & 0xffffffff

        def s_idiv0(mu, emu):
            mu.reg_write(UC_ARM_REG_R0, 0)  # GCC's div-by-zero trap: just return 0

        def s_aeabi_atexit(mu, emu):
            mu.reg_write(UC_ARM_REG_R0, 0)

        def s_llsl(mu, emu):  # long long logical shift left: r0:r1 << r2
            v = u64(mu.reg_read(UC_ARM_REG_R0), mu.reg_read(UC_ARM_REG_R1))
            shift = mu.reg_read(UC_ARM_REG_R2) & 63
            lo, hi = split64(v << shift)
            mu.reg_write(UC_ARM_REG_R0, lo); mu.reg_write(UC_ARM_REG_R1, hi)

        def s_llsr(mu, emu):  # long long logical shift right
            v = u64(mu.reg_read(UC_ARM_REG_R0), mu.reg_read(UC_ARM_REG_R1))
            shift = mu.reg_read(UC_ARM_REG_R2) & 63
            lo, hi = split64(v >> shift)
            mu.reg_write(UC_ARM_REG_R0, lo); mu.reg_write(UC_ARM_REG_R1, hi)

        def s_l2d(mu, emu):  # (long long r0:r1 signed) -> double, returned in r0:r1
            v = u64(mu.reg_read(UC_ARM_REG_R0), mu.reg_read(UC_ARM_REG_R1))
            sv = v - (1 << 64) if v & (1 << 63) else v
            lo, hi = split64(struct.unpack('<Q', struct.pack('<d', float(sv)))[0])
            mu.reg_write(UC_ARM_REG_R0, lo); mu.reg_write(UC_ARM_REG_R1, hi)

        def s_ul2d(mu, emu):  # (unsigned long long) -> double
            v = u64(mu.reg_read(UC_ARM_REG_R0), mu.reg_read(UC_ARM_REG_R1))
            lo, hi = split64(struct.unpack('<Q', struct.pack('<d', float(v)))[0])
            mu.reg_write(UC_ARM_REG_R0, lo); mu.reg_write(UC_ARM_REG_R1, hi)

        def s_d2ulz(mu, emu):  # double (r0:r1) -> unsigned long long, truncate toward 0
            bits = u64(mu.reg_read(UC_ARM_REG_R0), mu.reg_read(UC_ARM_REG_R1))
            d = struct.unpack('<d', struct.pack('<Q', bits))[0]
            iv = int(d) if d == d else 0  # NaN guard
            iv = max(0, iv) & 0xFFFFFFFFFFFFFFFF
            lo, hi = split64(iv)
            mu.reg_write(UC_ARM_REG_R0, lo); mu.reg_write(UC_ARM_REG_R1, hi)

        def s_f2lz(mu, emu):  # float (r0) -> long long, truncate toward 0
            f = struct.unpack('<f', struct.pack('<I', mu.reg_read(UC_ARM_REG_R0)))[0]
            iv = int(f) if f == f else 0
            lo, hi = split64(iv)
            mu.reg_write(UC_ARM_REG_R0, lo); mu.reg_write(UC_ARM_REG_R1, hi)

        def s_ldivmod(mu, emu):  # signed 64-bit a/b -> quotient in r0:r1, remainder via stack (rare path)
            a = u64(mu.reg_read(UC_ARM_REG_R0), mu.reg_read(UC_ARM_REG_R1))
            b = u64(mu.reg_read(UC_ARM_REG_R2), mu.reg_read(UC_ARM_REG_R3))
            sa = a - (1 << 64) if a & (1 << 63) else a
            sb = b - (1 << 64) if b & (1 << 63) else b
            q = int(sa / sb) if sb else 0
            lo, hi = split64(q)
            mu.reg_write(UC_ARM_REG_R0, lo); mu.reg_write(UC_ARM_REG_R1, hi)

        def s_uldivmod(mu, emu):
            a = u64(mu.reg_read(UC_ARM_REG_R0), mu.reg_read(UC_ARM_REG_R1))
            b = u64(mu.reg_read(UC_ARM_REG_R2), mu.reg_read(UC_ARM_REG_R3))
            q = (a // b) if b else 0
            lo, hi = split64(q)
            mu.reg_write(UC_ARM_REG_R0, lo); mu.reg_write(UC_ARM_REG_R1, hi)

        def s_calloc(mu, emu):
            nmemb = mu.reg_read(UC_ARM_REG_R0)
            size = mu.reg_read(UC_ARM_REG_R1)
            total = nmemb * size
            addr = emu.malloc(total if total else 1)
            if total:
                mu.mem_write(addr, b'\x00' * total)
            mu.reg_write(UC_ARM_REG_R0, addr)

        def s_realloc(mu, emu):
            # bump allocator: just allocate new + copy old contents. We don't
            # track allocation sizes, so copy a generous guess; harmless if it
            # over-reads a little since it's all inside our own heap region.
            old_ptr = mu.reg_read(UC_ARM_REG_R0)
            new_size = mu.reg_read(UC_ARM_REG_R1)
            new_ptr = emu.malloc(new_size if new_size else 1)
            if old_ptr:
                copy_n = min(new_size, 4096)
                try:
                    mu.mem_write(new_ptr, bytes(mu.mem_read(old_ptr, copy_n)))
                except Exception:
                    pass
            mu.reg_write(UC_ARM_REG_R0, new_ptr)

        def s_abort(mu, emu):
            raise RuntimeError("guest code called abort()")

        def s_strchr(mu, emu):
            ptr = mu.reg_read(UC_ARM_REG_R0)
            ch = mu.reg_read(UC_ARM_REG_R1) & 0xff
            i = 0
            while True:
                c = bytes(mu.mem_read(ptr + i, 1))[0]
                if c == ch:
                    mu.reg_write(UC_ARM_REG_R0, ptr + i)
                    return
                if c == 0:
                    mu.reg_write(UC_ARM_REG_R0, 0)
                    return
                i += 1

        def s_strncmp(mu, emu):
            a = mu.reg_read(UC_ARM_REG_R0)
            b = mu.reg_read(UC_ARM_REG_R1)
            n = mu.reg_read(UC_ARM_REG_R2)
            ba = bytes(mu.mem_read(a, n)).split(b'\x00', 1)[0]
            bb = bytes(mu.mem_read(b, n)).split(b'\x00', 1)[0]
            res = -1 if ba < bb else (1 if ba > bb else 0)
            mu.reg_write(UC_ARM_REG_R0, res & 0xffffffff)

        def s_strcpy(mu, emu):
            dst = mu.reg_read(UC_ARM_REG_R0)
            src = mu.reg_read(UC_ARM_REG_R1)
            i = 0
            while True:
                b = mu.mem_read(src + i, 1)
                mu.mem_write(dst + i, bytes(b))
                if bytes(b) == b'\x00':
                    break
                i += 1
            mu.reg_write(UC_ARM_REG_R0, dst)

        def make_double_math_stub(pyfunc):
            def stub(mu, emu):
                lo, hi = mu.reg_read(UC_ARM_REG_R0), mu.reg_read(UC_ARM_REG_R1)
                v = struct.unpack('<d', struct.pack('<II', lo, hi))[0]
                r = pyfunc(v)
                rlo, rhi = struct.unpack('<II', struct.pack('<d', r))
                mu.reg_write(UC_ARM_REG_R0, rlo); mu.reg_write(UC_ARM_REG_R1, rhi)
            return stub

        def make_float_math_stub(pyfunc):
            def stub(mu, emu):
                v = struct.unpack('<f', struct.pack('<I', mu.reg_read(UC_ARM_REG_R0)))[0]
                r = pyfunc(v)
                mu.reg_write(UC_ARM_REG_R0, struct.unpack('<I', struct.pack('<f', r))[0])
            return stub

        import math

        def s_memmove(mu, emu):
            dst = mu.reg_read(UC_ARM_REG_R0)
            src = mu.reg_read(UC_ARM_REG_R1)
            n = mu.reg_read(UC_ARM_REG_R2)
            if n:
                mu.mem_write(dst, bytes(mu.mem_read(src, n)))
            mu.reg_write(UC_ARM_REG_R0, dst)

        libc_impl = {
            'malloc': s_malloc, 'free': s_free, 'memset': s_memset, 'memcpy': s_memcpy,
            'memmove': s_memmove, 'memcmp': s_memcmp, 'strlen': s_strlen, 'strcmp': s_strcmp,
            'puts': s_puts, '__aeabi_idiv': s_idiv, '__aeabi_idivmod': s_idivmod,
            '__aeabi_idiv0': s_idiv0, '__aeabi_atexit': s_aeabi_atexit,
            '__aeabi_llsl': s_llsl, '__aeabi_llsr': s_llsr,
            '__aeabi_l2d': s_l2d, '__aeabi_ul2d': s_ul2d,
            '__aeabi_d2ulz': s_d2ulz, '__aeabi_f2lz': s_f2lz,
            '__aeabi_ldivmod': s_ldivmod, '__aeabi_uldivmod': s_uldivmod,
            'sqrt': s_sqrt,
            '__stack_chk_fail': s_noop_ok, '__cxa_atexit': s_noop_ok,
            '__cxa_finalize': s_noop_ok, '__aeabi_unwind_cpp_pr0': s_noop_ok,
            '__aeabi_unwind_cpp_pr1': s_noop_ok,
            # C++ runtime (mangled names -- these are what "new"/"delete" compile to)
            '_Znwj': s_op_new,           # operator new(unsigned int)
            '_Znaj': s_op_new,           # operator new[](unsigned int)
            '_ZdlPv': s_op_delete,       # operator delete(void*)
            '_ZdaPv': s_op_delete,       # operator delete[](void*)
            '_ZdlPvj': s_op_delete,      # operator delete(void*, unsigned int) (sized delete)
            '_ZdaPvj': s_op_delete,      # operator delete[](void*, unsigned int)
            '__cxa_guard_acquire': s_noop_one,   # "not yet initialized, go ahead"
            '__cxa_guard_release': s_noop_ok,
            '__cxa_guard_abort': s_noop_ok,
            '__cxa_pure_virtual': s_pure_virtual,
            '__gnu_Unwind_Find_exidx': s_noop_ok,
            # pthread_*: everything here runs on a single emulated "thread", so
            # every lock/condvar/etc. is a safe no-op that always "succeeds"
            # immediately. This unblocks the lazy static-initializer pattern
            # Skia/Android code loves (guarded singleton init on first touch).
            'pthread_mutex_init': s_noop_ok, 'pthread_mutex_lock': s_noop_ok,
            'pthread_mutex_unlock': s_noop_ok, 'pthread_mutex_destroy': s_noop_ok,
            'pthread_mutex_trylock': s_noop_ok,
            'pthread_cond_init': s_noop_ok, 'pthread_cond_destroy': s_noop_ok,
            'pthread_cond_wait': s_noop_ok, 'pthread_cond_timedwait': s_noop_ok,
            'pthread_cond_signal': s_noop_ok, 'pthread_cond_broadcast': s_noop_ok,
            'pthread_key_create': s_noop_ok, 'pthread_key_delete': s_noop_ok,
            'pthread_getspecific': s_noop_ok, 'pthread_setspecific': s_noop_ok,
            'pthread_self': s_noop_ok, 'pthread_equal': s_noop_ok,
            'pthread_attr_init': s_noop_ok, 'pthread_attr_destroy': s_noop_ok,
            'pthread_attr_setdetachstate': s_noop_ok, 'pthread_attr_getdetachstate': s_noop_ok,
            'pthread_attr_setstacksize': s_noop_ok, 'pthread_attr_getstacksize': s_noop_ok,
            'pthread_attr_setschedpolicy': s_noop_ok, 'pthread_attr_setschedparam': s_noop_ok,
            # pthread_create is the important one: this is EmojiFont spawning a
            # background loader thread, which is completely irrelevant to SPI
            # decoding. We do NOT actually run the thread function (real thread
            # emulation is a different, much bigger problem than this project
            # needs) - just report success and move on synchronously.
            'pthread_create': s_noop_ok,
            'pthread_join': s_noop_ok, 'pthread_detach': s_noop_ok,
            'pthread_exit': s_noop_ok, 'pthread_cancel': s_noop_ok,
        }
        self.libc_stub_addrs = {name: self._new_stub(name, fn) for name, fn in libc_impl.items()}

        # pthread_once needs to actually invoke the init callback (skipping it
        # can leave real global state unset up and cause a confusing crash much
        # later). We can't safely nest emu_start from inside a hook, so instead
        # we redirect the CPU: on entry, remember the real caller's LR, then
        # jump PC straight into the init function with LR pointed at a small
        # "finish" trampoline; when the init function returns, that trampoline
        # restores the real return address. No recursion, single continuous run.
        self._once_return_stack = []

        def s_pthread_once_finish(mu, emu):
            orig_lr = emu._once_return_stack.pop()
            mu.reg_write(UC_ARM_REG_R0, 0)
            mu.reg_write(UC_ARM_REG_PC, orig_lr)

        finish_addr = self._new_stub('pthread_once$finish', s_pthread_once_finish)

        def s_pthread_once(mu, emu):
            once_ctrl = mu.reg_read(UC_ARM_REG_R0)
            init_func = mu.reg_read(UC_ARM_REG_R1)
            already = bytes(mu.mem_read(once_ctrl, 1))[0]
            if already:
                mu.reg_write(UC_ARM_REG_R0, 0)
                return  # falls through to this stub's own bx lr, normal return
            mu.mem_write(once_ctrl, bytes([1]))
            emu._once_return_stack.append(mu.reg_read(UC_ARM_REG_LR))
            mu.reg_write(UC_ARM_REG_LR, finish_addr)
            mu.reg_write(UC_ARM_REG_PC, init_func)  # thumb bit already baked into the fn ptr

        self.libc_stub_addrs['pthread_once'] = self._new_stub('pthread_once', s_pthread_once)
        # __stack_chk_guard is a *data* symbol some code reads, not calls
        guard_addr = self.malloc(4)
        self.mu.mem_write(guard_addr, struct.pack('<I', 0xDEADBEEF))
        self.data_stub_addrs = {'__stack_chk_guard': guard_addr}

    def _resolve(self, name):
        if name in self.global_syms:
            return self.global_syms[name]
        if name in self.libc_stub_addrs:
            return self.libc_stub_addrs[name]
        if name in self.data_stub_addrs:
            return self.data_stub_addrs[name]
        # unknown symbol: point at a trap stub that raises with its real name
        return self._new_stub(name or '<unnamed>')

    def _apply_all_relocations(self):
        for img in self.images.values():
            for addr, rtype, sym_name in img.relocs:
                if rtype == R_ARM_RELATIVE:
                    old = struct.unpack('<I', self.mu.mem_read(addr, 4))[0]
                    self.mu.mem_write(addr, struct.pack('<I', (old + img.base) & 0xffffffff))
                elif rtype in (R_ARM_ABS32, R_ARM_GLOB_DAT, R_ARM_JUMP_SLOT):
                    target = self._resolve(sym_name)
                    self.mu.mem_write(addr, struct.pack('<I', target & 0xffffffff))
                # other reloc types not expected in these libs; ignore

    # -- thumb-aware "call a function" helper --------------------------------
    def call(self, func_addr, args=()):
        """func_addr must include the thumb bit (odd) if it's a thumb function,
        matching how the symbol table / disassembly addresses were given above."""
        thumb = func_addr & 1
        real_addr = func_addr & ~1
        regs = [UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3]
        for i, a in enumerate(args[:4]):
            self.mu.reg_write(regs[i], a & 0xffffffff)
        if len(args) > 4:
            raise NotImplementedError("stack args not needed for this project")
        sp = STACK_BASE + STACK_SIZE - 0x100
        self.mu.reg_write(UC_ARM_REG_SP, sp)
        self.mu.reg_write(UC_ARM_REG_LR, RETURN_ADDR)
        self.mu.reg_write(UC_ARM_REG_PC, real_addr | (1 if thumb else 0))
        # cpsr thumb bit handled automatically by unicorn based on bit0 of PC on entry
        self._stop = False
        try:
            self.mu.emu_start(real_addr | (1 if thumb else 0), RETURN_ADDR, timeout=0, count=0)
        except Exception as e:
            self._dump_crash_diagnostics(e)
            raise
        return self.mu.reg_read(UC_ARM_REG_R0)

    def _sym_near(self, addr):
        """Best-effort 'addr is inside/near which known function' for error messages."""
        best_name, best_addr = None, -1
        for name, a in self.global_syms.items():
            base = a & ~1
            if base <= addr and base > best_addr:
                best_addr, best_name = base, name
        if best_name is not None and addr - best_addr < 0x2000:
            return f"{best_name}+0x{addr - best_addr:x}"
        stub_name = self.stub_names.get(addr & ~1)
        if stub_name:
            return f"<stub: {stub_name}>"
        return "<unknown>"

    def _dump_crash_diagnostics(self, exc):
        mu = self.mu
        print("\n" + "=" * 70)
        print(f"[!] emulation stopped: {exc!r}")
        try:
            pc = mu.reg_read(UC_ARM_REG_PC)
            lr = mu.reg_read(UC_ARM_REG_LR)
            sp = mu.reg_read(UC_ARM_REG_SP)
            regs = {
                "r0": mu.reg_read(UC_ARM_REG_R0), "r1": mu.reg_read(UC_ARM_REG_R1),
                "r2": mu.reg_read(UC_ARM_REG_R2), "r3": mu.reg_read(UC_ARM_REG_R3),
            }
            print(f"    pc = 0x{pc:08x}  ({self._sym_near(pc)})")
            print(f"    lr = 0x{lr:08x}  ({self._sym_near(lr)})")
            print(f"    sp = 0x{sp:08x}")
            print("    " + "  ".join(f"{k}=0x{v:08x}" for k, v in regs.items()))
            try:
                raw = mu.mem_read(pc & ~0xF, 32)
                print("    bytes at pc (16-aligned, 32 bytes):", raw.hex())
            except Exception as mem_e:
                print(f"    (could not read memory at pc: {mem_e})")
        except Exception as inner:
            print(f"    (could not read registers for diagnostics: {inner})")
        print("=" * 70 + "\n")
        print(">>> Copy everything between the ==== lines above and send it back — "
              "that's enough to pinpoint the exact instruction without guessing.")


# ---------------------------------------------------------------------------
# Fake SkStream: vtable with just the 3 slots decodeSPI actually calls
#   +0x0c read(buf,size)->size_t   +0x14 rewind()->bool   +0x34 getLength()->size_t
# ---------------------------------------------------------------------------
def make_fake_stream(emu, file_bytes):
    mu = emu.mu
    data_addr = emu.malloc(len(file_bytes))
    mu.mem_write(data_addr, file_bytes)
    state = {'pos': 0}

    def s_getLength(mu, e):
        mu.reg_write(UC_ARM_REG_R0, len(file_bytes))

    def s_rewind(mu, e):
        state['pos'] = 0
        mu.reg_write(UC_ARM_REG_R0, 1)  # true

    def s_read(mu, e):
        # this=r0, buf=r1, size=r2  (SkStream::read(void* buffer, size_t size))
        buf = mu.reg_read(UC_ARM_REG_R1)
        size = mu.reg_read(UC_ARM_REG_R2)
        remaining = len(file_bytes) - state['pos']
        n = min(size, remaining) if buf else remaining  # SkStream: buf==NULL means "skip"
        if buf and n:
            mu.mem_write(buf, file_bytes[state['pos']:state['pos'] + n])
        state['pos'] += n
        mu.reg_write(UC_ARM_REG_R0, n)

    vtable_words = 16  # generous; only slots 0x0c/3, 0x14/5, 0x34/13 matter
    vtable_addr = emu.malloc(vtable_words * 4)
    trap = emu._new_stub("SkStream::<unused vtable slot>")
    for i in range(vtable_words):
        mu.mem_write(vtable_addr + i * 4, struct.pack('<I', trap))

    read_stub = emu._new_stub("SkStream::read", s_read)
    rewind_stub = emu._new_stub("SkStream::rewind", s_rewind)
    len_stub = emu._new_stub("SkStream::getLength", s_getLength)
    mu.mem_write(vtable_addr + 0x0c, struct.pack('<I', read_stub))
    mu.mem_write(vtable_addr + 0x14, struct.pack('<I', rewind_stub))
    mu.mem_write(vtable_addr + 0x34, struct.pack('<I', len_stub))

    obj_addr = emu.malloc(4)
    mu.mem_write(obj_addr, struct.pack('<I', vtable_addr))
    return obj_addr


# ---------------------------------------------------------------------------
# _SXPI_IMGB reader + PNG writer
# ---------------------------------------------------------------------------
def read_imgb_and_save(emu, imgb_ptr, out_path):
    mu = emu.mu
    if imgb_ptr == 0:
        raise RuntimeError("decodeSPI returned NULL — decode failed (check console for details)")

    def i32(off):
        return struct.unpack('<i', mu.mem_read(imgb_ptr + off, 4))[0]

    def u32(off):
        return struct.unpack('<I', mu.mem_read(imgb_ptr + off, 4))[0]

    w = i32(0x10)
    h = i32(0x14)
    cs = i32(0x20)
    pw0 = i32(0x24)
    ph0 = i32(0x34)
    pa0 = u32(0x44)
    print(f"[imgb] w={w} h={h} cs={cs} pw0={pw0} ph0={ph0} pa0=0x{pa0:x}")

    from PIL import Image

    if cs in (400, 401):  # packed, 1 plane, 3 bytes/px
        stride = pw0 if pw0 else w * 3
        raw = mu.mem_read(pa0, stride * h)
        img = Image.frombytes('RGB', (w, h), bytes(raw), 'raw', 'RGB', stride, 1)
    elif cs in (500, 501, 502, 503):  # packed, 1 plane, 4 bytes/px
        stride = pw0 if pw0 else w * 4
        raw = mu.mem_read(pa0, stride * h)
        img = Image.frombytes('RGBA', (w, h), bytes(raw), 'raw', 'RGBA', stride, 1)
    elif cs in (13, 43):  # planar, 3 or 4 planes (treat as separate Y/plane images for now)
        n_planes = 3 if cs == 13 else 4
        planes = []
        for p in range(n_planes):
            pw = i32(0x24 + p * 4)
            ph = i32(0x34 + p * 4)
            pa = u32(0x44 + p * 4)
            raw = bytes(mu.mem_read(pa, pw * ph))
            planes.append(Image.frombytes('L', (pw, ph), raw))
        # Best-effort: if plane 0 is full-res luma and it's really just a grey
        # keyguard mask, this alone may be exactly what you want. Save all
        # planes so you can inspect which is which.
        base = Path(out_path)
        for i, im in enumerate(planes):
            p_out = base.with_name(base.stem + f"_plane{i}" + base.suffix)
            im.save(p_out)
            print(f"[imgb] wrote plane {i} -> {p_out}")
        img = planes[0]
    else:
        raise RuntimeError(f"unhandled cs={cs} — tell me this value and I'll add the branch")

    img.save(out_path)
    print(f"[imgb] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spi_file")
    ap.add_argument("out_png")
    ap.add_argument("--libskia", default="libskia.so")
    ap.add_argument("--libmaet", default="libmaet.so")
    ap.add_argument("--libsxqk", default="libsxqk_skia.so")
    args = ap.parse_args()

    emu = Emulator(args.libskia, args.libmaet, args.libsxqk)

    # Addresses are the *file offsets from readelf*, thumb bit (|1) added
    # because every SkSPIImageDecoder method disassembled so far is Thumb code.
    CREATE_SPI_DECODER = BASE_SKIA + 0x157b4d | 1
    DECODE_SPI = BASE_SKIA + 0x156b11 | 1
    kDecodePixels_Mode = 1  # SkImageDecoder::Mode enum; 0 = bounds-only

    spi_bytes = Path(args.spi_file).read_bytes()
    stream_obj = make_fake_stream(emu, spi_bytes)

    print("[*] calling CreateSPIImageDecoder()...")
    decoder = emu.call(CREATE_SPI_DECODER, ())
    if decoder == 0:
        sys.exit("CreateSPIImageDecoder returned NULL")
    print(f"[*] decoder = 0x{decoder:x}")

    print("[*] calling SkSPIImageDecoder::decodeSPI(stream, kDecodePixels_Mode)...")
    imgb_ptr = emu.call(DECODE_SPI, (decoder, stream_obj, kDecodePixels_Mode))
    print(f"[*] decodeSPI returned 0x{imgb_ptr:x}")

    read_imgb_and_save(emu, imgb_ptr, args.out_png)


if __name__ == "__main__":
    main()
