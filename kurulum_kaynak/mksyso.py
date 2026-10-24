"""Write a COFF .syso with an icon resource (RT_ICON + RT_GROUP_ICON) for the Go linker."""
import struct, sys

ico = open(sys.argv[1], 'rb').read()
_, _, n = struct.unpack_from('<HHH', ico, 0)
images, grp = [], struct.pack('<HHH', 0, 1, n)
for i in range(n):
    w, h, c, r, planes, bpp, size, off = struct.unpack_from('<BBBBHHII', ico, 6 + 16 * i)
    images.append(ico[off:off + size])
    grp += struct.pack('<BBBBHHIH', w, h, c, r, planes, bpp, size, i + 1)

def directory(entries):  # entries: list of (id, offset, is_dir)
    b = struct.pack('<IIHHHH', 0, 0, 0, 0, 0, len(entries))
    for id_, off, isdir in entries:
        b += struct.pack('<II', id_, off | (0x80000000 if isdir else 0))
    return b

# layout: root | type dirs | name dirs | lang dirs | data entries | data blobs
types = [(3, [(i + 1, images[i]) for i in range(n)]), (14, [(1, grp)])]
dsz = lambda k: 16 + 8 * k
root_off = 0
off = dsz(len(types))
type_off = []
for t, items in types:
    type_off.append(off); off += dsz(len(items))
name_off = []
for t, items in types:
    name_off.append([]);
    for _ in items:
        name_off[-1].append(off); off += dsz(1)
entry_off = []
for t, items in types:
    entry_off.append([])
    for _ in items:
        entry_off[-1].append(off); off += 16
blob_off = []
for t, items in types:
    blob_off.append([])
    for _, data in items:
        off = (off + 7) & ~7
        blob_off[-1].append(off); off += len(data)
total = (off + 7) & ~7
buf = bytearray(total)
relocs = []
buf[0:dsz(len(types))] = directory([(t, type_off[k], True) for k, (t, _) in enumerate(types)])
for k, (t, items) in enumerate(types):
    d = directory([(id_, name_off[k][j], True) for j, (id_, _) in enumerate(items)])
    buf[type_off[k]:type_off[k] + len(d)] = d
    for j, (id_, data) in enumerate(items):
        d = directory([(0x0409, entry_off[k][j], False)])
        buf[name_off[k][j]:name_off[k][j] + len(d)] = d
        e = struct.pack('<IIII', blob_off[k][j], len(data), 0, 0)
        buf[entry_off[k][j]:entry_off[k][j] + 16] = e
        relocs.append(entry_off[k][j])
        buf[blob_off[k][j]:blob_off[k][j] + len(data)] = data
hdr_size, sec_size = 20, 40
raw_ptr = hdr_size + sec_size
rel_ptr = raw_ptr + total
sym_ptr = rel_ptr + 10 * len(relocs)
out = struct.pack('<HHIIIHH', 0x8664, 1, 0, sym_ptr, 1, 0, 0)
out += struct.pack('<8sIIIIIIHHI', b'.rsrc', 0, 0, total, raw_ptr, rel_ptr, 0, len(relocs), 0, 0x40000040)
out += bytes(buf)
for r in relocs:
    out += struct.pack('<IIH', r, 0, 3)   # IMAGE_REL_AMD64_ADDR32NB against symbol 0 (.rsrc)
out += struct.pack('<8sIhHBB', b'.rsrc', 0, 1, 0, 3, 0)
out += struct.pack('<I', 4)
open(sys.argv[2], 'wb').write(out)
print('ok', n, 'icons', len(out), 'bytes')
