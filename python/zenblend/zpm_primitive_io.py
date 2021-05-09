import numpy as np
import struct


def funpack(fmt, f):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def readzpm(path):
    attrs = {}
    conns = [], [], [], []

    with open(path, 'rb') as f:
        signature = f.read(8)
        assert signature == b'\x7fZPMv001', signature

        size, count = funpack('Ni', f)
        assert count < 1024, count

        #print('size =', size)

        attrinfos = []

        for i in range(count):
            type = f.read(4).strip(b'\0').decode()
            namelen, = funpack('N', f)
            assert namelen < 1024, namelen
            namebuf = f.read(namelen)
            name = namebuf.decode()
            #print('attr', name, 'is', type)
            attrinfos.append((type, name))

        for type, name in attrinfos:
            n = struct.calcsize(type) * size
            arr = np.frombuffer(f.read(n), dtype=type)
            attrs[name] = arr

    # TODO: read topology connections too
    return attrs, conns
