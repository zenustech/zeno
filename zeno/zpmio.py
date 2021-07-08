import numpy as np
import struct


def fread(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def fwrite(f, fmt, *args):
    return f.write(struct.pack(fmt, *args))


def readzpm(path):
    attrs = {}
    conns = [None, None, None, None]

    with open(path, 'rb') as f:
        signature = f.read(8)
        assert signature == b'\x7fZPMv001', signature

        size, count = fread(f, 'Ni')
        assert count < 1024, count
        #print('size =', size)

        attrinfos = []
        for i in range(count):
            type = f.read(4).strip(b'\0').decode()
            namelen, = fread(f, 'N')
            assert namelen < 1024, namelen
            namebuf = f.read(namelen)
            name = namebuf.decode()
            #print('attr', name, 'is', type)
            attrinfos.append((type, name))

        for type, name in attrinfos:
            n = struct.calcsize(type) * size
            data = f.read(n)
            arr = np.frombuffer(data, dtype=type)
            attrs[name] = arr

        for dim in [1, 2, 3, 4]:
            size, = fread(f, 'N')
            type = '{}I'.format(dim if dim != 1 else '')
            n = struct.calcsize(type) * size
            data = f.read(n)
            arr = np.frombuffer(data, dtype=type)
            conns[dim - 1] = arr

    return attrs, conns


def writezpm(path, attrs, conns=None):
    if conns is None:
        conns = [], [], [], []

    size = None
    for arr in attrs.values():
        if size is None:
            size = len(arr)
        else:
            assert size == len(arr), (size, len(arr))
    assert size is not None
    count = len(attrs.keys())

    with open(path, 'wb') as f:
        f.write(b'\x7fZPMv001')

        fwrite(f, 'Ni', size, count)

        attrinfos = []
        for name, arr in attrs.items():
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr, dtype=np.float32)

            type = arr.dtype.char
            if len(arr.shape) >= 2:
                assert len(arr.shape) == 2
                type = str(arr.shape[1]) + type
            else:
                assert len(arr.shape) == 1

            typebuf = type.encode()
            while len(typebuf) < 4:
                typebuf += b'\0'
            f.write(typebuf)

            namebuf = name.encode()
            namelen = len(namebuf)
            fwrite(f, 'N', namelen)
            f.write(namebuf)

            attrinfos.append((type, arr))

        for type, arr in attrinfos:
            data = arr.tobytes()
            f.write(data)

        for dim in [1, 2, 3, 4]:
            arr = conns[dim - 1]
            type = '{}I'.format(dim if dim != 1 else '')
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr, dtype=type)
            data = arr.tobytes()
            fwrite(f, 'N', len(arr))
            f.write(data)


'''
attrs, conns = readzpm('/tmp/zenio/000000/result.zpm')
print(attrs)
writezpm('/tmp/a.zpm', attrs, conns)
'''
