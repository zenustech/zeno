import numpy as np


def _tri_append(faces, indices):
    if len(indices) == 3:
        faces.append(indices)
    elif len(indices) == 4:
        faces.append([indices[0], indices[1], indices[2]])
        faces.append([indices[2], indices[3], indices[0]])
    elif len(indices) > 4:
        for n in range(1, len(indices) - 1):
            faces.append([indices[0], indices[n], indices[n + 1]])
    else:
        assert False, len(indices)


def readobj(path, orient='xyz', scale=None, simple=False, usemtl=True, quadok=False):
    v = []
    vt = []
    vn = []
    faces = []
    usemtls = []
    mtllib = None

    if callable(getattr(path, 'read', None)):
        lines = path.readlines()
    else:
        with open(path, 'rb') as myfile:
            lines = myfile.readlines()

    # cache vertices
    for line in lines:
        line = line.strip()
        assert isinstance(line, bytes), f'BytesIO expected! (got {type(line)})'
        try:
            type, fields = line.split(maxsplit=1)
            fields = [float(_) for _ in fields.split()]
        except ValueError:
            continue

        if type == b'v':
            v.append(fields)
        elif type == b'vt':
            vt.append(fields)
        elif type == b'vn':
            vn.append(fields)

    # cache faces
    for line in lines:
        line = line.strip()
        try:
            type, fields = line.split(maxsplit=1)
            fields = fields.split()
        except ValueError:
            continue

        if type == b'mtllib':
            mtllib = fields[0]
            continue

        if type == b'usemtl':
            usemtls.append([len(faces), fields[0]])
            continue

        # line looks like 'f 5/1/1 1/2/1 4/3/1'
        # or 'f 314/380/494 382/400/494 388/550/494 506/551/494' for quads
        if type != b'f':
            continue

        # a field should look like '5/1/1'
        # for vertex/vertex UV coords/vertex Normal  (indexes number in the list)
        # the index in 'f 5/1/1 1/2/1 4/3/1' STARTS AT 1 !!!
        indices = [[int(_) - 1 if _ else 0 for _ in field.split(b'/')] for field in fields]

        if quadok:
            faces.append(indices)
        else:
            _tri_append(faces, indices)

    ret = {}
    ret['v'] = np.array([[0, 0, 0]], dtype=np.float32) if len(v) == 0 else np.array(v, dtype=np.float32)
    ret['vt'] = np.array([[0, 0]], dtype=np.float32) if len(vt) == 0 else np.array(vt, dtype=np.float32)
    ret['vn'] = np.array([[0, 0, 0]], dtype=np.float32) if len(vn) == 0 else np.array(vn, dtype=np.float32)
    ret['f'] = np.zeros((1, 3, 3), dtype=np.int32) if len(faces) == 0 else np.array(faces, dtype=np.int32)
    if usemtl:
        ret['usemtl'] = usemtls
        ret['mtllib'] = mtllib

    if orient is not None:
        objorient(ret, orient)
    if scale is not None:
        if scale == 'auto':
            objautoscale(ret)
        else:
            ret['v'] *= scale

    if simple:
        return ret['v'], ret['f'][:, :, 0]

    return ret


def writeobj(path, obj):
    if callable(getattr(path, 'write', None)):
        f = path
    else:
        f = open(path, 'w')
    with f:
        f.write('# OBJ file saved by tina.writeobj\n')
        f.write('# https://github.com/taichi-dev/taichi_three\n')
        for pos in obj['v']:
            f.write(f'v {" ".join(map(str, pos))}\n')
        if 'vt' in obj:
            for pos in obj['vt']:
                f.write(f'vt {" ".join(map(str, pos))}\n')
        if 'vn' in obj:
            for pos in obj['vn']:
                f.write(f'vn {" ".join(map(str, pos))}\n')
        if 'f' in obj:
            if len(obj['f'].shape) >= 3:
                for i, face in enumerate(obj['f']):
                    f.write(f'f {" ".join("/".join(map(str, f + 1)) for f in face)}\n')
            else:
                for i, face in enumerate(obj['f']):
                    f.write(f'f {" ".join("/".join([str(f + 1)] * 3) for f in face)}\n')


def objunpackmtls(obj):
    faces = obj['f']
    parts = {}
    ends = []
    for end, name in obj['usemtl']:
        ends.append(end)
    ends.append(len(faces))
    ends.pop(0)
    for end, (beg, name) in zip(ends, obj['usemtl']):
        if name in parts:
            parts[name] = np.concatenate([parts[name], faces[beg:end]], axis=0)
        else:
            parts[name] = faces[beg:end]
    for name in parts.keys():
        cur = {}
        cur['f'] = parts[name]
        cur['v'] = obj['v']
        cur['vn'] = obj['vn']
        cur['vt'] = obj['vt']
        parts[name] = cur
    return parts


def objmtlids(obj):
    faces = obj['f']
    mids = np.zeros(shape=len(faces), dtype=np.int32)
    ends = []
    for end, name in obj['usemtl']:
        ends.append(end)
    ends.append(len(faces))
    ends.pop(0)
    names = []
    for end, (beg, name) in zip(ends, obj['usemtl']):
        if name not in names:
            mids[beg:end] = len(names) + 1
            names.append(name)
        else:
            mids[beg:end] = names.index(name) + 1
    return mids


def objverts(obj):
    return obj['v'][obj['f'][:, :, 0]]


def objnorms(obj):
    return obj['vn'][obj['f'][:, :, 2]]


def objcoors(obj):
    return obj['vt'][obj['f'][:, :, 1]]


def objautoscale(obj):
    obj['v'] -= np.average(obj['v'], axis=0)
    obj['v'] /= np.max(np.abs(obj['v']))


def objorient(obj, orient):
    flip = False
    if orient.startswith('-'):
        flip = True
        orient = orient[1:]

    x, y, z = ['xyz'.index(o.lower()) for o in orient]
    fx, fy, fz = [o.isupper() for o in orient]

    if x != 0 or y != 1 or z != 2:
        obj['v'][:, (0, 1, 2)] = obj['v'][:, (x, y, z)]
        obj['vn'][:, (0, 1, 2)] = obj['vn'][:, (x, y, z)]

    for i, fi in enumerate([fx, fy, fz]):
        if fi:
            obj['v'][:, i] = -obj['v'][:, i]
            obj['vn'][:, i] = -obj['vn'][:, i]

    if flip:
        obj['f'][:, ::-1, :] = obj['f'][:, :, :]


def objmknorm(obj):
    fip = obj['f'][:, :, 0]
    fit = obj['f'][:, :, 1]
    p = obj['v'][fip]
    nrm = np.cross(p[:, 2] - p[:, 0], p[:, 1] - p[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    fin = np.arange(obj['f'].shape[0])[:, np.newaxis]
    fin = np.concatenate([fin for i in range(3)], axis=1)
    newf = np.array([fip, fit, fin]).swapaxes(1, 2).swapaxes(0, 2)
    obj['vn'] = nrm
    obj['f'] = newf


def readply(path):
    from plyfile import PlyData

    ply = PlyData.read(path)
    verts = ply.elements[0]
    faces = ply.elements[1]
    verts = np.array([list(v)[:3] for v in verts], dtype=np.float32)
    faces = np.array([list(f[0]) for f in faces], dtype=np.int32)

    return verts, faces
