import zen
zen.loadLibrary('../FastFLIP/build/libFLIPlib.so')
zen.loadLibrary('../zenbase/build/libzenbase.so')
zen.loadLibrary('../OpenglVis/build/libzenvis.so')

def execute(frame):
        import zen
        if frame == 0: zen.addNode('ReadObjMesh', 'No1')
        zen.setNodeParam('No1', 'path', str(f'../monkey.obj'))
        zen.applyNode('No1')
        if frame == 0: zen.addNode('ViewMesh', 'No4')
        zen.setNodeInput('No4', 'mesh', 'No1::mesh')
        zen.applyNode('No4')

for frame in range(895):
        print('[Zen] executing frame', frame)
        execute(frame)

zen.finalize()
