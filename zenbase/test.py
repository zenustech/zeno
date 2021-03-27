import zen
zen.loadLibrary('build/libzenbase.so')
print(zen.dumpDescriptors())
# ===============
def execute(frame):
        import zen
        if frame == 0: zen.addNode('ReadObjMesh', 'No1')
        zen.setNodeParam('No1', 'path', str(f'../monkey.obj'))
        zen.applyNode('No1')
        if frame == 0: zen.addNode('ViewMesh', 'No4')
        zen.setNodeInput('No4', 'mesh', 'No1::mesh')
        zen.applyNode('No4')
        if frame == 0: zen.addNode('SleepMilis', 'No5')
        zen.setNodeParam('No5', 'ms', int(1000))
        zen.applyNode('No5')
        if frame == 0: zen.addNode('EndFrame', 'endFrame')
        zen.applyNode('endFrame')

for frame in range(64):
        print('[Zen] executing frame', frame)
        execute(frame)
# ===============
