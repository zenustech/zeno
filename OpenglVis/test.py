import zen
import time
zen.loadLibrary('build/libzenvis.so')
zen.loadLibrary('../zenbase/build/libzenbase.so')

# ===============
def execute(frame):
        import zen
        if frame == 0: zen.addNode('ReadParticles', 'No1')
        zen.setNodeParam('No1', 'path', str(f'../monkey.obj'))
        zen.applyNode('No1')
        if frame == 0: zen.addNode('SimpleSolver', 'No8')
        zen.setNodeInput('No8', 'ini_pars', 'No1::pars')
        zen.setNodeParam('No8', 'dt', float(0.04))
        zen.setNodeParam('No8', 'G', zen.float3(0, 0, 1))
        zen.applyNode('No8')
        if frame == 0: zen.addNode('ViewParticles', 'No4')
        zen.setNodeInput('No4', 'pars', 'No8::pars')
        zen.applyNode('No4')

zen.initialize()
for frame in range(64):
        print('[Zen] executing frame', frame)
        execute(frame)
        time.sleep(0.5)  # to show that GUI won't freeze when computation takes long
zen.finalize()
# ===============
