You may use the FluidSolver node to simulate fluids, example:
```python
emitModel = LoadModel(path='<emitter model path>.obj')  # specify your model path here
emitModel = Transform(in=emitModel, translate=(0, 0, 1))  # place at Z=1, up
colliderModel = LoadModel('<collider model path>.obj')  # specify your model path here
colliderModel = Transform(in=colliderModel, translate=(0, 0, -1))     # place at Z=-1, down
fluidSolver = FluidSolver(collider=colliderModel, emitter=emitModel)  # create the fluid simulation
nframes = 100  # start solving for 100 frames, replace this with your desired number of frames
for frame in range(1, nframes + 1):
    fluidParticles = fluidSolver.getParticles()  # get fluid particles
    fluidSurface = ParticlesToSurface(particles=fluidParticles)  # rebuild surface mesh from particles
    ExportModel(path=f'/tmp/{frame:06d}.obj', model=fluidSurface)  # Export from /tmp/000001.obj to /tmp/000100.obj
    fluidSolver.step(dt=0.01)  # run solver step, step into next frame state
```
