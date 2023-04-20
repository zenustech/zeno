To simulate with coupled rigid and fluid, create both a FluidSolver and a RigidSolver:
```python
fluidSolver = FluidSolver(collider=<collider model>, emitter=<emitter model>)
rigidSolver = RigidSolver()
rigidSolver.addRigid(<add your rigids here>)
nframes = 100  # simulate for 100 frames
for frame in range(1, nframes + 1):
    objList = rigidSolver.getRigidObjects()
    particles = fluidSolver.getParticles()
    fluidSurf = ParticleToSurface(particles=particles)
    mergedObj = MergeObjects(objects=[*objList, fluidSurf])  # both rigids list and fluid surface are encountered
    ExportModel(path=f'/tmp/{frame:06d}.obj', model=mergedObj)  # Export from /tmp/000001.obj to /tmp/000100.obj
    rigidSolver.step(dt=0.01, couple_fluid=fluidSolver)
    fluidSolver.step(dt=0.01, couple_rigid=rigidSolver)
```

See also:
- Read FluidSimulation.md for more details about FluidSolver.
- Read RigidFragmentileDestruction.md for more details about RigidSolver.
