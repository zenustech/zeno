Benchmark Settings
==================
Benchmark settings of each project are set within its directory (e.g. *Projects/MGSP*) independently.

General setup
------------------
Most configurations regarding the simulation are set in *settings.h*.

Data structure related setup:
    - **DOMAIN_BITS**: the resolution of the background grid in each dimension, e.g. 8 refers to a :math:`256 \times 256 \times 256` grid.
    - **MAX_PPC**: the maximum number of particles held within a cell.
    - **g_bin_capacity**: the number of particles in each particle bin.
    - **g_max_particle_num**: the maximum number of particles in each GPU.
    - **g_max_active_block**: the maximum number of blocks in each GPU. This affects the memory consumption of most important data structures (*particles*, *grid*, *partition*, etc).

Physical parameter setup:
    - **g_gravity**: the gravity constant.
    - **MODEL_PPC**: particle-per-cell for sampling particles from a SDF model. per-particle-volume is computed from this and the grid resolution.
    - **DENSITY**: the density constant of particles.
    - **get_material_type(int did) -> material**: the type of particles for each GPU. Assume each GPU handles only one category of particles.

Multi-GPU related setup:
    - **g_device_num**: the number of GPUs used.
    - **get_domain(int did) -> domain**: the spatial partition domain for each GPU. It can be set time-dependently.

Additional material setup
------------------
In *particle_buffer.cuh*, there are currently four types of particles supported. They are named *JFluid*, *FixedCorotated*, *Sand*, and *NACC*. 
Within each **ParticleBuffer**, there are various material-dependent constant parameters that could be configured.

Initial model setup
------------------
Initial particle models for all GPUs are set in the **init_models** function in *main.cu*.