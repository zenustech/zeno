import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import count
import taichi as ti

ti.init(arch=ti.cpu)

df_fac = 1.3
dx = 0.2
dh = dx * df_fac

###### Scene parameters ########
w = 20
h = 10
w_bound = 22
h_bound = 12

bottom_bound = 0.0
top_bound = 0.0
left_bound = 0.0
right_bound = 0.0

assert w_bound > w
assert h_bound > h
x_min = (w_bound - w) / 2.0
y_min = (h_bound - h) / 2.0
x_max = w_bound - (w_bound - w) / 2.0
y_max = h_bound - (h_bound - h) / 2.0

screen_res = (800, 400)
screen_to_world_ratio = 35.0
bg_color = 0x112f41
particle_color = 0x068587
boundary_color = 0xebaca2
particle_radius = 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio


def setup():
    def computeGridIndex(x, y):
        idx = np.floor(x / (2 * dh)).astype(int)
        idy = np.floor(y / (2 * dh)).astype(int)
        return idx, idy

    def placeParticles(position_list, paticle_list, wall_mark, bound=0):
        # position_list: [start_x, start_y, end_x, end_y]
        start_x, start_y, end_x, end_y = position_list
        vel_x, vel_y, p, rho = 0.0, 0.0, 0.0, 1000.0
        for pos_x in np.arange(start_x, end_x, dx):
            for pos_y in np.arange(start_y, end_y, dx):
                paticle_list.append([pos_x, pos_y])
                if bound:
                    wall_mark.append(0)
                else:
                    wall_mark.append(1)
    particle_list = []
    wall_mark = []

    #### Dam break #######
    start_x = x_min + 0.5 * dx
    start_y = y_min - 1.0 * dx
    end_x = start_x + 0.5 * h
    end_y = start_y + 0.6 * h

    ## Constrcut wall
    # Bottom square
    b_start_x = 0.0
    b_start_y = 0.0
    b_end_x = b_start_x + w
    b_end_y = b_start_y + y_min - 2 * dx

    bottom_bound = b_end_y

    # Top square
    t_start_x = 0.0
    t_start_y = h - y_min + 2 * dx
    t_end_x = t_start_x + w
    t_end_y = h

    top_bound = t_start_y

    # left square
    l_start_x = 0.0
    l_start_y = y_min - 2 * dx
    l_end_x = l_start_x + x_min - 2 * dx
    l_end_y = l_start_y + h - 2 * y_min + 4 * dx

    left_bound = l_end_x

    # right square
    r_start_x = w - x_min + 2 * dx
    r_start_y = y_min - 2 * dx
    r_end_x = w - dx
    r_end_y = r_start_y + h - 2 * y_min + 4 * dx

    right_bound = r_start_x

    pos_list_fluid = [start_x, start_y, end_x, end_y]
    placeParticles(pos_list_fluid, particle_list, wall_mark)

    # These are boundaries
    pos_list_bs = [b_start_x, b_start_y, b_end_x, b_end_y]
    placeParticles(pos_list_bs, particle_list, wall_mark, bound=1)

    pos_list_ts = [t_start_x, t_start_y, t_end_x, t_end_y]
    placeParticles(pos_list_ts, particle_list, wall_mark, bound=1)

    pos_list_ls = [l_start_x, l_start_y, l_end_x, l_end_y]
    placeParticles(pos_list_ls, particle_list, wall_mark, bound=1)

    pos_list_rs = [r_start_x, r_start_y, r_end_x, r_end_y]
    placeParticles(pos_list_rs, particle_list, wall_mark, bound=1)

    return particle_list,wall_mark, top_bound, bottom_bound, left_bound, right_bound

def makeGrid():
    grid_size = 2*dh
    num_x = np.ceil(w_bound / grid_size).astype(int)
    num_y = np.ceil(h_bound / grid_size).astype(int)

    grid_x = num_x
    grid_y = num_y
    return grid_x, grid_y

@ti.data_oriented
class sph_solver:
    def __init__(self, particle_list,
                 wall_mark,
                 grid,
                 bound, dx=0.2, max_time=10000, max_steps=1000,gui=None):
        ######## Solver parameters ##########
        self.max_time = max_time
        self.max_steps = max_steps
        self.gui = gui
        # Gravity
        self.g = -9.80
        # viscosity
        self.mu = 1e-3
        # reference density
        self.rho_0 = 1000.0
        # CFL coefficient
        self.CFL = 0.20

        # Smooth kernel norm factor
        self.kernel_norm = 1.0

        # Pressure state function parameters
        self.gamma = 7.0
        self.c_0 = 20.0

        ###### Scene parameters ########
        self.w = 20
        self.h = 10
        self.w_bound = 22
        self.h_bound = 12

        assert self.w_bound > self.w
        assert self.h_bound > self.h
        self.x_min = (self.w_bound - self.w) / 2.0
        self.y_min = (self.h_bound - self.h) / 2.0
        self.x_max = self.w_bound - (self.w_bound - self.w) / 2.0
        self.y_max = self.h_bound - (self.h_bound - self.h) / 2.0

        self.top_bound = bound[0] # top_bound
        self.bottom_bound = bound[1] #bottom_bound
        self.left_bound = bound[2] #left_bound
        self.right_bound = bound[3] #right_bound

        self.df_fac = 1.3
        self.dx = 0.2
        self.dh = self.dx * self.df_fac
        self.kernel_norm = 10. / (7. * np.pi * self.dh ** 2)

        ###### Particles #######
        self.dim = 2
        self.particle_numbers = len(particle_list)

        self.grid_x = grid[0]
        self.grid_y = grid[1]

        # Fluid particles
        self.old_positions = ti.Vector(self.dim, dt=ti.f32)
        self.particle_positions = ti.Vector(self.dim, dt=ti.f32)
        self.particle_velocity = ti.Vector(self.dim, dt=ti.f32)
        self.particle_pressure = ti.Vector(1, dt=ti.f32)
        self.particle_density = ti.Vector(1, dt=ti.f32)
        self.wall_mark_list = ti.Vector(1, dt=ti.f32)

        self.d_velocity = ti.Vector(self.dim, dt=ti.f32)
        self.d_density = ti.Vector(1, dt=ti.f32)

        self.dx = dx
        self.m = self.dx**2 * 1000

        self.particle_list = np.array(particle_list)
        self.wall_mark = np.array(wall_mark)

        self.grid_num_particles = ti.var(ti.i32)
        self.grid2particles = ti.var(ti.i32)
        self.particle_num_neighbors = ti.var(ti.i32)
        self.particle_neighbors = ti.var(ti.i32)

        self.max_num_particles_per_cell = 100
        self.max_num_neighbors = 100

        ti.root.dense(ti.i, self.particle_numbers).place(self.old_positions, self.particle_positions,
                                                         self.particle_velocity, self.particle_pressure,
                                                         self.particle_density, self.d_velocity, self.d_density,
                                                         self.wall_mark_list)

        grid_snode = ti.root.dense(ti.ij, (self.grid_x, self.grid_y))
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.k, self.max_num_particles_per_cell).place(self.grid2particles)

        nb_node = ti.root.dense(ti.i, self.particle_numbers)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, self.max_num_neighbors).place(self.particle_neighbors)

    @ti.kernel
    def init(self, p_list:ti.ext_arr(), w_list:ti.ext_arr()):
        for i in range(self.particle_numbers):
            for j in ti.static(range(self.dim)):
                self.particle_positions[i][j] = p_list[i,j]
                self.particle_velocity[i][j] = ti.cast(0.0, ti.f32)
            self.d_velocity[i][0] = ti.cast(0.0, ti.f32)
            self.d_velocity[i][1] = ti.cast(-9.8, ti.f32)

            self.wall_mark_list[i][0] = w_list[i]
            self.d_density[i][0] = ti.cast(0.0, ti.f32)
            self.particle_pressure[i][0] = ti.cast(0.0, ti.f32)
            self.particle_density[i][0] = ti.cast(1000.0, ti.f32)

    @ti.func
    def computeGridIndex(self, pos):
        return (pos / (2 * dh)).cast(int)

    @ti.kernel
    def allocateParticles(self):
        # Ref to pbf2d example from by Ye Kuang (k-ye)
        # https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py
        # allocate particles to grid
        for p_i in self.particle_positions:
            # Compute the grid index on the fly
            cell = self.computeGridIndex(self.particle_positions[p_i])
            offs = self.grid_num_particles[cell].atomic_add(1)
            self.grid2particles[cell, offs] = p_i

    @ti.func
    def is_in_grid(self, c):
        # Ref to pbf2d example from by Ye Kuang (k-ye)
        # https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py
        return 0 <= c[0] and c[0] < self.grid_x and 0 <= c[1] and c[1] < self.grid_y

    @ti.func
    def isFluid(self, p):
        # check fluid particle or bound particle
        return self.wall_mark_list[p][0]

    @ti.kernel
    def search_neighbors(self):
        # Ref to pbf2d example from by Ye Kuang (k-ye)
        # https://github.com/taichi-dev/taichi/blob/master/examples/pbf2d.py
        for p_i in self.particle_positions:
            pos_i = self.particle_positions[p_i]
            nb_i = 0
            if self.isFluid(p_i) == 1:
                # Compute the grid index on the fly
                cell = self.computeGridIndex(self.particle_positions[p_i])
                for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
                    cell_to_check = cell + offs
                    if self.is_in_grid(cell_to_check):
                        for j in range(self.grid_num_particles[cell_to_check]):
                            p_j = self.grid2particles[cell_to_check, j]
                            if nb_i < self.max_num_neighbors and p_j != p_i and (
                                    pos_i - self.particle_positions[p_j]).norm() < self.dh * 2.00:
                                self.particle_neighbors[p_i, nb_i] = p_j
                                nb_i += 1
            self.particle_num_neighbors[p_i] = nb_i

    @ti.func
    def cubicKernel(self, r, h):
        # value of cubic spline smoothing kernel
        k = 10. / (7. * np.pi * h ** 2)
        q = r / h
        assert q >= 0.0
        res = ti.cast(0.0, ti.f32)
        if q <= 1.0:
            res = k * (1 - 1.5 * q ** 2 + 0.75 * q ** 3)
        elif q < 2.0:
            res = k * 0.25 * (2 - q) ** 3
        return res

    @ti.func
    def cubicKernelDerivative(self, r, h):
        # derivative of cubcic spline smoothing kernel
        k = 10. / (7. * np.pi * h ** 2)
        q = r / h
        assert q > 0.0
        res = ti.cast(0.0, ti.f32)
        if q < 1.0:
            res =  (k / h) * (-3 * q + 2.25 * q ** 2)
        elif q < 2.0:
            res = -0.75 * (k / h) * (2 - q) ** 2
        return res

    @ti.func
    def rhoDerivative(self, ptc_i, ptc_j, r, r_mod, h):
        # density delta
        return self.m * self.cubicKernelDerivative(r_mod, h) \
               * (self.particle_velocity[ptc_i]- self.particle_velocity[ptc_j]).dot(r / r_mod)

    @ti.func
    def pUpdate(self, rho, rho_0=1000, gamma=7.0, c_0=20.0):
        # Weakly compressible, tait function
        b = rho_0 * c_0 ** 2 / gamma
        return b * ((rho / rho_0) ** gamma - 1.0)

    @ti.func
    def pressureForce(self, ptc_i, ptc_j, r, r_mod, h, mirror_pressure=0):
        # Compute the pressure force contribution, Symmetric Formula
        res = ti.Vector([0.0, 0.0])
        # Disable the mirror force, use collision instead
        # Use pressure mirror method to handle boundary leak
        # if mirror_pressure == 1:
        #     res = - self.m * (self.particle_pressure[ptc_i][0]/ self.particle_density[ptc_i][0] ** 2
        #                       + self.particle_pressure[ptc_i][0]/self.rho_0**2)* self.cubicKernelDerivative(r_mod, h) * r / r_mod
        # else:
        res =  -self.m * (self.particle_pressure[ptc_i][0] / self.particle_density[ptc_i][0] ** 2
                          + self.particle_pressure[ptc_j][0] / self.particle_density[ptc_j][0] ** 2) \
               * self.cubicKernelDerivative(r_mod, h) * r / r_mod
        return res

    @ti.func
    def viscosityForce(self, ptc_i, ptc_j, r, r_mod, h, mu=1e-3):
        # Compute the viscosity force contribution, Symmetric Formula
        res = ti.Vector([0.0, 0.0])
        res =  mu * (1.0 / self.particle_density[ptc_i][0] ** 2 + 1.0 / self.particle_density[ptc_j][0] ** 2) \
               * self.cubicKernelDerivative(r_mod, h) * (self.particle_velocity[ptc_i]- self.particle_velocity[ptc_j]) / r_mod
        return res

    @ti.func
    def simualteCollisions(self, ptc_i, vec, d):
        # Collision factor, assume roughly 50% velocity loss after collision, i.e. m_f /(m_f + m_b)
        c_f = 0.5
        self.particle_positions[ptc_i] += vec * d
        self.particle_velocity[ptc_i] -= (1.0+c_f) * self.particle_velocity[ptc_i].dot(vec) * vec

    @ti.kernel
    def enforceBoundary(self):
        for p_i in self.particle_positions:
            if self.isFluid(p_i) == 1:
                pos = self.particle_positions[p_i]
                if pos[0] < self.left_bound:
                    self.simualteCollisions(p_i, ti.Vector([1.0, 0.0]), self.left_bound - pos[0])
                if pos[0] > self.right_bound:
                    self.simualteCollisions(p_i, ti.Vector([-1.0, 0.0]), pos[0] - self.right_bound)
                if pos[1] > self.top_bound:
                    self.simualteCollisions(p_i, ti.Vector([0.0, -1.0]), pos[1] - self.top_bound)
                if pos[1] < self.bottom_bound:
                    self.simualteCollisions(p_i, ti.Vector([0.0, 1.0]), self.bottom_bound - pos[1])

    @ti.kernel
    def computeDeltas(self):
        for p_i in self.particle_positions:
            pos_i = self.particle_positions[p_i]
            d_v = ti.Vector([0.0, 0.0])
            d_rho = 0.0
            # if self.isFluid(p_i) == 1:
            #     d_v = ti.Vector([0.0, -9.8])
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                pos_j = self.particle_positions[p_j]

                # Disable mirror force
                # mirror_pressure = 0
                # if self.isFluid(p_j) == 0:
                #     mirror_pressure = 1

                # Compute distance and its mod
                r = pos_i - pos_j
                r_mod = r.norm()

                # Compute Density change
                d_rho += self.rhoDerivative(p_i, p_j, r, r_mod, self.dh)

                if self.isFluid(p_i) == 1:
                    # Compute Viscosity force contribution
                    d_v += self.viscosityForce(p_i, p_j, r, r_mod, self.dh, self.mu)

                    # Compute Pressure force contribution
                    d_v += self.pressureForce(p_i, p_j, r, r_mod, self.dh)

            # Add body force
            if self.isFluid(p_i) == 1:
                d_v += ti.Vector([0.0, -9.8])
            self.d_velocity[p_i] = d_v
            self.d_density[p_i][0] = d_rho

    @ti.kernel
    def updateTimeStep(self):
        # Simple Forward Euler currently
        for p_i in self.particle_positions:
            if self.isFluid(p_i) == 1:
                self.particle_positions[p_i] += self.dt * self.particle_velocity[p_i]
                self.particle_velocity[p_i] += self.dt * self.d_velocity[p_i]
            self.particle_density[p_i][0] += self.dt * self.d_density[p_i][0]
            self.particle_pressure[p_i][0] = self.pUpdate(self.particle_density[p_i][0], self.rho_0, self.gamma, self.c_0)

    def solve(self):
        # Compute dt, a naive initial test value
        self.dt = 0.1 * self.dh / self.c_0
        print("Time step: ", self.dt)
        print("Domain: (%s, %s, %s, %s)" % (self.x_min, self.x_max, self.y_min, self.y_max), )
        print("Fluid area: (%s, %s, %s, %s)"%(self.left_bound, self.right_bound, self.bottom_bound, self.top_bound))
        print("Grid: (%d, %d)"%(self.grid_x, self.grid_y))

        step = 1
        t = 0.0
        total_start = time.process_time()
        while t < self.max_time and step < self.max_steps:
            curr_start = time.process_time()
            self.solveUpdate()
            max_v = np.max(np.linalg.norm(self.particle_velocity.to_numpy(),2, axis=1))
            max_a = np.max(np.linalg.norm(self.d_velocity.to_numpy(),2, axis=1))
            max_rho = np.max(self.particle_density.to_numpy())
            max_pressure = np.max(self.particle_pressure.to_numpy())

            curr_end = time.process_time()
            t += self.dt
            step += 1

            # CFL analysis, adaptive dt
            dt_cfl = self.dh / max_v
            dt_f = np.sqrt(self.dh / max_a)
            dt_a = self.dh / (self.c_0 * np.sqrt((max_rho / self.rho_0)**self.gamma))
            self.dt = self.CFL * np.min([dt_cfl, dt_f, dt_a])
            if step % 10 == 0:
                print("Step: %d, physics time: %s, progress: %s %%, time used: %s, total time used: %s"
                      % (step, t, 100*np.max([t / self.max_time, step / self.max_steps]), curr_end-curr_start, curr_end-total_start))
                print("Max velocity: %s, Max acceleration: %s, Max density: %s, Max pressure: %s" % (max_v, max_a, max_rho, max_pressure))
                print("Adaptive time step: ", self.dt)
            self.render(step, self.gui)
        total_end = time.process_time()
        print("Total time used: %s " % (total_end - total_start))

    def solveUpdate(self):
        self.grid_num_particles.fill(0)
        self.particle_neighbors.fill(-1)
        self.allocateParticles()
        self.search_neighbors()
        # Compute deltas
        self.computeDeltas()
        # timestep Update
        self.updateTimeStep()
        # Handle potential leak particles
        self.enforceBoundary()

    def isFluidNP(self, p):
        # ti.func cannot be called in python scope
        # for render use
        return self.wall_mark[p]

    def render(self, step, gui):
        canvas = gui.canvas
        canvas.clear(bg_color)
        pos_np = self.particle_positions.to_numpy()
        fluid_p = []
        wall_p = []
        for i, pos in enumerate(pos_np):
            if self.isFluidNP(i) == 1:
                fluid_p.append(pos)
            else:
                wall_p.append(pos)
        fluid_p = np.array(fluid_p)
        wall_p = np.array(wall_p)

        for pos in fluid_p:
            for j in range(self.dim):
                pos[j] *= screen_to_world_ratio / screen_res[j]

        for pos in wall_p:
            for j in range(self.dim):
                pos[j] *= screen_to_world_ratio / screen_res[j]

        gui.circles(fluid_p, radius=particle_radius, color=particle_color)
        gui.circles(wall_p, radius=particle_radius, color=boundary_color)
        gui.show()

def main():
    gui = ti.GUI('SPH2D', screen_res)
    grid_shape = makeGrid()
    particle_list,wall_mark, u, b, l, r = setup()
    sph = sph_solver(particle_list, wall_mark, grid_shape, [u,b,l,r], dx = dx, gui=gui, max_steps=10000)
    sph.init(sph.particle_list, sph.wall_mark)
    sph.solve()
    print('done')

if __name__ == '__main__':
    main()










