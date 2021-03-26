#ifndef FLUID_SIM_H
#define FLUID_SIM_H

#include "util.h"
//#include "Sparse_buffer.h"
#include "FLIP_particle.h"
#include "sparse_matrix.h"
#include "levelset_util.h"
#include "pcg_solver.h"
#include "GeometricLevelGen.h"
#include "AlgebraicMultigrid.h"
#include "BPS3D.h"
#include "volumeMeshTools.h"

class moving_solids
{
public:
    moving_solids() : voxel_size(1.0/100.0), sdf(nullptr),
                      vel_func([](int frame)->LosTopos::Vec3f{return LosTopos::Vec3f(0.0);}),
                      pos_func([](int frame)->LosTopos::Vec3f{return LosTopos::Vec3f(0.0);})
    {}
    moving_solids(float h, openvdb::FloatGrid::Ptr sdf_in,
                  std::function<LosTopos::Vec3f(int framenum)> vel_func_in,
                  std::function<LosTopos::Vec3f(int framenum)> pos_func_in) :
                  voxel_size(h), sdf(sdf_in), vel_func(vel_func_in), pos_func(pos_func_in)
    {}
    ~moving_solids() = default;
    openvdb::FloatGrid::Ptr sdf;
    std::function<LosTopos::Vec3f(int framenum)> vel_func;
    std::function<LosTopos::Vec3f(int framenum)> pos_func;
    float voxel_size;
    void updateSolid(int framenum)
    {
        LosTopos::Vec3f solid_pos = pos_func(framenum);
        openvdb::math::Mat4f transMat;
        transMat.setToScale(openvdb::Vec3f(voxel_size));
        transMat.setTranslation(openvdb::Vec3f(solid_pos[0], solid_pos[1], solid_pos[2]));
        sdf->setTransform(openvdb::math::Transform::createLinearTransform(transMat));
    }
};

template<int N>
struct sparse_fluid_3D;

typedef sparse_fluid_3D<8> sparse_fluid8x8x8;
typedef sparse_fluid_3D<6> sparse_fluid6x6x6;
class FluidSim {
public:
	FluidSim(BPS3D* sim = nullptr, bool sync_with_bem = false);
    ~FluidSim(){}

    static float H(float r)
    {
        float res = 0;
        if(r>=-1 && r<0) res = 1+r;
        if(r>=0 && r<=1) res = 1-r;
        return res;
    }
    static float compute_weight(LosTopos::Vec3f gp, LosTopos::Vec3f pp, float dx)
    {
        //k(x,y,z) = H(dx/hx)H(dy/hx)H(dz/hx)
        //H(r) = 1-r 0<=r<1  1+r -1<=r<0 0 else;
        LosTopos::Vec3f dd = gp - pp;
        return H(dd[0]/dx)*H(dd[1]/dx)*H(dd[2]/dx);
    }

	struct boxEmitter {
		LosTopos::Vec3f vel;
		LosTopos::Vec3f bmin;
		LosTopos::Vec3f bmax;
		boxEmitter()
		{
			vel = LosTopos::Vec3f(0);
			bmin = LosTopos::Vec3f(0);
			bmax = LosTopos::Vec3f(0);
		}
		boxEmitter(const boxEmitter &e)
		{
			vel = e.vel;
			bmin = e.bmin;
			bmax = e.bmax;
		}
	};

    BPS3D * bps() { return m_bps; }
	void initialize(double _dx);
	void set_boundary(float(*phi)(const LosTopos::Vec3f&));
	void set_boundary(std::function<float(const LosTopos::Vec3f&)> phi);
	void set_liquid(const LosTopos::Vec3f &bmin, const LosTopos::Vec3f &bmax, std::function<float(const LosTopos::Vec3f&)> phi);
	void init_domain();
	void resolveParticleBoundaryCollision();
	void setSolverRegion(const LosTopos::Vec3f &bmin, const LosTopos::Vec3f &bmax)
	{
		regionMin = bmin;
		regionMax = bmax;
	}
	void setEmitter(LosTopos::Vec3f &bmin, LosTopos::Vec3f &bmax, LosTopos::Vec3f &vel)
	{
		boxEmitter e;
		e.bmin = bmin;
		e.bmax = bmax;
		e.vel = vel;
		emitters.push_back(e);
	}
	//void advance(float dt, float(*phi)(const LosTopos::Vec3f&));
	void advance(float dt, std::function<float(const LosTopos::Vec3f&)> boundary_phi);
	bool isIsolatedParticle(LosTopos::Vec3f &pos);

	//Grid dimensions
	float dx;
	uint total_frame;

	//Eulerian Fluid
	std::shared_ptr<sparse_fluid8x8x8> eulerian_fluids;
    std::shared_ptr<sparse_fluid8x8x8> resample_field;

	//[bulk_idx][index_of_minimum_FLIP_particle_in_this_bulk]
    std::vector<std::vector<uint>> particle_bulks;
	std::vector<minimum_FLIP_particle> particles;
	std::vector<minimum_FLIP_particle> m_new_particles;
	std::vector<minimum_FLIP_particle> m_new_boundary_particles;
	float particle_radius;
	float cfl_dt;
	
    // mesh levelset from outside
    std::vector<moving_solids> mesh_vec;

	SparseMatrixf matrix;
	std::vector<float> rhs;
	std::vector<float> Dofs;


	struct FLIP_options {

		float m_waterline;
		bool m_use_waterline_for_boundary_layer;
		float m_sea_level_gridsize;
		//the interior box size that really holds the FLIP particles
		LosTopos::Vec3f m_inner_boundary_min;
		LosTopos::Vec3f m_inner_boundary_max;
		//the exterior box size that is slightly larger than the FLIP region
		//the gap is reserved for boundary layer
		LosTopos::Vec3f m_exterior_boundary_min;
		LosTopos::Vec3f m_exterior_boundary_max;
	};

	FLIP_options m_flip_options;

	LosTopos::Vec3f regionMin, regionMax;
	std::vector<boxEmitter> emitters;
	

	void reorder_particles();

	struct sort_particle_by_bulks_reducer {
		//the initial constructor
		sort_particle_by_bulks_reducer(
			const std::vector< minimum_FLIP_particle>& in_minimum_FLIP_particles,
			size_t number_of_bulks,
			std::shared_ptr<sparse_fluid8x8x8> sparse_bulk,
			float in_dx, const std::vector<moving_solids>& in_mesh_vec
		):m_unsorted_particles(in_minimum_FLIP_particles),mesh_vec(in_mesh_vec){
			m_sparse_bulk = sparse_bulk;
			m_bin_bulk_particle.resize(number_of_bulks);
			dx = in_dx;
		};

		//the split constructor
		sort_particle_by_bulks_reducer(sort_particle_by_bulks_reducer& x, tbb::split) :m_unsorted_particles(x.m_unsorted_particles),mesh_vec(x.mesh_vec) {
			m_sparse_bulk = x.m_sparse_bulk;
			m_bin_bulk_particle.resize(x.m_bin_bulk_particle.size());
			dx = x.dx;
		}

		void operator()(const tbb::blocked_range<size_t>& r);

		void join(const sort_particle_by_bulks_reducer& other) {
			tbb::parallel_for((size_t)0, m_bin_bulk_particle.size(), [&](size_t bulk_idx) {
				m_bin_bulk_particle[bulk_idx].insert(
					m_bin_bulk_particle[bulk_idx].end(),
					other.m_bin_bulk_particle[bulk_idx].begin(),
					other.m_bin_bulk_particle[bulk_idx].end());
				});
		}

		//reference to particles
		const std::vector<minimum_FLIP_particle>& m_unsorted_particles;
		//reduction result
		//[index of bulk][ith particle in this bulk]
		std::vector<std::vector<unsigned int>> m_bin_bulk_particle;
		std::shared_ptr<sparse_fluid8x8x8> m_sparse_bulk;
		float dx;
		const std::vector<moving_solids>& mesh_vec;
	};

	void assign_particle_to_bulks();
	void emitFluids(float dt, float(*phi)(const LosTopos::Vec3f&));

	//LosTopos::Vec3f get_dvelocity(const LosTopos::Vec3f & position);
	//void compute_delta(Array3f & u, Array3f &u_old, Array3f &u_temp);
	void particle_interpolate(float alpha);
	void FLIP_advection(float dt);
	void particle_to_grid();
    void particle_to_grid(sparse_fluid8x8x8 &_eulerian_fluid, std::vector<minimum_FLIP_particle> &_particles, float dx);
	void fusion_p2g_liquid_phi();
    void extrapolate(sparse_fluid8x8x8 &_eulerian_fluid, int times);

	LosTopos::Vec3f trace_rk3(const LosTopos::Vec3f& position, float dt);

	float cfl();

	void advect_particles(float dt);
	
	void add_force(float dt);
	void project(float dt);
	void extrapolate(int times);
	void constrain_velocity();

	////helpers for pressure projection
	void compute_weights();
	void solve_pressure(float dt);
	void solve_pressure_morton(float dt);
	void solve_pressure_parallel_build(float dt);
	void compute_phi();
	////void computeGradPhi(Array3f & u_temp, int dir);
	////void advectPotential(float dt);

	// BEM and FLIP simulator
    BPS3D * m_bps;
    bool sync_with_bem;
    std::vector<LosTopos::Vec3f> solid_upos;
    std::vector<LosTopos::Vec3f> solid_vpos;
    std::vector<LosTopos::Vec3f> solid_wpos;
    std::vector<LosTopos::Vec3f> solid_u;
    std::vector<LosTopos::Vec3f> solid_v;
    std::vector<LosTopos::Vec3f> solid_w;
    std::vector<float> solid_uweight;
    std::vector<float> solid_vweight;
    void bem_boundaryvel();
	
	void const_velocity_volume(const std::vector<LosTopos::Vec3f>& in_pos, std::vector<LosTopos::Vec3f>& out_vel, LosTopos::Vec3f vel);
	void handle_boundary_layer();
	void seed_and_remove_boundary_layer_particles();
	void assign_boundary_layer_solid_velocities();


	//return if the position is in the boundary layer and below the boundary waterline
	//the waterline could possibly be a volume object
	bool in_boundary_volume(const LosTopos::Vec3d& pos);

	float get_sea_level(const LosTopos::Vec3f& pos);
	bool below_waterline_or_sealevel(const LosTopos::Vec3f& pos);
	void set_sea_level_from_BEM();
	std::vector<std::vector<float>> sea_level_at_voxel_xz;

	bool bulk_contain_boundary(const LosTopos::Vec3i& bulk_ijk);
	void in_boundary_volume(const std::vector<LosTopos::Vec3d>& positions, std::vector<bool>& out_result);

	void resampleVelocity(std::vector<minimum_FLIP_particle>& _particles, float _dx, std::vector<minimum_FLIP_particle>& _resample_pos);

};





#endif