#include "FLIPwrapper.h"
#include "array3_utils.h"
#include "volumeMeshTools.h"
#include "array3_utils.h"
#include "SimOptions.h"
#include <algorithm>
#include "fluidsim.h"
#include "Sparse_buffer.h"
float FLIPwrapper::box_phi(const LosTopos::Vec3f& position, const LosTopos::Vec3f& centre, LosTopos::Vec3f& b)
{
	//vec3 d = abs(p) - b;
	//return min(max(d.x,max(d.y,d.z)),0.0) +
	//	length(max(d,0.0));


	LosTopos::Vec3f p = position - centre;
	LosTopos::Vec3f d = LosTopos::fabs(p) - b;
	return std::min(std::max(d[0], std::max(d[1], d[2])), 0.0f)
		+ dist(LosTopos::Vec3f(std::max(d[0], 0.0f), std::max(d[1], 0.0f), std::max(d[2], 0.0f)), LosTopos::Vec3f(0, 0, 0));
}

float FLIPwrapper::sphere_phi(const LosTopos::Vec3f& position, const LosTopos::Vec3f& centre, float radius) {
	return dist(position, centre) - radius;
}

float FLIPwrapper::boundary_phi(const LosTopos::Vec3f& position)
{
	//    return -sphere_phi(position, LosTopos::Vec3f(0,0,0),0.7f);
	LosTopos::Vec3f b(0.4f, 0.4f, 0.4f);
	return -box_phi(position, LosTopos::Vec3f(0, 0.25, 0), b);
}


float FLIPwrapper::tank_phi(const LosTopos::Vec3f& position) {
	//	return sphere_phi(position, LosTopos::Vec3f(0,0,0), 0.5f);
	LosTopos::Vec3f b(0.4f, 0.1f, 0.4f);
	return box_phi(position, LosTopos::Vec3f(0, -0.1, 0), b);
}

float FLIPwrapper::waterdrop_phi(const LosTopos::Vec3f &position)
{
    LosTopos::Vec3f center(0, 0.2, 0);
    return sphere_phi(position, center, 0.06);
}

static float cylinder_tank_phi(const LosTopos::Vec3f& position)
{
	LosTopos::Vec3d bottom_center{ 0,-Options::doubleValue("FLIPpool-depth") ,0 };
	double radius = Options::doubleValue("FLIPpool-radius");
	double height = Options::doubleValue("FLIPpool-depth");
	double close_threshold = 5e-5;
	bool is_container = false;
	solid::cylinder_solid cylinder_water(bottom_center, radius, height, close_threshold, is_container);

	
	float temp = (float)cylinder_water.sdf(LosTopos::Vec3d{ position });
	return  temp;
}

FLIPwrapper::FLIPwrapper(BPS3D * sim, bool sync_with_bem) {
	openvdb::initialize();
	m_fluidsim = std::make_shared<FluidSim>(sim, sync_with_bem);
	
	//initialize_crownsplash_scene();
	initialize_rectancular_scene();
}

void FLIPwrapper::addMeshLevelset(float h, openvdb::FloatGrid::Ptr mesh_ls,
                                  std::function<LosTopos::Vec3f(int framenum)> vel_func_in,
                                  std::function<LosTopos::Vec3f(int framenum)> pos_func_in)
{
    moving_solids m_solid(h, mesh_ls, vel_func_in, pos_func_in);
    m_fluidsim->mesh_vec.push_back(m_solid);
}

void FLIPwrapper::initialize_crownsplash_scene()
{
	std::vector<openvdb::Vec3s> points;
	std::vector<openvdb::Vec3I> triangles;

	double dx = Options::doubleValue("FLIPdx");
	m_fluidsim->initialize(dx);

	m_outpath = "./flipsim";
	printf("Initializing liquid\n");
	auto drop_phi = [](const LosTopos::Vec3f& pos) {
		LosTopos::Vec3f center(0, Options::doubleValue("crownsplash-drop-pos"), 0);
		return sphere_phi(pos, center, Options::doubleValue("radius"));
	};

	double radius = Options::doubleValue("FLIPpool-radius");
	double height = Options::doubleValue("FLIPpool-depth");
	double drop_radius = Options::doubleValue("radius");
	double drop_pos_y = Options::doubleValue("crownsplash-drop-pos");
	m_fluidsim->set_liquid(LosTopos::Vec3f(- 2 * radius, - 2 * height, -2  * radius), LosTopos::Vec3f(2 * radius, 3 * height, 2 * radius), cylinder_tank_phi);

	LosTopos::Vec3f drop_pad = LosTopos::Vec3f{ 1,1,1 }*drop_radius * 1.1;
	LosTopos::Vec3f drop_center = LosTopos::Vec3f{ 0,(float)drop_pos_y,0 };
	m_fluidsim->set_liquid(drop_center - drop_pad, drop_center + drop_pad, drop_phi);
	std::cout << "particles:" << m_fluidsim->particles.size() << std::endl;
	m_fluidsim->setSolverRegion(LosTopos::Vec3f(-3.0, -3.0, -3.0), LosTopos::Vec3f(3.0, 3.0, 3.0));
	//vdbToolsWapper::export_VDB(m_outpath, 0, m_fluidsim->particles, m_fluidsim->particle_radius, points, triangles, m_fluidsim->eulerian_fluids);

	//	sim.set_liquid(LosTopos::Vec3f(0.15,0.18,0.025),LosTopos::Vec3f(0.35,0.5,0.475));
	m_fluidsim->init_domain();
	printf("Initializing boundary\n");


	printf("Exporting initial data\n");
	m_frame = 0;

	//boundaries
	m_FLIP_solid_manager = std::make_shared<BPS3D_SolidManager>();
	{
		LosTopos::Vec3d cylinder_bottom = LosTopos::Vec3d{ 0,-Options::doubleValue("FLIPpool-depth"),0 };
		double radius = Options::doubleValue("FLIPpool-radius");
		double height = 5 * Options::doubleValue("FLIPpool-depth");
		double close_threshold = 5e-3;//not useful for now
		bool is_container = true;
		m_FLIP_solid_manager->add_cylinder(cylinder_bottom, radius, height, close_threshold, is_container);
	}

}

void FLIPwrapper::initialize_rectancular_scene()
{
	std::vector<openvdb::Vec3s> points;
	std::vector<openvdb::Vec3I> triangles;

	double voxel_dx = Options::doubleValue("FLIPdx");

	m_fluidsim->initialize(voxel_dx);

	m_outpath = "./flipsim";
	printf("Initializing liquid\n");

	auto drop_phi = [](const LosTopos::Vec3f& pos) {
		LosTopos::Vec3f center(0, Options::doubleValue("crownsplash-drop-pos"), 0);
		return sphere_phi(pos, center, Options::doubleValue("radius"));
	};

	//The flip domain size
	double minx = Options::doubleValue("FLIPpool-minx");
	double miny = Options::doubleValue("FLIPpool-miny");
	double minz = Options::doubleValue("FLIPpool-minz");
	double maxx = Options::doubleValue("FLIPpool-maxx");
	double maxy = Options::doubleValue("FLIPpool-maxy");
	double maxz = Options::doubleValue("FLIPpool-maxz");

	//re-set the maximum value so that the max-min is multiple of bulksize
	double bulk_size = 8*voxel_dx;

	//align the minimum with the voxel
	minx = voxel_dx * std::floor(minx / voxel_dx);
	miny = voxel_dx * std::floor(miny / voxel_dx);
	minz = voxel_dx * std::floor(minz / voxel_dx);

	//align the maximum with the fluid bulk
	maxx = minx + bulk_size * std::ceil((maxx - minx) / bulk_size);
	maxy = miny + bulk_size * std::ceil((maxy - miny) / bulk_size);
	maxz = minz + bulk_size * std::ceil((maxz - minz) / bulk_size);

	//boundaries
	m_FLIP_solid_manager = std::make_shared<BPS3D_SolidManager>();
	{
		double close_threshold = 5e-3;//not useful for now
		bool is_container = true;
		m_FLIP_solid_manager->add_box(LosTopos::Vec3d(minx, miny, minz), LosTopos::Vec3d(maxx, maxy, maxz), close_threshold, is_container);

		//add cylinder obstacle
		LosTopos::Vec3d cylinder_bottom = LosTopos::Vec3d{ 0,-Options::doubleValue("FLIPpool-depth"),0 };
		double radius = Options::doubleValue("FLIPpool-radius");
		double height = 2 * Options::doubleValue("FLIPpool-depth");
		//m_FLIP_solid_manager->add_cylinder(cylinder_bottom, radius, height, close_threshold, /*is_container*/ false);
	}

	m_fluidsim->m_flip_options.m_inner_boundary_min = LosTopos::Vec3f(minx, miny, minz);
	m_fluidsim->m_flip_options.m_inner_boundary_max = LosTopos::Vec3f(maxx, maxy, maxz);
	double waterline = Options::doubleValue("FLIPpool-waterline");

	//set the sea level for the simulation and possibly use it for the new particle seeding
	m_fluidsim->m_flip_options.m_waterline = waterline;
	m_fluidsim->m_flip_options.m_use_waterline_for_boundary_layer = false;

	//set the original fluid filled below the waterline
	m_fluidsim->set_liquid(
		LosTopos::Vec3f(minx, std::min(miny, waterline), minz), 
		LosTopos::Vec3f(maxx, std::min(maxy, waterline), maxz), 
		[&](const LosTopos::Vec3f& pos) {return -float(m_FLIP_solid_manager->sdf(LosTopos::Vec3d(pos))); });
	
	double drop_radius = Options::doubleValue("radius");
	double drop_pos_y = Options::doubleValue("crownsplash-drop-pos");
	LosTopos::Vec3f drop_pad = LosTopos::Vec3f{ 1,1,1 }*drop_radius * 1.1;
	LosTopos::Vec3f drop_center = LosTopos::Vec3f{ 0,(float)drop_pos_y,0 };
//	m_fluidsim->set_liquid(drop_center - drop_pad, drop_center + drop_pad, drop_phi);

	std::cout << "particles:" << m_fluidsim->particles.size() << std::endl;

	tbb::parallel_for(size_t(0), m_fluidsim->particles.size(), [&](size_t pidx) {
		m_fluidsim->particles[pidx].vel[0] = Options::doubleValue("tank_flow_x");
		m_fluidsim->particles[pidx].vel[2] = Options::doubleValue("tank_flow_z");
		});

	m_fluidsim->m_flip_options.m_exterior_boundary_min = LosTopos::Vec3f(minx, miny, minz) - LosTopos::Vec3f(1, 1, 1) * 1.5 * bulk_size;
	m_fluidsim->m_flip_options.m_exterior_boundary_max = LosTopos::Vec3f(maxx, maxy, maxz) + LosTopos::Vec3f(1, 1, 1) * 1.5 * bulk_size;
	m_fluidsim->setSolverRegion(
		m_fluidsim->m_flip_options.m_exterior_boundary_min,
		m_fluidsim->m_flip_options.m_exterior_boundary_max);

	//vdbToolsWapper::export_VDB(m_outpath, 0, m_fluidsim->particles, m_fluidsim->particle_radius, points, triangles, m_fluidsim->eulerian_fluids);

	//	sim.set_liquid(LosTopos::Vec3f(0.15,0.18,0.025),LosTopos::Vec3f(0.35,0.5,0.475));
	m_fluidsim->init_domain();
	//Simulate
	m_fluidsim->handle_boundary_layer();
	printf("Initializing boundary\n");
	//set the initial volume velocity
	m_fluidsim->fusion_p2g_liquid_phi();
	m_fluidsim->extrapolate(4);
	tbb::parallel_for((size_t)0,
		(size_t)m_fluidsim->eulerian_fluids->n_bulks,
		(size_t)1,
		[&](size_t index)
		{
			for (int i = 0; i < m_fluidsim->eulerian_fluids->n_perbulk; i++)
			{
				m_fluidsim->eulerian_fluids->fluid_bulk[index].u_save.data[i]
					= m_fluidsim->eulerian_fluids->fluid_bulk[index].u.data[i];
				m_fluidsim->eulerian_fluids->fluid_bulk[index].v_save.data[i]
					= m_fluidsim->eulerian_fluids->fluid_bulk[index].v.data[i];
				m_fluidsim->eulerian_fluids->fluid_bulk[index].w_save.data[i]
					= m_fluidsim->eulerian_fluids->fluid_bulk[index].w.data[i];
			}

		});

	printf("Exporting initial data\n");
	m_frame = 0;

	

}

void FLIPwrapper::step()
{
	m_frame++;
	printf("--------------------\nFrame %d\n--------------------\n", m_frame);

	//Simulate
	auto sdf = [this](const LosTopos::Vec3f& pos) {
		return (float)m_FLIP_solid_manager->sdf(LosTopos::Vec3d{ pos });
	};
	printf("Simulating liquid\n");
	{
		float dt = Options::doubleValue("time-step");
		m_fluidsim->advance(dt, sdf);
	}

	//vdbToolsWapper::outputBgeo(m_outpath, m_frame, m_fluidsim->solid_upos, m_fluidsim->solid_u, m_fluidsim->solid_uweight, m_fluidsim->solid_vweight);
	//vdbToolsWapper::export_VDB(m_outpath, m_frame, m_fluidsim->particles,m_fluidsim->particle_radius, points, triangles, m_fluidsim->eulerian_fluids);
	//vdbToolsWapper::outputBgeo(m_outpath, m_frame, m_fluidsim->particles);
    //vdbToolsWapper::export_VDBmesh(m_outpath, m_frame, m_fluidsim->mesh_vec);
	//vdbToolsWapper::outputgeo(m_outpath, m_frame, m_fluidsim->particles);
//	vdbToolsWapper::outputBgeo(m_outpath, m_frame, points, m_fluidsim->eulerian_fluids);
	//printf("Exporting particle data\n");

    printf("--------------------\nFinish Frame %d\n--------------------\n", m_frame);
}

float FLIPwrapper::get_particle_radius() const
{
	return m_fluidsim->particle_radius;
}

openvdb::FloatGrid::Ptr FLIPwrapper::raw_particle_levelset_ptr() const
{
	double voxelSize =  m_fluidsim->particle_radius / 1.001 * 2.0 / sqrt(3.0) / 2.0;
	double halfWidth = 2.0;
	openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, halfWidth);
	openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, openvdb::Index32> raster(*ls);
	MyParticleList pa(m_fluidsim->particles.size(), 1, 1);
	tbb::parallel_for(0, (int)m_fluidsim->particles.size(), 1, [&](int p)
		{
			LosTopos::Vec3f pos = m_fluidsim->particles[p].pos;
			pa.set(p, openvdb::Vec3R((double)(pos[0]), (double)(pos[1]), (double)(pos[2])), m_fluidsim->particle_radius);
		});

	//raster.setGrainSize(1);//a value of zero disables threading
	raster.rasterizeSpheres(pa);
	openvdb::CoordBBox bbox = pa.getBBox(*ls);
	std::cout << bbox.min() << std::endl;
	std::cout << bbox.max() << std::endl;
	raster.finalize(true);

	return ls;
}

LosTopos::Vec3f FLIPwrapper::regionmin() const
{
	return m_fluidsim->regionMin;
}

LosTopos::Vec3f FLIPwrapper::regionmax() const
{
	return m_fluidsim->regionMax;
}

