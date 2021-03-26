#pragma once
#include "fluidsim.h"
#include "array3_utils.h"
#include "volumeMeshTools.h"
#include "BPS3D_SolidManager.h"
#include <vector>
#include <memory>
#include <string>

//wraps the flip simulation
//It maintains a fluid simulation environment
class FLIPwrapper {
public:
	FLIPwrapper(BPS3D * sim = nullptr, bool sync_with_bem = false);
	~FLIPwrapper(){}


	void initialize_crownsplash_scene();
	void initialize_rectancular_scene();

	void step();
	float get_particle_radius() const;
	openvdb::FloatGrid::Ptr raw_particle_levelset_ptr() const;
	LosTopos::Vec3f regionmin() const;
	LosTopos::Vec3f regionmax() const;
	void addMeshLevelset(float h, openvdb::FloatGrid::Ptr mesh_ls,
                         std::function<LosTopos::Vec3f(int framenum)> vel_func_in,
                         std::function<LosTopos::Vec3f(int framenum)> pos_func_in);

	std::shared_ptr<FluidSim> m_fluidsim;
protected:
	std::vector<openvdb::Vec3s> points;
	std::vector<openvdb::Vec3I> triangles;
	int m_frame;
	std::string m_outpath;
	std::shared_ptr<BPS3D_SolidManager> m_FLIP_solid_manager;
private:
	static float box_phi(const LosTopos::Vec3f& position, const LosTopos::Vec3f& centre, LosTopos::Vec3f& b);
	static float sphere_phi(const LosTopos::Vec3f& position, const LosTopos::Vec3f& centre, float radius);
	static float boundary_phi(const LosTopos::Vec3f& position);
	static float tank_phi(const LosTopos::Vec3f& position);
	static float waterdrop_phi(const LosTopos::Vec3f& position);
};