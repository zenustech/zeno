#ifndef VOLUMETOOLS_H
#define VOLUMETOOLS_H
#include <vector>
#include "vec.h"
#include <openvdb/math/Math.h>
#include <openvdb/openvdb.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/tree/LeafNode.h>
#include "openvdb/tools/ParticlesToLevelSet.h"
#include "openvdb/tools/LevelSetFilter.h"
#include "openvdb/tools/VolumeToMesh.h"
#include "openvdb/tools/MeshToVolume.h"
#include "openvdb/tools/Interpolation.h"
#include "openvdb/tools/GridOperators.h"
#include "openvdb/tools/LevelSetSphere.h"
#include "openvdb/tools/LevelSetPlatonic.h"
#include <openvdb/util/Util.h>
#include <glm/vec3.hpp>
#ifdef USE_PARTIO
#include <Partio.h>
#endif

#include "FLIP_particle.h"
#include <memory>
template<int N>
struct sparse_fluid_3D;

struct FLIP_particle;

class moving_solids;

typedef sparse_fluid_3D<8> sparse_fluid8x8x8;
typedef sparse_fluid_3D<6> sparse_fluid6x6x6;
class MyParticleList
{

protected:
	struct MyParticle {

		openvdb::Vec3R p, v;
		openvdb::Real  r;
		MyParticle()
		{
			p = openvdb::Vec3R(0);
			v = openvdb::Vec3R(0);
			r = 0;
		}
		MyParticle(const MyParticle &_p)
		{
			p = _p.p;
			v = _p.v;
			r = _p.r;
		}
	};
	openvdb::Real           mRadiusScale;
	openvdb::Real           mVelocityScale;
	std::vector<MyParticle> mParticleList;
public:
    typedef openvdb::Vec3R  PosType ;
	typedef openvdb::Vec3R  value_type;

	MyParticleList(size_t size, openvdb::Real rScale = 1, openvdb::Real vScale = 1)
		: mRadiusScale(rScale), mVelocityScale(vScale) {
		mParticleList.resize(size);
	}
	MyParticleList(openvdb::Real rScale = 1, openvdb::Real vScale = 1)
		: mRadiusScale(rScale), mVelocityScale(vScale) {
		mParticleList.resize(0);
	}
	void free()
	{
		//mParticleList=std::vector<MyParticle>(0);
	}
	void set(int i, const openvdb::Vec3R &p, const openvdb::Real &r,
		const openvdb::Vec3R &v = openvdb::Vec3R(0, 0, 0))
	{
		MyParticle pa;
		pa.p = p;
		pa.r = r;
		pa.v = v;
		mParticleList[i] = pa;
	}
	void add(const openvdb::Vec3R &p, const openvdb::Real &r,
		const openvdb::Vec3R &v = openvdb::Vec3R(0, 0, 0))
	{
		MyParticle pa;
		pa.p = p;
		pa.r = r;
		pa.v = v;
		mParticleList.push_back(pa);
	}
	/// @return coordinate bbox in the space of the specified transfrom
	openvdb::CoordBBox getBBox(const openvdb::GridBase& grid) {
		openvdb::CoordBBox bbox;
		openvdb::Coord &min = bbox.min(), &max = bbox.max();
		openvdb::Vec3R pos;
		openvdb::Real rad, invDx = 1.0 / grid.voxelSize()[0];
		for (size_t n = 0, e = this->size(); n<e; ++n) {
			this->getPosRad(n, pos, rad);
			const openvdb::Vec3d xyz = grid.worldToIndex(pos);
			const openvdb::Real   r = rad * invDx;
			for (int i = 0; i<3; ++i) {
				min[i] = openvdb::math::Min(min[i], openvdb::math::Floor(xyz[i] - r));
				max[i] = openvdb::math::Max(max[i], openvdb::math::Ceil(xyz[i] + r));
			}
		}
		return bbox;
	}
	//typedef int AttributeType;
	// The methods below are only required for the unit-tests
	openvdb::Vec3R pos(int n)   const { return mParticleList[n].p; }
	openvdb::Vec3R vel(int n)   const { return mVelocityScale*mParticleList[n].v; }
	openvdb::Real radius(int n) const { return mRadiusScale*mParticleList[n].r; }

	//////////////////////////////////////////////////////////////////////////////
	/// The methods below are the only ones required by tools::ParticleToLevelSet
	/// @note We return by value since the radius and velocities are modified
	/// by the scaling factors! Also these methods are all assumed to
	/// be thread-safe.

	/// Return the total number of particles in list.
	///  Always required!
	size_t size() const { return mParticleList.size(); }

	/// Get the world space position of n'th particle.
	/// Required by ParticledToLevelSet::rasterizeSphere(*this,radius).
	void getPos(size_t n, openvdb::Vec3R&pos) const { pos = mParticleList[n].p; }


	void getPosRad(size_t n, openvdb::Vec3R& pos, openvdb::Real& rad) const {
		pos = mParticleList[n].p;
		rad = mRadiusScale*mParticleList[n].r;
	}
	void getPosRadVel(size_t n, openvdb::Vec3R& pos, openvdb::Real& rad, openvdb::Vec3R& vel) const {
		pos = mParticleList[n].p;
		rad = mRadiusScale*mParticleList[n].r;
		vel = mVelocityScale*mParticleList[n].v;
	}
	// The method below is only required for attribute transfer
	void getAtt(size_t n, openvdb::Index32& att) const { att = openvdb::Index32(n); }
};
struct vdbToolsWapper{
	static void writeObj(const std::string& objname, const std::vector<openvdb::Vec3f>& verts, const std::vector <openvdb::Vec4I>& faces);

	static openvdb::FloatGrid::Ptr readMeshToLevelset(const std::string& filename, float h);
    
	static openvdb::FloatGrid::Ptr particleToLevelset(const std::vector<FLIP_particle>& particles, double radius, double voxelSize);

	static void export_VDB(std::string path, int frame, const std::vector<FLIP_particle>& particles, double radius, std::vector<openvdb::Vec3s>& points, std::vector<openvdb::Vec3I>& triangles, std::shared_ptr<sparse_fluid8x8x8> eulerian_fluids);
    static void export_VDBmesh(std::string path, int frame, const std::vector<moving_solids>& mesh_ls);

	static void outputgeo(std::string path, int frame, const std::vector<FLIP_particle>& particles);
	static void outputBgeo(std::string path, int frame, const std::vector<LosTopos::Vec3f>& p_pos, const std::vector<LosTopos::Vec3f>& p_vel, const std::vector<float>& p_w, const std::vector<float>& p_v);

	static void outputBgeo(std::string path, int frame, const std::vector<openvdb::Vec3s>& points, std::shared_ptr<sparse_fluid8x8x8> eulerian_fluids);
	static void outputBgeo(std::string path, int frame, const std::vector<FLIP_particle>& flip_p);
	static void outputBgeo(std::string path, const std::vector<glm::vec3> &pos, const std::vector<glm::vec3> &vel);
};
#endif //VOLUMETOOLS_H
