/*
#include <zeno/zeno.h>
#include <zeno/NumericObject.h>
#include <vector>
#include <zeno/VDBGrid.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/points/PointAdvect.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/LevelSetTracker.h>
#include <openvdb/tools/Filter.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/ChangeBackground.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <zeno/PrimitiveObject.h>

namespace zeno{
class MyParticleList
{

protected:
	struct MyParticle {

		openvdb::Vec3R p;//, v;
		//openvdb::Real  r;
		MyParticle()
		{
			p = openvdb::Vec3R(0);
			//v = openvdb::Vec3R(0);
			//r = 0;
		}
		MyParticle(const MyParticle &_p)
		{
			p = _p.p;
			//v = _p.v;
			////r = _p.r;
		}
	};
	openvdb::Real           mRadius = 1;
	std::vector<MyParticle> mParticleList;
public:
    typedef openvdb::Vec3R  PosType ;
	typedef openvdb::Vec3R  value_type;

	MyParticleList(size_t size = 0, openvdb::Real radius = 1)
		: mRadius(radius) {
		mParticleList.resize(size);
	}
	void free()
	{
		mParticleList=std::vector<MyParticle>(0);
	}
	void set(int i, const openvdb::Vec3R &p)
	{
		MyParticle pa;
		pa.p = p;
		mParticleList[i] = pa;
	}
	void add(const openvdb::Vec3R &p)
	{
		MyParticle pa;
		pa.p = p;
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
	openvdb::Vec3R vel(int n)   const { return {0, 0, 0}; }
	openvdb::Real radius(int n) const { return mRadius; }

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
		rad = mRadius;
	}
	void getPosRadVel(size_t n, openvdb::Vec3R& pos, openvdb::Real& rad, openvdb::Vec3R& vel) const {
		pos = mParticleList[n].p;
        rad = mRadius;
        vel = {0, 0, 0};
		//rad = mRadiusScale*mParticleList[n].r;
		//vel = mVelocityScale*mParticleList[n].v;
	}
	// The method below is only required for attribute transfer
	void getAtt(size_t n, openvdb::Index32& att) const { att = openvdb::Index32(n); }
};

openvdb::FloatGrid::Ptr rasterizeParticleToVDB(const std::vector<zeno::vec3f>& pos, double radius, double voxelSize)
{
    MyParticleList pa(pos.size(), radius);
    tbb::parallel_for(0, (int)pos.size(), 1, [&](int p)
        {
            auto ppos = pos[p];
            pa.set(p, openvdb::Vec3R(ppos[0], ppos[1], ppos[2]));
        });
    openvdb::FloatGrid::Ptr ls = openvdb::createLevelSet<openvdb::FloatGrid>(voxelSize, 4.0);
    openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, openvdb::Index32> raster(*ls);

    raster.setGrainSize(1);//a value of zero disables threading
    raster.rasterizeSpheres(pa);
    //openvdb::CoordBBox bbox = pa.getBBox(*ls);
    raster.finalize(true);
    return ls;
}

struct RasterizeParticleToVDB : zeno::INode{
    virtual void apply() override {
        auto par = get_input("Particles")->as<zeno::PrimitiveObject>();
        auto radius = get_input("Radius")->as<zeno::NumericObject>()->get<float>();
        float dx = radius/2.0f;
        if(has_input("Dx")) dx = get_input("Dx")->as<zeno::NumericObject>()->get<float>();
        auto result = zeno::IObject::make<VDBFloatGrid>();
        result->m_grid = rasterizeParticleToVDB(par->attr<zeno::vec3f>("pos"), radius, dx);
        set_output("SurfaceSDF", result);
    }
};
ZENDEFNODE(RasterizeParticleToVDB, {
    {"Particles", {"float","Radius"}, {"float","Dx"}},
    {"SurfaceSDF"},
    {},
    {"openvdb"},
});
}
*/
