#include <zeno/zeno.h>
#include <zeno/ParticlesObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/VDBGrid.h>
#include "tbb/concurrent_vector.h"
#include "tbb/parallel_for.h"
#include "tbb/scalable_allocator.h"
namespace zeno {

struct GetVDBPoints : zeno::INode {
  virtual void apply() override {
    auto grid = get_input("grid")->as<VDBPointsGrid>()->m_grid;

    std::vector<openvdb::points::PointDataTree::LeafNodeType*> leafs;
    grid->tree().getNodes(leafs);
    printf("GetVDBPoints: particle leaf nodes: %ld\n", leafs.size());

    auto transform = grid->transformPtr();

    auto ret = zeno::IObject::make<ParticlesObject>();

    for (auto const &leaf: leafs) {
      //attributes
      // Attribute reader
      // Extract the position attribute from the leaf by name (P is position).
      openvdb::points::AttributeArray& positionArray =
        leaf->attributeArray("P");
      // Extract the velocity attribute from the leaf by name (v is velocity).
      openvdb::points::AttributeArray& velocityArray =
        leaf->attributeArray("v");

      using PositionCodec = openvdb::points::FixedPointCodec</*one byte*/false>;
      using VelocityCodec = openvdb::points::TruncateCodec;
      // Create read handles for position and velocity
      openvdb::points::AttributeHandle<openvdb::Vec3f, PositionCodec> positionHandle(positionArray);
      openvdb::points::AttributeHandle<openvdb::Vec3f, VelocityCodec> velocityHandle(velocityArray);

      for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
        openvdb::Vec3R p = positionHandle.get(*iter);
        p += iter.getCoord().asVec3d();
        // https://people.cs.clemson.edu/~jtessen/cpsc8190/OpenVDB-dpawiki.pdf
        p = transform->indexToWorld(p);
        openvdb::Vec3R v = velocityHandle.get(*iter);
        ret->pos.push_back(glm::vec3(p[0], p[1], p[2]));
        ret->vel.push_back(glm::vec3(v[0], v[1], v[2]));
      }
    }
    set_output("pars", ret);
  }
};

static int defGetVDBPoints = zeno::defNodeClass<GetVDBPoints>("GetVDBPoints",
    { /* inputs: */ {
        "grid",
    }, /* outputs: */ {
        "pars",
    }, /* params: */ {
    }, /* category: */ {
      "deprecated",
    }});

#if 0
struct GetVDBPointsLeafCount : zeno::INode {
  virtual void apply() override {
    auto grid = get_input("grid")->as<VDBPointsGrid>()->m_grid;
    std::vector<openvdb::points::PointDataTree::LeafNodeType*> leafs;
    grid->tree().getNodes(leafs);
    auto ret = std::make_shared<zeno::NumericObject>();
    ret->set((int)leafs.size());
    set_output("leafCount", std::move(ret));
  }
};
#endif


//TODO: parallelize using concurrent vector
struct VDBPointsToPrimitive : zeno::INode {
  virtual void apply() override {
    auto grid = get_input("grid")->as<VDBPointsGrid>()->m_grid;

    std::vector<openvdb::points::PointDataTree::LeafNodeType*> leafs;
    grid->tree().getNodes(leafs);
    printf("GetVDBPoints: particle leaf nodes: %ld\n", leafs.size());

    auto transform = grid->transformPtr();

    auto ret = zeno::IObject::make<zeno::PrimitiveObject>();
    auto &retpos = ret->add_attr<zeno::vec3f>("pos");
    auto &retvel = ret->add_attr<zeno::vec3f>("vel");

    //tbb::concurrent_vector<std::tuple<zeno::vec3f,zeno::vec3f>> data(0);
    std::vector<std::vector<std::tuple<zeno::vec3f,zeno::vec3f>>> data(leafs.size());
    for(int i=0;i<leafs.size();i++)
    {
      data[i].resize(0);
      data[i].reserve(512*32);
    }
    tbb::parallel_for((size_t)0, (size_t)leafs.size(), (size_t)1, [&](size_t index)
    //for (auto const &leaf: leafs)
    {
      auto &leaf = leafs[index];
      //attributes
      // Attribute reader
      // Extract the position attribute from the leaf by name (P is position).
      openvdb::points::AttributeArray& positionArray =
        leaf->attributeArray("P");
      // Extract the velocity attribute from the leaf by name (v is velocity).
      openvdb::points::AttributeArray& velocityArray =
        leaf->attributeArray("v");

      using PositionCodec = openvdb::points::FixedPointCodec</*one byte*/false>;
      using VelocityCodec = openvdb::points::TruncateCodec;
      // Create read handles for position and velocity
      openvdb::points::AttributeHandle<openvdb::Vec3f, PositionCodec> positionHandle(positionArray);
      openvdb::points::AttributeHandle<openvdb::Vec3f, VelocityCodec> velocityHandle(velocityArray);

      for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
        openvdb::Vec3R p = positionHandle.get(*iter);
        p += iter.getCoord().asVec3d();
        // https://people.cs.clemson.edu/~jtessen/cpsc8190/OpenVDB-dpawiki.pdf
        p = transform->indexToWorld(p);
        openvdb::Vec3R v = velocityHandle.get(*iter);
        //retpos.emplace_back(p[0], p[1], p[2]);
        //retvel.emplace_back(v[0], v[1], v[2]);
        data[index].emplace_back(std::make_tuple(zeno::vec3f(p[0],p[1],p[2]), zeno::vec3f(v[0],v[1],v[2])));
      }
    });
    std::vector<size_t> sum_table(data.size()+1);
    sum_table[0] = 0;
    for(size_t i=0;i<data.size();i++)
    {
      sum_table[i+1] = sum_table[i] +data[i].size();
    }
    size_t count = sum_table[sum_table.size()-1];
    std::vector<std::tuple<zeno::vec3f,zeno::vec3f>> data2;
    data2.resize(0);
    data2.reserve(count);
    ret->resize(count);
    for(size_t i=0;i<data.size();i++)
    {
      data2.insert(data2.end(), data[i].begin(), data[i].end());
    }
    tbb::parallel_for((size_t)0, (size_t)ret->size(), (size_t)1, 
    [&](size_t index)
    {
      retpos[index] = std::get<0>(data2[index]);
      retvel[index] = std::get<1>(data2[index]);
    });
    set_output("prim", ret);
  }
};

static int defVDBPointsToPrimitive = zeno::defNodeClass<VDBPointsToPrimitive>("VDBPointsToPrimitive",
    { /* inputs: */ {
        "grid",
    }, /* outputs: */ {
        "prim",
    }, /* params: */ {
    }, /* category: */ {
      "openvdb",
    }});




struct GetVDBPointsDroplets : zeno::INode {
  virtual void apply() override {
    auto grid = get_input("grid")->as<VDBPointsGrid>()->m_grid;
    auto sdf = get_input("sdf")->as<VDBFloatGrid>()->m_grid;
    auto dx = sdf->voxelSize()[0];
    std::vector<openvdb::points::PointDataTree::LeafNodeType*> leafs;
    grid->tree().getNodes(leafs);
    printf("GetVDBPoints: particle leaf nodes: %ld\n", leafs.size());

    auto transform = grid->transformPtr();

    auto ret = zeno::IObject::make<zeno::PrimitiveObject>();
    auto &retpos = ret->add_attr<zeno::vec3f>("pos");
    auto &retvel = ret->add_attr<zeno::vec3f>("vel");

    tbb::concurrent_vector<std::tuple<zeno::vec3f,zeno::vec3f>> data(0);
    tbb::parallel_for((size_t)0, (size_t)leafs.size(), (size_t)1, [&](size_t index)
    //for (auto const &leaf: leafs)
    {
      auto &leaf = leafs[index];
      //attributes
      // Attribute reader
      // Extract the position attribute from the leaf by name (P is position).
      openvdb::points::AttributeArray& positionArray =
        leaf->attributeArray("P");
      // Extract the velocity attribute from the leaf by name (v is velocity).
      openvdb::points::AttributeArray& velocityArray =
        leaf->attributeArray("v");

      using PositionCodec = openvdb::points::FixedPointCodec</*one byte*/false>;
      using VelocityCodec = openvdb::points::TruncateCodec;
      // Create read handles for position and velocity
      openvdb::points::AttributeHandle<openvdb::Vec3f, PositionCodec> positionHandle(positionArray);
      openvdb::points::AttributeHandle<openvdb::Vec3f, VelocityCodec> velocityHandle(velocityArray);

      for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
        openvdb::Vec3R p = positionHandle.get(*iter);
        p += iter.getCoord().asVec3d();
        // https://people.cs.clemson.edu/~jtessen/cpsc8190/OpenVDB-dpawiki.pdf
        p = transform->indexToWorld(p);
        openvdb::Vec3R v = velocityHandle.get(*iter);
        //retpos.emplace_back(p[0], p[1], p[2]);
        //retvel.emplace_back(v[0], v[1], v[2]);
        auto p2 = sdf->worldToIndex(p);
        auto val = openvdb::tools::BoxSampler::sample(sdf->tree(), p2);
        if(val>dx)
          data.emplace_back(std::make_tuple(zeno::vec3f(p[0],p[1],p[2]), zeno::vec3f(v[0],v[1],v[2])));
      }
    });
    ret->resize(data.size());
    tbb::parallel_for((size_t)0, (size_t)ret->size(), (size_t)1, 
    [&](size_t index)
    {
      retpos[index] = std::get<0>(data[index]);
      retvel[index] = std::get<1>(data[index]);
    });
    set_output("prim", ret);
  }
};

static int defGetVDBPointsDroplets = zeno::defNodeClass<GetVDBPointsDroplets>("GetVDBPointsDroplets",
    { /* inputs: */ {
        "grid","sdf"
    }, /* outputs: */ {
        "prim",
    }, /* params: */ {
    }, /* category: */ {
      "openvdb",
    }});


struct ConvertTo_VDBPointsGrid_PrimitiveObject : VDBPointsToPrimitive {
    virtual void apply() override {
        VDBPointsToPrimitive::apply();
        get_input<PrimitiveObject>("prim")->move_assign(std::move(smart_any_cast<std::shared_ptr<IObject>>(outputs.at("prim"))).get());
    }
};

ZENO_DEFOVERLOADNODE(ConvertTo, _VDBPointsGrid_PrimitiveObject, typeid(VDBPointsGrid).name(), typeid(PrimitiveObject).name())({
        {"grid", "prim"},
        {},
        {},
        {"primitive"},
});

#if 0
// TODO: ToVisualize is deprecated in zeno2, please impl this directly in the zenovis module later...
struct ToVisualize_VDBPointsGrid : VDBPointsToPrimitive {
    virtual void apply() override {
        VDBPointsToPrimitive::apply();
        auto path = get_param<std::string>("path");
        auto prim = std::move(smart_any_cast<std::shared_ptr<IObject>>(outputs.at("prim")));
        if (auto node = graph->getOverloadNode("ToVisualize", {std::move(prim)}); node) {
            node->inputs["path:"] = std::make_shared<StringObject>(path);
            node->doApply();
        }
    }
};

ZENO_DEFOVERLOADNODE(ToVisualize, _VDBPointsGrid, typeid(VDBPointsGrid).name())({
        {"grid"},
        {},
        {{"string", "path", ""}},
        {"primitive"},
});
#endif


}
