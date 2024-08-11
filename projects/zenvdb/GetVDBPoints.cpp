#include <zeno/zeno.h>
#include <zeno/ParticlesObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/utils/log.h>
#include <tbb/parallel_for.h>
#include <thread>
#include <map>
#include "reflect/reflection.generated.hpp"


namespace zeno {

struct GetVDBPoints : zeno::INode {
  virtual void apply() override {
    auto grid = get_input("grid")->as<VDBPointsGrid>()->m_grid;

    std::vector<openvdb::points::PointDataTree::LeafNodeType*> leafs;
    grid->tree().getNodes(leafs);
    zeno::log_info("GetVDBPoints: particle leaf nodes: {}", leafs.size());

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
        {"VDBGrid", "grid", "", zeno::Socket_ReadOnly},
    }, /* outputs: */ {
        {"object", "pars"},
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

struct VDBPointsToPrimitive : zeno::INode {
  virtual void apply() override {
    auto grid = get_input("grid")->as<VDBPointsGrid>()->m_grid;

    std::vector<openvdb::points::PointDataTree::LeafNodeType *> leafs;
    grid->tree().getNodes(leafs);
    zeno::log_info("VDBPointsToPrimitive: particle leaf nodes: {}\n", leafs.size());
    auto transform = grid->transformPtr();

    auto ret = zeno::IObject::make<zeno::PrimitiveObject>();
    size_t count = openvdb::points::pointCount(grid->tree());
    zeno::log_info("VDBPointsToPrimitive: particles: {}\n", count);
    ret->resize(count);
    auto &retpos = ret->add_attr<zeno::vec3f>("pos");

    auto hasVel = !leafs.size() || leafs[0]->hasAttribute("v") && leafs[0]->hasAttribute("P");

    //tbb::concurrent_vector<std::tuple<zeno::vec3f,zeno::vec3f>> data(0);
    if (hasVel) {
      auto &retvel = ret->add_attr<zeno::vec3f>("vel");

#if 0
      using MapT = std::map<std::thread::id, size_t>;
      using IterT = typename MapT::iterator;
      MapT pars;
      MapT offsets;
      std::mutex mutex;

      auto counter = [&](auto &range) {
          IterT iter;
          {
              std::lock_guard<std::mutex> lk(mutex);
              bool tag;
              std::tie(iter, tag) = pars.insert(std::make_pair(std::this_thread::get_id(), 0));
          }
          size_t &inc = iter->second;
          for (auto leafIter = range.begin(); leafIter; ++leafIter) {
              size_t count = leafIter->pointCount();
              inc += count;
          }
          //std::cout << "thread [" << std::this_thread::get_id() << "]: " << lcnt << "leafs, " << inc << "pars in total\n";
      };
      {
        openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>> leafman(grid->tree());
        tbb::parallel_for(leafman.leafRange(), counter, tbb::static_partitioner());
      }

      size_t offset = 0;
      for (const auto &[tid, sz] : pars) {
        offsets.emplace(tid, offset);
        offset += sz;
      }

      auto converter = [&](auto &range) {
          auto offset = offsets[std::this_thread::get_id()];

          for (auto leafIter = range.begin(); leafIter; ++leafIter) {

              const openvdb::points::AttributeArray &positionArray = leafIter->constAttributeArray("P");
              const openvdb::points::AttributeArray &velocityArray = leafIter->constAttributeArray("v");

              openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(positionArray);
              openvdb::points::AttributeHandle<openvdb::Vec3f> velocityHandle(velocityArray);

              for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
                  openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);
                  const openvdb::Vec3d xyz = indexIter.getCoord().asVec3d();
                  openvdb::Vec3f worldPosition = grid->transform().indexToWorld(voxelPosition + xyz);

                  openvdb::Vec3f particleVel = velocityHandle.get(*indexIter);

                  retpos[offset] = zeno::vec3f(worldPosition[0], worldPosition[1], worldPosition[2]);
                  retvel[offset++] = zeno::vec3f(particleVel[0], particleVel[1], particleVel[2]);
              }
          }
      };
      {
        openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>> leafman(grid->tree());
        tbb::parallel_for(leafman.leafRange(), converter, tbb::static_partitioner());
      }

#else
      size_t i = 0;
      for (auto leafIter = grid->tree().cbeginLeaf(); leafIter; ++leafIter) {
        const openvdb::points::AttributeArray &positionArray = leafIter->constAttributeArray("P");
        const openvdb::points::AttributeArray &velocityArray = leafIter->constAttributeArray("v");

        openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(positionArray);
        openvdb::points::AttributeHandle<openvdb::Vec3f> velocityHandle(velocityArray);

        for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
            openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);
            const openvdb::Vec3d xyz = indexIter.getCoord().asVec3d();
            openvdb::Vec3f worldPosition = grid->transform().indexToWorld(voxelPosition + xyz);

            openvdb::Vec3f particleVel = velocityHandle.get(*indexIter);

            retpos[i] = zeno::vec3f{worldPosition[0], worldPosition[1], worldPosition[2]};
            retvel[i] = zeno::vec3f{particleVel[0], particleVel[1], particleVel[2]};
            i++;
        }
      }
#endif
    } else {

#if 0
      using MapT = std::map<std::thread::id, size_t>;
      using IterT = typename MapT::iterator;
      MapT pars;
      MapT offsets;
      std::mutex mutex;

      auto counter = [&](auto &range) {
          IterT iter;
          {
              std::lock_guard<std::mutex> lk(mutex);
              bool tag;
              std::tie(iter, tag) = pars.insert(std::make_pair(std::this_thread::get_id(), 0));
          }
          size_t &inc = iter->second;
          for (auto leafIter = range.begin(); leafIter; ++leafIter) {
              size_t count = leafIter->pointCount();
              inc += count;
          }
          //std::cout << "thread [" << std::this_thread::get_id() << "]: " << lcnt << "leafs, " << inc << "pars in total\n";
      };
      {
        openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>> leafman(grid->tree());
        tbb::parallel_for(leafman.leafRange(), counter, tbb::static_partitioner());
      }

      size_t offset = 0;
      for (const auto &[tid, sz] : pars) {
        offsets.emplace(tid, offset);
        offset += sz;
      }

      auto converter = [&](auto &range) {
          auto offset = offsets[std::this_thread::get_id()];

          for (auto leafIter = range.begin(); leafIter; ++leafIter) {
              const openvdb::points::AttributeArray &positionArray = leafIter->constAttributeArray("P");

              openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(positionArray);

              for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
                  openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);
                  const openvdb::Vec3d xyz = indexIter.getCoord().asVec3d();
                  openvdb::Vec3f worldPosition = grid->transform().indexToWorld(voxelPosition + xyz);

                  retpos[offset++] = zeno::vec3f(worldPosition[0], worldPosition[1], worldPosition[2]);
              }
          }
      };
      {
        openvdb::tree::LeafManager<std::decay_t<decltype(grid->tree())>> leafman(grid->tree());
        tbb::parallel_for(leafman.leafRange(), converter, tbb::static_partitioner());
      }
#else
      size_t i = 0;
      for (auto leafIter = grid->tree().cbeginLeaf(); leafIter; ++leafIter) {
        const openvdb::points::AttributeArray &positionArray = leafIter->constAttributeArray("P");

        openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(positionArray);

        for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
            openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);
            const openvdb::Vec3d xyz = indexIter.getCoord().asVec3d();
            openvdb::Vec3f worldPosition = grid->transform().indexToWorld(voxelPosition + xyz);

            retpos[i] = zeno::vec3f{worldPosition[0], worldPosition[1], worldPosition[2]};
            i++;
        }
      }
#endif
    }

    zeno::log_info("VDBPointsToPrimitive: complete\n");
    set_output("prim", ret);
  }
};

static int defVDBPointsToPrimitive = zeno::defNodeClass<VDBPointsToPrimitive>("VDBPointsToPrimitive",
    { /* inputs: */ {
        {"VDBGrid", "grid", "", zeno::Socket_ReadOnly},
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
    zeno::log_info("GetVDBPointsDroplets: particle leaf nodes: {}", leafs.size());

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
        {"VDBGrid", "grid", "", zeno::Socket_ReadOnly},
        {"VDBGrid", "sdf", "", zeno::Socket_ReadOnly},
    }, /* outputs: */ {
        "prim",
    }, /* params: */ {
    }, /* category: */ {
      "openvdb",
    }});


struct ConvertTo_VDBPointsGrid_PrimitiveObject : VDBPointsToPrimitive {
    virtual void apply() override {
        VDBPointsToPrimitive::apply();
        //get_input<PrimitiveObject>("prim")->move_assign(std::move(smart_any_cast<std::shared_ptr<IObject>>(anyToZAny(get_output_obj("prim"), zeno::types::gParamType_Primitive))).get());
    }
};

ZENO_DEFOVERLOADNODE(ConvertTo, _VDBPointsGrid_PrimitiveObject, typeid(VDBPointsGrid).name(), typeid(PrimitiveObject).name())({
        {
            {"VDBGrid", "grid", "", zeno::Socket_ReadOnly},
            {"", "prim", "", zeno::Socket_ReadOnly},
        },
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
