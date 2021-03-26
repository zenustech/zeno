#if 1
#include <zen/zen.h>
#include "volumeMeshTools.h"
#include "SimOptions.h"
#include "FLIP_vdb.h"
#include <zen/VDBGrid.h>  // ScriptExample

class FLIPSolver : public zen::INode {
  std::unique_ptr<FLIP_vdb> flip;

  double dt{0.04};
  double dx{0.08};
  double simTime{100.0};
  openvdb::Vec3f gravity{0, 0, 12};
  openvdb::Vec3f tankFlow{0, 0, 1};
  openvdb::Vec3f bmin{-1};
  openvdb::Vec3f bmax{+1};

  zenbase::VDBPointsGrid *mParticles;

  void init();
  bool step();

  bool initialized{false};
 public:
  virtual void apply() override;
};


void FLIPSolver::apply() {
  if (!initialized) {
    initialized = true;
    init();
  } else {
    step();
  }

  /*
  auto solidsdf = get_inputs("solidsdf")->as<zenbase::VDBFloatGrid>();
  flip->add_solid_sdf(solidsdf->m_grid);
  */

  /*
  auto dropsdf = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(0.4,
  openvdb::Vec3f(0, 0, 0), dx);
  flip->seed_liquid(dropsdf, openvdb::Vec3f(0, 0, 0));
  */
}

void FLIPSolver::init() {
  dt = std::get<float>(get_param("dt"));
  dx = std::get<float>(get_param("dx"));
  gravity = zen::get_float3<openvdb::Vec3f>(get_param("gravity"));
  tankFlow = zen::get_float3<openvdb::Vec3f>(get_param("tankFlow"));
  bmin = zen::get_float3<openvdb::Vec3f>(get_param("boundMin"));
  bmax = zen::get_float3<openvdb::Vec3f>(get_param("boundMax"));

  Options::addDoubleOption("time-step", dt);
  Options::addDoubleOption("simulation-time", simTime);
  Options::addDoubleOption("FLIP_acc_x", gravity[0]);
  Options::addDoubleOption("FLIP_acc_y", gravity[1]);
  Options::addDoubleOption("FLIP_acc_z", gravity[2]);
  Options::addDoubleOption("tank_flow_x", tankFlow[0]);
  Options::addDoubleOption("tank_flow_y", tankFlow[1]);
  Options::addDoubleOption("tank_flow_z", tankFlow[2]);

  FLIP_vdb::init_arg_t args(dx, bmin, bmax);
  flip = std::make_unique<FLIP_vdb>(args);

  mParticles = new_member<zenbase::VDBPointsGrid>("mParticles");
  mParticles->m_grid = flip->get_particles();

  if (has_input("solidSDF")) {
    auto solidsdf = get_input("solidSDF")->as<zenbase::VDBFloatGrid>();
    flip->add_solid_sdf(solidsdf->m_grid);
  }

  /*auto dropsdf = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(0.4,
      openvdb::Vec3f(0, 0, 0), dx);
  flip->seed_liquid(dropsdf, openvdb::Vec3f(0, 0, 0));*/
}

bool FLIPSolver::step() {
  if (!flip->test())
    return false;

  /*printf("%d\n", flip->get_framenumber());
  auto transform = openvdb::math::Transform::createLinearTransform(dx);
  transform->postRotate(flip->get_framenumber() * 0.04, openvdb::math::Y_AXIS);
  mSolidSDF->ptr->setTransform(transform);
  //mSolidSDF->someMethod();*/
    /*static int i; i++;
    auto transform = openvdb::math::Transform::createLinearTransform(0.08);
    transform->postRotate(i * 0.04, openvdb::math::Y_AXIS);
    mSolidSDF->ptr->setTransform(transform);*/

  printf("===================================\n");
  return true;
}

#if 1
static int defFLIPSolver = zen::defNodeClass<FLIPSolver>("FLIPSolver",
    { /* inputs: */ {
        "solidSDF",
    }, /* outputs: */ {
        "mParticles",
    }, /* params: */ {
        {"float", "dt", "0.08"},
        {"float", "dx", "0.04",},
        {"float3", "gravity", "0 0 12"},
        {"float3", "tankFlow", "0 0 1"},
        {"float3", "boundMin", "-1 -1 -1"},
        {"float3", "boundMax", "1 1 1"},
    }, /* category: */ {
        "solver",
    }});
#endif

#if 0
std::unique_ptr<DensePoints> FLIPSolver::get_particles() {
  std::vector<openvdb::points::PointDataTree::LeafNodeType*> pars;
  flip->get_particles()->tree().getNodes(pars);
  printf("particle leaf nodes: %d\n", pars.size());

  auto ret = std::make_unique<DensePoints>();

  for (auto const &leaf: pars) {
    //attributes
    // Attribute reader
    // Extract the position attribute from the leaf by name (P is position).
    openvdb::points::AttributeArray& positionArray =
      leaf->attributeArray("P");
    // Extract the velocity attribute from the leaf by name (v is velocity).
    openvdb::points::AttributeArray& velocityArray =
      leaf->attributeArray("v");

    // Create read handles for position and velocity
    openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::PositionCodec> positionHandle(positionArray);
    openvdb::points::AttributeHandle<openvdb::Vec3f, FLIP_vdb::VelocityCodec> velocityHandle(velocityArray);

    for (auto iter = leaf->beginIndexOn(); iter; ++iter) {
      openvdb::Vec3R p = iter.getCoord().asVec3d() + positionHandle.get(*iter);
      openvdb::Vec3R v = iter.getCoord().asVec3d() + velocityHandle.get(*iter);
      ret->positions.push_back(Vec3f(p[0], p[1], p[2]));
      ret->velocities.push_back(Vec3f(v[0], v[1], v[2]));
    }
  }

  return ret;
}
#endif
#endif
