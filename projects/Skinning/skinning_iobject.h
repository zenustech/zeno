#pragma once

#include <zeno/zeno.h>
#include <zeno/logger.h>
#include <zeno/ListObject.h>
#include <zeno/NumericObject.h>
#include <zeno/PrimitiveObject.h>
#include <zeno/utils/UserData.h>
#include <zeno/StringObject.h>

#include <Eigen/Geometry>
#include <Eigen/src/Geometry/Transform.h>

namespace {
using namespace zeno;

typedef
  std::vector<Eigen::Quaterniond,Eigen::aligned_allocator<Eigen::Quaterniond> >
  RotationList;

struct PosesAnimationFrame : zeno::IObject {
    PosesAnimationFrame() = default;
    RotationList posesFrame;
};

struct SkinningWeight : zeno::IObject {
    SkinningWeight() = default;
    Eigen::MatrixXd weight;
};

};