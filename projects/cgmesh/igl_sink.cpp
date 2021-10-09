#include <igl/copyleft/cgal/mesh_boolean.h>
#if 0
#include <igl/copyleft/cgal/trim_with_solid.h>
#endif
#include <zeno/utils/Exception.h>
#include "igl_sink.h"

namespace zeno {

void igl_mesh_boolean(
    Eigen::MatrixXd const &VA,
    Eigen::MatrixXi const &FA,
    Eigen::MatrixXd const &VB,
    Eigen::MatrixXi const &FB,
    std::string const &op_type,
    Eigen::MatrixXd &VC,
    Eigen::MatrixXi &FC,
    Eigen::VectorXi &J) {

    auto const *pVA = &VA;
    auto const *pVB = &VB;
    auto const *pFA = &FA;
    auto const *pFB = &FB;

        igl::MeshBooleanType boolean_type;
        if (op_type == "Union") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_UNION;
        } else if (op_type == "Intersect") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_INTERSECT;
        } else if (op_type == "Minus") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_MINUS;
        } else if (op_type == "RevMinus") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_MINUS;
          std::swap(pVA, pVB); std::swap(pFA, pFB);
        } else if (op_type == "XOR") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_XOR;
        } else if (op_type == "Resolve") {
          boolean_type = igl::MESH_BOOLEAN_TYPE_RESOLVE;
        } else {
          throw Exception("bad boolean op type: " + op_type);
        }
        igl::copyleft::cgal::mesh_boolean(*pVA, *pFA, *pVB, *pFB, boolean_type, VC, FC, J);

}

#if 0
void igl_trim_with_sold(
    Eigen::MatrixXd const &VA,
    Eigen::MatrixXi const &FA,
    Eigen::MatrixXd const &VB,
    Eigen::MatrixXi const &FB,
    Eigen::MatrixXd &VC,
    Eigen::MatrixXi &FC,
    Eigen::VectorXi &D,
    Eigen::VectorXi &J) {

    igl::copyleft::cgal::trim_with_solid(VA, FA, FB, FB, VC, FC, D, J);

}
#endif

}
