#pragma once

#include <string>
#include <Eigen/Core>

namespace zeno {

void igl_mesh_boolean(
    Eigen::MatrixXd const &VA,
    Eigen::MatrixXi const &FA,
    Eigen::MatrixXd const &VB,
    Eigen::MatrixXi const &FB,
    std::string const &op_type,
    Eigen::MatrixXd &VC,
    Eigen::MatrixXi &FC,
    Eigen::VectorXi &J);

#if 0
void igl_trim_with_sold(
    Eigen::MatrixXd const &VA,
    Eigen::MatrixXi const &FA,
    Eigen::MatrixXd const &VB,
    Eigen::MatrixXi const &FB,
    Eigen::MatrixXd &VC,
    Eigen::MatrixXi &FC,
    Eigen::VectorXi &D,
    Eigen::VectorXi &J);
#endif

}
