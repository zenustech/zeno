#ifndef MESHFIX_H
#define MESHFIX_H

#include <exttrimesh.h>
// Run the meshfix program to clean a given mesh
//
// Inputs:
//   epsilon_angle  used to change default of JMesh::acos_tolerance (0.0 -->
//     use default value)
//   keep_all_components  whether to keep all components of input
//   tin  input messy mesh (changed in place)
// Outputs:
//   tin  see output
// Returns true on success
bool meshfix(
  const double epsilon_angle, 
  const bool keep_all_components, 
  ExtTriMesh & tin);

#ifdef MESHFIX_WITH_EIGEN

#include <Eigen/Core>
// Run meshfix on (V,F) producing (W,G)
// 
// Inputs:
//   V  #V by 3 vertex positions
//   F  #F by 3 face indices into V
// Outputs:
//   W  #W by 3 vertex positions
//   G  #G by 3 face indices into W
// Returns true on success
inline bool meshfix(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  Eigen::MatrixXd & W,
  Eigen::MatrixXi & G);
// Implementation

#include "mesh_to_ext_tri_mesh.h"
#include "ext_tri_mesh_to_mesh.h"

inline bool meshfix(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  Eigen::MatrixXd & W,
  Eigen::MatrixXi & G)
{
  ExtTriMesh tin;
  mesh_to_ext_tri_mesh(V,F,tin);
  // These **must** be called after loading, before calling `meshfix`
  tin.removeVertices();
  tin.cutAndStitch();
  tin.forceNormalConsistence();
  tin.duplicateNonManifoldVertices();
  tin.removeDuplicatedTriangles();
  if(!meshfix(0,false,tin))
  {
    return false;
  }
  ext_tri_mesh_to_mesh(tin,W,G);
  return true;
}

#endif

#endif
