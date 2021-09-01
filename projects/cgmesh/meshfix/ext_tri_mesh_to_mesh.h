#ifndef EXT_TRI_MESH_TO_MESH_H
#define EXT_TRI_MESH_TO_MESH_H
#include <exttrimesh.h>
#include <Eigen/Core>
// Unload a mesh to matrix of vertex positions V and a matrix of face indices F
// into V. 
//
// Inputs:
//   tin  ExtTriMesh containing (V,F)
// Outputs:
//   V  #V by 3 vertex positions
//   F  #F by 3 face indices into V
// Returns 0 iff success
template <typename DerivedV, typename DerivedF>
inline void ext_tri_mesh_to_mesh(
  const ExtTriMesh & tin,
  Eigen::PlainObjectBase<DerivedV> & V, 
  Eigen::PlainObjectBase<DerivedF> & F);

// Implementation

#include "ext_tri_mesh_to_mesh.h"
#include <unordered_map>

template <typename DerivedV, typename DerivedF>
inline void ext_tri_mesh_to_mesh(
  const ExtTriMesh & tin,
  Eigen::PlainObjectBase<DerivedV> & V, 
  Eigen::PlainObjectBase<DerivedF> & F)
{
  V.resize(tin.V.numels(),3);
  F.resize(tin.T.numels(),3);
  {
    int i;
    Node *n;
    Vertex *v;
    std::unordered_map<const Vertex *,int> v2i;
    v2i.reserve(V.rows());
    {
      int i = 0;
      for(
        n = tin.V.head(), v = (n)?((Vertex *)n->data):NULL; 
        n != NULL; 
        n=n->next(), v = (n)?((Vertex *)n->data):NULL)
      {
        V(i,0) = v->x;
        V(i,1) = v->y;
        V(i,2) = v->z;
        v2i[v] = i;
        i++;
      }
    }
    {
      int i = 0;
      for ((n) = (tin.T).head(); (n) != NULL; (n)=(n)->next())
      {
        F(i,0) = v2i[((Triangle *)n->data)->v1()];
        F(i,1) = v2i[((Triangle *)n->data)->v2()];
        F(i,2) = v2i[((Triangle *)n->data)->v3()];
        i++;
      }
    }
  }
}
#endif
