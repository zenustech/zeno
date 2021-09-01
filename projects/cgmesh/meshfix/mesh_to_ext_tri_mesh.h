#ifndef MESH_TO_EXT_TRI_MESH_H
#define MESH_TO_EXT_TRI_MESH_H
#include <exttrimesh.h>
#include <Eigen/Core>

// Load from a matrix of vertex positions V and a matrix of face indices F
// into V. 
//
// Inputs:
//   V  #V by 3 vertex positions
//   F  #F by 3 face indices into V
// Outputs:
//   tin  ExtTriMesh containing (V,F)
template <typename DerivedV, typename DerivedF>
inline void mesh_to_ext_tri_mesh(
  const Eigen::PlainObjectBase<DerivedV> & V, 
  const Eigen::PlainObjectBase<DerivedF> & F, 
  ExtTriMesh & tin);

// Implementation

#include "mesh_to_ext_tri_mesh.h"

template <typename DerivedV, typename DerivedF>
inline void mesh_to_ext_tri_mesh(
  const Eigen::PlainObjectBase<DerivedV> & V, 
  const Eigen::PlainObjectBase<DerivedF> & F, 
  ExtTriMesh & tin)
{
  const int n = V.rows();
  const int m = F.rows();
  // Create list of vertices
  for(int i=0; i<n; i++)
  {
    tin.V.appendTail(new Vertex(V(i,0),V(i,1),V(i,2)));
  }
  // Allocate Vertices
  ExtVertex **var = (ExtVertex **)malloc(sizeof(ExtVertex *)*n);
  {
    Vertex *v;
    Node *node;
    int i=0;
    for(
      node = tin.V.head(), v = (node)?((Vertex *)node->data):NULL; 
      node != NULL; 
      node=node->next(), v = (node)?((Vertex *)node->data):NULL)
    {
      var[i++] = new ExtVertex(v);
    }
  }
  for(int i=0; i<m; i++)
  {
    tin.CreateIndexedTriangle(var,F(i,0),F(i,1),F(i,2));
  }
  // Clean up
  for(int i=0; i<n; i++)
  {
    delete var[i];
  }
  free(var);
}
#endif
