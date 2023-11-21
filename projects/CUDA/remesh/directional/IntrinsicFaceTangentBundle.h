// This file is part of Directional, a library for directional field processing.
// Copyright (C) 2022 Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#ifndef DIRECTIONAL_INTRINSIC_FACE_TANGENT_BUNDLE_H
#define DIRECTIONAL_INTRINSIC_FACE_TANGENT_BUNDLE_H

#include <vector>
#include <queue>
#include <set>
#include <unordered_map>
#include <iostream>
#include <Eigen/Core>
#include "../SurfaceMesh.h"


/***
This class represents piecewise-constant face-based tangent bundles, where tangent spaces identify with the natural plane to every triangle of a 2-manifold mesh, connections are across (dual) edges, and local cycles are around vertices, with curvature being discrete angle defect.
 ***/

namespace zeno::directional {

    class IntrinsicFaceTangentBundle {
    public:
        const zeno::pmp::SurfaceMesh* mesh;

        int num_v, num_e, num_f;

        // combinatorics and topology
        Eigen::VectorXi inner_edges;                   // Edges that are not boundary
        Eigen::SparseMatrix<float> cycles;             // Adjacent matrix of cycles
        // TODO(@seeeagull): Y is cycles float but not int? it represent adjacency info, doesn't it?
        Eigen::VectorXf cycle_curv;                    // Curvature of cycles.
        Eigen::VectorXi vertex2cycle;                  // Map from vertex to general cycles

        // geometry
        // the connection between adjacent tangent space. That is, a field is parallel between EF(i,0) and EF(i,1) when complex(intField.row(adjSpaceS(i,0))*connection(i))=complex(intField.row(adjSpaceS(i,1))
        Eigen::VectorXcf connection;                  // #V, metric connection between adjacent spaces

        IntrinsicFaceTangentBundle(){}
        ~IntrinsicFaceTangentBundle(){}

        void init(const SurfaceMesh& _mesh, const std::vector<vec3f>& pos, const std::vector<vec3f>& lines);

        // projecting intrinsic to extrinsic
        Eigen::MatrixXf project_to_extrinsic(const Eigen::VectorXi& tangentSpaces, const Eigen::MatrixXf& intDirectionals) const;
    
    private:
    // Construct a dual tree (tris as verts and adjacent edges as connecting edges) from a mesh.
    // Create a primal tree first, then construct the dual tree with edges that are not in the primal tree.
    //  e_intree: whether the edge is in primal tree or dual tree.
    //  f_fa_edge: edges leading to every face vert in the tree. -1 for the root.
    //  f_depth: depth of every face vert in the dual tree. used for later LCA finding.
    void dual_tree(std::vector<int>& e_intree,
                   std::vector<int>& f_fa_edge,
                   std::vector<int>& f_depth);
    
    // Constuct boundary vertices loops.
    void get_boundary_loops(std::vector<std::vector<int>>& loops);

    // Creates the set of independent dual cycles (closed loops of connected faces that cannot be morphed to each other) on a mesh. Primarily used for index prescription.
    // The basis cycle matrix first contains #V-#b cycles for every inner vertex (by order), then #b boundary cycles, and finally 2*g generator cycles around all handles. Total #c cycles.The cycle matrix sums information on the dual edges between the faces, and is indexed into the inner edges alone (excluding boundary)
    void dual_cycles();
        
    };

}

#endif
