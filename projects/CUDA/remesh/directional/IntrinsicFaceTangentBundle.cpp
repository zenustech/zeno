//This file is part of Directional, a library for directional field processing.
// Copyright (C) 2016 Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "./IntrinsicFaceTangentBundle.h"

namespace zeno::directional {
    void IntrinsicFaceTangentBundle::dual_tree(std::vector<int>& e_intree,
                                                std::vector<int>& f_fa_edge,
                                                std::vector<int>& f_depth) {
        // TODO(@seeeagull): It seems that we only construct one single tree at once.
        // So it should not be able to handle non-manifold cases.
        auto& ev_deleted = mesh->prim->lines.add_attr<int>("ev_deleted", 0);
        auto& ef_deleted = mesh->prim->lines.add_attr<int>("ef_deleted", 0);
        
        // construct primal tree with ev adj info
        for (int e = 0; e < num_e; ++e) {
            if (mesh->is_boundary_v(mesh->to_vertex(e<<1)) ||
                mesh->is_boundary_v(mesh->to_vertex(e<<1|1)))
                ev_deleted[e] = 1;
        }

        std::vector<int> v_valance(num_v, 0);
        std::vector<std::vector<int>> ve(num_v);

        for (int i = 0; i < num_e; ++i){
            if (ev_deleted[i] == 1)
                continue;
            int v0 = mesh->to_vertex(i<<1), v1 = mesh->to_vertex(i<<1|1);
            ++v_valance[v0];
            ++v_valance[v1];
            ve[v0].push_back(i);
            ve[v1].push_back(i);
        }

        std::set<int> visited{};
        std::queue<std::pair<int,int>> edge_verts{};

        // find a possible root for the primal tree
        for (int i = 0; i < num_v; ++i) {
            if (v_valance[i] > 0) {
                edge_verts.push(std::make_pair(-1, i));
                break;
            }
        }

        while (!edge_verts.empty()) {
            auto cur_edge = edge_verts.front();
            edge_verts.pop();

            if (visited.count(cur_edge.second) > 0)
                continue;
            visited.insert(cur_edge.second);

            if (cur_edge.first != -1) {
                ef_deleted[cur_edge.first] = 1; // dual edges do not cross edges in the primal tree
                e_intree[cur_edge.first] = 1;
            }

            //inserting the new unused vertices
            for (int ee : ve[cur_edge.second]) {
                int v0 = mesh->to_vertex(ee<<1), v1 = mesh->to_vertex(ee<<1|1);
                int next_v = ((v0 == cur_edge.second) ? v1 : v0);
                if (visited.count(next_v) == 0)
                    edge_verts.push(std::make_pair(ee, next_v));
            }
        }

        visited.clear();

        // construct dual tree with ef adj info
        std::vector<int> tree_edges(num_f - 1);
        std::vector<int> f_valance(num_f, 0);
        std::vector<std::vector<int>> fe(num_f);

        for (int i = 0; i < num_e; ++i){
            if (ef_deleted[i] == 1)
                continue;
            int f0 = mesh->get_face(i<<1), f1 = mesh->get_face(i<<1|1);
            ++f_valance[f0];
            ++f_valance[f1];
            fe[f0].push_back(i);
            fe[f1].push_back(i);
        }

        f_fa_edge.resize(num_f);
        int edge_id = 0;

        for (int i = 0; i < num_f; ++i) {
            f_fa_edge[i] = -2;
            f_depth[i] = -1;
        }
        // find a possible root for the dual tree
        for (int i = 0; i < num_f; ++i) {
            if (f_valance[i] > 0) {
                edge_verts.push(std::make_pair(-1, i));
                f_depth[i] = 0;
                break;
            }
        }

        while (!edge_verts.empty()) {
            auto cur_edge = edge_verts.front();
            edge_verts.pop();

            if (visited.count(cur_edge.second) > 0)
                continue;
            visited.insert(cur_edge.second);

            if (cur_edge.first != -1)
                tree_edges[edge_id++] = cur_edge.first;
            f_fa_edge[cur_edge.second] = cur_edge.first;

            //inserting the new unused vertices
            for (int ee : fe[cur_edge.second]) {
                int f0 = mesh->get_face(ee<<1), f1 = mesh->get_face(ee<<1|1);
                int next_v = (f0 == cur_edge.second ? f1 : f0);
                if (f_depth[next_v] == -1 || f_depth[next_v] > f_depth[cur_edge.second] + 1)
                    f_depth[next_v] = f_depth[cur_edge.second] + 1;
                if (visited.count(next_v) == 0)
                    edge_verts.push(std::make_pair(ee, next_v));
            }
        }

        tree_edges.resize(visited.size() - 1);
        for (int e : tree_edges)
            e_intree[e] = 1;

        mesh->prim->lines.erase_attr("ev_deleted");
        mesh->prim->lines.erase_attr("ef_deleted");
    }

    void IntrinsicFaceTangentBundle::get_boundary_loops(std::vector<std::vector<int>>& loops) {
        std::set<int> boundary_edges{};
        std::unordered_map<int, zeno::vec2i> ve{};
        for (int e = 0; e < num_e; ++e) {
            if (mesh->is_boundary_e(e)) {
                boundary_edges.insert(e);
                int v0 = mesh->to_vertex(e << 1);
                int v1 = mesh->to_vertex(e << 1 | 1);
                if (ve.count(v0) == 0)
                    ve.insert({v0, zeno::vec2i(e, -1)});
                else
                    ve[v0][1] = e;
                if (ve.count(v1) == 0)
                    ve.insert({v1, zeno::vec2i(e, -1)});
                else
                    ve[v1][1] = e;
            }
        }
        while (!boundary_edges.empty()) {
            std::vector<int> loop{};
            int e_start = *(boundary_edges.begin());
            int e_cur = e_start;
            int v_cur = mesh->to_vertex(e_cur<<1), v_next;
            do {
                boundary_edges.erase(e_cur);
                loop.push_back(v_cur);
                v_next = ((mesh->to_vertex(e_cur<<1) == v_cur) ? mesh->to_vertex(e_cur<<1|1) : mesh->to_vertex(e_cur<<1));
                e_cur = ((ve[v_cur][0] == e_cur) ? ve[v_cur][1] : ve[v_cur][0]);
                v_cur = v_next;
            } while (e_cur != e_start);
            loops.push_back(loop);
        }
    }

  
    void IntrinsicFaceTangentBundle::dual_cycles() {
        // TODO(@seeeagull): according to the author, it may go wrong when there is intersections of boundary cycles and generator cycles
        std::vector<std::vector<int>> boundary_loops{};
        get_boundary_loops(boundary_loops);
        int num_boundaries = boundary_loops.size();
        int num_generators = 2 - num_boundaries - (num_v - num_e + num_f);
        
        auto& ev = mesh->prim->lines;
        std::vector<Eigen::Triplet<float>> basis_cycle_tris(num_e * 2);
        // all 1-ring cycles, including boundaries
        for (int e = 0; e < num_e; ++e) {
            basis_cycle_tris.push_back(Eigen::Triplet<float>(ev[e][0], e, -1.f));
            basis_cycle_tris.push_back(Eigen::Triplet<float>(ev[e][1], e, 1.f));
        }
        
        int cur_generator = 0;
        if (num_generators != 0) {
            auto& e_intree = mesh->prim->lines.add_attr<int>("e_intree", 0);
            // construct dual tree with ev adj info
            std::vector<int> dual_fa_edges, dual_depth;
            dual_tree(e_intree, dual_fa_edges, dual_depth);

            std::set<int> inner{};
            for (int e = 0; e < num_e; ++e)
                if (!(mesh->is_boundary_v(ev[e][0])) && !(mesh->is_boundary_v(ev[e][1])))
                inner.insert(e);
            
            // building tree co-tree based homological cycles
            // finding dual edge which are not in the tree, and following their faces to the end
            for (int e = 0; e < num_e; e++) {
                if (e_intree[e] == 1 || mesh->is_boundary_e(e))
                    continue;

                // if it is not a boundary edge,
                // begin from both faces to their LCA and gets a dual cycle
                std::vector<zeno::vec2f> candidates{};
                int cur_f0 = mesh->get_face(e<<1), cur_f1 = mesh->get_face(e<<1|1);
                bool is_boundary_cycle = true;

                // TODO(@seeeagull): what if one or both faces are not on the dual tree?
                int cur_edge;
                float sgn;
                while (cur_f0 != cur_f1) {
                    if (dual_depth[cur_f0] > dual_depth[cur_f1]) {
                        cur_edge = dual_fa_edges[cur_f0];
                        int next_f0 = mesh->get_face(cur_edge<<1), next_f1 = mesh->get_face(cur_edge<<1|1);
                        sgn = (next_f0 != cur_f0) ? 1.f : -1.f;
                        candidates.push_back(zeno::vec2f(cur_edge, sgn));
                        cur_f0 = (next_f0 == cur_f0) ? next_f1 : next_f0;
                    } else {
                        cur_edge = dual_fa_edges[cur_f1];
                        int next_f0 = mesh->get_face(cur_edge<<1), next_f1 = mesh->get_face(cur_edge<<1|1);
                        sgn = (next_f0 == cur_f1) ? 1.f : -1.f;
                        candidates.push_back(zeno::vec2f(cur_edge, sgn));
                        cur_f1 = (next_f0 == cur_f1) ? next_f1 : next_f0;
                    }
                }

                for (auto &t : candidates)
                    if (inner.count(t[1]) > 0) {
                        is_boundary_cycle = false;
                        break;
                    }
                
                if (is_boundary_cycle)
                    continue; // boundary cycles were already calculated above
                
                int cycle_cnt = num_v + cur_generator;
                cur_generator++;
                
                basis_cycle_tris.push_back(Eigen::Triplet<float>(cycle_cnt, e, 1.0f));
                for (auto &c : candidates) {
                    basis_cycle_tris.push_back(Eigen::Triplet<float>(cycle_cnt, c[0], c[1]));
                }
            }
          
            mesh->prim->lines.erase_attr("e_intree");
        }
        
        num_generators = cur_generator;

        Eigen::SparseMatrix<float> bound_loops(num_v + num_boundaries + num_generators, num_v + num_generators);
        std::vector<Eigen::Triplet<float>> bound_loop_tris{};
        std::vector<int> inner_vert_list{}, inner_edge_list{};
        Eigen::VectorXi remain_rows, remain_cols;
        
        // mask for boundary vertices
        vertex2cycle.conservativeResize(num_v);
        for (int v = 0; v < num_v; ++v) {
            if (!mesh->is_boundary_v(v)) {
                vertex2cycle(v) = inner_vert_list.size();
                inner_vert_list.push_back(v);
                bound_loop_tris.push_back(Eigen::Triplet<float>(v, v, 1.f));
            } else {
                bound_loop_tris.push_back(Eigen::Triplet<float>(v, v, 0.f));
            }
        }
        // boundary loops
        for (int i = 0; i < num_boundaries; ++i)
            for (int j : boundary_loops[i]) {
                vertex2cycle(j) = inner_vert_list.size() + i;
                bound_loop_tris.push_back(Eigen::Triplet<float>(num_v + i, j, 1.f));
            }
        // just passing generators through
        for (int i = num_v; i < num_v + num_generators; ++i)
           bound_loop_tris.push_back(Eigen::Triplet<float>(i, i, 1.f));
        bound_loops.setFromTriplets(bound_loop_tris.begin(), bound_loop_tris.end());
        
        cycles.resize(num_v + num_generators, num_e);
        cycles.setFromTriplets(basis_cycle_tris.begin(), basis_cycle_tris.end());
        cycles = bound_loops * cycles;
        
        // remove rows and columns
        for (int e = 0; e < num_e; ++e)
            if (!((mesh->is_boundary_v(ev[e][0])) && (mesh->is_boundary_v(ev[e][1]))))
                inner_edge_list.push_back(e);
        remain_rows.resize(inner_vert_list.size() + num_boundaries + num_generators);
        remain_cols.resize(inner_edge_list.size());
        for (int i = 0; i < inner_vert_list.size(); i++)
          remain_rows(i) = inner_vert_list[i];
        for (int i = 0; i < num_boundaries + num_generators; i++)
          remain_rows(inner_vert_list.size() + i) = num_v + i;
        for (int i = 0; i < inner_edge_list.size(); i++)
          remain_cols(i) = inner_edge_list[i];
        
        // create slicing matrices
        std::vector<Eigen::Triplet<float>> row_slice_tris, col_slice_tris;
        for (int i = 0; i < remain_rows.size(); i++)
          row_slice_tris.push_back(Eigen::Triplet<float>(i, remain_rows(i), 1.0));
        for (int i = 0; i < remain_cols.size(); i++)
          col_slice_tris.push_back(Eigen::Triplet<float>(remain_cols(i), i, 1.0));
        Eigen::SparseMatrix<float> row_slice_mat(remain_rows.rows(), cycles.rows());
        row_slice_mat.setFromTriplets(row_slice_tris.begin(), row_slice_tris.end());
        Eigen::SparseMatrix<float> col_slice_mat(cycles.cols(), remain_cols.rows());
        col_slice_mat.setFromTriplets(col_slice_tris.begin(), col_slice_tris.end());
        cycles = row_slice_mat * cycles * col_slice_mat;
        
        inner_edges.conservativeResize(inner_edge_list.size());
        for (int i = 0; i < inner_edge_list.size(); i++)
          inner_edges(i) = inner_edge_list[i];
        
        // correct computation of cycle curvature by adding angles
        // getting corner angle sum
        auto &pos = mesh->prim->attr<zeno::vec3f>("pos");
        auto &faces = mesh->prim->tris;
        Eigen::VectorXf all_angles(3 * num_f);
        for (int i = 0; i < num_f; i++){
            for (int j = 0; j < 3; j++){
                zeno::vec3f e0 = pos[faces[i][(j+1)%3]] - pos[faces[i][j]];
                zeno::vec3f e1 = pos[faces[i][(j+2)%3]] - pos[faces[i][j]];
                all_angles(3*i+j) = std::acos(dot(normalize(e0), normalize(e1)));
            }
        }
        
        // for each cycle, summing up all its internal angles negatively  + either 2*pi*|cycle| for internal cycles or pi*|cycle| for boundary cycles.
        cycle_curv = Eigen::VectorXf::Zero(cycles.rows());
        Eigen::VectorXi is_big_cycle = Eigen::VectorXi::Ones(cycles.rows());  //TODO: retain it rather then reverse-engineer...
        for (int i = 0; i < num_v; i++)  //inner cycles
            if (!mesh->is_boundary_v(i))
                is_big_cycle(vertex2cycle(i)) = 0;
        
        // getting the 4 corners of each edge to allocated later to cycles according to the sign of the edge.
        std::vector<std::set<int>> corner_sets(cycles.rows());
        std::vector<std::set<int>> vertex_sets(cycles.rows());
        Eigen::MatrixXi edge_corners(inner_edges.size(), 4);
        for (int i = 0; i < inner_edge_list.size(); ++i) {
            int e = inner_edge_list[i];
            int f0 = mesh->get_face(e<<1), f1 = mesh->get_face(e<<1|1);
            int i00 = 0, i10 = 0;
            while (faces[f0][i00] != ev[e][0])
                i00 = (i00 + 1) % 3;
            int i01 = (faces[f0][(i00+1)%3] == ev[e][1]) ? (i00+1)%3 : (i00+2)%3;
            while (faces[f1][i10] != ev[e][0])
                i10 = (i10 + 1) % 3;
            int i11 = (faces[f1][(i10+1)%3] == ev[e][1]) ? (i10+1)%3 : (i10+2)%3;
            edge_corners(i,0) = f0 * 3 + i00;
            edge_corners(i,1) = f1 * 3 + i10;
            edge_corners(i,2) = f0 * 3 + i01;
            edge_corners(i,3) = f1 * 3 + i11;
        }
        
        for (int k = 0; k < cycles.outerSize(); ++k)
            for (Eigen::SparseMatrix<float>::InnerIterator it(cycles, k); it; ++it){
                corner_sets[it.row()].insert(edge_corners(it.col(), it.value()<0 ? 0 : 2));
                corner_sets[it.row()].insert(edge_corners(it.col(), it.value()<0 ? 1 : 3));
                vertex_sets[it.row()].insert(ev[inner_edges(it.col())][it.value()<0 ? 0 : 1]);
            }
        
        for (int i = 0; i < corner_sets.size(); i++){
            if (is_big_cycle(i))
                cycle_curv(i) = M_PI * (float)(vertex_sets[i].size());
            else
                cycle_curv(i) = 2.0 * M_PI;
            for (std::set<int>::iterator si = corner_sets[i].begin(); si != corner_sets[i].end(); si++)
                cycle_curv(i) -= all_angles(*si);
        }
    
    }

    IntrinsicFaceTangentBundle::IntrinsicFaceTangentBundle(zeno::pmp::SurfaceMesh* surface_mesh) {
        mesh = surface_mesh;
        num_v = mesh->n_vertices();
        num_e = mesh->n_lines();
        num_f = mesh->n_faces();
        auto &pos = mesh->prim->attr<zeno::vec3f>("pos");
        auto &lines = mesh->prim->lines;
        auto &fbx = mesh->prim->tris.attr<zeno::vec3f>("fbx");
        auto &fby = mesh->prim->tris.attr<zeno::vec3f>("fby");
        // TODO(@seeeagull): remember to erase connection attr somewhere finally
        // connection: the angle rotated from the local basis of ef(i,0) to the local basis of ef(i,1) in a unit circle,
        // represented in complex form
        connection.resize(num_e, 1);
        for (int i = 0; i < num_e; i++) {
            if (mesh->is_boundary_e(i))
                continue;
            zeno::vec3f edge_vec = normalize(pos[lines[i][1]] - pos[lines[i][0]]);
            int f0 = mesh->get_face(i<<1), f1 = mesh->get_face(i<<1|1);
            std::complex<float> e2f(dot(edge_vec, fbx[f0]), dot(edge_vec, fby[f0]));
            std::complex<float> e2g(dot(edge_vec, fbx[f1]), dot(edge_vec, fby[f1]));
            connection[i] = e2g / e2f;
        }

        dual_cycles();
    }

    Eigen::MatrixXf IntrinsicFaceTangentBundle::project_to_extrinsic(const Eigen::VectorXi& tan_spaces, const Eigen::MatrixXf& int_directionals) const {

        assert(tan_spaces.rows() == int_directionals.rows() || tan_spaces.rows() == 0);
        Eigen::VectorXi actual_tan_spaces;
        if (tan_spaces.rows()==0)
            actual_tan_spaces = Eigen::VectorXi::LinSpaced(num_f, 0, num_f - 1);
        else
            actual_tan_spaces = tan_spaces;

        auto &fbx = mesh->prim->tris.attr<zeno::vec3f>("fbx");
        auto &fby = mesh->prim->tris.attr<zeno::vec3f>("fby");
        Eigen::MatrixXf ext_directionals(actual_tan_spaces.rows(), 3);

        ext_directionals.conservativeResize(int_directionals.rows(), int_directionals.cols() * 3 / 2);
        for (int i = 0; i < int_directionals.rows(); i++) {
            Eigen::Vector3f fbxi = Eigen::Vector3f(fbx[actual_tan_spaces(i)].data());
            Eigen::Vector3f fbyi = Eigen::Vector3f(fby[actual_tan_spaces(i)].data());
            for (int j = 0; j < int_directionals.cols(); j += 2)
                ext_directionals.block(i,3*j/2,1,3) = fbxi * int_directionals(i,j) + fbyi * int_directionals(i,j+1);
        }

        return ext_directionals;
    }

    Eigen::MatrixXf IntrinsicFaceTangentBundle::project_to_intrinsic(const Eigen::VectorXi& tan_spaces, const Eigen::MatrixXf& ext_directionals) const{
        assert(tan_spaces.rows() == ext_directionals.rows());

        int N = ext_directionals.cols() / 3;
        Eigen::MatrixXf int_directionals(tan_spaces.rows(), 2 * N);

        auto &fbx = mesh->prim->tris.attr<zeno::vec3f>("fbx");
        auto &fby = mesh->prim->tris.attr<zeno::vec3f>("fby");
        for (int i = 0; i < tan_spaces.rows(); i++) {
            Eigen::Vector3f fbxi = Eigen::Vector3f(fbx[tan_spaces(i)].data());
            Eigen::Vector3f fbyi = Eigen::Vector3f(fby[tan_spaces(i)].data());
            for (int j = 0; j < N; j++) {
                Eigen::MatrixXf ext = ext_directionals.block(i, 3*j, 1, 3);
                int_directionals.block(i, 2*j, 1, 2) << (ext.array() * fbxi.array()).sum(), (ext.array() * fbyi.array()).sum();
            }
        }

        return int_directionals;
    }
}
