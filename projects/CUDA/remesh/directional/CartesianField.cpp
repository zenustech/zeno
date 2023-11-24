// This file is part of Directional, a library for directional field processing.
// Copyright (C) 2018 Amir Vaxman <avaxman@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <queue>
#include "./CartesianField.h"

namespace zeno::directional {
    void CartesianField::effort_to_indices() {
        Eigen::VectorXf effortInner(tb->inner_edges.size());
        for (int i = 0; i < tb->inner_edges.size(); i++)
            effortInner(i) = effort(tb->inner_edges(i));
        Eigen::VectorXf dIndices = ((tb->cycles * effortInner + N * tb->cycle_curv).array() / (2.f * M_PI));  //this should already be an integer up to numerical precision

        std::vector<int> indices(tb->vertex2cycle.size());
        for (int i = 0; i < tb->vertex2cycle.size(); i++)
            indices[i] = std::round(dIndices(tb->vertex2cycle(i)));

        sing_local_cycles.clear();
        for (int i = 0; i < tb->vertex2cycle.size(); i++)
            if (indices[i] != 0){
                sing_local_cycles.push_back(i);
            }
    }

    void CartesianField::principal_matching(bool update_matching) {

        matching.conservativeResize(tb->num_e);
        matching.setConstant(-1);

        effort = Eigen::VectorXf::Zero(tb->num_e);
        for (int i = 0; i < tb->num_e; i++) {
            if (tb->mesh->is_boundary_e(i))
                continue;

            float min_angle = 10000.0;
            int min_id = 0;

            std::complex<float> effortc(1.f, 0.f);
            // finding where the 0 vector in f0 goes to with smallest rotation angle in f1, computing the effort, and then adjusting the matching to have principal effort.
            int f0 = tb->mesh->get_face(i<<1), f1 = tb->mesh->get_face(i<<1|1);
            std::complex<float> f0_vec0(intField(f0, 0), intField(f0, 1));
            std::complex<float> b1_f0_vec0 = f0_vec0 * tb->connection(i);
            for (int j = 0; j < N; j++) {
                // f0_vecj: field j of f0 in its local basis
                std::complex<float> f0_vecj(intField(f0, j<<1),intField(f0, j<<1|1));
                // f1_vecj: field j of f1 in its local basis
                std::complex<float> f1_vecj(intField(f1, j<<1),intField(f1, j<<1|1));
                // b1_f1_vecj: f0_vecj in f1's local basis
                std::complex<float> b1_f1_vecj = f0_vecj * tb->connection(i);
                effortc *= (f1_vecj / b1_f1_vecj);
                float cur_angle = arg(f1_vecj / b1_f0_vec0);
                if (fabs(cur_angle) < fabs(min_angle)){
                    min_id = j;
                    min_angle = cur_angle;
                }

            }
            // effort for different matchings differ by 2*PI.
            // arg(complex) returns the angle in [-PI, PI), which is the principal effort.
            effort(i) = arg(effortc);
            // @seeeagull): simplified case for 4-RoSy field
            if (update_matching)
                matching(i) = min_id;
        }

        // getting final singularities and their indices
        effort_to_indices();

    }

    void CartesianField::dijkstra (const int &source,
                                  const std::set<int> &targets,
                                  std::vector<int> &path) {
        using node = std::pair<float, int>;

        std::vector<float> min_distance(tb->num_v, std::numeric_limits<float>::max());
        std::vector<int> previous(tb->num_v, -1);

        min_distance[source] = 0;
        auto cmp = [](node left, node right) { return left.first > right.first; };
        std::priority_queue<node, std::vector<node>, decltype(cmp)> vertex_queue(cmp);
        vertex_queue.push(std::make_pair(min_distance[source], source));

        while (!vertex_queue.empty()) {
            int u = vertex_queue.top().second;
            if (min_distance[u] < vertex_queue.top().first) {
                vertex_queue.pop();
                continue;
            }
            vertex_queue.pop();

            // we find an targets, so we save the path and return
            if (targets.count(u) > 0) {
                while (u != source) {
                    path.push_back(u);
                    u = previous[u];
                }
                path.push_back(source);
                return;
            }

            for (auto v : tb->mesh->vertices(u)) {
                // @seeeagull: we use edge num as distance for now, we can also change to edge length sum
                float dist_uv = min_distance[u] + 1.f;
                if (dist_uv < min_distance[v]) {
                    min_distance[v] = dist_uv;
                    previous[v] = u;
                    vertex_queue.push(std::make_pair(dist_uv, v));
                }
            }
        }
        path.push_back(source);
    }

    void CartesianField::cut_mesh_with_singularities(const std::vector<int>& singularities,
                                                     Eigen::MatrixXi& cuts) {
        
        std::set<int> vertices_in_cut{};
        for (int i = 0; i < singularities.size(); ++i) {
            // add a singularity into the vertices_in_cut set using Dijkstra's algorithm
            std::vector<int> path{};
            dijkstra(singularities[i], vertices_in_cut, path);
            vertices_in_cut.insert(path.begin(), path.end());
            
            // insert adjacent faces and edges in path to cuts
            auto &faces = tb->mesh->prim->tris;
            for (int ii = 1; ii < path.size(); ++ii) {
                const int &v0 = path[ii - 1];
                const int &v1 = path[ii];
                
                std::vector<int> common_face{};
                std::set<int> vf0{};
                for (auto ff : tb->mesh->faces(v0))
                    vf0.insert(ff);
                for (auto ff : tb->mesh->faces(v1))
                    if (vf0.count(ff) > 0) {
                        common_face.push_back(ff);
                    }
                assert(common_face.size() == 2);
                
                for (int j = 0; j < 2; ++j) {
                    int f = common_face[j];
                    for (int k = 0; k < 3; ++k)
                        if (((faces[f][k] == v0) && (faces[f][(k+1)%3] == v1)) ||
                            ((faces[f][k] == v1) && (faces[f][(k+1)%3] == v0))) {
                            cuts(f, k) = 1;
                            break;
                        }
                }
            }
        }
    }

    void CartesianField::combing(directional::CartesianField& combed_field,
                                 const Eigen::MatrixXi& cuts) {
        combed_field.init(*(tb), fieldTypeEnum::RAW_FIELD, N);
        std::vector<int> space_turn(intField.rows(), 0);
        
        // flood-filling through the matching to comb field
        // dual tree to find combing routes
        std::set<int> visited_spaces{};
        std::queue<std::pair<int,int> > space_queue{};
        space_queue.push(std::pair<int,int>(0,0));
        Eigen::MatrixXf combed_int_field(combed_field.intField.rows(), combed_field.intField.cols());
        do {
            auto cur = space_queue.front();
            space_queue.pop();
            if (visited_spaces.count(cur.first) > 0)
                continue;
            visited_spaces.insert(cur.first);
            
            // combing field to start from the matching index
            int l1 = 2 * cur.second, l2 = 2 * (N - cur.second);
            combed_int_field.block(cur.first, 0, 1, l2) = intField.block(cur.first, l1, 1, l2);
            combed_int_field.block(cur.first, l2, 1, l1) = intField.block(cur.first, 0, 1, l1);
            
            space_turn[cur.first] = cur.second;
            
            int h = tb->mesh->get_halfedge_f(cur.first);
            for (int i = 0; i < 3; ++i) {
                int e = h >> 1;
                int f1 = tb->mesh->get_face(h), f2 = tb->mesh->get_face(h^1);
                int next_face = (f1 == cur.first ? f2 : f1);
                if ((next_face != -1) && (visited_spaces.count(next_face) == 0) && (!cuts(cur.first, i))) {
                    int next_matching = matching(e) * (f1 == cur.first ? 1 : -1);
                    next_matching = (next_matching + cur.second + 10 * N) % N;  // killing negatives
                    space_queue.push(std::pair<int,int>(next_face, next_matching));
                }
                tb->mesh->next_halfedge(h);
              
            }
        } while (!space_queue.empty());
        
        combed_field.set_intrinsic_field(combed_int_field);
        combed_field.matching.resize(tb->num_e);
        // combed matching
        for (int i = 0; i < tb->num_e;i++){
            if (tb->mesh->is_boundary_e(i))
                combed_field.matching(i) = -1;
            else {
                int f0 = tb->mesh->get_face(i<<1), f1 = tb->mesh->get_face(i<<1|1);
                combed_field.matching(i) = (space_turn[f0] - space_turn[f1] + matching(i) + 10 * N) % N;
            }
        }

        // only update effort.
        combed_field.principal_matching(false);
    }

    void CartesianField::setup_integration(IntegrationData& intData,
                                           Eigen::MatrixXf& cut_verts,
                                           Eigen::MatrixXi& cut_faces,
                                           directional::CartesianField& combedField) {
        assert(tb->discTangType() == discTangTypeEnum::FACE_SPACES && "setup_integration() only works with face-based fields");
        // cutting mesh and combing field.
        intData.face2cut.conservativeResize(tb->num_f, 3);
        cut_mesh_with_singularities(sing_local_cycles, intData.face2cut);
        combing(combedField, intData.face2cut);

        auto &pos = tb->mesh->prim->attr<zeno::vec3f>("pos");
        auto &faces = tb->mesh->prim->tris;

        // mark vertices as being a singularity vertex of the vector field
        auto &vsingular = tb->mesh->prim->verts.add_attr<int>("v_singularity", 0);
        int sing_num = 0;
        for (int i = 0; i < sing_local_cycles.size(); i++)
            vsingular[sing_local_cycles[i]] = 1;
        for (int i = 0 ; i < tb->num_v; ++i) {
            if (tb->mesh->is_boundary_v(i))
                vsingular[i] = 0; // boundary vertices cannot be singular
            if (vsingular[i] == 1)
                ++sing_num;
        }

        intData.constrainedVertices = Eigen::VectorXi::Zero(tb->num_v);

        // here we compute a permutation matrix
        std::vector<Eigen::MatrixXi> const_perm_mt(intData.N);
        Eigen::MatrixXi unit_perm_mt = Eigen::MatrixXi::Zero(intData.N, intData.N);
        for (int i = 0; i < intData.N; i++)
            unit_perm_mt((i + 1) % intData.N, i) = 1;
        // generate all the members of the permutation group
        const_perm_mt[0] = Eigen::MatrixXi::Identity(intData.N, intData.N);
        for (int i = 1; i < intData.N; i++)
            const_perm_mt[i] = unit_perm_mt * const_perm_mt[i - 1];

        // calculate the cut edge valance for verts, record whether every edge is a seam
        auto &vcuts = tb->mesh->prim->verts.add_attr<int>("v_cut_valance", 0);
        auto &eseam = tb->mesh->prim->lines.add_attr<int>("e_seam", 0);
        auto &eclaimed = tb->mesh->prim->lines.add_attr<int>("e_claimed", 0);
        for (int f = 0; f < tb->num_f; f++) {
            for (int i = 0; i < 3; i++)
                if (intData.face2cut(f, i)) {
                    int v0 = faces[f][i], v1 = faces[f][(i+1)%3];
                    int h = tb->mesh->get_halfedge_f(f);
                    int tmp = 0;
                    while (!(v0 == tb->mesh->to_vertex(h) && v1 == tb->mesh->from_vertex(h)) &&
                           !(v1 == tb->mesh->to_vertex(h) && v0 == tb->mesh->from_vertex(h))) {
                        h = tb->mesh->next_halfedge(h);
                    }
                    eseam[h>>1] = 1;
                    ++vcuts[v0];
                    ++vcuts[v1];
                }
        }

        // establish transition variables by tracing cut curves
        Eigen::VectorXi h2transition = Eigen::VectorXi::Constant(tb->num_e*2, 32767);

        // cutting the mesh, record the cut verts
        std::vector<int> cut2whole{};
        std::vector<zeno::vec3f> cut_verts_list{};
        cut_faces.resize(tb->num_f, 3);
        for (int i = 0; i < tb->num_v; i++) {
            int begin = tb->mesh->get_halfedge_v(i);
            int cur = begin;
            // find the first cut edge (for non-boundary vert) or first boundary edge (for boundary vert)
            if (!tb->mesh->is_boundary_v(i)) {
                do {
                    if (eseam[cur>>1] != 0)
                        break;
                    cur = tb->mesh->next_halfedge(cur^1);
                } while (begin != cur);
            } else {
                while (!tb->mesh->is_boundary_e(cur>>1))
                    cur = tb->mesh->next_halfedge(cur^1);
            }
            begin = cur;
            do {
                if ((eseam[cur>>1] != 0) || (begin == cur)) {
                    cut2whole.push_back(i);
                    cut_verts_list.push_back(pos[i]);
                }
                for (int j = 0; j < 3; j++)
                    if (faces[tb->mesh->get_face(cur)][j] == i)
                        cut_faces(tb->mesh->get_face(cur), j) = cut2whole.size() - 1;
                cur = tb->mesh->prev_halfedge(cur)^1;
            } while ((begin != cur) && (cur != -1));
        }

        cut_verts.resize(cut_verts_list.size(), 3);
        for(int i = 0; i < cut_verts_list.size(); i++)
            cut_verts.row(i) = Eigen::RowVector3f(cut_verts_list[i].data());

        // trace cut curves from every cut-graph node
        int cur_trans = 1;
        for(int i = 0; i < tb->num_v; i++) {
            if (((vcuts[i] == 2) && (vsingular[i] == 0)) || (vcuts[i] == 0))
                continue;  // either mid-cut curve or non at all

            // tracing curves until next node, if not already filled
            int begin = tb->mesh->get_halfedge_v(i);
            int cur = begin;
            if (tb->mesh->is_boundary_v(i)) {
                while (!tb->mesh->is_boundary_e(cur>>1))
                    cur = tb->mesh->next_halfedge(cur^1);
            }

            begin = cur;
            do {
                // find an unclaimed inner halfedge
                if ((eseam[cur>>1] == 1) && (eclaimed[cur>>1] == 0) && (!tb->mesh->is_boundary_e(cur>>1))) {
                    int next = cur;
                    h2transition(next) = cur_trans;
                    h2transition(next^1) = -cur_trans;
                    eclaimed[next>>1] = 1;
                    int next_vert = tb->mesh->to_vertex(next);
                    // advance on the cut until next node
                    while ((vcuts[next_vert] == 2) && (vsingular[next_vert] == 0) && (!tb->mesh->is_boundary_v(next_vert))) {
                        int in_begin = tb->mesh->get_halfedge_v(next_vert);
                        int in_cur = in_begin;
                        next = -1;
                        do {
                            // move to next unclaimed cut halfedge
                            if ((eseam[in_cur] == 1) && (eclaimed[in_cur] == 0)) {
                                next = in_cur;
                                break;
                            }
                            in_cur = tb->mesh->prev_halfedge(in_cur)^1;
                        } while (in_begin != in_cur);
                        h2transition(next) = cur_trans;
                        h2transition(next^1) = -cur_trans;
                        eclaimed[next>>1] = 1;
                        next_vert = tb->mesh->to_vertex(next);
                    }
                    cur_trans++;
                }
                cur = tb->mesh->prev_halfedge(cur)^1;
            } while ((begin != cur) && (cur != -1));
        }

        int num_trans = cur_trans - 1;
        std::vector<Eigen::Triplet<float>> vert_trans2cut_tris, const_tris;
        std::vector<Eigen::Triplet<int>> vert_trans2cut_tris_int, const_tris_int;
        // form the constraints and the singularity positions
        int cur_const = 0;
        // this loop set up the transtions (vector field matching) across the cuts
        for (int i = 0; i < tb->num_v; i++) {
            std::vector<Eigen::MatrixXi> perm_mt{};
            std::vector<int> perm_id{};  // in the space #V + #transitions
            // the initial corner gets the identity without any transition
            perm_mt.push_back(Eigen::MatrixXi::Identity(intData.N, intData.N));
            perm_id.push_back(i);

            int begin = tb->mesh->get_halfedge_v(i);
            int cur = begin;
            if (!tb->mesh->is_boundary_v(i)) {
                do {
                    if (eseam[cur] == 1)
                        break;
                    cur = tb->mesh->next_halfedge(cur^1);
                } while (begin != cur);
            } else {
                while (!tb->mesh->is_boundary_e(cur>>1))
                    cur = tb->mesh->next_halfedge(cur^1);
            }

            // set the beginning to the edge on the cut or on the boundary
            begin = cur;
            do {
                int cur_face = tb->mesh->get_face(cur); // face containing the half-edge
                int new_vert = -1;
                // find position of the vertex i in the face of the initial mesh
                for (int j = 0; j < 3; j++) {
                    if (faces[cur_face][j] == i)
                        new_vert = cut_faces(cur_face, j);
                }

                // currCorner gets the permutations so far
                if (new_vert != -1) {
                    for(int ii = 0; ii < perm_id.size(); ii++) {
                        // place the perumtation matrix in a bigger matrix, we need to know how things are connected along the cut, no?
                        for(int j = 0; j < intData.N; j++)
                            for(int k = 0; k < intData.N; k++){
                                vert_trans2cut_tris.emplace_back(intData.N * new_vert + j, intData.N * perm_id[ii] + k, (float) perm_mt[ii](j, k));
                                vert_trans2cut_tris_int.emplace_back(intData.N * new_vert + j, intData.N * perm_id[ii] + k, perm_mt[ii](j, k));
                            }
                    }
                }

                // reached a boundary
                if (tb->mesh->prev_halfedge(cur) == PMP_MAX_INDEX) {
                    break;
                }
                // update the matrices for the next corner
                int next = tb->mesh->prev_halfedge(cur)^1;

                // const_perm_mt contains all the members of the permutation group
                int hmatching = (next & 1) ? -combedField.matching(next>>1) : combedField.matching(next>>1);
                hmatching = (intData.N + (hmatching % intData.N)) % intData.N;
                Eigen::MatrixXi next_perm_mt = const_perm_mt[hmatching];
                // no update needed
                if(eseam[next] == 0) {
                    cur = next;
                    continue;
                }

                // otherwise, update matrices with transition
                int next_trans = h2transition(next);
                if (next_trans > 0) { // Pe*f + Je
                    for(int j = 0; j < perm_mt.size(); j++)
                        perm_mt[j] = next_perm_mt * perm_mt[j];
                    //and identity on the fresh transition
                    perm_mt.push_back(Eigen::MatrixXi::Identity(intData.N, intData.N));
                    perm_id.push_back(tb->num_v + next_trans - 1);
                } else { // (Pe*(f-Je))  matrix is already inverse since halfedge matching is minused
                    //reverse order
                    perm_mt.push_back(-Eigen::MatrixXi::Identity(intData.N, intData.N));
                    perm_id.push_back(tb->num_v - next_trans - 1);
                    for(int j = 0; j < perm_mt.size(); j++)
                        perm_mt[j] = next_perm_mt * perm_mt[j];
                }
                cur = next;
            } while ((cur != begin) && (cur != -1));

            // clean parmMatrices and perm_id to see if there is a constraint or reveal singularity-from-transition
            std::set<int> clean_perm_id_set(perm_id.begin(), perm_id.end());
            std::vector<int> clean_perm_id(clean_perm_id_set.begin(), clean_perm_id_set.end());
            std::vector<Eigen::MatrixXi> clean_perm_mt(clean_perm_id.size());

            for (int j = 0; j < clean_perm_id.size(); j++) {
                clean_perm_mt[j] = Eigen::MatrixXi::Zero(intData.N, intData.N);
                for (int k = 0;k < perm_id.size(); k++)
                    if (clean_perm_id[j] == perm_id[k])
                        clean_perm_mt[j] += perm_mt[k];
                if (clean_perm_id[j] == i)
                    clean_perm_mt[j] -= Eigen::MatrixXi::Identity(intData.N, intData.N);
            }

            // if not all matrices are zero, there is a constraint
            bool is_constraint = false;
            for (int j = 0; j < clean_perm_mt.size(); j++)
                if (clean_perm_mt[j].cwiseAbs().maxCoeff() != 0)
                    is_constraint = true;

            if ((is_constraint) && (!tb->mesh->is_boundary_v(i))) {
                for(int j = 0; j < clean_perm_mt.size(); j++) {
                    for(int k = 0; k < intData.N; k++)
                        for(int l = 0; l < intData.N; l++) {
                            const_tris.emplace_back(intData.N * cur_const + k, intData.N * clean_perm_id[j] + l, (float) clean_perm_mt[j](k, l));
                            const_tris_int.emplace_back(intData.N * cur_const + k, intData.N * clean_perm_id[j] + l, clean_perm_mt[j](k, l));
                        }
                }
                cur_const++;
                intData.constrainedVertices(i) = 1;
            }
        }

        std::vector<Eigen::Triplet<float>> clean_tris{};
        std::vector<Eigen::Triplet<int>> clean_tris_int{};

        intData.vertexTrans2CutMat.resize(intData.N * cut_verts.rows(), intData.N * (tb->num_v + num_trans));
        intData.vertexTrans2CutMatInteger.resize(intData.N * cut_verts.rows(), intData.N * (tb->num_v + num_trans));
        clean_tris.clear();
        clean_tris_int.clear();
        for(int i = 0; i < vert_trans2cut_tris.size(); i++) {
            if(vert_trans2cut_tris_int[i].value() != 0) {
                clean_tris_int.push_back(vert_trans2cut_tris_int[i]);
                clean_tris.push_back(vert_trans2cut_tris[i]);
            }
        }
        intData.vertexTrans2CutMat.setFromTriplets(clean_tris.begin(), clean_tris.end());
        intData.vertexTrans2CutMatInteger.setFromTriplets(clean_tris_int.begin(), clean_tris_int.end());

        intData.constraintMat.resize(intData.N * cur_const, intData.N * (tb->num_v + num_trans));
        intData.constraintMatInteger.resize(intData.N * cur_const, intData.N * (tb->num_v + num_trans));
        clean_tris.clear();
        clean_tris_int.clear();
        for(int i = 0; i < const_tris.size(); i++) {
            if(const_tris_int[i].value() != 0) {
                clean_tris_int.push_back(const_tris_int[i]);
                clean_tris.push_back(const_tris[i]);
            }
        }
        intData.constraintMat.setFromTriplets(clean_tris.begin(), clean_tris.end());
        intData.constraintMatInteger.setFromTriplets(clean_tris_int.begin(), clean_tris_int.end());

        // do the integer spanning matrix
        intData.intSpanMat.resize(intData.n * (tb->num_v + num_trans), intData.n * (tb->num_v + num_trans));
        intData.intSpanMatInteger.resize(intData.n * (tb->num_v + num_trans), intData.n * (tb->num_v + num_trans));
        std::vector<Eigen::Triplet<float>> in_span_tris;
        std::vector<Eigen::Triplet<int>> in_span_tris_int;
        for (int i = 0; i < intData.n * num_trans; i += intData.n) {
            for (int k = 0; k < intData.n; k++)
                for (int l = 0; l < intData.n; l++) {
                    if (intData.periodMat(k,l) != 0) {
                        in_span_tris.emplace_back(intData.n * tb->num_v+i+k, intData.n * tb->num_v+i+l, (float)intData.periodMat(k,l));
                        in_span_tris_int.emplace_back(intData.n * tb->num_v+i+k, intData.n * tb->num_v+i+l, intData.periodMat(k,l));
                    }
                }
        }
        for (int i = 0; i < intData.n * tb->num_v; i++) {
            in_span_tris.emplace_back(i,i,1.0);
            in_span_tris_int.emplace_back(i,i,1);
        }

        intData.intSpanMat.setFromTriplets(in_span_tris.begin(), in_span_tris.end());
        intData.intSpanMatInteger.setFromTriplets(in_span_tris_int.begin(), in_span_tris_int.end());

        // filter out barycentric symmetry, including sign symmetry. The parameterization should always only include n dof for the surface
        // TODO: this assumes n divides N!
        intData.linRedMat.resize(intData.N * (tb->num_v + num_trans), intData.n * (tb->num_v + num_trans));
        intData.linRedMatInteger.resize(intData.N * (tb->num_v + num_trans), intData.n * (tb->num_v + num_trans));
        std::vector<Eigen::Triplet<float>> lin_red_tris;
        std::vector<Eigen::Triplet<int>> lin_red_tris_int;
        for (int i = 0; i < intData.N*(tb->num_v + num_trans); i += intData.N)
            for (int k = 0; k < intData.N; k++)
                for (int l = 0; l < intData.n; l++) {
                    if (intData.linRed(k,l) != 0) {
                        lin_red_tris.emplace_back(i + k, i*intData.n/intData.N + l, (float)intData.linRed(k,l));
                        lin_red_tris_int.emplace_back(i + k, i*intData.n/intData.N + l, intData.linRed(k,l));
                    }
                }

        intData.linRedMat.setFromTriplets(lin_red_tris.begin(), lin_red_tris.end());
        intData.linRedMatInteger.setFromTriplets(lin_red_tris_int.begin(), lin_red_tris_int.end());

        // integer variables are per single "d" packet, and the rounding is done for the N functions with projection over linRed
        intData.integerVars.resize(num_trans);
        intData.integerVars.setZero();
        for(int i = 0; i < num_trans; i++)
            intData.integerVars(i) = tb->num_v + i;

        // fixed values
        intData.fixedIndices.resize(intData.n);
        if (sing_num == 0) {  //no inner singular vertices; vertex 0 is set to (0....0)
            for (int j = 0; j < intData.n; j++)
                intData.fixedIndices(j) = j;
        } else {  //fixing first singularity to (0.5,....0.5)
            int first_sing;
            for (first_sing = 0; first_sing < tb->num_v; first_sing++)
                if (vsingular[first_sing] == 1)
                    break;
            for (int j = 0; j < intData.n; j++)
                intData.fixedIndices(j) = intData.n * first_sing + j;
        }

        // create list of singular corners and singular integer matrix
        Eigen::VectorXi singularIndices(intData.n * sing_num);
        int counter = 0;
        for (int i = 0; i < tb->num_v; i++) {
            if (vsingular[i] == 1)
                for (int j = 0; j < intData.n; j++)
                    singularIndices(counter++) = intData.n * i + j;
        }

        // do the integer spanning matrix
        intData.singIntSpanMat.resize(intData.n * (tb->num_v + num_trans), intData.n * (tb->num_v + num_trans));
        intData.singIntSpanMatInteger.resize(intData.n * (tb->num_v + num_trans), intData.n * (tb->num_v + num_trans));
        std::vector<Eigen::Triplet<float>> sing_lin_span_tris;
        std::vector<Eigen::Triplet<int>> sing_lin_span_tris_int;
        for (int i = 0; i < tb->num_v; i++) {
            if (vsingular[i] == 0) {
                for (int j = 0; j < intData.n; j++) {
                    sing_lin_span_tris.emplace_back(intData.n * i + j, intData.n * i + j, 1);
                    sing_lin_span_tris_int.emplace_back(intData.n * i + j, intData.n * i + j, 1);
                }
            } else {
                for(int k = 0; k < intData.n; k++)
                    for(int l = 0; l < intData.n; l++) {
                        if (intData.periodMat(k,l) != 0) {
                            sing_lin_span_tris.emplace_back(intData.n*i+k, intData.n*i+l, (float)intData.periodMat(k,l));
                            sing_lin_span_tris_int.emplace_back(intData.n*i+k, intData.n*i+l, intData.periodMat(k,l));
                        }
                    }
            }
        }
        for (int i = intData.n * tb->num_v ; i < intData.n * (tb->num_v+num_trans); i++) {
            sing_lin_span_tris.emplace_back(i,i,1.0);
            sing_lin_span_tris_int.emplace_back(i,i,1);
        }
        intData.singIntSpanMat.setFromTriplets(sing_lin_span_tris.begin(), sing_lin_span_tris.end());
        intData.singIntSpanMatInteger.setFromTriplets(sing_lin_span_tris_int.begin(), sing_lin_span_tris_int.end());

        intData.singularIndices = singularIndices;
        intData.fixedValues.resize(intData.n);
        intData.fixedValues.setConstant(0);

        tb->mesh->prim->verts.erase_attr("v_singularity");
        tb->mesh->prim->verts.erase_attr("v_cut_valance");
        tb->mesh->prim->lines.erase_attr("e_seam");
        tb->mesh->prim->lines.erase_attr("e_claimed");
    }
}
