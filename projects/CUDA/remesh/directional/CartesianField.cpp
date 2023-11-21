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
    void CartesianField::effort_to_indices(directional::CartesianField& field) {
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

        effort = VectorXf::Zero(tb->num_e);
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
        effort_to_indices(field);

    }

    // TODO(@seeeagull): I will change Eigen::VectorXi &singularities to vector, since it involves in no math calculation but requires a replication
    void CartesianField::dijkstra(const int &source,
                                  const std::set<int> &targets,
                                  std::vecotr<int> &path) {
        using node = std::pair<float, int>;

        std::vector<float> min_distance(tb->num_v, std::numeric_limits<float>::max());
        std::vector<int> previous(tb->num_v, -1);

        min_distance[source] = 0;
        auto cmp = [](node left, node right) { return left.first > right.first; };
        std::priority_queue<node, std::vector<node>, decltype(cmp)> vertex_queue(cmp);
        vertex_queue.insert(std::make_pair(min_distance[source], source));

        while (!vertex_queue.empty()) {
            int u = vertex_queue.top()->second;
            if (min_distance[u] < vertex_queue.front()->first) {
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

    void CartesianField::cut_mesh_with_singularities(const Eigen::VectorXi& singularities,
                                                     Eigen::MatrixXi& cuts) {
        
        std::set<int> vertices_in_cut;
        for (int i = 0; i < singularities.rows(); ++i) {
            // add a singularity into the vertices_in_cut set using Dijkstra's algorithm
            std::vector<int> path();
            int vertex_found = dijkstra(singularities[i], vertices_in_cut, path);
            vertices_in_cut.insert(path.begin(), path.end());
            
            // insert adjacent faces and edges in path to cuts
            for (int ii = 1; ii < path.size(); ++ii) {
                const int &v0 = path[ii - 1];
                const int &v1 = path[ii];
                
                std::vector<int> common_face();
                std::set<int> vf0();
                for (auto ff : tb->mesh->faces(v0))
                    vf0.insert(ff);
                for (auto ff : tb->mesh->faces(v1))
                    if (vf0.count(ff) > 0) {
                        common_face.insert(ff);
                    }
                assert(common_face.size() == 2);
                
                for (int j = 0; j < 2; ++j) {
                    const int f = common_face[j];
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
        // TODO(@seeeagull): we should make sure that the passed cuts matrix is at least initialized with zeros(but not empty)

        std::vector<int> space_turn(intField.rows(), 0);
        
        // flood-filling through the matching to comb field
        // dual tree to find combing routes
        std::set<int> visited_spaces();
        std::queue<std::pair<int,int> > space_queue;
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
            
            int h = mesh->get_halfedge_f(cur.first);
            for (int i = 0; i < 3; ++i){
                int e = h >> 1;
                int f1 = mesh->get_face(h), f2 = mesh->get_face(h ^ 1);
                int next_face = (f1 == cur.first ? f2 : f1);
                if ((next_face != -1) && (visited_spaces.count(next_face) == 0) && (!cuts(cur.first, i))) {
                    int next_matching = matching(e) * (f1 == cur.first ? 1 : -1);
                    next_matching = (next_matching + cur.second + 10 * N) % N;  //killing negatives
                    space_queue.push(std::pair<int,int>(next_face, next_matching));
                }
                mesh->get_next_halfedge(h);
              
            }
        } while (!space_queue.empty());
        
        combed_field.set_intrinsic_field(combed_int_field);
        combed_field.matching.resize(tb->num_e);
        // combed matching
        for (int i = 0; i < tb->num_e;i++){
            if (mesh->is_boundary_e(i))
                combed_field.matching(i) = -1;
            else {
                int f0 = tb->mesh->get_face(i<<1), f1 = tb->mesh->get_face(i<<1|1);
                combed_field.matching(i) = (space_turn(f0) - space_turn(f1) + matching(i) + 10 * N) % N;
            }
        }

        // only update effort.
        combed_field.principal_matching(false);
    }
}
