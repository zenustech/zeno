// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#pragma once

#include <vector>

#include <zeno/utils/vec.h>
#include "../SurfaceMesh.h"

namespace zeno {
namespace pmp {

class TriangleKdTree {
public:
    TriangleKdTree(const SurfaceMesh* mesh, unsigned int max_faces = 10,
                   unsigned int max_depth = 30);
    TriangleKdTree(const AttrVector<vec3i>& tris, const AttrVector<vec3f>& points,
                   unsigned int max_faces = 10, unsigned int max_depth = 30);

    ~TriangleKdTree() { delete root_; }

    struct NearestNeighbor {
        float dist;
        int face;
        vec3f nearest;
    };

    NearestNeighbor nearest(const vec3f& p) const;
    void faces_in_box(BoundingBox& box, std::vector<int>& faces);

private:

    // Node of the tree: contains parent, children and splitting plane
    struct Node {
        Node() = default;

        ~Node() {
            delete faces;
            delete left_child;
            delete right_child;
        }

        int axis;
        float split;
        std::vector<int>* faces{nullptr};
        Node* left_child{nullptr};
        Node* right_child{nullptr};
    };

    void build_recurse(Node* node, unsigned int max_handles,
                               unsigned int depth);

    void nearest_recurse(Node* node, const vec3f& point,
                         NearestNeighbor& data) const;
    void in_box_recursive(Node* node, BoundingBox& box,
                          std::vector<int>& faces);

    Node* root_;

    std::vector<std::array<vec3f, 3>> face_points_;
};

} // namespace pmp
} // namespace zeno
