// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include "./TriangleKdTree.h"

#include <limits>

#include "../BoundingBox.h"

namespace zeno {
namespace pmp {

float dist_point_line_segment(const vec3f& p, const vec3f& v0, const vec3f& v1,
                            vec3f& nearest_point) {
    vec3f d1(p - v0);
    vec3f d2(v1 - v0);
    vec3f min_v(v0);
    float t = dot(d2, d2);

    if (t > std::numeric_limits<float>::min()) {
        t = dot(d1, d2) / t;
        if (t > 1.0)
            d1 = p - (min_v = v1);
        else if (t > 0.0)
            d1 = p - (min_v = v0 + d2 * t);
    }

    nearest_point = min_v;
    return length(d1);
}

float dist_point_triangle(const vec3f& p, const vec3f& v0, const vec3f& v1,
                        const vec3f& v2, vec3f& nearest_point) {
    vec3f v0v1 = v1 - v0;
    vec3f v0v2 = v2 - v0;
    vec3f n = cross(v0v1, v0v2); // not normalized !
    float d = lengthSquared(n);

    // Check if the triangle is degenerated -> measure dist to line segments
    if (fabs(d) < std::numeric_limits<float>::min()) {
        vec3f q, qq;
        float d, dd(std::numeric_limits<float>::max());

        dd = dist_point_line_segment(p, v0, v1, qq);

        d = dist_point_line_segment(p, v1, v2, q);
        if (d < dd) {
            dd = d;
            qq = q;
        }

        d = dist_point_line_segment(p, v2, v0, q);
        if (d < dd) {
            dd = d;
            qq = q;
        }

        nearest_point = qq;
        return dd;
    }

    float inv_d = 1.0 / d;
    vec3f v1v2 = v2;
    v1v2 -= v1;
    vec3f v0p = p;
    v0p -= v0;
    vec3f t = cross(v0p, n);
    float a = dot(t, v0v2) * -inv_d;
    float b = dot(t, v0v1) * inv_d;
    float s01, s02, s12;

    // Calculate the distance to an edge or a corner vertex
    if (a < 0) {
        s02 = dot(v0v2, v0p) / lengthSquared(v0v2);
        if (s02 < 0.0) {
            s01 = dot(v0v1, v0p) / lengthSquared(v0v1);
            if (s01 <= 0.0)
                v0p = v0;
            else if (s01 >= 1.0)
                v0p = v1;
            else
                (v0p = v0) += (v0v1 *= s01);
        } else if (s02 > 1.0) {
            s12 = dot(v1v2, (p - v1)) / lengthSquared(v1v2);
            if (s12 >= 1.0)
                v0p = v2;
            else if (s12 <= 0.0)
                v0p = v1;
            else
                (v0p = v1) += (v1v2 *= s12);
        } else {
            (v0p = v0) += (v0v2 *= s02);
        }
    }

    // Calculate the distance to an edge or a corner vertex
    else if (b < 0.0) {
        s01 = dot(v0v1, v0p) / lengthSquared(v0v1);
        if (s01 < 0.0) {
            s02 = dot(v0v2, v0p) / lengthSquared(v0v2);
            if (s02 <= 0.0)
                v0p = v0;
            else if (s02 >= 1.0)
                v0p = v2;
            else
                (v0p = v0) += (v0v2 *= s02);
        } else if (s01 > 1.0) {
            s12 = dot(v1v2, (p - v1)) / lengthSquared(v1v2);
            if (s12 >= 1.0)
                v0p = v2;
            else if (s12 <= 0.0)
                v0p = v1;
            else
                (v0p = v1) += (v1v2 *= s12);
        } else {
            (v0p = v0) += (v0v1 *= s01);
        }
    }

    // Calculate the distance to an edge or a corner vertex
    else if (a + b > 1.0) {
        s12 = dot(v1v2, (p - v1)) / lengthSquared(v1v2);
        if (s12 >= 1.0) {
            s02 = dot(v0v2, v0p) / lengthSquared(v0v2);
            if (s02 <= 0.0)
                v0p = v0;
            else if (s02 >= 1.0)
                v0p = v2;
            else
                (v0p = v0) += (v0v2 *= s02);
        } else if (s12 <= 0.0) {
            s01 = dot(v0v1, v0p) / lengthSquared(v0v1);
            if (s01 <= 0.0)
                v0p = v0;
            else if (s01 >= 1.0)
                v0p = v1;
            else
                (v0p = v0) += (v0v1 *= s01);
        } else {
            (v0p = v1) += (v1v2 *= s12);
        }
    }

    // Calculate the distance to an interior point of the triangle
    else {
        n *= (dot(n, v0p) * inv_d);
        (v0p = p) -= n;
    }

    nearest_point = v0p;
    v0p -= p;
    return length(v0p);
}

TriangleKdTree::TriangleKdTree(const SurfaceMesh* mesh, unsigned int max_faces,
                               unsigned int max_depth) {
    // init
    root_ = new Node();
    root_->faces = new std::vector<int>();

    // collect triangles
    root_->faces->reserve(mesh->n_faces());
    face_points_.reserve(mesh->n_faces());
    auto &points = mesh->prim_->attr<vec3f>("pos");

    auto& fdeleted = mesh->prim_->tris.attr<int>("f_deleted");
    for (int fit = 0; fit < mesh->faces_size_; ++fit) {
        if (mesh->has_garbage_ && fdeleted[fit])
            continue;

        root_->faces->push_back(fit);

        auto fvIt = mesh->prim_->tris[fit];
        face_points_.push_back({points[fvIt[0]], points[fvIt[1]], points[fvIt[2]]});
    }

    // call recursive helper
    build_recurse(root_, max_faces, max_depth);
}

TriangleKdTree::TriangleKdTree(const AttrVector<vec3i>& tris, const AttrVector<vec3f>& points,
                               unsigned int max_faces, unsigned int max_depth) {
    root_ = new Node();
    root_->faces = new std::vector<int>();
 
    root_->faces->reserve(tris->size());
    face_points_.reserve(tris->size());

    for (int fit = 0; fit < tris->size(); ++fit) {
        root_->faces->push_back(fit);
        auto fvIt = tris[fit];
        face_points_.push_back({points[fvIt[0]], points[fvIt[1]], points[fvIt[2]]});
    }

    build_recurse(root_, max_faces, max_depth);
}

void TriangleKdTree::build_recurse(Node* node, unsigned int max_faces,
                                           unsigned int depth) {
    // should we stop at this level ?
    if ((depth == 0) || (node->faces->size() <= max_faces))
        return;

    // compute bounding box
    BoundingBox bbox;
    for (const auto& it : *node->faces) {
        bbox += face_points_[it][0];
        bbox += face_points_[it][1];
        bbox += face_points_[it][2];
    }

    // split longest side of bounding box
    vec3f bb = bbox.max() - bbox.min();
    float length = bb[0];
    int axis = 0;
    if (bb[1] > length)
        length = bb[(axis = 1)];
    if (bb[2] > length)
        length = bb[(axis = 2)];

    // split in the middle
    float split = bbox.center()[axis];

    // create children
    auto* left = new Node();
    left->faces = new std::vector<int>();
    left->faces->reserve(node->faces->size() / 2);
    auto* right = new Node();
    right->faces = new std::vector<int>();
    right->faces->reserve(node->faces->size() / 2);

    // partition for left and right child
    for (const auto& it : *node->faces) {
        bool l = false, r = false;

        const auto& pos =  face_points_[it];
        if (pos[0][axis] <= split)
            l = true;
        else
            r = true;
        if (pos[1][axis] <= split)
            l = true;
        else
            r = true;
        if (pos[2][axis] <= split)
            l = true;
        else
            r = true;

        if (l)
            left->faces->push_back(it);

        if (r)
            right->faces->push_back(it);
    }

    // stop here?
    if (left->faces->size() == node->faces->size() ||
        right->faces->size() == node->faces->size()) {
        // compact my memory
        node->faces->shrink_to_fit();

        // delete new nodes
        delete left;
        delete right;

        return;
    }

    // or recurse further?
    else {
        // free my memory
        delete node->faces;
        node->faces = nullptr;

        // store internal data
        node->axis = axis;
        node->split = split;
        node->left_child = left;
        node->right_child = right;

        // recurse to childen
        build_recurse(node->left_child, max_faces, depth - 1);
        build_recurse(node->right_child, max_faces, depth - 1);
    }
}

TriangleKdTree::NearestNeighbor TriangleKdTree::nearest(const vec3f& p) const {
    NearestNeighbor data;
    data.dist = std::numeric_limits<float>::max();
    nearest_recurse(root_, p, data);
    return data;
}

void TriangleKdTree::faces_in_box(BoundingBox& box, std::vector<int>& faces) {
    in_box_recursive(root_, box, faces);
}

void TriangleKdTree::nearest_recurse(Node* node, const vec3f& point,
                                     NearestNeighbor& data) const {
    // terminal node?
    if (!node->left_child) {
        for (const auto& f : *node->faces) {
            vec3f n;
            const auto& pos = face_points_[f];
            auto d = dist_point_triangle(point, pos[0], pos[1], pos[2], n);
            if (d < data.dist) {
                data.dist = d;
                data.face = f;
                data.nearest = n;
            }
        }
    }

    // non-terminal node
    else {
        float dist = point[node->axis] - node->split;

        if (dist <= 0.0) {
            nearest_recurse(node->left_child, point, data);
            if (fabs(dist) < data.dist)
                nearest_recurse(node->right_child, point, data);
        } else {
            nearest_recurse(node->right_child, point, data);
            if (fabs(dist) < data.dist)
                nearest_recurse(node->left_child, point, data);
        }
    }
}

void TriangleKdTree::in_box_recursive(Node* node, BoundingBox& box,
                                      std::vector<int>& faces) {
    if (!node->left_child) {
        for (const auto& f : *node->faces) {
            vec3f n;
            const auto& pos = face_points_[f];
            for (int i = 0; i < 3; ++i) {
                if (box.min()[0] <= pos[i][0] && pos[i][0] <= box.max()[1] ||
                    box.min()[1] <= pos[i][1] && pos[i][1] <= box.max()[1] ||
                    box.min()[2] <= pos[i][2] && pos[i][2] <= box.max()[2]) {
                        faces.push_back(f);
                        break;
                    }
            }
        }
        return;
    }

    if (box.min()[node->axis] <= node->split)
        in_box_recursive(node->left_child, box, faces);
    if (box.max()[node->axis] >= node->split)
        in_box_recursive(node->right_child, box, faces);
}

} // namespace pmp
} // namespace zeno
