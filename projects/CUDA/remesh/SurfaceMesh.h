// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#pragma once
#include <assert.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <zeno/utils/logger.h>
#include <zeno/utils/vec.h>
#include <map>
#include <set>
#include "./BoundingBox.h"

namespace zeno {
namespace pmp {

#define PMP_MAX_INDEX INT_LEAST32_MAX
#define PMP_ENABLE_PROFILE 0

class SurfaceMesh {

public:
    
    class VertexAroundVertexCirculator {
    public:
        VertexAroundVertexCirculator(const SurfaceMesh* mesh = nullptr,
                                     int v = PMP_MAX_INDEX)
            : mesh_(mesh), is_active_(true) {
            if (mesh_)
                halfedge_ = mesh_->vconn_[v].halfedge_;
        }

        bool operator==(const VertexAroundVertexCirculator& rhs) const {
            assert(mesh_ == rhs.mesh_);
            return (is_active_ && (halfedge_ == rhs.halfedge_));
        }

        bool operator!=(const VertexAroundVertexCirculator& rhs) const {
            return !operator==(rhs);
        }

        // pre-increment (rotate couter-clockwise)
        VertexAroundVertexCirculator& operator++() {
            halfedge_ = mesh_->hconn_[halfedge_].prev_halfedge_ ^ 1;
            is_active_ = true;
            return *this;
        }

        // pre-decrement (rotate clockwise)
        VertexAroundVertexCirculator& operator--() {
            halfedge_ = mesh_->hconn_[halfedge_^1].next_halfedge_;
            return *this;
        }

        int operator*() const {
            return mesh_->to_vertex(halfedge_);
        }

        // cast to bool: true if vertex is not isolated
        operator bool() const { return halfedge_ != PMP_MAX_INDEX; }

        // helper for C++11 range-based for-loops
        VertexAroundVertexCirculator& begin() {
            is_active_ = (halfedge_ == PMP_MAX_INDEX);
            return *this;
        }
        // helper for C++11 range-based for-loops
        VertexAroundVertexCirculator& end() {
            is_active_ = true;
            return *this;
        }

    private:
        const SurfaceMesh* mesh_;
        int halfedge_;
        bool is_active_; // helper for C++11 range-based for-loops
    };

    class HalfedgeAroundVertexCirculator {
    public:
        HalfedgeAroundVertexCirculator(const SurfaceMesh* mesh = nullptr,
                                       int v = PMP_MAX_INDEX)
            : mesh_(mesh), is_active_(true) {
            if (mesh_)
                halfedge_ = mesh_->vconn_[v].halfedge_;
        }

        bool operator==(const HalfedgeAroundVertexCirculator& rhs) const {
            assert(mesh_ == rhs.mesh_);
            return (is_active_ && (halfedge_ == rhs.halfedge_));
        }

        bool operator!=(const HalfedgeAroundVertexCirculator& rhs) const {
            return !operator==(rhs);
        }

        // pre-increment (rotate couter-clockwise)
        HalfedgeAroundVertexCirculator& operator++() {
            halfedge_ = mesh_->hconn_[halfedge_].prev_halfedge_ ^ 1;
            is_active_ = true;
            return *this;
        }

        // pre-decrement (rotate clockwise)
        HalfedgeAroundVertexCirculator& operator--() {
            halfedge_ = mesh_->hconn_[halfedge_^1].next_halfedge_;
            return *this;
        }

        // get the halfedge the circulator refers to
        int operator*() const { return halfedge_; }

        // cast to bool: true if vertex is not isolated
        operator bool() const { return halfedge_ != PMP_MAX_INDEX; }

        // helper for C++11 range-based for-loops
        HalfedgeAroundVertexCirculator& begin() {
            is_active_ = (halfedge_ == PMP_MAX_INDEX);
            return *this;
        }
        // helper for C++11 range-based for-loops
        HalfedgeAroundVertexCirculator& end() {
            is_active_ = true;
            return *this;
        }

    private:
        const SurfaceMesh* mesh_;
        int halfedge_;
        bool is_active_; // helper for C++11 range-based for-loops
    };

    class FaceAroundVertexCirculator {
    public:
        FaceAroundVertexCirculator(const SurfaceMesh* mesh = nullptr,
                                       int v = PMP_MAX_INDEX)
            : mesh_(mesh), is_active_(true) {
            if (mesh_) {
                halfedge_ = mesh_->vconn_[v].halfedge_;
                if (halfedge_ != PMP_MAX_INDEX && mesh_->hconn_[halfedge_].face_ == PMP_MAX_INDEX)
                    operator++();
            }
        }

        bool operator==(const FaceAroundVertexCirculator& rhs) const {
            assert(mesh_ == rhs.mesh_);
            return (is_active_ && (halfedge_ == rhs.halfedge_));
        }

        bool operator!=(const FaceAroundVertexCirculator& rhs) const {
            return !operator==(rhs);
        }

        // pre-increment (rotate couter-clockwise)
        FaceAroundVertexCirculator& operator++() {
            assert(mesh_ && (halfedge_ != PMP_MAX_INDEX));
            do {
                halfedge_ = mesh_->hconn_[halfedge_].prev_halfedge_ ^ 1;
            } while (mesh_->hconn_[halfedge_].face_ == PMP_MAX_INDEX);
            is_active_ = true;
            return *this;
        }

        // pre-decrement (rotate clockwise)
        FaceAroundVertexCirculator& operator--() {
            assert(mesh_ && (halfedge_ != PMP_MAX_INDEX));
            do {
                halfedge_ = mesh_->hconn_[halfedge_^1].next_halfedge_;
            } while (mesh_->hconn_[halfedge_].face_ == PMP_MAX_INDEX);
            return *this;
        }

        // get the halfedge the circulator refers to
        int operator*() const {
            assert(mesh_ && (halfedge_ != PMP_MAX_INDEX));
            return mesh_->hconn_[halfedge_].face_;
        }

        // cast to bool: true if vertex is not isolated
        operator bool() const { return halfedge_ != PMP_MAX_INDEX; }

        // helper for C++11 range-based for-loops
        FaceAroundVertexCirculator& begin() {
            is_active_ = (halfedge_ == PMP_MAX_INDEX);
            return *this;
        }
        // helper for C++11 range-based for-loops
        FaceAroundVertexCirculator& end() {
            is_active_ = true;
            return *this;
        }

    private:
        const SurfaceMesh* mesh_;
        int halfedge_;
        bool is_active_; // helper for C++11 range-based for-loops
    };

    SurfaceMesh(std::shared_ptr<zeno::PrimitiveObject> prim,
                std::string line_pick_tag);
    SurfaceMesh(const SurfaceMesh& rhs);
    ~SurfaceMesh();

    int add_tri(const vec3i& vertices);

    size_t n_faces() const { return faces_size_ - deleted_faces_; }

    int halfedge(int v) const { return vconn_[v].halfedge_; }
    bool is_boundary_v(int v) const {
        int h = halfedge(v);
        return (!(h != PMP_MAX_INDEX && hconn_[h].face_ != PMP_MAX_INDEX));
    }
    bool is_boundary_e(int e) const {
        return (hconn_[e << 1].face_ == PMP_MAX_INDEX || hconn_[e << 1 | 1].face_ == PMP_MAX_INDEX);
    }
    bool is_isolated(int v) const { return halfedge(v) == PMP_MAX_INDEX; }
    inline int to_vertex(int h) const { return hconn_[h].vertex_; }
    inline int from_vertex(int h) const { return to_vertex(h^1); }
    inline int next_halfedge(int h) const {
        return hconn_[h].next_halfedge_;
    }
    inline int prev_halfedge(int h) const {
        return hconn_[h].prev_halfedge_;
    }
    inline void set_next_halfedge(int h, int nh) {
        if (h != PMP_MAX_INDEX) {
            hconn_[h].next_halfedge_ = nh;
        }
        if (nh != PMP_MAX_INDEX) {
            hconn_[nh].prev_halfedge_ = h;
        }
    }
    inline void build_dup_list() {
        auto &vduplicate = prim_->verts.attr<int>("v_duplicate");
        dup_list_.clear();
        for (int v = 0; v < vertices_size_; ++v) {
            int src = vduplicate[v];
            if (dup_list_.count(src) == 0)
                dup_list_[src] = std::set<int>{};
            dup_list_[src].insert(v);
        }
    }
    inline std::set<int>& get_dup_list(int v) {
        return dup_list_[v];
    }
    inline void erase_dup_list(int v) {
        dup_list_.erase(v);
    }

    VertexAroundVertexCirculator vertices(int v) const {
        return VertexAroundVertexCirculator(this, v);
    }

    HalfedgeAroundVertexCirculator halfedges(int v) const {
        return HalfedgeAroundVertexCirculator(this, v);
    }

    FaceAroundVertexCirculator faces(int v) const {
        return FaceAroundVertexCirculator(this, v);
    }

    int find_halfedge(int start, int end) const;
    void is_collapse_ok(int v0v1, bool &hcol01, bool &hcol10, bool relaxed = false);
    void collapse(int h);
    void garbage_collection();
    int split(int e, int v, int& new_lines, int& new_faces);
    bool is_flip_ok(int e, bool relaxed = false) const;
    void flip(int e);

    size_t valence(int v) const;

    // clamp cotangent values as if angles are in [1, 179]
    inline float clamp_cot(const float v) {
        const float bound = 19.1; // 3 degrees
        return (v < -bound ? -bound : (v > bound ? bound : v));
    }

    float cotan_weight(int e) {
        float weight = 0.0;

        const int h0 = (e << 1);
        const int h1 = (e << 1 | 1);

        auto& pos = prim_->attr<vec3f>("pos");

        const vec3f p0 = pos[to_vertex(h0)];
        const vec3f p1 = pos[to_vertex(h1)];

        if (hconn_[h0].face_ != PMP_MAX_INDEX) {
            const vec3f p2 = pos[to_vertex(next_halfedge(h0))];
            const vec3f d0 = p0 - p2;
            const vec3f d1 = p1 - p2;
            const float area = length(cross(d0, d1));
            if (area > std::numeric_limits<float>::min()) {
                const float cot = dot(d0, d1) / area;
                weight += clamp_cot(cot);
            }
        }

        if (hconn_[h1].face_ != PMP_MAX_INDEX) {
            const vec3f p2 = pos[to_vertex(next_halfedge(h1))];
            const vec3f d0 = p0 - p2;
            const vec3f d1 = p1 - p2;
            const float area = length(cross(d0, d1));
            if (area > std::numeric_limits<float>::min()) {
                const float cot = dot(d0, d1) / area;
                weight += clamp_cot(cot);
            }
        }

        assert(!std::isnan(weight));
        assert(!std::isinf(weight));

        return weight;
    }

    float voronoi_area(int v) {
        float area(0.0);

        if (!is_isolated(v)) {
            int h0, h1, h2;
            vec3f p, q, r, pq, qr, pr;
            float dotp, dotq, dotr, triArea;
            float cotq, cotr;

            auto& pos = prim_->attr<vec3f>("pos");

            for (auto h : halfedges(v)) {
                h0 = h;
                h1 = next_halfedge(h0);
                h2 = next_halfedge(h1);

                if (hconn_[h0].face_ == PMP_MAX_INDEX)
                    continue;

                // three vertex positions
                p = pos[to_vertex(h2)];
                q = pos[to_vertex(h0)];
                r = pos[to_vertex(h1)];

                // edge vectors
                (pq = q) -= p;
                (qr = r) -= q;
                (pr = r) -= p;

                // compute and check triangle area
                triArea = length(cross(pq, pr));
                if (triArea <= std::numeric_limits<float>::min())
                    continue;

                // dot products for each corner (of its two emanating edge vectors)
                dotp = dot(pq, pr);
                dotq = -dot(qr, pq);
                dotr = dot(qr, pr);

                // angle at p is obtuse
                if (dotp < 0.0) {
                    area += 0.25 * triArea;
                }

                // angle at q or r obtuse
                else if (dotq < 0.0 || dotr < 0.0) {
                    area += 0.125 * triArea;
                }

                // no obtuse angles
                else {
                    // cot(angle) = cos(angle)/sin(angle) = dot(A,B)/length(cross(A,B))
                    cotq = dotq / triArea;
                    cotr = dotr / triArea;

                    // clamp cot(angle) by clamping angle to [1,179]
                    area += 0.125 * (lengthSquared(pr) * clamp_cot(cotq) +
                                    lengthSquared(pq) * clamp_cot(cotr));
                }
            }
        }

        assert(!std::isnan(area));
        assert(!std::isinf(area));

        return area;
    }

    BoundingBox bounds() {
        BoundingBox bb;
        auto pos = prim_->attr<vec3f>("pos");
        for (auto p : pos)
            bb += p;
        return bb;
    }


    private:
    friend class SurfaceRemeshing;
    friend class SurfaceCurvature;
    friend class SurfaceNormals;
    friend class TriangleKdTree;


    struct VertexConnectivity {
        int halfedge_ = PMP_MAX_INDEX;
    };

    struct HalfedgeConnectivity {
        int face_ = PMP_MAX_INDEX;
        int vertex_ = PMP_MAX_INDEX;
        int next_halfedge_ = PMP_MAX_INDEX;
        int prev_halfedge_ = PMP_MAX_INDEX;
    };

    struct FaceConnectivity {
        int halfedge_ = PMP_MAX_INDEX;
    };

    int new_vertex(const vec3f& p) {
        if (vertices_size_ == PMP_MAX_INDEX - 1) {
            zeno::log_error("remesh: cannot allocate vertex, max index reached");
            return PMP_MAX_INDEX;
        }

        prim_->verts.push_back(p);
        ++vertices_size_;
        if (vertices_size_ > vconn_.size()) {
            vconn_.resize(vertices_size_);
        }
        return vertices_size_ - 1;
    }

    int new_edge(int start, int end) {
        assert(start != end);

        if (halfedges_size_ >= PMP_MAX_INDEX - 2) {
            zeno::log_error("remesh: cannot allocate edge, max index reached");
            return PMP_MAX_INDEX;
        }

        prim_->lines.push_back(vec2i(start, end));
        ++lines_size_;
        
        halfedges_size_+=2;

        int h0 = halfedges_size_ - 2;
        int h1 = halfedges_size_ - 1;

        if (halfedges_size_ > hconn_.size()) {
            hconn_.resize(halfedges_size_);
        }
        hconn_[h0].vertex_ = end;
        hconn_[h1].vertex_ = start;

        return h0;
    }

    int new_halfedge(int start, int end, int line_id) {
        if ((int)(line_id << 1) < 0) {
            zeno::log_error("remesh: cannot allocate edge, max index reached");
            return PMP_MAX_INDEX;
        }
        assert(start != end);
        
        int h0 = line_id << 1;
        int h1 = line_id << 1 | 1;

        hconn_[h0].vertex_ = end;
        hconn_[h1].vertex_ = start;

        return h0;
    }

    int new_face(int v1, int v2, int v3) {
        if (faces_size_ == PMP_MAX_INDEX - 1) {
            zeno::log_error("remesh: cannot allocate face, max index reached");
            return PMP_MAX_INDEX;
        }

        prim_->tris.push_back(vec3i(v1, v2, v3));
        
        ++faces_size_;
        if (faces_size_ > fconn_.size()) {
            fconn_.resize(faces_size_);
        }
        return faces_size_ - 1;
    }

    void adjust_outgoing_halfedge(int v);
    void remove_edge_helper(int h);
    void remove_loop_helper(int h);

    std::shared_ptr<zeno::PrimitiveObject> prim_;

    size_t vertices_size_;
    size_t halfedges_size_;
    size_t lines_size_;
    size_t faces_size_;

    std::map<int, std::set<int>> dup_list_{};
    std::map<std::pair<int, int>, int> line_map_{};
    std::string line_pick_tag_;

    // connectivity information
    std::vector<VertexConnectivity> vconn_;
    std::vector<HalfedgeConnectivity> hconn_;
    std::vector<FaceConnectivity> fconn_;

    // numbers of deleted entities
    int deleted_vertices_;
    int deleted_lines_;
    int deleted_faces_;

    // indicate garbage present
    bool has_garbage_;

    // a constant for collapse and flip checks
    const float angle_thrd = (M_PI * 30.0f / 180.0f);

    // helper data for add_tri()
    typedef std::pair<int, int> NextCacheEntry;
    typedef std::vector<NextCacheEntry> NextCache;
    std::vector<int> add_face_vertices_;
    std::vector<int> add_face_halfedges_;
    std::vector<bool> add_face_is_new_;
    std::vector<bool> add_face_needs_adjust_;
    NextCache add_face_next_cache_;
};

} // namespace pmp
} // namespace zeno
