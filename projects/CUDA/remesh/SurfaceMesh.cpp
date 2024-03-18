// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include "./SurfaceMesh.h"

#include <cmath>
#include <algorithm>

namespace zeno {
namespace pmp {

SurfaceMesh::SurfaceMesh(std::shared_ptr<zeno::PrimitiveObject> prim,
                         std::string line_pick_tag)
                         : line_pick_tag_(line_pick_tag) {
    prim_ = prim;
    vconn_.clear();
    hconn_.clear();
    fconn_.clear();

    vertices_size_ = prim_->verts.size();
    lines_size_ = prim_->lines.size();
    faces_size_ = 0;
    halfedges_size_ = 0;

    vconn_.resize(vertices_size_);
    fconn_.resize(prim_->tris.size());
    
    auto& lines = prim_->lines;
    for (int i = 0; i < lines_size_; ++i) {
        auto line = std::make_pair(lines[i][0], lines[i][1]);
        line_map_[line] = i;
    }
    halfedges_size_ = lines_size_ * 2;
    hconn_.resize(halfedges_size_);

    prim_->verts.add_attr<int>("v_deleted", 0);
    prim_->lines.add_attr<int>("e_deleted", 0);
    prim_->tris.add_attr<int>("f_deleted", 0);

    for (auto& it : prim_->tris) {
        if (add_tri(it) == PMP_MAX_INDEX) {
            return;
        }
    }

    deleted_vertices_ = 0;
    deleted_lines_ = 0;
    deleted_faces_ = 0;
    has_garbage_ = false;
    
}

SurfaceMesh::SurfaceMesh(const SurfaceMesh& rhs) {
    prim_ = std::static_pointer_cast<PrimitiveObject>(rhs.prim_->clone());

    vertices_size_ = rhs.vertices_size_;
    halfedges_size_ = rhs.halfedges_size_;
    lines_size_ = rhs.lines_size_;
    faces_size_ = rhs.faces_size_;

    vconn_.clear();
    hconn_.clear();
    fconn_.clear();
    vconn_ = rhs.vconn_;
    hconn_ = rhs.hconn_;
    fconn_ = rhs.fconn_;

    deleted_vertices_ = rhs.deleted_vertices_;
    deleted_lines_ = rhs.deleted_lines_;
    deleted_faces_ = rhs.deleted_faces_;
    has_garbage_ = rhs.has_garbage_;
}

SurfaceMesh::~SurfaceMesh() = default;

int SurfaceMesh::find_halfedge(int start, int end) const {
    assert(start < vertices_size_ && end < vertices_size_);

    int h = halfedge(start);
    const int hh = h;

    if (h != PMP_MAX_INDEX) {
        do {
            if (to_vertex(h) == end)
                return h;
            h = next_halfedge(h ^ 1);
        } while (h != hh);
    }

    return PMP_MAX_INDEX;
}

void SurfaceMesh::adjust_outgoing_halfedge(int v) {
    int h = halfedge(v);
    const int hh = h;

    if (h != PMP_MAX_INDEX) {
        do {
            if (hconn_[h].face_ == PMP_MAX_INDEX) {
                vconn_[v].halfedge_ = h;
                return;
            }
            h = next_halfedge(h ^ 1);
        } while (h != hh);
    }
}

int SurfaceMesh::add_tri(const vec3i& vertices){
    int v;
    size_t i, ii, id;
    int innerNext, innerPrev, outerNext, outerPrev, boundaryNext,
        boundaryPrev, patchStart, patchEnd;

    // use global arrays to avoid new/delete of local arrays
    std::vector<int>& halfedges = add_face_halfedges_;
    std::vector<bool>& isNew = add_face_is_new_;
    std::vector<bool>& needsAdjust = add_face_needs_adjust_;
    NextCache& nextCache = add_face_next_cache_;
    halfedges.clear();
    halfedges.resize(3);
    isNew.clear();
    isNew.resize(3);
    needsAdjust.clear();
    needsAdjust.resize(3, false);
    nextCache.clear();
    nextCache.reserve(9);

    // test for topological errors
    for (i = 0, ii = 1; i < 3; ++i, ++ii, ii %= 3) {
        if (!is_boundary_v(vertices[i])) {
            zeno::log_error("SurfaceMesh::add_face: complex vertex {}", vertices[i]);
            return PMP_MAX_INDEX;
        }

        halfedges[i] = find_halfedge(vertices[i], vertices[ii]);
        isNew[i] = (halfedges[i] == PMP_MAX_INDEX);

        if (!isNew[i] && hconn_[halfedges[i]].face_ != PMP_MAX_INDEX) {
            zeno::log_error("SurfaceMesh::add_face: complex edge {}-{} ", vertices[i], vertices[ii]);
            return PMP_MAX_INDEX;
        }
    }

    // re-link patches if necessary
    for (i = 0, ii = 1; i < 3; ++i, ++ii, ii %= 3) {
        if (!isNew[i] && !isNew[ii]) {
            innerPrev = halfedges[i];
            innerNext = halfedges[ii];

            if (next_halfedge(innerPrev) != innerNext) {
                // here comes the ugly part... we have to relink a whole patch

                // search a free gap
                // free gap will be between boundaryPrev and boundaryNext
                outerPrev = innerNext^1;
                outerNext = innerPrev^1;
                boundaryPrev = outerPrev;
                do {
                    boundaryPrev = next_halfedge(boundaryPrev) ^ 1;
                } while (hconn_[boundaryPrev].face_ != PMP_MAX_INDEX ||
                         boundaryPrev == innerPrev);
                boundaryNext = next_halfedge(boundaryPrev);
                assert(hconn_[boundaryPrev].face_ == PMP_MAX_INDEX);
                assert(hconn_[boundaryNext].face_ == PMP_MAX_INDEX);

                if (boundaryNext == innerNext) {
                    zeno::log_error("SurfaceMeshT::add_face: patch re-linking failed {}", vertices);
                    return PMP_MAX_INDEX;
                }

                // other halfedges' handles
                patchStart = next_halfedge(innerPrev);
                patchEnd = prev_halfedge(innerNext);

                // relink
                nextCache.emplace_back(boundaryPrev, patchStart);
                nextCache.emplace_back(patchEnd, boundaryNext);
                nextCache.emplace_back(innerPrev, innerNext);
            }
        }
    }

    // create missing edges
    for (i = 0, ii = 1; i < 3; ++i, ++ii, ii %= 3) {
        if (isNew[i]) {
            int line_id;
            if (line_map_.count(std::make_pair(vertices[i], vertices[ii])) > 0) {
                line_id = line_map_[std::make_pair(vertices[i], vertices[ii])];
            } else {
                line_id = line_map_[std::make_pair(vertices[ii], vertices[i])];
            }
            halfedges[i] = new_halfedge(vertices[i], vertices[ii], line_id);
            if (halfedges[i] == PMP_MAX_INDEX) {
                return PMP_MAX_INDEX;
            }
        }
    }

    // create the face
    if (faces_size_ == PMP_MAX_INDEX - 1) {
        zeno::log_error("remesh: cannot allocate face, max index reached");
        return PMP_MAX_INDEX;
    }
    ++faces_size_;
    int f = faces_size_ - 1;
    fconn_[f].halfedge_ = halfedges[2];

    // setup halfedges
    for (i = 0, ii = 1; i < 3; ++i, ++ii, ii %= 3) {
        v = vertices[ii];
        innerPrev = halfedges[i];
        innerNext = halfedges[ii];

        id = 0;
        if (isNew[i])
            id |= 1;
        if (isNew[ii])
            id |= 2;

        if (id) {
            outerPrev = innerNext^1;
            outerNext = innerPrev^1;

            // set outer links
            switch (id) {
                case 1: // prev is new, next is old
                    boundaryPrev = prev_halfedge(innerNext);
                    nextCache.emplace_back(boundaryPrev, outerNext);
                    vconn_[v].halfedge_ = outerNext;
                    break;

                case 2: // next is new, prev is old
                    boundaryNext = next_halfedge(innerPrev);
                    nextCache.emplace_back(outerPrev, boundaryNext);
                    vconn_[v].halfedge_ = boundaryNext;
                    break;

                case 3: // both are new
                    if (vconn_[v].halfedge_ == PMP_MAX_INDEX) {
                        vconn_[v].halfedge_ = outerNext;
                        nextCache.emplace_back(outerPrev, outerNext);
                    } else {
                        boundaryNext = vconn_[v].halfedge_;
                        boundaryPrev = prev_halfedge(boundaryNext);
                        nextCache.emplace_back(boundaryPrev, outerNext);
                        nextCache.emplace_back(outerPrev, boundaryNext);
                    }
                    break;
            }

            // set inner link
            nextCache.emplace_back(innerPrev, innerNext);
        }
        else
            needsAdjust[ii] = (vconn_[v].halfedge_ == innerNext);

        // set face handle
        hconn_[halfedges[i]].face_ = f;
    }

    // process next halfedge cache
    NextCache::const_iterator ncIt(nextCache.begin()), ncEnd(nextCache.end());
    for (; ncIt != ncEnd; ++ncIt) {
        hconn_[ncIt->first].next_halfedge_ = ncIt->second;
        hconn_[ncIt->second].prev_halfedge_ = ncIt->first;
    }

    // adjust vertices' halfedge handle
    for (i = 0; i < 3; ++i) {
        if (needsAdjust[i])
            adjust_outgoing_halfedge(vertices[i]);
    }

    return f;
}

size_t SurfaceMesh::valence(int v) const {
    size_t count = 0;

    for (auto vv : vertices(v)) {
        assert(vv < vertices_size_);
        ++count;
    }

    return count;
}

int SurfaceMesh::split(int e, int v, int& new_lines, int& new_faces) {
    int h0 = e<<1;
    int o0 = e<<1|1;

    int v2 = to_vertex(o0);
    int v4 = to_vertex(h0);

    int e1 = new_edge(v, v2);
    if (e1 == PMP_MAX_INDEX) {
        return PMP_MAX_INDEX;
    }
    new_lines = 1;
    new_faces = 0;

    int t1 = e1^1;
    prim_->lines[e] = vec2i(v, v4);

    int f0 = hconn_[h0].face_;
    int f3 = hconn_[o0].face_;

    vconn_[v].halfedge_ = h0;
    hconn_[o0].vertex_ = v;

    if (f0 != PMP_MAX_INDEX) {
        int h1 = next_halfedge(h0);
        int h2 = next_halfedge(h1);

        int v1 = to_vertex(h1);

        int e0 = new_edge(v, v1);
        if (e0 == PMP_MAX_INDEX) {
            return PMP_MAX_INDEX;
        }
        new_lines += 1;
        int t0 = e0 ^ 1;

        int f1 = new_face(v, v1, v2);
        if (f1 == PMP_MAX_INDEX) {
            return PMP_MAX_INDEX;
        }
        new_faces += 1;
        fconn_[f0].halfedge_ = h0;
        fconn_[f1].halfedge_ = h2;

        hconn_[h1].face_ = f0;
        hconn_[t0].face_ = f0;
        hconn_[h0].face_ = f0;

        hconn_[h2].face_ = f1;
        hconn_[t1].face_ = f1;
        hconn_[e0].face_ = f1;

        set_next_halfedge(h0, h1);
        set_next_halfedge(h1, t0);
        set_next_halfedge(t0, h0);

        set_next_halfedge(e0, h2);
        set_next_halfedge(h2, t1);
        set_next_halfedge(t1, e0);

        prim_->tris[f0] = vec3i(v, v4, v1);
    } else {
        set_next_halfedge(prev_halfedge(h0), t1);
        set_next_halfedge(t1, h0);
        // halfedge handle of vh already is h0
    }

    if (f3 != PMP_MAX_INDEX) {
        int o1 = next_halfedge(o0);
        int o2 = next_halfedge(o1);

        int v3 = to_vertex(o1);

        int e2 = new_edge(v, v3);
        if (e2 == PMP_MAX_INDEX) {
            return PMP_MAX_INDEX;
        }
        new_lines += 1;
        int t2 = e2 ^ 1;

        int f2 = new_face(v, v2, v3);
        if (f2 == PMP_MAX_INDEX) {
            return PMP_MAX_INDEX;
        }
        new_faces += 1;
        fconn_[f2].halfedge_ = o1;
        fconn_[f3].halfedge_ = o0;

        hconn_[o1].face_ = f2;
        hconn_[t2].face_ = f2;
        hconn_[e1].face_ = f2;

        hconn_[o2].face_ = f3;
        hconn_[o0].face_ = f3;
        hconn_[e2].face_ = f3;

        set_next_halfedge(e1, o1);
        set_next_halfedge(o1, t2);
        set_next_halfedge(t2, e1);

        set_next_halfedge(o0, e2);
        set_next_halfedge(e2, o2);
        set_next_halfedge(o2, o0);

        prim_->tris[f3] = vec3i(v, v3, v4);
    } else {
        set_next_halfedge(e1, next_halfedge(o0));
        set_next_halfedge(o0, e1);
        vconn_[v].halfedge_ = e1;
    }

    if (halfedge(v2) == h0)
        vconn_[v2].halfedge_ = t1;

    return t1;
}

bool SurfaceMesh::is_flip_ok(int e, bool relaxed) const {
    // boundary edges cannot be flipped
    if (is_boundary_e(e))
        return false;

    // check if the flipped edge is already present in the mesh
    int h0 = e<<1;
    int h1 = e<<1|1;

    int v0 = to_vertex(next_halfedge(h0));
    int v1 = to_vertex(next_halfedge(h1));

    if (v0 == v1) // this is generally a bad sign !!!
        return false;

    if (find_halfedge(v0, v1) != PMP_MAX_INDEX)
        return false;

    if (relaxed) {
        // when removing caps, we don't need so strict normal checks
        return true;
    }

    // check about normals for the flip
    // if the normals change more than a given threshold angle,
    // it is very likely to be topologically incorrect

    auto& pos = prim_->attr<vec3f>("pos");
    int old_v0 = to_vertex(h0);
    int old_v1 = to_vertex(h1);

    vec3f old_n0 = normalize(cross(pos[old_v0] - pos[v0], pos[old_v1] - pos[v0]));
    vec3f old_n1 = normalize(cross(pos[old_v1] - pos[v1], pos[old_v0] - pos[v1]));
    vec3f new_n0 = normalize(cross(pos[v1] - pos[old_v0], pos[v0] - pos[old_v0]));
    vec3f new_n1 = normalize(cross(pos[v0] - pos[old_v1], pos[v1] - pos[old_v1]));
    if(acos(clamp(dot(old_n0, new_n0), -1, 1)) > angle_thrd) return false;
    if(acos(clamp(dot(old_n0, new_n1), -1, 1)) > angle_thrd) return false;
    if(acos(clamp(dot(old_n1, new_n0), -1, 1)) > angle_thrd) return false;
    if(acos(clamp(dot(old_n1, new_n1), -1, 1)) > angle_thrd) return false;

    return true;
}

void SurfaceMesh::flip(int e) {
    //let's make it sure it is actually checked
    assert(is_flip_ok(e));

    int a0 = e<<1;
    int b0 = e<<1|1;

    int a1 = next_halfedge(a0);
    int a2 = next_halfedge(a1);

    int b1 = next_halfedge(b0);
    int b2 = next_halfedge(b1);

    int va0 = to_vertex(a0);
    int va1 = to_vertex(a1);

    int vb0 = to_vertex(b0);
    int vb1 = to_vertex(b1);

    int fa = hconn_[a0].face_;
    int fb = hconn_[b0].face_;

    hconn_[a0].vertex_ = va1;
    hconn_[b0].vertex_ = vb1;

    set_next_halfedge(a0, a2);
    set_next_halfedge(a2, b1);
    set_next_halfedge(b1, a0);

    set_next_halfedge(b0, b2);
    set_next_halfedge(b2, a1);
    set_next_halfedge(a1, b0);

    hconn_[a1].face_ = fb;
    hconn_[b1].face_ = fa;

    fconn_[fa].halfedge_ = a0;
    fconn_[fb].halfedge_ = b0;

    prim_->lines[e] = vec2i(va1, vb1);
    prim_->tris[fa] = vec3i(va1, vb0, vb1);
    prim_->tris[fb] = vec3i(va1, vb1, va0);

    if (halfedge(va0) == b0)
        vconn_[va0].halfedge_ = a1;
    if (halfedge(vb0) == a0)
        vconn_[vb0].halfedge_ = b1;
}

void SurfaceMesh::is_collapse_ok(int v0v1, bool &hcol01, bool &hcol10, bool relaxed) {
    int v1v0 = v0v1 ^ 1;
    int v0 = to_vertex(v1v0);
    int v1 = to_vertex(v0v1);
    int vl = PMP_MAX_INDEX, vr = PMP_MAX_INDEX;
    int h1, h2;

    // the edges v1-vl and vl-v0 must not be both boundary edges
    if (hconn_[v0v1].face_ != PMP_MAX_INDEX) {
        vl = to_vertex(next_halfedge(v0v1));
        h1 = next_halfedge(v0v1);
        h2 = next_halfedge(h1);
        if (hconn_[h1^1].face_ == PMP_MAX_INDEX &&
            hconn_[h2^1].face_ == PMP_MAX_INDEX) {
            hcol01 = hcol10 = false;
            return;
        }
    }

    // the edges v0-vr and vr-v1 must not be both boundary edges
    if (hconn_[v1v0].face_ != PMP_MAX_INDEX) {
        vr = to_vertex(next_halfedge(v1v0));
        h1 = next_halfedge(v1v0);
        h2 = next_halfedge(h1);
        if (hconn_[h1^1].face_ == PMP_MAX_INDEX &&
            hconn_[h2^1].face_ == PMP_MAX_INDEX) {
            hcol01 = hcol10 = false;
            return;
        }
    }

    // if vl and vr are equal or both invalid -> fail
    if (vl == vr) {
        hcol01 = hcol10 = false;
        return;
    }

    // edge between two boundary vertices should be a boundary edge
    if (is_boundary_v(v0) && is_boundary_v(v1) && hconn_[v0v1].face_ != PMP_MAX_INDEX &&
        hconn_[v1v0].face_ != PMP_MAX_INDEX) {
        hcol01 = hcol10 = false;
        return;
    }

    // test intersection of the one-rings of v0 and v1
    for (int vv : vertices(v0)) {
        if (vv != v1 && vv != vl && vv != vr)
            if (find_halfedge(vv, v1) != PMP_MAX_INDEX) {
                hcol01 = hcol10 = false;
                return;
            }
    }

    if (relaxed)
        return;

    // check whether there are triangles flipped after collapsing
    auto& pos = prim_->attr<vec3f>("pos");
    vec3f pos0 = pos[v0], pos1 = pos[v1];

    int hv0 = halfedge(v0);
    const int hhv0 = hv0;
    if (hv0 != PMP_MAX_INDEX) {
        do {
            if (hconn_[hv0].face_ == PMP_MAX_INDEX) {
                break;
            }

            int v01 = to_vertex(hv0);
            int v02 = to_vertex(next_halfedge(hv0));
            vec3f nv0 = normalize(cross(pos[v01] - pos0, pos[v02] - pos0));
            vec3f nv1 = normalize(cross(pos[v01] - pos1, pos[v02] - pos1));
            if(acos(clamp(dot(nv0, nv1), -1, 1)) > angle_thrd) hcol01 = false;

            hv0 = next_halfedge(hv0 ^ 1);
        } while (hv0 != hhv0);
    }

    int hv1 = halfedge(v1);
    const int hhv1 = hv1;
    if (hv1 != PMP_MAX_INDEX) {
        do {
            if (hconn_[hv1].face_ == PMP_MAX_INDEX) {
                break;
            }

            int v11 = to_vertex(hv1);
            int v12 = to_vertex(next_halfedge(hv1));
            vec3f nv1 = normalize(cross(pos[v11] - pos1, pos[v12] - pos1));
            vec3f nv0 = normalize(cross(pos[v11] - pos0, pos[v12] - pos0));
            if(acos(clamp(dot(nv1, nv0), -1, 1)) > angle_thrd) hcol10 = false;

            hv1 = next_halfedge(hv1 ^ 1);
        } while (hv1 != hhv1);
    }

    return;
}

void SurfaceMesh::collapse(int h) {
    int h0 = h;
    int h1 = prev_halfedge(h0);
    int o0 = h0^1;
    int o1 = next_halfedge(o0);

    // remove edge
    remove_edge_helper(h0);

    // remove loops
    if (next_halfedge(next_halfedge(h1)) == h1)
        remove_loop_helper(h1);
    if (next_halfedge(next_halfedge(o1)) == o1)
        remove_loop_helper(o1);
}

void SurfaceMesh::remove_edge_helper(int h) {
    auto& vdeleted = prim_->verts.attr<int>("v_deleted");
    auto& edeleted = prim_->lines.attr<int>("e_deleted");
    
    int hn = next_halfedge(h);
    int hp = prev_halfedge(h);

    int o = h ^ 1;
    int on = next_halfedge(o);
    int op = prev_halfedge(o);

    int fh = hconn_[h].face_;
    int fo = hconn_[o].face_;

    int vh = to_vertex(h);
    int vo = to_vertex(o);

    // halfedge -> vertex
    for (const auto hc : halfedges(vo)) {
        hconn_[hc^1].vertex_ = vh;

        if (prim_->lines[hc>>1][0] == vo) {
            prim_->lines[hc>>1][0] = vh;
        } else {
            prim_->lines[hc>>1][1] = vh;
        }
        
        int fit = hconn_[hc].face_;
        if (fit != PMP_MAX_INDEX) {
            for (int i = 0; i < 3; ++i) {
                if (prim_->tris[fit][i] == vo) {
                    prim_->tris[fit][i] = vh;
                    break;
                }
            }
        }
        fit = hconn_[hc^1].face_;
        if (fit != PMP_MAX_INDEX) {
            for (int i = 0; i < 3; ++i) {
                if (prim_->tris[fit][i] == vo) {
                    prim_->tris[fit][i] = vh;
                    break;
                }
            }
        }
    }

    // halfedge -> halfedge
    set_next_halfedge(hp, hn);
    set_next_halfedge(op, on);

    // face -> halfedge
    if (fh != PMP_MAX_INDEX) {
        fconn_[fh].halfedge_ = hn;
    }
    if (fo != PMP_MAX_INDEX) {
        fconn_[fo].halfedge_ = on;
    }

    // vertex -> halfedge
    if (halfedge(vh) == o)
        vconn_[vh].halfedge_ = hn;
    adjust_outgoing_halfedge(vh);
    vconn_[vo].halfedge_ = PMP_MAX_INDEX;

    // delete stuff
    vdeleted[vo] = 1;
    ++deleted_vertices_;
    edeleted[h>>1] = 1;
    ++deleted_lines_;
    has_garbage_ = true;
}

void SurfaceMesh::remove_loop_helper(int h) {
    auto& edeleted = prim_->lines.attr<int>("e_deleted");
    auto& fdeleted = prim_->tris.attr<int>("f_deleted");

    int h0 = h;
    int h1 = next_halfedge(h0);

    int o0 = h0 ^ 1;
    int o1 = h1 ^ 1;

    int v0 = to_vertex(h0);
    int v1 = to_vertex(h1);

    int fh = hconn_[h0].face_;
    int fo = hconn_[o0].face_;

    // is it a loop ?
    assert((next_halfedge(h1) == h0) && (h1 != o0));

    // halfedge -> halfedge
    set_next_halfedge(h1, next_halfedge(o0));
    set_next_halfedge(prev_halfedge(o0), h1);

    // halfedge -> face
    hconn_[h1].face_ = fo;

    // vertex -> halfedge
    vconn_[v0].halfedge_ = h1;
    adjust_outgoing_halfedge(v0);
    vconn_[v1].halfedge_ = o1;
    adjust_outgoing_halfedge(v1);

    // face -> halfedge
    if (fo != PMP_MAX_INDEX && fconn_[fo].halfedge_ == o0)
        fconn_[fo].halfedge_ = h1;

    // delete stuff
    if (fh != PMP_MAX_INDEX) {
        fdeleted[fh] = 1;
        ++deleted_faces_;
    }
    edeleted[h>>1] = 1;
    ++deleted_lines_;
    has_garbage_ = true;
}

void SurfaceMesh::garbage_collection() {
    int i, i0, i1, nV(vertices_size_), nE(lines_size_), nH(halfedges_size_),
        nF(faces_size_);

    auto& pos = prim_->attr<vec3f>("pos");
    auto& lines = prim_->lines;
    auto& tris = prim_->tris;

    auto& vdeleted = prim_->verts.attr<int>("v_deleted");
    auto& edeleted = prim_->lines.attr<int>("e_deleted");
    auto& fdeleted = prim_->tris.attr<int>("f_deleted");
    auto& vduplicate = prim_->verts.attr<int>("v_duplicate");

    // setup handle mapping
    auto& vmap = prim_->verts.add_attr<int>("v_garbage_collection");
    auto& fmap = prim_->tris.add_attr<int>("f_garbage_collection");
    auto hmap = std::vector<int>();
    hmap.resize(nH);

    for (i = 0; i < nV; ++i)
        vmap[i] = i;
    for (i = 0; i < nH; ++i)
        hmap[i] = i;
    for (i = 0; i < nF; ++i)
        fmap[i] = i;

    // remove deleted vertices
    if (nV > 0) {
        i0 = 0;
        i1 = nV - 1;

        while (1) {
            // find first deleted and last un-deleted
            while (i0 < i1 && vdeleted[i0] == 0)
                ++i0;
            while (i0 < i1 && vdeleted[i1] == 1)
                --i1;
            if (i0 >= i1)
                break;

            prim_->verts.foreach_attr<zeno::AttrAcceptAll>([&] (auto const &key, auto &arr) {
                std::swap(arr[i0], arr[i1]);
            });
            std::swap(pos[i0], pos[i1]);
            std::swap(vconn_[i0], vconn_[i1]);
        };

        // remember new size
        nV = (vdeleted[i0] == 1) ? i0 : i0 + 1;
    }

    // remove deleted edges
    if (nE > 0) {
        i0 = 0;
        i1 = nE - 1;

        while (1) {
            // find first deleted and last un-deleted
            while (i0 < i1 && edeleted[i0] == 0)
                ++i0;
            while (i0 < i1 && edeleted[i1] == 1)
                --i1;
            if (i0 >= i1)
                break;

            prim_->lines.foreach_attr<zeno::AttrAcceptAll>([&] (auto const &key, auto &arr) {
                std::swap(arr[i0], arr[i1]);
            });
            std::swap(lines[i0], lines[i1]);

            // swap hconn_
            std::swap(hmap[i0<<1], hmap[i1<<1]);
            std::swap(hconn_[i0<<1], hconn_[i1<<1]);
            
            std::swap(hmap[i0<<1|1], hmap[i1<<1|1]);
            std::swap(hconn_[i0<<1|1], hconn_[i1<<1|1]);
        };

        // remember new size
        nE = (edeleted[i0] == 1) ? i0 : i0 + 1;
        nH = 2 * nE;
    }

    // remove deleted faces
    if (nF > 0) {
        i0 = 0;
        i1 = nF - 1;

        while (1) {
            // find 1st deleted and last un-deleted
            while (i0 < i1 && fdeleted[i0] == 0)
                ++i0;
            while (i0 < i1 && fdeleted[i1] == 1)
                --i1;
            if (i0 >= i1)
                break;

            prim_->tris.foreach_attr<zeno::AttrAcceptAll>([&] (auto const &key, auto &arr) {
                std::swap(arr[i0], arr[i1]);
            });
            std::swap(tris[i0], tris[i1]);
            std::swap(fconn_[i0], fconn_[i1]);
        };

        // remember new size
        nF = (fdeleted[i0] == 1) ? i0 : i0 + 1;
    }

    // update vertex connectivity
    for (int v = 0; v < nV; ++v) {
        vduplicate[v] = vmap[vduplicate[v]];
        if (!is_isolated(v)) {
            vconn_[v].halfedge_ = hmap[halfedge(v)];
        }
    }

    // update halfedge connectivity
    for (int h = 0; h < nH; ++h) {
        hconn_[h].vertex_ = vmap[to_vertex(h)];
        set_next_halfedge(h, hmap[next_halfedge(h)]);
        if (hconn_[h].face_ != PMP_MAX_INDEX)
            hconn_[h].face_ = fmap[hconn_[h].face_];
    }

    // update handles of faces
    for (int f = 0; f < nF; ++f) {
        fconn_[f].halfedge_ = hmap[fconn_[f].halfedge_];
    }

    // update prim
    for (int e = 0; e < nE; ++e) {
        vec2i old = prim_->lines[e];
        prim_->lines[e] = vec2i(vmap[old[0]], vmap[old[1]]);
    }
    for (int f = 0; f < nF; ++f) {
        vec3i old = prim_->tris[f];
        prim_->tris[f] = vec3i(vmap[old[0]], vmap[old[1]], vmap[old[2]]);
    }

    // remove handle maps
    prim_->verts.erase_attr("v_garbage_collection");
    prim_->tris.erase_attr("f_garbage_collection");

    // finally resize arrays
    prim_->verts.foreach_attr<zeno::AttrAcceptAll>([&] (auto const &key, auto &arr) {
        arr.resize(nV);
    });
    pos.resize(nV);
    vconn_.resize(nV);
    vertices_size_ = nV;
    
    hconn_.resize(nH);
    halfedges_size_ = nH;

    prim_->lines.foreach_attr<zeno::AttrAcceptAll>([&] (auto const &key, auto &arr) {
        arr.resize(nE);
    });
    lines.resize(nE);
    lines_size_ = nE;
    
    prim_->tris.foreach_attr<zeno::AttrAcceptAll>([&] (auto const &key, auto &arr) {
        arr.resize(nF);
    });
    tris.resize(nF);
    fconn_.resize(nF);
    faces_size_ = nF;

    deleted_vertices_ = deleted_lines_ = deleted_faces_ = 0;
    has_garbage_ = false;
}

} // namespace pmp
} // namespace zeno