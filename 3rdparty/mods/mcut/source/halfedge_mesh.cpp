/**
 * Copyright (c) 2020-2021 CutDigital Ltd.
 * All rights reserved.
 * 
 * NOTE: This file is licensed under GPL-3.0-or-later (default). 
 * A commercial license can be purchased from CutDigital Ltd. 
 *  
 * License details:
 * 
 * (A)  GNU General Public License ("GPL"); a copy of which you should have 
 *      recieved with this file.
 * 	    - see also: <http://www.gnu.org/licenses/>
 * (B)  Commercial license.
 *      - email: contact@cut-digital.com
 * 
 * The commercial license options is for users that wish to use MCUT in 
 * their products for comercial purposes but do not wish to release their 
 * software products under the GPL license. 
 * 
 * Author(s)     : Floyd M. Chitalu
 */

#include "mcut/internal/halfedge_mesh.h"

#include <algorithm>
#include <cstdio>

#define ENABLE_EDGE_DESCRIPTOR_TRICK 1

namespace mcut
{
    //
    // array_iterator_t
    //
    template <>
    vertex_array_iterator_t vertex_array_iterator_t::cbegin(bool account_for_removed_elems, id_<vertex_array_iterator_t>)
    {
        return mesh_ptr->vertices_begin(account_for_removed_elems);
    }
    
    template <>
    vertex_array_iterator_t vertex_array_iterator_t::cend(id_<vertex_array_iterator_t>)
    {
        return mesh_ptr->vertices_end();
    }

    template <>
    edge_array_iterator_t edge_array_iterator_t::cbegin(bool account_for_removed_elems, id_<edge_array_iterator_t>)
    {
        return mesh_ptr->edges_begin(account_for_removed_elems);
    }

    template <>
    edge_array_iterator_t edge_array_iterator_t::cend(id_<edge_array_iterator_t>)
    {
        return mesh_ptr->edges_end();
    }

    template <>
    halfedge_array_iterator_t halfedge_array_iterator_t::cbegin(bool account_for_removed_elems, id_<halfedge_array_iterator_t>)
    {
        return mesh_ptr->halfedges_begin(account_for_removed_elems);
    }

    template <>
    halfedge_array_iterator_t halfedge_array_iterator_t::cend(id_<halfedge_array_iterator_t>)
    {
        return mesh_ptr->halfedges_end();
    }

    template <>
    face_array_iterator_t face_array_iterator_t::cbegin(bool account_for_removed_elems, id_<face_array_iterator_t>)
    {
        return mesh_ptr->faces_begin(account_for_removed_elems);
    }
    template <>
    face_array_iterator_t face_array_iterator_t::cend(id_<face_array_iterator_t>)
    {
        return mesh_ptr->faces_end();
    }

    //
    // mesh_t
    //

    mesh_t::mesh_t()
    {
    }
    mesh_t::~mesh_t() {}

    // static member functions
    // -----------------------

    vertex_descriptor_t mesh_t::null_vertex()
    {
        return vertex_descriptor_t();
    }

    halfedge_descriptor_t mesh_t::null_halfedge()
    {
        return halfedge_descriptor_t();
    }

    edge_descriptor_t mesh_t::null_edge()
    {
        return edge_descriptor_t();
    }

    face_descriptor_t mesh_t::null_face()
    {
        return face_descriptor_t();
    }

    // regular member functions
    // ------------------------

    int mesh_t::number_of_vertices() const
    {
        return number_of_internal_vertices() - number_of_vertices_removed();
    }

    int mesh_t::number_of_edges() const
    {
        return number_of_internal_edges() - number_of_edges_removed();
    }

    int mesh_t::number_of_halfedges() const
    {
        return number_of_internal_halfedges() - number_of_halfedges_removed();
    }

    int mesh_t::number_of_faces() const
    {
        return number_of_internal_faces() - number_of_faces_removed();
    }

    vertex_descriptor_t mesh_t::source(const halfedge_descriptor_t &h) const
    {
        MCUT_ASSERT((size_t)h < m_halfedges.size() /*h != null_halfedge()*/);
        const halfedge_data_t &hd = m_halfedges.at(h);
        MCUT_ASSERT((size_t)hd.o < m_halfedges.size() /*hd.o != null_halfedge()*/);
        const halfedge_data_t &ohd = m_halfedges.at(hd.o); // opposite
        return ohd.t;
    }

    vertex_descriptor_t mesh_t::target(const halfedge_descriptor_t &h) const
    {
        MCUT_ASSERT(h != null_halfedge());
        MCUT_ASSERT((size_t)h < m_halfedges.size() /*m_halfedges.count(h) == 1*/);
        const halfedge_data_t &hd = m_halfedges.at(h);
        return hd.t;
    }

    halfedge_descriptor_t mesh_t::opposite(const halfedge_descriptor_t &h) const
    {
        MCUT_ASSERT(h != null_halfedge());
        MCUT_ASSERT((size_t)h < m_halfedges.size() /*m_halfedges.count(h) == 1*/);
#if ENABLE_EDGE_DESCRIPTOR_TRICK
        return halfedge_descriptor_t((h % 2 == 0) ? h + 1 : h - 1);
#else
        const halfedge_data_t &hd = m_halfedges.at(h);
        return hd.o;
#endif
    }

    halfedge_descriptor_t mesh_t::prev(const halfedge_descriptor_t &h) const
    {
        MCUT_ASSERT(h != null_halfedge());
        MCUT_ASSERT((size_t)h < m_halfedges.size() /*m_halfedges.count(h) == 1*/);
        const halfedge_data_t &hd = m_halfedges.at(h);
        return hd.p;
    }

    halfedge_descriptor_t mesh_t::next(const halfedge_descriptor_t &h) const
    {
        MCUT_ASSERT(h != null_halfedge());
        MCUT_ASSERT((size_t)h < m_halfedges.size() /*m_halfedges.count(h) == 1*/);
        const halfedge_data_t &hd = m_halfedges.at(h);
        return hd.n;
    }

    void mesh_t::set_next(const halfedge_descriptor_t &h, const halfedge_descriptor_t &nxt)
    {
        MCUT_ASSERT(h != null_halfedge());
        MCUT_ASSERT(nxt != null_halfedge());
        MCUT_ASSERT((size_t)h < m_halfedges.size() /*m_halfedges.count(h) == 1*/);
        halfedge_data_t &hd = m_halfedges.at(h);
        hd.n = nxt;
        set_previous(nxt, h);
    }

    void mesh_t::set_previous(const halfedge_descriptor_t &h, const halfedge_descriptor_t &prev)
    {
        MCUT_ASSERT(h != null_halfedge());
        MCUT_ASSERT(prev != null_halfedge());
        MCUT_ASSERT((size_t)h < m_halfedges.size() /*m_halfedges.count(h) == 1*/);
        halfedge_data_t &hd = m_halfedges.at(h);
        hd.p = prev;
    }

    edge_descriptor_t mesh_t::edge(const halfedge_descriptor_t &h) const
    {
        MCUT_ASSERT(h != null_halfedge());
        MCUT_ASSERT((size_t)h < m_halfedges.size() /*m_halfedges.count(h) == 1*/);
#if ENABLE_EDGE_DESCRIPTOR_TRICK
        return edge_descriptor_t(h / 2);
#else
        const halfedge_data_t &hd = m_halfedges.at(h);
        return hd.e;
#endif
    }

    face_descriptor_t mesh_t::face(const halfedge_descriptor_t &h) const
    {
        MCUT_ASSERT(h != null_halfedge());
        MCUT_ASSERT((size_t)h < m_halfedges.size() /*m_halfedges.count(h) == 1*/);
        const halfedge_data_t &hd = m_halfedges.at(h);
        return hd.f;
    }

    vertex_descriptor_t mesh_t::vertex(const edge_descriptor_t e, const int v) const
    {
        MCUT_ASSERT(e != null_edge());
        MCUT_ASSERT(v == 0 || v == 1);
        MCUT_ASSERT((size_t)e < m_edges.size() /*m_edges.count(e) == 1*/);
#if ENABLE_EDGE_DESCRIPTOR_TRICK
        return target(halfedge_descriptor_t((e * 2) + v));
#else
        const edge_data_t &ed = m_edges.at(e);
        const halfedge_descriptor_t h = ed.h;
        MCUT_ASSERT((size_t)h < m_halfedges.size() /*m_halfedges.count(h) == 1*/);
        const halfedge_data_t &hd = m_halfedges.at(h);
        vertex_descriptor_t v_out = hd.t; // assuming v ==0

        if (v == 1)
        {
            const halfedge_descriptor_t opp = hd.o;
            MCUT_ASSERT((size_t)opp < m_halfedges.size() /*m_halfedges.count(opp) == 1*/);
            const halfedge_data_t &ohd = m_halfedges.at(opp);
            v_out = ohd.t;
        }

        return v_out;
#endif
    }

    bool mesh_t::is_border(const halfedge_descriptor_t h)
    {
        MCUT_ASSERT(h != null_halfedge());
        return face(h) == null_face();
    }

    bool mesh_t::is_border(const edge_descriptor_t e)
    {
        MCUT_ASSERT(e != null_edge());
        halfedge_descriptor_t h0 = halfedge(e, 0);
        MCUT_ASSERT(h0 != null_halfedge());
        halfedge_descriptor_t h1 = halfedge(e, 1);
        MCUT_ASSERT(h1 != null_halfedge());

        return is_border(h0) || is_border(h1);
    }

    halfedge_descriptor_t mesh_t::halfedge(const edge_descriptor_t e, const int i) const
    {
        MCUT_ASSERT(i == 0 || i == 1);
        MCUT_ASSERT(e != null_edge());
        MCUT_ASSERT((size_t)e < m_edges.size() /*m_edges.count(e) == 1*/);
#if ENABLE_EDGE_DESCRIPTOR_TRICK
        return halfedge_descriptor_t(e * 2 + i);
#else
        const edge_data_t &ed = m_edges.at(e);
        halfedge_descriptor_t h = ed.h; // assuming i ==0

        MCUT_ASSERT(h != null_halfedge());

        if (i == 1)
        {
            MCUT_ASSERT((size_t)h < m_halfedges.size() /*m_halfedges.count(h) == 1*/);

            const halfedge_data_t &hd = m_halfedges.at(h);
            h = hd.o;

            MCUT_ASSERT(h != null_halfedge());
        }

        return h;
#endif
    }

    halfedge_descriptor_t mesh_t::halfedge(const vertex_descriptor_t s, const vertex_descriptor_t t, bool strict_check) const
    {
        MCUT_ASSERT((size_t)s < m_vertices.size()); // MCUT_ASSERT(m_vertices.count(s) == 1);
        const vertex_data_t &svd = m_vertices.at(s);
        const std::vector<halfedge_descriptor_t> &s_halfedges = svd.m_halfedges;
        MCUT_ASSERT((size_t)t < m_vertices.size()); // MCUT_ASSERT(m_vertices.count(t) == 1);
        const vertex_data_t &tvd = m_vertices.at(t);
        const std::vector<halfedge_descriptor_t> &t_halfedges = tvd.m_halfedges;
        std::vector<edge_descriptor_t> t_edges;
        t_edges.reserve(t_halfedges.size());

        for (std::vector<halfedge_descriptor_t>::const_iterator i = t_halfedges.cbegin(); i != t_halfedges.cend(); i++)
        {
            edge_descriptor_t e = edge(*i);
            MCUT_ASSERT(e != null_edge());
            t_edges.push_back(e);
        }

        halfedge_descriptor_t result = null_halfedge();
        for (std::vector<halfedge_descriptor_t>::const_iterator i = s_halfedges.cbegin(); i != s_halfedges.cend(); ++i)
        {
            edge_descriptor_t s_edge = edge(*i);
            if (std::find(t_edges.cbegin(), t_edges.cend(), s_edge) != t_edges.cend()) // belong to same edge?
            {
                result = *i; // assume source(*i) and target(*i) match "s" and "t"

                // check if we need to return the opposite halfedge
                if ((source(*i) == s && target(*i) == t) == false)
                {
                    MCUT_ASSERT(source(*i) == t);
                    MCUT_ASSERT(target(*i) == s);

                    halfedge_descriptor_t h = opposite(*i);

                    if (face(h) != null_face() || strict_check)
                    { // "strict_check" ensures that we return the halfedge matching the input vertices
                        result = h;
                        break;
                    }
                }
            }
        }
        return result;
    }

    edge_descriptor_t mesh_t::edge(const vertex_descriptor_t s, const vertex_descriptor_t t, bool strict_check) const
    {
        halfedge_descriptor_t h = halfedge(s, t, strict_check);
        return (h == mesh_t::null_halfedge() ? mesh_t::null_edge() : edge(h));
    }

    vertex_descriptor_t mesh_t::add_vertex(const math::vec3 &point)
    {
        const math::real_number_t x = point.x();
        const math::real_number_t y = point.y();
        const math::real_number_t z = point.z();
        return add_vertex(x, y, z);
    }

#if defined(MCUT_WITH_ARBITRARY_PRECISION_NUMBERS)
    vertex_descriptor_t mesh_t::add_vertex(const math::fast_vec3 &point)
    {
        return add_vertex(point.x(), point.y(), point.z());
    }
#endif // #if defined(MCUT_WITH_ARBITRARY_PRECISION_NUMBERS)

    vertex_descriptor_t mesh_t::add_vertex(const math::real_number_t &x, const math::real_number_t &y, const math::real_number_t &z)
    {
        vertex_descriptor_t vd = mesh_t::null_vertex();
        vertex_data_t *data_ptr = nullptr;
        bool reusing_removed_descr = (!m_vertices_removed.empty());

        if (reusing_removed_descr) // can we re-use a slot?
        {
            std::vector<vertex_descriptor_t>::iterator it = m_vertices_removed.begin(); // take the oldest unused slot (NOTE: important for user data mapping)
            vd = *it;
            m_vertices_removed.erase(it);
            MCUT_ASSERT((size_t)vd < m_vertices.size()); // MCUT_ASSERT(m_vertices.find(vd) != m_vertices.cend());
            data_ptr = &m_vertices.at(vd);
        }
        else
        {
            vd = static_cast<vertex_descriptor_t>(number_of_vertices());
            //std::pair<typename std::map<vertex_descriptor_t, vertex_data_t>::iterator, bool> ret = m_vertices.insert(std::make_pair(vd, vertex_data_t()));
            //MCUT_ASSERT(ret.second == true);
            m_vertices.push_back(vertex_data_t());
            data_ptr = &m_vertices.back(); // &ret.first->second;
        }

        MCUT_ASSERT(vd != mesh_t::null_vertex());

        data_ptr->p = math::vec3(x, y, z);

        return vd;
    }

    halfedge_descriptor_t mesh_t::add_edge(const vertex_descriptor_t v0, const vertex_descriptor_t v1)
    {
        MCUT_ASSERT(v0 != null_vertex());
        MCUT_ASSERT(v1 != null_vertex());

        // primary halfedge(0) of edge
        halfedge_descriptor_t h0_idx(static_cast<face_descriptor_t::index_type>(number_of_halfedges())); // primary halfedge of new edge to be created
        bool reusing_removed_h0_descr = (!m_halfedges_removed.empty());

        if (reusing_removed_h0_descr) // can we re-use a slot?
        {
            std::vector<halfedge_descriptor_t>::iterator hIter = m_halfedges_removed.begin(); // take the oldest unused slot (NOTE: important for user data mapping)
            h0_idx = *hIter;
            m_halfedges_removed.erase(hIter);
            MCUT_ASSERT((size_t)h0_idx < m_halfedges.size() /*m_halfedges.find(h0_idx) != m_halfedges.cend()*/);
        }

        halfedge_data_t *halfedge0_data_ptr = nullptr;
        if (reusing_removed_h0_descr)
        {
            //halfedge0_data_ptr = &m_halfedges.at(h0_idx);
        }
        else
        {
            // create new halfedge --> h0
            //std::pair<typename std::map<halfedge_descriptor_t, halfedge_data_t>::iterator, bool> h0_ret = m_halfedges.insert(std::make_pair(h0_idx, halfedge_data_t()));
            //MCUT_ASSERT(h0_ret.second == true);
            //halfedge0_data_ptr = &h0_ret.first->second;
            m_halfedges.emplace_back(halfedge_data_t());
        }

        // second halfedge(1) of edge
        halfedge_descriptor_t h1_idx(static_cast<face_descriptor_t::index_type>(number_of_halfedges())); // second halfedge of new edge to be created (opposite of h0_idx)
        bool reusing_removed_h1_descr = (!m_halfedges_removed.empty());

        if (reusing_removed_h1_descr) // can we re-use a slot?
        {
            std::vector<halfedge_descriptor_t>::iterator hIter = m_halfedges_removed.begin() /*+ (m_halfedges_removed.size() - 1)*/; // take the most recently removed
            h1_idx = *hIter;
            m_halfedges_removed.erase(hIter);
            MCUT_ASSERT((size_t)h1_idx < m_halfedges.size() /*m_halfedges.find(h1_idx) != m_halfedges.cend()*/);
        }

        halfedge_data_t *halfedge1_data_ptr = nullptr;
        if (reusing_removed_h1_descr)
        {
            //halfedge1_data_ptr = &m_halfedges.at(h1_idx);
        }
        else
        {
            // create new halfedge --> h1
            //std::pair<typename std::map<halfedge_descriptor_t, halfedge_data_t>::iterator, bool> h1_ret = m_halfedges.insert(std::make_pair(h1_idx, halfedge_data_t()));
            //MCUT_ASSERT(h1_ret.second == true);
            //halfedge1_data_ptr = &h1_ret.first->second;
            m_halfedges.emplace_back(halfedge_data_t());
        }

        // https://stackoverflow.com/questions/34708189/c-vector-of-pointer-loses-the-reference-after-push-back
        halfedge0_data_ptr = &m_halfedges[h0_idx];
        halfedge1_data_ptr = &m_halfedges[h1_idx];

        // edge

        edge_descriptor_t e_idx(static_cast<face_descriptor_t::index_type>(number_of_edges())); // index of new edge
        bool reusing_removed_edge_descr = (!m_edges_removed.empty());

        if (reusing_removed_edge_descr) // can we re-use a slot?
        {
            std::vector<edge_descriptor_t>::iterator eIter = m_edges_removed.begin(); // take the oldest unused slot (NOTE: important for user data mapping)
            e_idx = *eIter;
            m_edges_removed.erase(eIter);
            MCUT_ASSERT((size_t)e_idx < m_edges.size() /*m_edges.find(e_idx) != m_edges.cend()*/);
        }

        edge_data_t *edge_data_ptr = nullptr;
        if (reusing_removed_edge_descr)
        {
            edge_data_ptr = &m_edges.at(e_idx);
        }
        else
        {
            //std::pair<typename std::map<edge_descriptor_t, edge_data_t>::iterator, bool> eret = m_edges.insert(std::make_pair(e_idx, edge_data_t())); // create a new edge
            //MCUT_ASSERT(eret.second == true);
            //edge_data_ptr = &eret.first->second;
            m_edges.emplace_back(edge_data_t());
            edge_data_ptr = &m_edges.back();
        }

        // update incidence information

        //edge_data_t& edge_data = eret.first->second;
        edge_data_ptr->h = h0_idx; // even/primary halfedge
        //eret.first->h = h0_idx; // even/primary halfedge

        //halfedge_data_t& halfedge0_data = h0_ret.first->second;
        halfedge0_data_ptr->t = v1;     // target vertex of h0
        halfedge0_data_ptr->o = h1_idx; // ... because opp has idx differing by 1
        halfedge0_data_ptr->e = e_idx;

        //halfedge_data_t& halfedge1_data = h1_ret.first->second;
        halfedge1_data_ptr->t = v0;     // target vertex of h1
        halfedge1_data_ptr->o = h0_idx; // ... because opp has idx differing by 1
        halfedge1_data_ptr->e = e_idx;

        //MCUT_ASSERT(h1_ret.first->first == halfedge0_data_ptr->o); // h1 comes just afterward (its index)

        // update vertex incidence

        // v0
        MCUT_ASSERT((size_t)v0 < m_vertices.size()); //MCUT_ASSERT(m_vertices.count(v0) == 1);
        vertex_data_t &v0_data = m_vertices.at(v0);
        //MCUT_ASSERT();
        if (std::find(v0_data.m_halfedges.cbegin(), v0_data.m_halfedges.cend(), h1_idx) == v0_data.m_halfedges.cend())
        {
            v0_data.m_halfedges.push_back(h1_idx); // halfedge whose target is v0
        }
        // v1
        MCUT_ASSERT((size_t)v1 < m_vertices.size()); //MCUT_ASSERT(m_vertices.count(v1) == 1);
        vertex_data_t &v1_data = m_vertices.at(v1);
        //MCUT_ASSERT();
        if (std::find(v1_data.m_halfedges.cbegin(), v1_data.m_halfedges.cend(), h0_idx) == v1_data.m_halfedges.cend())
        {
            v1_data.m_halfedges.push_back(h0_idx); // halfedge whose target is v1
        }

        return static_cast<halfedge_descriptor_t>(h0_idx); // return halfedge whose target is v1
    }

    face_descriptor_t mesh_t::add_face(const std::vector<vertex_descriptor_t> &vi)
    {
        const int face_vertex_count = static_cast<int>(vi.size());
        MCUT_ASSERT(face_vertex_count >= 3);

        const int face_count = number_of_faces();
        face_data_t new_face_data;
        face_descriptor_t new_face_idx(static_cast<face_descriptor_t::index_type>(face_count));
        bool reusing_removed_face_descr = (!m_faces_removed.empty());

        if (reusing_removed_face_descr) // can we re-use a slot?
        {
            std::vector<face_descriptor_t>::iterator fIter = m_faces_removed.begin(); // take the oldest unused slot (NOTE: important for user data mapping)
            new_face_idx = *fIter;
            m_faces_removed.erase(fIter); // slot is going to be used again

            MCUT_ASSERT((size_t)new_face_idx < m_faces.size() /*m_faces.find(new_face_idx) != m_faces.cend()*/);
        }

        face_data_t *face_data_ptr = reusing_removed_face_descr ? &m_faces.at(new_face_idx) : &new_face_data;
        face_data_ptr->m_halfedges.clear();

        for (int i = 0; i < face_vertex_count; ++i)
        {
            const vertex_descriptor_t v0 = vi.at(i); // i.e. src

            MCUT_ASSERT(v0 != null_vertex());

            const vertex_descriptor_t v1 = vi.at((i + 1) % face_vertex_count); // i.e. tgt

            MCUT_ASSERT(v1 != null_vertex());

            // check if edge exists between v0 and v1 (using halfedges incident to either v0 or v1)
            // TODO: use the halfedge(..., true) function
            //vertex_data_t& v0_data = m_vertices.at(v0);
            //vertex_data_t& v1_data = m_vertices.at(v1);

#if 0
        vertex_data_t& v0_data = m_vertices.at(v0);
        vertex_data_t& v1_data = m_vertices.at(v1);

        bool connecting_edge_exists = false;
        halfedge_descriptor_t v0_h = null_halfedge();
        halfedge_descriptor_t v1_h = null_halfedge();

        for (int v0_h_iter = 0; v0_h_iter < static_cast<int>(v0_data.m_halfedges.size()); ++v0_h_iter) {
            v0_h = v0_data.m_halfedges.at(v0_h_iter);
            const edge_descriptor_t v0_e = edge(v0_h);

            for (int v1_h_iter = 0; v1_h_iter < static_cast<int>(v1_data.m_halfedges.size()); ++v1_h_iter) {

                v1_h = v1_data.m_halfedges.at(v1_h_iter);
                const edge_descriptor_t v1_e = edge(v1_h);
                const bool same_edge = (v0_e == v1_e);

                if (same_edge) {
                    connecting_edge_exists = true;
                    break;
                }
            }

            if (connecting_edge_exists) {
                break;
            }
        }
#endif
            halfedge_descriptor_t v1_h = halfedge(v0, v1, true);
            bool connecting_edge_exists = v1_h != null_halfedge();

            //halfedge_descriptor_t v1_h = null_halfedge();

            // we use v1 in the following since v1 is the target (vertices are associated with halfedges which point to them)

            halfedge_data_t *v1_hd_ptr = nullptr; // refer to halfedge whose tgt is v1

            if (connecting_edge_exists) // edge connecting v0 and v1
            {
                MCUT_ASSERT((size_t)v1_h < m_halfedges.size() /*m_halfedges.count(v1_h) == 1*/);
                v1_hd_ptr = &m_halfedges.at(v1_h);
            }
            else
            { // there exists no edge between v0 and v1, so we create it
                v1_h = add_edge(v0, v1);
                MCUT_ASSERT((size_t)v1_h < m_halfedges.size() /*m_halfedges.count(h) == 1*/);
                v1_hd_ptr = &m_halfedges.at(v1_h); // add to vertex list since v1 is the target of h
            }

            if(v1_hd_ptr->f != null_face())
            {
                return null_face(); // face is incident to a non-manifold edge
            }

            v1_hd_ptr->f = new_face_idx; // associate halfedge with face
            face_data_ptr->m_halfedges.push_back(v1_h);
        }

        if (!reusing_removed_face_descr)
        {
            MCUT_ASSERT((size_t)new_face_idx == m_faces.size() /*m_faces.count(new_face_idx) == 0*/);
            //m_faces.insert(std::make_pair(new_face_idx, *face_data_ptr));
            m_faces.emplace_back(*face_data_ptr);
        }

        // update halfedges (next halfedge)
        //const std::vector<halfedge_descriptor_t>& halfedges_around_new_face = get_halfedges_around_face(new_face_idx);
        const int num_halfedges = static_cast<int>(face_data_ptr->m_halfedges.size());

        for (int i = 0; i < num_halfedges; ++i)
        {
            const halfedge_descriptor_t h = face_data_ptr->m_halfedges.at(i);
            const halfedge_descriptor_t nh = face_data_ptr->m_halfedges.at((i + 1) % num_halfedges);
            set_next(h, nh);
        }

        return new_face_idx;
    }

    const math::vec3 &mesh_t::vertex(const vertex_descriptor_t &vd) const
    {
        MCUT_ASSERT(vd != null_vertex());
        MCUT_ASSERT((size_t)vd < m_vertices.size());
        const vertex_data_t &vdata = m_vertices.at(vd);
        return vdata.p;
    }

    std::vector<vertex_descriptor_t> mesh_t::get_vertices_around_face(const face_descriptor_t f, uint32_t prepend_offset) const
    {
        MCUT_ASSERT(f != null_face());

        const std::vector<halfedge_descriptor_t> &halfedges_on_face = get_halfedges_around_face(f);
        std::vector<vertex_descriptor_t> vertex_descriptors(halfedges_on_face.size());
        for (int i = 0; i < (int)halfedges_on_face.size(); ++i)
        {
            const halfedge_descriptor_t h = halfedges_on_face[i];
            //MCUT_ASSERT((size_t)h < m_halfedges.size() /*m_halfedges.count(h) == 1*/);
            //const halfedge_data_t& hd = m_halfedges.at(h);
            vertex_descriptors[i] = vertex_descriptor_t(prepend_offset + target(h) /*hd.t*/);
        }
        return vertex_descriptors;
    }

    std::vector<vertex_descriptor_t> mesh_t::get_vertices_around_vertex(const vertex_descriptor_t v) const
    {
        MCUT_ASSERT(v != null_vertex());
        // halfedges whoe target is 'v'
        const std::vector<halfedge_descriptor_t> &halfedges = get_halfedges_around_vertex(v);
        std::vector<vertex_descriptor_t> out;
        for (std::vector<halfedge_descriptor_t>::const_iterator h = halfedges.cbegin(); h != halfedges.cend(); ++h)
        {
            vertex_descriptor_t src = source(*h);
            out.push_back(src);
        }
        return out;
    }

    const std::vector<halfedge_descriptor_t> &mesh_t::get_halfedges_around_face(const face_descriptor_t f) const
    {
        MCUT_ASSERT(f != null_face());
        MCUT_ASSERT((size_t)f < m_faces.size() /*m_faces.count(f) == 1*/);
        return m_faces.at(f).m_halfedges;
    }

    const std::vector<face_descriptor_t> mesh_t::get_faces_around_face(const face_descriptor_t f, const std::vector<halfedge_descriptor_t> *halfedges_around_face_) const
    {
        MCUT_ASSERT(f != null_face());

        std::vector<face_descriptor_t> faces_around_face;
        const std::vector<halfedge_descriptor_t> &halfedges_on_face = (halfedges_around_face_ != nullptr) ? *halfedges_around_face_ : get_halfedges_around_face(f);

        for (int i = 0; i < (int)halfedges_on_face.size(); ++i)
        {

            const halfedge_descriptor_t h = halfedges_on_face.at(i);
            MCUT_ASSERT((size_t)h < m_halfedges.size() /*m_halfedges.count(h) == 1*/);
            const halfedge_data_t &hd = m_halfedges.at(h);

            if (hd.o != null_halfedge())
            {
                MCUT_ASSERT((size_t)hd.o < m_halfedges.size() /*m_halfedges.count(hd.o) == 1*/);
                const halfedge_data_t &ohd = m_halfedges.at(hd.o);

                if (ohd.f != null_face())
                {
                    faces_around_face.push_back(ohd.f);
                }
            }
        }
        return faces_around_face;
    }

    const std::vector<halfedge_descriptor_t> &mesh_t::get_halfedges_around_vertex(const vertex_descriptor_t v) const
    {
        MCUT_ASSERT(v != mesh_t::null_vertex());
        MCUT_ASSERT((size_t)v < m_vertices.size());
        const vertex_data_t &vd = m_vertices.at(v);
        const std::vector<halfedge_descriptor_t> &incoming_halfedges = vd.m_halfedges;
        return incoming_halfedges;
    }

    vertex_array_iterator_t mesh_t::vertices_begin(bool account_for_removed_elems) const
    {
        vertex_array_t::const_iterator it = m_vertices.cbegin();
        if (account_for_removed_elems)
        {
            uint32_t index = 0;
            while (it != m_vertices.cend() && is_removed(vertex_descriptor_t(index++)) /*is_removed(it)*/)
            {
                ++it; // shift the pointer to the first valid mesh element
            }
        }
        return vertex_array_iterator_t(it, this);
    }

    vertex_array_iterator_t mesh_t::vertices_end() const
    {
        return vertex_array_iterator_t(m_vertices.cend(), this);
    }

    edge_array_iterator_t mesh_t::edges_begin(bool account_for_removed_elems) const
    {
        edge_array_t::const_iterator it = m_edges.cbegin();
        if (account_for_removed_elems)
        {
            uint32_t index = 0;
            while (it != m_edges.cend() && is_removed(edge_descriptor_t(index++) /*it->first*/))
            {
                ++it; // shift the pointer to the first valid mesh element
            }
        }
        return edge_array_iterator_t(it, this);
    }

    edge_array_iterator_t mesh_t::edges_end() const
    {
        return edge_array_iterator_t(m_edges.cend(), this);
    }

    halfedge_array_iterator_t mesh_t::halfedges_begin(bool account_for_removed_elems) const
    {
        halfedge_array_t::const_iterator it = m_halfedges.cbegin();
        if (account_for_removed_elems)
        {
            uint32_t index = 0;
            while (it != m_halfedges.cend() && is_removed(halfedge_descriptor_t(index++) /*it->first*/))
            {
                ++it; // shift the pointer to the first valid mesh element
            }
        }
        return halfedge_array_iterator_t(it, this);
    }

    halfedge_array_iterator_t mesh_t::halfedges_end() const
    {
        return halfedge_array_iterator_t(m_halfedges.cend(), this);
    }

    face_array_iterator_t mesh_t::faces_begin(bool account_for_removed_elems) const
    {
        face_array_t::const_iterator it = m_faces.cbegin();
        if (account_for_removed_elems)
        {
            uint32_t index = 0;
            while (it != m_faces.cend() && is_removed(face_descriptor_t(index++) /*it->first*/))
            {
                ++it; // shift the pointer to the first valid mesh element
            }
        }
        return face_array_iterator_t(it, this);
    }

    face_array_iterator_t mesh_t::faces_end() const
    {
        return face_array_iterator_t(m_faces.cend(), this);
    }

    // also disassociates (not remove) any halfedges(s) and vertices incident to face
    void mesh_t::remove_face(const face_descriptor_t f)
    {
        MCUT_ASSERT(f != null_face());
        MCUT_ASSERT(std::find(m_faces_removed.cbegin(), m_faces_removed.cend(), f) == m_faces_removed.cend());

        face_data_t &fd = m_faces.at(f);

        std::vector<vertex_descriptor_t> face_vertices; // ... that are used by face

        // disassociate halfedges

        for (std::vector<halfedge_descriptor_t>::const_iterator it = fd.m_halfedges.cbegin(); it != fd.m_halfedges.cend(); ++it)
        {
            halfedge_data_t &hd = m_halfedges.at(*it);
            MCUT_ASSERT(hd.f != null_face());
            hd.f = null_face();

            // NOTE: "next" and "previous" are only meaningful when the halfedge is used by a
            // face. So we reset that information here since the halfedge is not longer used
            // by a face
            if (hd.n != null_halfedge())
            { // disassociate "next"
                const halfedge_descriptor_t hn = hd.n;
                halfedge_data_t &hnd = m_halfedges.at(hn);
                MCUT_ASSERT(hnd.p == *it);
                hnd.p = null_halfedge();
                //
                hd.n = null_halfedge();
            }

            if (hd.p != null_halfedge())
            { // disassociate "previous"
                const halfedge_descriptor_t hp = hd.p;
                halfedge_data_t &hpd = m_halfedges.at(hp);
                MCUT_ASSERT(hpd.n == *it);
                hpd.n = null_halfedge();
                //
                hd.p = null_halfedge();
            }

            face_vertices.push_back(hd.t);
        }

        // disassociate vertices

        // for each vertex used by face
        for (std::vector<vertex_descriptor_t>::const_iterator it = face_vertices.cbegin(); it != face_vertices.cend(); ++it)
        {
            vertex_descriptor_t face_vertex = *it;
            vertex_data_t &vd = m_vertices.at(face_vertex);

            std::vector<face_descriptor_t>::iterator fIter = std::find(vd.m_faces.begin(), vd.m_faces.end(), f);

            if (fIter != vd.m_faces.cend())
            {
                vd.m_faces.erase(fIter); // remove association
            }
        }

        m_faces_removed.push_back(f);
    }

    // also disassociates (not remove) the halfedges(s) and vertex incident to this halfedge
    void mesh_t::remove_halfedge(halfedge_descriptor_t h)
    {
        MCUT_ASSERT(h != null_halfedge());
        MCUT_ASSERT(std::find(m_halfedges_removed.cbegin(), m_halfedges_removed.cend(), h) == m_halfedges_removed.cend());

        halfedge_data_t &hd = m_halfedges.at(h);

        MCUT_ASSERT(hd.e == null_edge()); // there must not be an edge dependent on h if we are to remove h
        MCUT_ASSERT(hd.f == null_face()); // there must not be a face dependent on h if we are to remove h

        if (hd.n != null_halfedge())
        { // disassociate next
            const halfedge_descriptor_t hn = hd.n;
            halfedge_data_t &hnd = m_halfedges.at(hn);
            MCUT_ASSERT(hnd.p == h);
            hnd.p = null_halfedge();
            //
            hd.n = null_halfedge();
        }

        if (hd.o != null_halfedge())
        { // disassociate opposite
            const halfedge_descriptor_t ho = hd.o;
            halfedge_data_t &hod = m_halfedges.at(ho);
            MCUT_ASSERT(hod.o == h);
            hod.o = null_halfedge();
            //
            hd.o = null_halfedge();
        }

        if (hd.p != null_halfedge())
        { // disassociate previous
            const halfedge_descriptor_t hp = hd.p;
            halfedge_data_t &hpd = m_halfedges.at(hp);
            MCUT_ASSERT(hpd.n == h);
            hpd.n = null_halfedge();
            //
            hd.p = null_halfedge();
        }

        MCUT_ASSERT(hd.t != null_vertex()); // every h has a target vertex which is effectively dependent on h

        // disassociate target vertex
        vertex_data_t &htd = m_vertices.at(hd.t);
        std::vector<halfedge_descriptor_t>::iterator hIter = std::find(htd.m_halfedges.begin(), htd.m_halfedges.end(), h);

        MCUT_ASSERT(hIter != htd.m_halfedges.end()); // because not yet removed h

        htd.m_halfedges.erase(hIter); // remove association

        m_halfedges_removed.push_back(h);
    }

    // also disassociates (not remove) any face(s) incident to edge via its halfedges, and also disassociates the halfedges
    void mesh_t::remove_edge(const edge_descriptor_t e, bool remove_halfedges)
    {
        MCUT_ASSERT(e != null_edge());
        MCUT_ASSERT(std::find(m_edges_removed.cbegin(), m_edges_removed.cend(), e) == m_edges_removed.cend());

        edge_data_t &ed = m_edges.at(e);
        std::vector<halfedge_descriptor_t> halfedges = {ed.h, opposite(ed.h)}; // both halfedges incident to edge must be disassociated

        for (std::vector<halfedge_descriptor_t>::const_iterator it = halfedges.cbegin(); it != halfedges.cend(); ++it)
        {
            const halfedge_descriptor_t h = *it;
            MCUT_ASSERT(h != null_halfedge());

            // disassociate halfedge
            halfedge_data_t &hd = m_halfedges.at(h);
            MCUT_ASSERT(hd.e == e);
            hd.e = null_edge();
            if (remove_halfedges)
            {
                remove_halfedge(h);
            }
        }

        ed.h = null_halfedge(); // we are removing the edge so every associated data element must be nullified

        m_edges_removed.push_back(e);
    }

    void mesh_t::remove_vertex(const vertex_descriptor_t v)
    {
        MCUT_ASSERT(v != null_vertex());
        MCUT_ASSERT((size_t)v < m_vertices.size());
        MCUT_ASSERT(std::find(m_vertices_removed.cbegin(), m_vertices_removed.cend(), v) == m_vertices_removed.cend());
        MCUT_ASSERT(m_vertices.at(v).m_faces.empty());
        MCUT_ASSERT(m_vertices.at(v).m_halfedges.empty());

        m_vertices_removed.push_back(v);
    }

    void mesh_t::remove_elements()
    {
        for (face_array_iterator_t i = faces_begin(); i != faces_end(); ++i)
        {
            remove_face(*i);
        }

        for (edge_array_iterator_t i = edges_begin(); i != edges_end(); ++i)
        {
            remove_edge(*i);
        }

        for (halfedge_array_iterator_t i = halfedges_begin(); i != halfedges_end(); ++i)
        {
            remove_halfedge(*i);
        }

        for (vertex_array_iterator_t i = vertices_begin(); i != vertices_end(); ++i)
        {
            remove_vertex(*i);
        }
    }

    void mesh_t::reset()
    {
        m_vertices.clear();
        m_vertices.shrink_to_fit();
        m_vertices_removed.clear();
        m_vertices_removed.shrink_to_fit();
        m_halfedges.clear();
        m_halfedges.shrink_to_fit();
        m_halfedges_removed.clear();
        m_halfedges_removed.shrink_to_fit();
        m_edges.clear();
        m_edges.shrink_to_fit();
        m_edges_removed.clear();
        m_edges_removed.shrink_to_fit();
        m_faces.clear();
        m_faces.shrink_to_fit();
        m_faces_removed.clear();
        m_faces_removed.shrink_to_fit();
    }

    int mesh_t::number_of_internal_faces() const
    {
        return static_cast<int>(m_faces.size());
    }

    int mesh_t::number_of_internal_edges() const
    {
        return static_cast<int>(m_edges.size());
    }

    int mesh_t::number_of_internal_halfedges() const
    {
        return static_cast<int>(m_halfedges.size());
    }

    int mesh_t::number_of_internal_vertices() const
    {
        return static_cast<int>(m_vertices.size());
    }

    //
    int mesh_t::number_of_vertices_removed() const
    {
        return (int)this->m_vertices_removed.size();
    }

    int mesh_t::number_of_edges_removed() const
    {
        return (int)this->m_edges_removed.size();
    }

    int mesh_t::number_of_halfedges_removed() const
    {
        return (int)this->m_halfedges_removed.size();
    }

    int mesh_t::number_of_faces_removed() const
    {
        return (int)this->m_faces_removed.size();
    }

    bool mesh_t::is_removed(face_descriptor_t f) const
    {
        return std::find(m_faces_removed.cbegin(), m_faces_removed.cend(), f) != m_faces_removed.cend();
    }

    bool mesh_t::is_removed(edge_descriptor_t e) const
    {
        return std::find(m_edges_removed.cbegin(), m_edges_removed.cend(), e) != m_edges_removed.cend();
    }

    bool mesh_t::is_removed(halfedge_descriptor_t h) const
    {
        return std::find(m_halfedges_removed.cbegin(), m_halfedges_removed.cend(), h) != m_halfedges_removed.cend();
    }

    bool mesh_t::is_removed(vertex_descriptor_t v) const
    {
        return std::find(m_vertices_removed.cbegin(), m_vertices_removed.cend(), v) != m_vertices_removed.cend();
    }

    void mesh_t::reserve_for_additional_vertices(std::uint32_t n)
    {
        m_vertices.reserve(number_of_internal_vertices() + n);
    }

    void mesh_t::reserve_for_additional_edges(std::uint32_t n)
    {
        m_edges.reserve(number_of_internal_edges() + n);
    }

    void mesh_t::reserve_for_additional_halfedges(std::uint32_t n)
    {
        m_halfedges.reserve(number_of_internal_halfedges() + n);
    }

    void mesh_t::reserve_for_additional_faces(std::uint32_t n)
    {
        m_faces.reserve(number_of_internal_faces() + n);
    }

    void mesh_t::reserve_for_additional_elements(std::uint32_t n)
    {
        const std::uint32_t nv = n;
        reserve_for_additional_vertices(nv);
        const std::uint32_t nf = nv * 2;
        reserve_for_additional_faces(nf);
        const std::uint32_t ne = std::uint32_t((3.0 / 2.0) * (double)nf);
        reserve_for_additional_edges(ne);
        const std::uint32_t nh = ne * 2;
        reserve_for_additional_halfedges(nh);
    }

    const std::vector<vertex_descriptor_t> &mesh_t::get_removed_elements(id_<array_iterator_t<vertex_array_t>>) const
    {
        return get_removed_vertices();
    }

    const std::vector<edge_descriptor_t> &mesh_t::get_removed_elements(id_<array_iterator_t<edge_array_t>>) const
    {
        return get_removed_edges();
    }

    const std::vector<halfedge_descriptor_t> &mesh_t::get_removed_elements(id_<array_iterator_t<halfedge_array_t>>) const
    {
        return get_removed_halfedges();
    }

    const std::vector<face_descriptor_t> &mesh_t::get_removed_elements(id_<array_iterator_t<face_array_t>>) const
    {
        return get_removed_faces();
    }

    const std::vector<vertex_descriptor_t> &mesh_t::get_removed_vertices() const
    {
        return m_vertices_removed;
    }

    const std::vector<edge_descriptor_t> &mesh_t::get_removed_edges() const
    {
        return m_edges_removed;
    }

    const std::vector<halfedge_descriptor_t> &mesh_t::get_removed_halfedges() const
    {
        return m_halfedges_removed;
    }

    const std::vector<face_descriptor_t> &mesh_t::get_removed_faces() const
    {
        return m_faces_removed;
    }

    const vertex_array_iterator_t mesh_t::elements_begin_(id_<array_iterator_t<vertex_array_t>>, bool account_for_removed_elems) const
    {
        return vertices_begin(account_for_removed_elems);
    }

    const edge_array_iterator_t mesh_t::elements_begin_(id_<array_iterator_t<edge_array_t>>, bool account_for_removed_elems) const
    {
        return edges_begin(account_for_removed_elems);
    }

    const halfedge_array_iterator_t mesh_t::elements_begin_(id_<array_iterator_t<halfedge_array_t>>, bool account_for_removed_elems) const
    {
        return halfedges_begin(account_for_removed_elems);
    }

    const face_array_iterator_t mesh_t::elements_begin_(id_<array_iterator_t<face_array_t>>, bool account_for_removed_elems) const
    {
        return faces_begin(account_for_removed_elems);
    }

    void write_off(const char *fpath, const mcut::mesh_t &mesh)
    {

        std::ofstream outfile(fpath);

        if (!outfile.is_open())
        {
            printf("error: could not open file: %s\n", fpath);
            std::exit(1);
        }

        //
        // file header
        //
        outfile << "OFF\n";

        //
        // #vertices, #faces, #edges
        //
        outfile << mesh.number_of_vertices() << " " << mesh.number_of_faces() << " " << 0 /*mesh.number_of_edges()*/ << "\n";

        //
        // vertices
        //
        for (mcut::vertex_array_iterator_t iter = mesh.vertices_begin(); iter != mesh.vertices_end(); ++iter)
        {
            //const vertex_data_t& vdata = iter.second;
            const math::vec3 &point = mesh.vertex(*iter);
            outfile << (double)point.x() << " " << (double)point.y() << " " << (double)point.z() << "\n";
        }

        //
        // edges
        //

#if 0
    for (typename mesh_t::edge_iterator_t iter = mesh.edges_begin(); iter != mesh.edges_end(); ++iter) {
        const mesh_t::edge_descriptor_t ed = iter.first;
        const mesh_t::vertex_descriptor_t& v0 = vertex(ed, 0);
        const mesh_t::vertex_descriptor_t& v1 = vertex(ed, 1);
        // TODO
    }
#endif

        //
        // faces
        //
        for (mcut::face_array_iterator_t iter = mesh.faces_begin(); iter != mesh.faces_end(); ++iter)
        {
            //const typename mesh_t::face_descriptor_t& fd = iter.first;
            const std::vector<vertex_descriptor_t> vertices_around_face = mesh.get_vertices_around_face(*iter);

            MCUT_ASSERT(!vertices_around_face.empty());

            outfile << vertices_around_face.size() << " ";

            for (std::vector<vertex_descriptor_t>::const_iterator i = vertices_around_face.cbegin(); i != vertices_around_face.cend(); ++i)
            {
                outfile << (*i) << " ";
            }
            outfile << " \n";
        }

        outfile.close();
    }

    void read_off(mcut::mesh_t &mesh, const char *fpath)
    {
        auto next_line = [&](std::ifstream &f, std::string &s) -> bool
        {
            while (getline(f, s))
            {
                if (s.length() > 1 && s[0] != '#')
                {
                    return true;
                }
            }
            return false;
        };

        std::ifstream infile(fpath);

        if (!infile.is_open())
        {
            printf("error: could not open file: %s\n", fpath);
            std::exit(1);
        }

        //
        // file header
        //
        std::string header;
        if (!next_line(infile, header))
        {
            printf("error: .off file header not found\n");
            std::exit(1);
        }

        if (header != "OFF")
        {
            printf("error: unrecognised .off file header\n");
            std::exit(1);
        }

        //
        // #vertices, #faces, #edges
        //
        std::string info;
        if (!next_line(infile, info))
        {
            printf("error: .off element count not found\n");
            std::exit(1);
        }

        std::istringstream info_stream;
        info_stream.str(info);

        int nvertices;
        int nfaces;
        int nedges;
        info_stream >> nvertices >> nfaces >> nedges;

        //
        // vertices
        //
        std::map<int, vd_t> vmap;
        for (int i = 0; i < nvertices; ++i)
        {
            if (!next_line(infile, info))
            {
                printf("error: .off vertex not found\n");
                std::exit(1);
            }
            std::istringstream vtx_line_stream(info);

            double x;
            double y;
            double z;
            vtx_line_stream >> x >> y >> z;
            vmap[i] = mesh.add_vertex(x, y, z);
        }

        //
        // edges
        //
        for (int i = 0; i < nedges; ++i)
        {
            // TODO
        }

        //
        // faces
        //
        for (auto i = 0; i < nfaces; ++i)
        {
            if (!next_line(infile, info))
            {
                printf("error: .off file face not found\n");
                std::exit(1);
            }
            std::istringstream face_line_stream(info);
            int n; // number of vertices in face
            int index;
            face_line_stream >> n;

            if (n < 3)
            {
                printf("error: invalid polygon vertex count in file (%d)\n", n);
                std::exit(1);
            }

            typename std::vector<vd_t> face;
            face.resize(n);
            for (int j = 0; j < n; ++j)
            {
                info_stream >> index;
                face[j] = vmap[index];
            }

            mesh.add_face(face);
        }

        infile.close();
    }

} // namespace mcut

#if 1
namespace std
{
#if 1
    template <>
    typename mcut::edge_array_iterator_t::difference_type distance(
        mcut::edge_array_iterator_t first,
        mcut::edge_array_iterator_t last)
    {
        MCUT_ASSERT(first.get_mesh_ptr() == last.get_mesh_ptr());
        mcut::edge_array_iterator_t it = first;
        mcut::edge_array_iterator_t::difference_type dist = last - first;

        uint32_t r = it.get_mesh_ptr()->count_removed_elements_in_range(first, last);
        if (r > 0)
        {
            dist = dist - r;
        }

        MCUT_ASSERT(dist >= 0);

        return dist;
    }
#endif

#if 0
    template<>
    void advance(
        mcut::mesh_t::array_iterator_t<mcut::mesh_t::edge_array_t>& iter, 
        typename std::iterator_traits<mcut::mesh_t::array_iterator_t<mcut::mesh_t::edge_array_t>>::difference_type n){

        const mcut::mesh_t::array_iterator_t<mcut::mesh_t::edge_array_t> it = iter;
        iter += n; // raw ptr shift (i.e. ignoring that there may be removed elements)
#if 0
        uint32_t r = iter.get_mesh_ptr()->count_removed_elements_in_range(it, iter);
        uint32_t i =0;
        while(i < r)
        {
            ++iter;
            ++i;
        }
#endif
    }
#endif
}
#endif
