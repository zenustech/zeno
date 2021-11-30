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

#ifndef MCUT_GEOM_H_
#define MCUT_GEOM_H_

#include "mcut/internal/math.h"

// Shewchuk predicates : shewchuk.c
extern "C"
{
    void exactinit();
    double orient2d(const double *pa, const double *pb, const double *pc);
    double orient3d(const double *pa, const double *pb, const double *pc, const double *pd);
    double incircle(const double *pa, const double *pb, const double *pc, const double *pd);
    double insphere(const double *pa, const double *pb, const double *pc, const double *pd, const double *pe);
}

namespace mcut
{
    namespace geom
    {

        mcut::math::real_number_t orient2d(const mcut::math::vec2 &pa, const mcut::math::vec2 &pb, const mcut::math::vec2 &pc);
        mcut::math::real_number_t orient3d(const mcut::math::vec3 &pa, const mcut::math::vec3 &pb, const mcut::math::vec3 &pc, const mcut::math::vec3 &pd);

        // Compute a polygon's plane coefficients (i.e. normal and d parameters).
        // The computed normal is not normalized. This function returns the largest component of the normal.
        int compute_polygon_plane_coefficients(
            math::vec3 &normal,
            math::real_number_t &d_coeff,
            const math::vec3 *polygon_vertices,
            const int polygon_vertex_count);

        // Compute the intersection point between a line (not a segment) and a plane defined by a polygon.
        //
        // Parameters:
        //  'p' : output intersection point (computed if line does indeed intersect the plane)
        //  'q' : first point defining your line
        //  'r' : second point defining your line
        //  'polygon_vertices' : the vertices of the polygon defineing the plane (assumed to not be degenerate)
        //  'polygon_vertex_count' : number of olygon vertices
        //  'polygon_normal_max_comp' : largest component of polygon normal.
        //  'polygon_plane_normal' : normal of the given polygon
        //  'polygon_plane_d_coeff' : the distance coefficient of the plane equation corresponding to the polygon's plane
        //
        // Return values:
        // '0': line is parallel to plane or polygon is degenerate (within available precision)
        // '1': an intersection exists.
        // 'p': q and r lie in the plane (technically they are parallel to the plane too).
        char compute_line_plane_intersection(
            math::vec3 &p, //intersection point
            const math::vec3 &q,
            const math::vec3 &r,
            const math::vec3 *polygon_vertices,
            const int polygon_vertex_count,
            const int polygon_normal_max_comp,
            const math::vec3 &polygon_plane_normal);

        // Test if a line segment intersects with a plane, and yeild the intersection point if so.
        //
        // Return values:
        // 'p': The segment lies wholly within the plane.
        // 'q': The(first) q endpoint is on the plane (but not 'p').
        // 'r' : The(second) r endpoint is on the plane (but not 'p').
        // '0' : The segment lies strictly to one side or the other of the plane.
        // '1': The segment intersects the plane, and none of {p, q, r} hold.
        char compute_segment_plane_intersection(
            math::vec3 &p,
            const math::vec3 &normal,
            const math::real_number_t &d_coeff,
            const math::vec3 &q,
            const math::vec3 &r);

        // Similar to "compute_segment_plane_intersection" but simply checks the [type] of intersection using
        // exact arithmetic
        //
        // Return values:
        // 'p': The segment lies wholly within the plane.
        // 'q': The(first) q endpoint is on the plane (but not 'p').
        // 'r' : The(second) r endpoint is on the plane (but not 'p').
        // '0' : The segment lies strictly to one side or the other of the plane.
        // '1': The segment intersects the plane, and none of {p, q, r} hold.
        char compute_segment_plane_intersection_type(
            const math::vec3 &q,
            const math::vec3 &r,
            const std::vector<math::vec3> &polygon_vertices,
            const int polygon_normal_max_comp);

        // Test if a point 'q' (in 2D) lies inside or outside a given polygon (count the number ray crossings).
        //
        // Return values:
        // 'i': q is strictly interior
        // 'o': q is strictly exterior (outside).
        // 'e': q is on an edge, but not an endpoint.
        // 'v': q is a vertex.
        char compute_point_in_polygon_test(
            const math::vec2 &q,
            const std::vector<math::vec2> &polygon_vertices);

        // Test if a point 'q' (in 3D) lies inside or outside a given polygon (count the number ray crossings).
        //
        // Return values:
        // 'i': q is strictly interior
        // 'o': q is strictly exterior (outside).
        // 'e': q is on an edge, but not an endpoint.
        // 'v': q is a vertex.
        char compute_point_in_polygon_test(
            const math::vec3 &p,
            const std::vector<math::vec3> &polygon_vertices,
            const int polygon_plane_normal_largest_component);

        // project a 3d polygon to 3d by eliminating the largest component of its normal
        void project2D(
            std::vector<math::vec2> &out,
            const std::vector<math::vec3> &polygon_vertices,
            const int polygon_plane_normal_largest_component);

        bool coplaner(const mcut::math::vec3 &pa, const mcut::math::vec3 &pb, const mcut::math::vec3 &pc, const mcut::math::vec3 &pd);

        bool collinear(const math::vec2 &a, const math::vec2 &b, const math::vec2 &c, math::real_number_t &predResult);

        bool collinear(const math::vec2 &a, const math::vec2 &b, const math::vec2 &c);

        /*
     Compute the intersection of two line segments. Can also be used to calculate where the respective lines intersect.

     Parameters:
       'a' and 'b': end points of first segment
       'c' and 'd': end points of second segment
       'p': the intersection point
       's': the parameter for parametric equation of segment a,b (0..1)
       't': the parameter for parametric equation of segment c,d (0..1)

     Return values:

       'e': The segments collinearly overlap, sharing a point; 'e' stands for 'edge.'
       'v': An endpoint of one segment is on the other segment, but 'e' doesn't hold; 'v' stands for 'vertex.'
       '1': The segments intersect properly (i.e., they share a point and neither 'v' nor 'e' holds); '1' stands for TRUE.
       '0': The segments do not intersect (i.e., they share no points); '0' stands for FALSE
   */
        char compute_segment_intersection(const math::vec2 &a, const math::vec2 &b, const math::vec2 &c, const math::vec2 &d, math::vec2 &p, math::real_number_t &s, math::real_number_t &t);

        template <typename vector_type>
        struct bounding_box_t
        {

            vector_type m_minimum;
            vector_type m_maximum;

            bounding_box_t(const vector_type &minimum, const vector_type &maximum)
            {
                m_minimum = minimum;
                m_maximum = maximum;
            }

            bounding_box_t()
            {
                m_minimum = vector_type(std::numeric_limits<double>::max());
                m_maximum = vector_type(-std::numeric_limits<double>::max());
            }

            inline const vector_type &minimum() const
            {
                return m_minimum;
            }

            inline const vector_type &maximum() const
            {
                return m_maximum;
            }

            inline void expand(const vector_type &point)
            {
                m_maximum = compwise_max(m_maximum, point);
                m_minimum = compwise_min(m_minimum, point);
            }

            inline void expand(const bounding_box_t<vector_type> &bbox)
            {
                m_maximum = compwise_max(m_maximum, bbox.maximum());
                m_minimum = compwise_min(m_minimum, bbox.minimum());
            }

            inline void enlarge(const typename vector_type::element_type &eps_)
            {
                m_maximum = m_maximum + eps_;
                m_minimum = m_minimum - eps_;
            }

            float SurfaceArea() const
            {
                vector_type d = m_maximum - m_minimum;
                return typename vector_type::element_type(2.0) * (d.x() * d.y() + d.x() * d.z() + d.y() * d.z());
            }

            int MaximumExtent() const
            {
                vector_type diag = m_maximum - m_minimum;
                if (diag.x() > diag.y() && diag.x() > diag.z())
                    return 0;
                else if (diag.y() > diag.z())
                    return 1;
                else
                    return 2;
            }
        };

        template <typename T>
        inline bool intersect_bounding_boxes(const bounding_box_t<math::vec3_<T>> &a, const bounding_box_t<math::vec3_<T>> &b)
        {
            const math::vec3_<T> &amin = a.minimum();
            const math::vec3_<T> &amax = a.maximum();
            const math::vec3_<T> &bmin = b.minimum();
            const math::vec3_<T> &bmax = b.maximum();
            return (amin.x() <= bmax.x() && amax.x() >= bmin.x()) && //
                   (amin.y() <= bmax.y() && amax.y() >= bmin.y()) && //
                   (amin.z() <= bmax.z() && amax.z() >= bmin.z());
        }

        bool point_in_bounding_box(const math::vec2 &point, const bounding_box_t<math::vec2> &bbox);

        bool point_in_bounding_box(const math::vec3 &point, const bounding_box_t<math::vec3> &bbox);

        template <typename vector_type>
        void make_bbox(bounding_box_t<vector_type> &bbox, const vector_type *vertices, const int num_vertices)
        {
            MCUT_ASSERT(vertices != nullptr);
            MCUT_ASSERT(num_vertices >= 3);

            for (int i = 0; i < num_vertices; ++i)
            {
                const vector_type &vertex = vertices[i];
                bbox.expand(vertex);
            }
        }
    } // namespace geom {
} // namespace mcut {

#endif // MCUT_GEOM_H_
