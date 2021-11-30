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

#include "mcut/internal/geom.h"

namespace mcut
{
    namespace geom
    {

        mcut::math::real_number_t orient2d(const mcut::math::vec2 &pa, const mcut::math::vec2 &pb, const mcut::math::vec2 &pc)
        {
            const double pa_[2] = {static_cast<double>(pa.x()), static_cast<double>(pa.y())};
            const double pb_[2] = {static_cast<double>(pb.x()), static_cast<double>(pb.y())};
            const double pc_[2] = {static_cast<double>(pc.x()), static_cast<double>(pc.y())};

            return ::orient2d(pa_, pb_, pc_); // shewchuk predicate
        }

        mcut::math::real_number_t orient3d(const mcut::math::vec3 &pa, const mcut::math::vec3 &pb, const mcut::math::vec3 &pc, const mcut::math::vec3 &pd)
        {
            const double pa_[3] = {static_cast<double>(pa.x()), static_cast<double>(pa.y()), static_cast<double>(pa.z())};
            const double pb_[3] = {static_cast<double>(pb.x()), static_cast<double>(pb.y()), static_cast<double>(pb.z())};
            const double pc_[3] = {static_cast<double>(pc.x()), static_cast<double>(pc.y()), static_cast<double>(pc.z())};
            const double pd_[3] = {static_cast<double>(pd.x()), static_cast<double>(pd.y()), static_cast<double>(pd.z())};

            return ::orient3d(pa_, pb_, pc_, pd_); // shewchuk predicate
        }

#if 0
    void polygon_normal(math::vec3& normal, const math::vec3* vertices, const int num_vertices)
    {
      normal = math::vec3(0.0);
      for (int i = 0; i < num_vertices; ++i) {
        normal = normal + cross_product(vertices[i] - vertices[0], vertices[(i + 1) % num_vertices] - vertices[0]);
      }
      normal = normalize(normal);
    }
#endif

        int compute_polygon_plane_coefficients(
            math::vec3 &normal,
            math::real_number_t &d_coeff,
            const math::vec3 *polygon_vertices,
            const int polygon_vertex_count)
        {
            // compute polygon normal (http://cs.haifa.ac.il/~gordon/plane.pdf)
            normal = math::vec3(0.0);
            for (int i = 1; i < polygon_vertex_count - 1; ++i)
            {
                normal = normal + cross_product(polygon_vertices[i] - polygon_vertices[0], polygon_vertices[(i + 1) % polygon_vertex_count] - polygon_vertices[0]);
            }

            // In our calculations we need the normal be be of unit length so that the d-coeff
            // represents the distance of the plane from the origin
            //normal = math::normalize(normal);

            d_coeff = math::dot_product(polygon_vertices[0], normal);

            math::real_number_t largest_component(0.0);
            math::real_number_t tmp(0.0);
            int largest_component_idx = 0;

            for (int i = 0; i < 3; ++i)
            {
                tmp = math::absolute_value(normal[i]);
                if (tmp > largest_component)
                {
                    largest_component = tmp;
                    largest_component_idx = i;
                }
            }

            return largest_component_idx;
        }

        // Intersect a line segment with a plane
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
            const math::vec3 &r)
        {

            //math::vec3 planeNormal;
            //math::real_number_t planeDCoeff;
            //planeNormalLargestComponent = compute_polygon_plane_coefficients(planeNormal, planeDCoeff, polygonVertices, polygonVertexCount);

            math::real_number_t num = d_coeff - math::dot_product(q, normal);
            const math::vec3 rq = (r - q);
            math::real_number_t denom = math::dot_product(rq, normal);

            if (denom == mcut::math::real_number_t(0.0) /* Segment is parallel to plane.*/)
            {
                if (num == mcut::math::real_number_t(0.0))
                {               // 'q' is on plane.
                    return 'p'; // The segment lies wholly within the plane
                }
                else
                {
                    return '0';
                }
            }

            math::real_number_t t = num / denom;

            for (int i = 0; i < 3; ++i)
            {
                p[i] = q[i] + t * (r[i] - q[i]);
            }

            if ((mcut::math::real_number_t(0.0) < t) && (t < mcut::math::real_number_t(1.0)))
            {
                return '1'; // The segment intersects the plane, and none of {p, q, r} hold
            }
            else if (num == mcut::math::real_number_t(0.0)) // t==0
            {
                return 'q'; // The (first) q endpoint is on the plane (but not 'p').
            }
            else if (num == denom) // t==1
            {
                return 'r'; // The (second) r endpoint is on the plane (but not 'p').
            }
            else
            {
                return '0'; // The segment lies strictly to one side or the other of the plane
            }
        }

        bool determine_three_noncollinear_vertices(
            int &i,
            int &j,
            int &k,
            const std::vector<math::vec3> &polygon_vertices,

            const int polygon_normal_max_comp)
        {
            const int polygon_vertex_count = (int)polygon_vertices.size();
            MCUT_ASSERT(polygon_vertex_count >= 3);

            std::vector<math::vec2> x;
            project2D(x, polygon_vertices, polygon_normal_max_comp);
            MCUT_ASSERT(x.size() == (size_t)polygon_vertex_count);

            // get any three vertices that are not collinear
            i = 0;
            j = i + 1;
            k = j + 1;
            bool found = false;
            for (; i < polygon_vertex_count; ++i)
            {
                for (; j < polygon_vertex_count; ++j)
                {
                    for (; k < polygon_vertex_count; ++k)
                    {
                        if (!collinear(x[i], x[j], x[k]))
                        {
                            found = true;
                            break;
                        }
                    }
                    if (found)
                    {
                        break;
                    }
                }
                if (found)
                {
                    break;
                }
            }

            return found;
        }

        char compute_segment_plane_intersection_type(
            const math::vec3 &q,
            const math::vec3 &r,
            const std::vector<math::vec3> &polygon_vertices,

            const int polygon_normal_max_comp)
        {
            const int polygon_vertex_count = (int)polygon_vertices.size();
            // ... any three vertices that are not collinear
            int i = 0;
            int j = 1;
            int k = 2;
            if (polygon_vertex_count > 3)
            { // case where we'd have the possibility of noncollinearity
                bool b = determine_three_noncollinear_vertices(i, j, k, polygon_vertices, polygon_normal_max_comp);

                if (!b)
                {
                    return '0'; // all polygon points are collinear
                }
            }

            mcut::math::real_number_t qRes = orient3d(polygon_vertices[i], polygon_vertices[j], polygon_vertices[k], q);
            mcut::math::real_number_t rRes = orient3d(polygon_vertices[i], polygon_vertices[j], polygon_vertices[k], r);

            if (qRes == mcut::math::real_number_t(0.0) && rRes == mcut::math::real_number_t(0.0))
            {
                return 'p';
            }
            else if (qRes == mcut::math::real_number_t(0.0))
            {
                return 'q';
            }
            else if (rRes == mcut::math::real_number_t(0.0))
            {
                return 'r';
            }
            else if ((rRes < mcut::math::real_number_t(0.0) && qRes < mcut::math::real_number_t(0.0)) || (rRes > mcut::math::real_number_t(0.0) && qRes > mcut::math::real_number_t(0.0)))
            {
                return '0';
            }
            else
            {
                return '1';
            }
        }

        char compute_segment_line_plane_intersection_type(
            const math::vec3 &q,
            const math::vec3 &r,
            const std::vector<math::vec3> &polygon_vertices,

            const int polygon_normal_max_comp,
            const math::vec3 &polygon_plane_normal)
        {
            const int polygon_vertex_count = (int)polygon_vertices.size();
            // ... any three vertices that are not collinear
            int i = 0;
            int j = 1;
            int k = 2;
            if (polygon_vertex_count > 3)
            { // case where we'd have the possibility of noncollinearity
                bool b = determine_three_noncollinear_vertices(i, j, k, polygon_vertices, polygon_normal_max_comp);

                if (!b)
                {
                    return '0'; // all polygon points are collinear
                }
            }

            mcut::math::real_number_t qRes = orient3d(polygon_vertices[i], polygon_vertices[j], polygon_vertices[k], q);
            mcut::math::real_number_t rRes = orient3d(polygon_vertices[i], polygon_vertices[j], polygon_vertices[k], r);

            if (qRes == mcut::math::real_number_t(0.0) && rRes == mcut::math::real_number_t(0.0))
            {
                // both points used to define line lie on plane therefore we have an in-plane intersection
                // or the polygon is a degenerate triangle
                return 'p';
            }
            else
            {
                if ((rRes < mcut::math::real_number_t(0.0) && qRes < mcut::math::real_number_t(0.0)) || (rRes > mcut::math::real_number_t(0.0) && qRes > mcut::math::real_number_t(0.0)))
                { // both points used to define line lie on same side of plane
                    // check if line is parallel to plane
                    //const math::real_number_t num = polygon_plane_d_coeff - math::dot_product(q, polygon_plane_normal);
                    const math::vec3 rq = (r - q);
                    const math::real_number_t denom = math::dot_product(rq, polygon_plane_normal);

                    if (denom == mcut::math::real_number_t(0.0) /* Segment is parallel to plane.*/)
                    {
                        //MCUT_ASSERT(num != 0.0); // implies 'q' is on plane (see: "compute_segment_plane_intersection(...)") but we have already established that q and r are on same side.
                        return '0';
                    }
                }
            }

            // q and r are on difference sides of the plane, therefore we have an intersection
            return '1';
        }

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
        // '0': line is parallel to plane (or polygon is degenerate ... within available precision)
        // '1': an intersection exists.
        // 'p': q and r lie in the plane (technically they are parallel to the plane too but we need to report this because it violates GP).
        char compute_line_plane_intersection(
            math::vec3 &p, //intersection point
            const math::vec3 &q,
            const math::vec3 &r,
            const std::vector<math::vec3> &polygon_vertices,
            const int polygon_normal_max_comp,
            const math::vec3 &polygon_plane_normal,
            const math::real_number_t &polygon_plane_d_coeff)
        {

            const int polygon_vertex_count = (int)polygon_vertices.size();
            // ... any three vertices that are not collinear
            int i = 0;
            int j = 1;
            int k = 2;
            if (polygon_vertex_count > 3)
            { // case where we'd have the possibility of noncollinearity
                bool b = determine_three_noncollinear_vertices(i, j, k, polygon_vertices, polygon_normal_max_comp);

                if (!b)
                {
                    return '0'; // all polygon points are collinear
                }
            }

            mcut::math::real_number_t qRes = orient3d(polygon_vertices[i], polygon_vertices[j], polygon_vertices[k], q);
            mcut::math::real_number_t rRes = orient3d(polygon_vertices[i], polygon_vertices[j], polygon_vertices[k], r);

            if (qRes == mcut::math::real_number_t(0.) && rRes == mcut::math::real_number_t(0.))
            {
                return 'p'; // both points used to define line lie on plane therefore we have an in-plane intersection
            }
            else
            {

                const math::real_number_t num = polygon_plane_d_coeff - math::dot_product(q, polygon_plane_normal);
                const math::vec3 rq = (r - q);
                const math::real_number_t denom = math::dot_product(rq, polygon_plane_normal);

                if ((rRes < mcut::math::real_number_t(0.) && qRes < mcut::math::real_number_t(0.)) || (rRes > mcut::math::real_number_t(0.) && qRes > mcut::math::real_number_t(0.)))
                { // both q an r are on same side of plane
                    if (denom == mcut::math::real_number_t(0.) /* line is parallel to plane.*/)
                    {
                        return '0';
                    }
                }

                // q and r are on difference sides of the plane, therefore we have an intersection

                // compute the intersection point
                const math::real_number_t t = num / denom;

                for (int it = 0; it < 3; ++it)
                {
                    p[it] = q[it] + t * (r[it] - q[it]);
                }

                return '1';
            }
        }

        // Count the number ray crossings to determine if a point 'q' lies inside or outside a given polygon.
        //
        // Return values:
        // 'i': q is strictly interior
        // 'o': q is strictly exterior (outside).
        // 'e': q is on an edge, but not an endpoint.
        // 'v': q is a vertex.
        char compute_point_in_polygon_test(
            const math::vec2 &q,
            const std::vector<math::vec2> &polygon_vertices)
        {
            const int polygon_vertex_count = (int)polygon_vertices.size();

#if 0 // Algorithm 1 in : https://pdf.sciencedirectassets.com/271512/1-s2.0-S0925772100X00545/1-s2.0-S0925772101000128/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAEaCXVzLWVhc3QtMSJHMEUCIBr6Fu%2F%2FtscErn%2Fl4pn2UGNA45sAw9vggQscK7Tnl0ssAiEAzfnyy4B4%2BXkZp8xcEk7utDBrxgmGyH7pyqk0efKOUEoq%2BgMIKhAEGgwwNTkwMDM1NDY4NjUiDHQT4kOfk4%2B6kBB0xCrXAzd8SWlnELJbM5l93HSvv4nUgtO85%2FGyx5%2BOoHmYoUuVwJjCXChmLJmlz%2BxYUQE%2Fr8vQa2hPlUEPfiTVGgoHaK8NkSMP6LRs%2F3WjyZ9OxzHbqSwZixNceW34OAkq0E1QgYLdGVjpPKxNK1haXDfBTMN6bF2IOU9dKi9wTv3uTlep0kHDa1BNNl6M6yZk5QlF2bPF9XmNjjZCpFQLhr%2BPoHo%2Bx4xy39aH8hCkkTqGdy2KrKGN6lv0%2FduIaItyZfqalYS%2BwW6cll2F5G11g0tSu7yKu6mug94LUTzsRmzD0UhzeGl2WV6Ev2qhw26mwFEKgnTMqGst8XAHjFjjAQyMzXNHCQqNBRIevHIzVWuUY4QZMylSRsodo0dfwwCFhzl0%2BJA1ZXb0%2BoyB8c11meQBO8FpMSshngNcbiAYUtIOFcHNCTCUMk0JPOZ%2FxvANsopnivbrPARL71zU4PaTujg5jfC2zoO6ZDUL8E6Vn%2BNtfb%2BuQV7DwtIH51Bv%2F%2F1m6h5mjbpRiGYcH%2F5SDD%2Fk%2BpHfKRQzKu7jJc%2B0XO0bQvoLSrAte0Qk10PwOvDl5jMIMdmxTBDDiDGErRykYxMQhq5EwjyiWPXzM3ll9uK59Vy0bAEj5Qemj5by1jCDht6IBjqlAV4okAPQO5wWdWojCYeKvluKfXCvudrUxrLhsxb7%2BTZNMODTG%2Fn%2Fbw875Yr6fOQ42u40pOsnPB%2FTP3cWwjiB%2BEHzDqN8AhCVQyoedw7QrU3OBWlSl6lB%2BGLAVqrhcizgFUiM2nj3HaVP2m7S%2FXqpv%2FoWlEXt4gR8iI9XsIlh6L6SBE22FqbsU5ewCxXaqip19VXhAGnlvjTihXUg6yZGWhExHj%2BKcA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20210814T103958Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY3PPQSW2L%2F20210814%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=94403dce5c12ede9507e9612c000db5772607e59a2134d30d4e5b44212056dc7&hash=cada083b165f9786e6042e21d3d22147ec08b1e2c8fa2572ceeb2445cb5e730a&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0925772101000128&tid=spdf-f9e7c4b7-808d-4a8e-a474-6430ff687798&sid=8dddff5b32fff2499908bf6501965aec3d0fgxrqb&type=client

            const math::real_number_t zero(0.0);
            int i = 0;
            int k = 0;

            math::real_number_t f = zero;
            math::real_number_t u1 = 0.0;
            math::real_number_t v1 = 0.0;
            math::real_number_t u2 = 0.0;
            math::real_number_t v2 = 0.0;
            const int n = polygon_vertex_count;
            for (i = 0; i < n; ++i)
            {
                v1 = polygon_vertices[i].y() - q.y();
                v2 = polygon_vertices[(i + 1) % n].y() - q.y();

                if ((v1 < zero && v2 < zero) || (v1 > zero && v2 > zero))
                {
                    continue;
                }

                u1 = polygon_vertices[i].x() - q.x();
                u2 = polygon_vertices[(i + 1) % n].x() - q.x();

                if (v2 > zero && v1 <= zero)
                {
                    f = u1 * v2 - u2 * v1;
                    if (f > zero)
                    {
                        k = k + 1;
                    }
                    else if (f == zero)
                    {
                        return 'e'; //-1; // boundary
                    }
                }
                else if (v1 > zero && v2 <= zero)
                {
                    f = u1 * v2 - u2 * v1;
                    if (f < zero)
                    {
                        k = k + 1;
                    }
                    else if (f == zero)
                    {
                        return 'e'; //-1; // boundary
                    }
                }
                else if (v2 == zero && v1 < zero)
                {
                    f = u1 * v2 - u2 * v1;
                    if (f == zero)
                    {
                        return 'e'; //-1; // // boundary
                    }
                }
                else if (v1 == zero && v2 < zero)
                {
                    f = u1 * v2 - u2 * v1;
                    if (f == zero)
                    {
                        return 'e'; //-1; // boundary
                    }
                }
                else if (v1 == zero && v2 == zero)
                {
                    if (u2 <= zero && u1 >= zero)
                    {
                        return 'e'; //-1; // boundary
                    }
                    else if (u1 <= zero && u2 >= zero)
                    {
                        return 'e'; //-1; // boundary
                    }
                }
            }
            if (k % 2 == 0)
            {
                return 'o'; //0; // outside
            }
            else
            {
                return 'i'; //1; // inside
            }
#else // http://www.science.smith.edu/~jorourke/books/compgeom.html

            std::vector<math::vec2> vertices(polygon_vertex_count, math::vec2());
            // Shift so that q is the origin. Note this destroys the polygon.
            // This is done for pedagogical clarity.
            for (int i = 0; i < polygon_vertex_count; i++)
            {
                for (int d = 0; d < 2; d++)
                {
                    const mcut::math::real_number_t &a = polygon_vertices[i][d];
                    const mcut::math::real_number_t &b = q[d];
                    mcut::math::real_number_t &c = vertices[i][d];
                    c = a - b;
                }
            }

            int Rcross = 0; /* number of right edge/ray crossings */
            int Lcross = 0; /* number ofleft edge/ray crossings */

            /* For each edge e = (i�lj), see if crosses ray. */
            for (int i = 0; i < polygon_vertex_count; i++)
            {

                /* First check if q = (0, 0) is a vertex. */
                if (vertices[i].x() == mcut::math::real_number_t(0.) && vertices[i].y() == mcut::math::real_number_t(0.))
                {
                    return 'v';
                }

                int il = (i + polygon_vertex_count - 1) % polygon_vertex_count;

                // Check if e straddles x axis, with bias above/below.

                // Rstrad is TRUE iff one endpoint of e is strictly above the x axis and the other is not (i.e., the other is on or below)
                bool Rstrad = (vertices[i].y() > mcut::math::real_number_t(0.)) != (vertices[il].y() > mcut::math::real_number_t(0.));
                bool Lstrad = (vertices[i].y() < mcut::math::real_number_t(0.)) != (vertices[il].y() < mcut::math::real_number_t(0.));

                if (Rstrad || Lstrad)
                {
                    /* Compute intersection of e with x axis. */

                    // The computation of x is needed whenever either of these straddle variables is TRUE, which
                    // only excludes edges passing through q = (0, 0) (and incidentally protects against division by 0).
                    math::real_number_t x = (vertices[i].x() * vertices[il].y() - vertices[il].x() * vertices[i].y()) / (vertices[il].y() - vertices[i].y());
                    if (Rstrad && x > mcut::math::real_number_t(0.))
                    {
                        Rcross++;
                    }
                    if (Lstrad && x < mcut::math::real_number_t(0.))
                    {
                        Lcross++;
                    }
                } /* end straddle computation*/
            }     // end for

            /* q on an edge if L/Rcross counts are not the same parity.*/
            if ((Rcross % 2) != (Lcross % 2))
            {
                return 'e';
            }

            /* q inside iff an odd number of crossings. */
            if ((Rcross % 2) == 1)
            {
                return 'i';
            }
            else
            {
                return 'o';
            }
#endif
        }

        void project2D(
            std::vector<math::vec2> &out,
            const std::vector<math::vec3> &polygon_vertices,
            const int polygon_plane_normal_largest_component)
        {
            const int polygon_vertex_count = (int)polygon_vertices.size();
            out.clear();
            out.resize(polygon_vertex_count);
            for (int i = 0; i < polygon_vertex_count; ++i)
            { // for each vertex
                math::vec2 &Tp = out[i];
                int k = 0;
                for (int j = 0; j < 3; j++)
                { // for each component
                    if (j != polygon_plane_normal_largest_component)
                    { /* skip largest coordinate */

                        Tp[k] = polygon_vertices[i][j];
                        k++;
                    }
                }
            }
        }

        // TODO: update this function to use "project2D" for projection step
        char compute_point_in_polygon_test(
            const math::vec3 &p,
            const std::vector<math::vec3> &polygon_vertices,
            const int polygon_plane_normal_largest_component)
        {
            const int polygon_vertex_count = (int)polygon_vertices.size();
            /* Project out coordinate m in both p and the triangular face */

            int k = 0;
            math::vec2 pp; /*projected p */
            for (int j = 0; j < 3; j++)
            { // for each component
                if (j != polygon_plane_normal_largest_component)
                { /* skip largest coordinate */
                    pp[k] = p[j];
                    k++;
                }
            }

            std::vector<math::vec2> polygon_vertices2d(polygon_vertex_count, math::vec2());

            for (int i = 0; i < polygon_vertex_count; ++i)
            { // for each vertex
                math::vec2 &Tp = polygon_vertices2d[i];
                k = 0;
                for (int j = 0; j < 3; j++)
                { // for each component
                    if (j != polygon_plane_normal_largest_component)
                    { /* skip largest coordinate */

                        Tp[k] = polygon_vertices[i][j];
                        k++;
                    }
                }
            }

            return compute_point_in_polygon_test(pp, polygon_vertices2d);
        }

        inline bool Between(const math::vec2 &a, const math::vec2 &b, const math::vec2 &c)
        {
            math::vec2 ba, ca;
            /* If ab not vertical check betweenness on x; else on y. */
            if (a[0] != b[0])
                return ((a[0] <= c[0]) && (c[0] <= b[0])) || //
                       ((a[0] >= c[0]) && (c[0] >= b[0]));
            else
                return ((a[1] <= c[1]) && (c[1] <= b[1])) || //
                       ((a[1] >= c[1]) && (c[1] >= b[1]));
        }

        bool coplaner(const mcut::math::vec3 &pa, const mcut::math::vec3 &pb, const mcut::math::vec3 &pc, const mcut::math::vec3 &pd)
        {
            const math::real_number_t val = orient3d(pa, pb, pc, pd);
            typedef std::numeric_limits<double> dbl;

            //double d = 3.14159265358979;
            std::cout.precision(dbl::max_digits10);
            
            // NOTE: thresholds are chosen based on benchmark meshes that are used for testing.
            // It is extremely difficult to get this right because of intermediate conversions 
            // between exact and fixed precision representations during cutting.
            // Thus, general advise is for user to ensure that the input polygons are really
            // co-planer. It might be possible for MCUT to help here (see eps used during poly 
            // partitioning).
            return math::absolute_value(val) <= math::real_number_t(4e-7);
        }

        bool collinear(const math::vec2 &a, const math::vec2 &b, const math::vec2 &c, math::real_number_t &predResult)
        {
            predResult = mcut::geom::orient2d(a, b, c);
            return predResult == mcut::math::real_number_t(0.);
        }

        bool collinear(const math::vec2 &a, const math::vec2 &b, const math::vec2 &c)
        {
            return mcut::geom::orient2d(a, b, c) == mcut::math::real_number_t(0.);
        }

        char Parallellnt(const math::vec2 &a, const math::vec2 &b, const math::vec2 &c, const math::vec2 &d, math::vec2 &p)
        {
            if (!collinear(a, b, c))
            {
                return '0';
            }

            if (Between(a, b, c))
            {
                p = c;
                return 'e';
            }

            if (Between(a, b, d))
            {
                p = d;
                return 'e';
            }

            if (Between(c, d, a))
            {
                p = a;
                return 'e';
            }

            if (Between(c, d, b))
            {
                p = b;
                return 'e';
            }

            return '0';
        }

        char compute_segment_intersection(const math::vec2 &a, const math::vec2 &b, const math::vec2 &c, const math::vec2 &d, math::vec2 &p, math::real_number_t &s, math::real_number_t &t)
        {
            //math::real_number_t s, t; /* The two parameters of the parametric eqns. */
            math::real_number_t num, denom; /* Numerator and denominator of equations. */
            char code = '?';                /* Return char characterizing intersection.*/

            denom = a[0] * (d[1] - c[1]) + //
                    b[0] * (c[1] - d[1]) + //
                    d[0] * (b[1] - a[1]) + //
                    c[0] * (a[1] - b[1]);

            /* If denom is zero, then segments are parallel: handle separately. */
            if (denom == math::real_number_t(0.0))
            {
                return Parallellnt(a, b, c, d, p);
            }

            num = a[0] * (d[1] - c[1]) + //
                  c[0] * (a[1] - d[1]) + //
                  d[0] * (c[1] - a[1]);

            if ((num == math::real_number_t(0.0)) || (num == denom))
            {
                code = 'v';
            }

            s = num / denom;

            num = -(a[0] * (c[1] - b[1]) + //
                    b[0] * (a[1] - c[1]) + //
                    c[0] * (b[1] - a[1]));

            if ((num == math::real_number_t(0.0)) || (num == denom))
            {
                code = 'v';
            }

            t = num / denom;

            if ((math::real_number_t(0.0) < s) && (s < math::real_number_t(1.0)) && (math::real_number_t(0.0) < t) && (t < math::real_number_t(1.0)))
            {
                code = '1';
            }
            else if ((math::real_number_t(0.0) > s) || (s > math::real_number_t(1.0)) || (math::real_number_t(0.0) > t) || (t > math::real_number_t(1.0)))
            {
                code = '0';
            }

            p[0] = a[0] + s * (b[0] - a[0]);
            p[1] = a[1] + s * (b[1] - a[1]);

            return code;
        }

        inline bool point_in_bounding_box(const math::vec2 &point, const bounding_box_t<math::vec2> &bbox)
        {
            if ((point.x() < bbox.m_minimum.x() || point.x() > bbox.m_maximum.x()) || //
                (point.y() < bbox.m_minimum.y() || point.y() > bbox.m_maximum.y()))
            {
                return false;
            }
            else
            {
                return true;
            }
        }

        inline bool point_in_bounding_box(const math::vec3 &point, const bounding_box_t<math::vec3> &bbox)
        {
            if ((point.x() < bbox.m_minimum.x() || point.x() > bbox.m_maximum.x()) || //
                (point.y() < bbox.m_minimum.y() || point.y() > bbox.m_maximum.y()) || //
                (point.z() < bbox.m_minimum.z() || point.z() > bbox.m_maximum.z()))
            { //
                return false;
            }
            else
            {
                return true;
            }
        }
    } // namespace mcut {
} //namespace geom {
