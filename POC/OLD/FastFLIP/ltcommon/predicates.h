#ifndef PREDICATES_H
#define PREDICATES_H

#include <vec.h>

namespace LosTopos {

const int MAX_COORD_2D=619925131;
const int MAX_COORD_3D=577053;


//======================================================================================================
// double precision floating point versions; they err on the side of safety (indicate collision when
// it's too close to tell for sure, or exactly touches)
//======================================================================================================

// x0 is the point, x1-x2 is the segment. Take care to specify x1,x2 in sorted order of index!
bool fe_point_segment_collision(const Vec2d& x0, const Vec2d& xnew0,
                                const Vec2d& x1, const Vec2d& xnew1,
                                const Vec2d& x2, const Vec2d& xnew2);

bool fe_point_segment_collision(const Vec2d& x0, const Vec2d& xnew0,
                                const Vec2d& x1, const Vec2d& xnew1,
                                const Vec2d& x2, const Vec2d& xnew2,
                                double& edge_bary, Vec2d& normal, double& t,
                                double& relative_normal_displacement );

// x0-x1 and x2-x3 are the segments. Order is irrelevant.
bool fe_segment_segment_intersection(const Vec2d& x0, const Vec2d& x1,
                                     const Vec2d& x2, const Vec2d& x3);

bool fe_segment_segment_intersection(const Vec2d& x0, const Vec2d& x1,
                                     const Vec2d& x2, const Vec2d& x3,
                                     double& alpha, double& beta );

// x0 is the point, x1-x2-x3 is the triangle. Take care to specify x1,x2,x3 in sorted order of index!
bool fe_point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0,
                                 const Vec3d& x1, const Vec3d& xnew1,
                                 const Vec3d& x2, const Vec3d& xnew2,
                                 const Vec3d& x3, const Vec3d& xnew3);

// x0 is the point, x1-x2-x3 is the triangle. Take care to specify x1,x2,x3 in sorted order of index!
// If there is a collision, returns true and sets bary1, bary2, bary3 to the barycentric coordinates of
// the collision point, sets normal to the collision point, t to the collision time, and the relative
// normal displacement (in terms of point 0 minus triangle 1-2-3)
bool fe_point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0,
                                 const Vec3d& x1, const Vec3d& xnew1,
                                 const Vec3d& x2, const Vec3d& xnew2,
                                 const Vec3d& x3, const Vec3d& xnew3,
                                 double& bary1, double& bary2, double& bary3,
                                 Vec3d& normal,
                                 double& relative_normal_displacement,
                                 bool verbose = false );

// x0-x1 and x2-x3 are the segments. Take care to specify x0,x1 and x2,x3 in sorted order of index!
bool fe_segment_segment_collision(const Vec3d& x0, const Vec3d& xnew0,
                                  const Vec3d& x1, const Vec3d& xnew1,
                                  const Vec3d& x2, const Vec3d& xnew2,
                                  const Vec3d& x3, const Vec3d& xnew3);

// x0-x1 and x2-x3 are the segments. Take care to specify x0,x1 and x2,x3 in sorted order of index!
// If there is a collision, returns true and sets bary0 and bary2 to parts of the barycentric coordinates of
// the collision point, sets normal to the collision point, t to the collision time, and the relative
// normal displacement (in terms of edge 0-1 minus edge 2-3)
bool fe_segment_segment_collision(const Vec3d& x0, const Vec3d& xnew0,
                                  const Vec3d& x1, const Vec3d& xnew1,
                                  const Vec3d& x2, const Vec3d& xnew2,
                                  const Vec3d& x3, const Vec3d& xnew3,
                                  double& bary0, double& bary2,
                                  Vec3d& normal,
                                  double& relative_normal_displacement,
                                  bool verbose = false );

bool fe_segment_triangle_intersection(const Vec3d& x0, const Vec3d& x1,
                                      const Vec3d& x2, const Vec3d& x3, const Vec3d& x4, 
                                      double& a, double& b, double& c, double& s, double& t, 
                                      bool /*degenerate_counts_as_intersection*/,
                                      bool /*verbose*/ );

// x0-x1 is the segment and and x2-x3-x4 is the triangle. Order is irrelevant.
bool fe_segment_triangle_intersection(const Vec3d& x0, const Vec3d& x1,
                                      const Vec3d& x2, const Vec3d& x3, const Vec3d& x4, 
                                      bool degenerate_counts_as_intersection,
                                      bool verbose = false );

// x0 is the point and x1-x2-x3-x4 is the tetrahedron. Order is irrelevant.
bool fe_point_tetrahedron_intersection(const Vec3d& x0, const Vec3d& x1,
                                       const Vec3d& x2, const Vec3d& x3, const Vec3d& x4);

}

#endif
