// ---------------------------------------------------------
//
//  ccd_wrapper.h
//  Tyson Brochu 2009
//  Christopher Batty, Fang Da 2014
//
//  General interface for collision and intersection queries.
//
// ---------------------------------------------------------


#ifndef CCD_WRAPPER_H
#define CCD_WRAPPER_H

#include <vec.h>

namespace LosTopos {
// --------------------------------------------------------------------------------------------------
// 2D continuous collision detection
// --------------------------------------------------------------------------------------------------

// x0 is the point, x1-x2 is the segment. Take care to specify x1,x2 in sorted order of index!
bool point_segment_collision(const Vec2d& x0, const Vec2d& xnew0, size_t index0,
                             const Vec2d& x1, const Vec2d& xnew1, size_t index1,
                             const Vec2d& x2, const Vec2d& xnew2, size_t index2);

bool point_segment_collision(const Vec2d& x0, const Vec2d& xnew0, size_t index0,
                             const Vec2d& x1, const Vec2d& xnew1, size_t index1,
                             const Vec2d& x2, const Vec2d& xnew2, size_t index2,
                             double& edge_alpha, Vec2d& normal, double& rel_disp);

// --------------------------------------------------------------------------------------------------
// 2D static intersection detection
// --------------------------------------------------------------------------------------------------

bool segment_segment_intersection(const Vec2d& x0, size_t index0, 
                                  const Vec2d& x1, size_t index1,
                                  const Vec2d& x2, size_t index2,
                                  const Vec2d& x3, size_t index3);

bool segment_segment_intersection(const Vec2d& x0, size_t index0, 
                                  const Vec2d& x1, size_t index1,
                                  const Vec2d& x2, size_t index2,
                                  const Vec2d& x3, size_t index3,
                                  double &s0, double& s2 );

// --------------------------------------------------------------------------------------------------
// 3D continuous collision detection
// --------------------------------------------------------------------------------------------------

// x0 is the point, x1-x2-x3 is the triangle. Take care to specify x1,x2,x3 in sorted order of index!
bool point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0, size_t index0,
                              const Vec3d& x1, const Vec3d& xnew1, size_t index1,
                              const Vec3d& x2, const Vec3d& xnew2, size_t index2,
                              const Vec3d& x3, const Vec3d& xnew3, size_t index3);

// x0 is the point, x1-x2-x3 is the triangle. Take care to specify x1,x2,x3 in sorted order of index!
// If there is a collision, returns true and sets bary1, bary2, bary3 to the barycentric coordinates of
// the collision point, sets normal to the collision point, t to the collision time, and the relative
// normal displacement (in terms of point 0 minus triangle 1-2-3)
bool point_triangle_collision(const Vec3d& x0, const Vec3d& xnew0, size_t index0,
                              const Vec3d& x1, const Vec3d& xnew1, size_t index1,
                              const Vec3d& x2, const Vec3d& xnew2, size_t index2,
                              const Vec3d& x3, const Vec3d& xnew3, size_t index3,
                              double& bary1, double& bary2, double& bary3,
                              Vec3d& normal,
                              double& relative_normal_displacement );

// x0-x1 and x2-x3 are the segments. Take care to specify x0,x1 and x2,x3 in sorted order of index!
bool segment_segment_collision(const Vec3d& x0, const Vec3d& xnew0, size_t index0,
                               const Vec3d& x1, const Vec3d& xnew1, size_t index1,
                               const Vec3d& x2, const Vec3d& xnew2, size_t index2,
                               const Vec3d& x3, const Vec3d& xnew3, size_t index3);

// x0-x1 and x2-x3 are the segments. Take care to specify x0,x1 and x2,x3 in sorted order of index!
// If there is a collision, returns true and sets bary0 and bary2 to parts of the barycentric coordinates of
// the collision point, sets normal to the collision point, t to the collision time, and the relative
// normal displacement (in terms of edge 0-1 minus edge 2-3)
bool segment_segment_collision(const Vec3d& x0, const Vec3d& xnew0, size_t index0,
                               const Vec3d& x1, const Vec3d& xnew1, size_t index1,
                               const Vec3d& x2, const Vec3d& xnew2, size_t index2,
                               const Vec3d& x3, const Vec3d& xnew3, size_t index3,
                               double& bary0, double& bary2,
                               Vec3d& normal,
                               double& relative_normal_displacement );


// --------------------------------------------------------------------------------------------------
// 3D static intersection detection
// --------------------------------------------------------------------------------------------------

// x0-x1 is the segment and and x2-x3-x4 is the triangle.
bool segment_triangle_intersection(const Vec3d& x0, size_t index0,
                                   const Vec3d& x1, size_t index1,
                                   const Vec3d& x2, size_t index2,
                                   const Vec3d& x3, size_t index3,
                                   const Vec3d& x4, size_t index4,
                                   bool degenerate_counts_as_intersection,
                                   bool verbose = false );

bool segment_triangle_intersection(const Vec3d& x0, size_t index0,
                                   const Vec3d& x1, size_t index1,
                                   const Vec3d& x2, size_t index2,
                                   const Vec3d& x3, size_t index3,
                                   const Vec3d& x4, size_t index4,
                                   double& bary0, double& bary1, double& bary2, double& bary3, double& bary4,
                                   bool degenerate_counts_as_intersection,
                                   bool verbose = false );


// x0 is the point and x1-x2-x3-x4 is the tetrahedron. Order is irrelevant.
bool point_tetrahedron_intersection(const Vec3d& x0, size_t index0,
                                    const Vec3d& x1, size_t index1,
                                    const Vec3d& x2, size_t index2,
                                    const Vec3d& x3, size_t index3,
                                    const Vec3d& x4, size_t index4);

}

#endif

