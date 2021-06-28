#ifndef COLLISIONQUERIES_H
#define COLLISIONQUERIES_H

#include <vec.h>

namespace LosTopos {
// 2D ====================================================================================================


void check_point_edge_proximity(bool update, const Vec2d &x0, const Vec2d &x1, const Vec2d &x2,
                                double &distance);

void check_point_edge_proximity(bool update, const Vec2d &x0, const Vec2d &x1, const Vec2d &x2,
                                double &distance, double &s, Vec2d &normal, double normal_multiplier);

// 3D ====================================================================================================

void check_point_edge_proximity(bool update, const Vec3d &x0, const Vec3d &x1, const Vec3d &x2,
                                double &distance);
void check_point_edge_proximity(bool update, const Vec3d &x0, const Vec3d &x1, const Vec3d &x2,
                                double &distance, double &s, Vec3d &normal, double normal_multiplier);

void check_edge_edge_proximity(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                               double &distance);
void check_edge_edge_proximity(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                               double &distance, double &s0, double &s2, Vec3d &normal);

void check_point_triangle_proximity(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                    double &distance);
void check_point_triangle_proximity(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3,
                                    double &distance, double &s1, double &s2, double &s3, Vec3d &normal);


double signed_volume(const Vec3d &x0, const Vec3d &x1, const Vec3d &x2, const Vec3d &x3);

}

#endif
