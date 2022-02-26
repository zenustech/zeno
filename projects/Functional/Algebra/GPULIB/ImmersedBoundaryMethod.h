/**
 * @file ImmersedBoundaryMethod.h
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-12-07
 * 
 * @copyright Copyright (c) 2021  Ma Pengfei
 * 
 */
#ifndef __IMMERSEDBOUNDARYMETHOD_H__
#define __IMMERSEDBOUNDARYMETHOD_H__
#include "utilities.h"
#include "../Timer.h"

namespace gpu{
void distribute_force_cpu(
    const double3* solid_forces,                        /// solid_forces
    const double4* quadrature_rules,                    /// quadrature quadrature_rules
          double3* fluid_forces,
    int num, double h, int3 dim);


void interpolate_velocity_cpu(
          double3* solid_velocities, 
    const double4* quadrature_rules, 
    const double3* fluid_velocities, 
    int num, double h, int3 dim);

}

#endif