#pragma once
#include "openvdb/openvdb.h"
class vdb_velocity_extrapolator {
public:
	static void extrapolate(int n_layer, openvdb::Vec3fGrid::Ptr in_out_vel_grid);
	static void extrapolate(int n_layer, openvdb::FloatGrid::Ptr in_out_scalar_grid);
	static void union_extrapolate(int n_layer, openvdb::FloatGrid::Ptr vx, openvdb::FloatGrid::Ptr vy, openvdb::FloatGrid::Ptr vz, const openvdb::FloatTree* target_topo = nullptr);
};