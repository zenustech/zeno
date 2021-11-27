#pragma once
// FLIP simulation based on openvdb points and grid structure
//#include "openvdb/openvdb.h"
#include "openvdb/points/PointConversion.h"
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>



static inline float frand(unsigned int i) {
	unsigned int value = (i ^ 61) ^ (i >> 16);
	value *= 9;
	value ^= value << 4;
	value *= 0x27d4eb2d;
	value ^= value >> 15;	
    return (float)value / (float)4294967296;
}
static inline struct MakeZhxxHappyRandomTable {
    float operator[](unsigned int i) { return frand(i)-0.5; }
} randomTable;

struct FLIP_vdb {
  using vec_tree_t = openvdb::Vec3fGrid::TreeType;
  using scalar_tree_t = openvdb::FloatGrid::TreeType;
  using point_tree_t = openvdb::points::PointDataGrid::TreeType;

  // attribute encoding types
  using PositionCodec = openvdb::points::FixedPointCodec</*one byte*/ false>;
  // using PositionCodec = openvdb::points::TruncateCodec;
  // using PositionCodec = openvdb::points::NullCodec;
  using position_attribute =
      openvdb::points::TypedAttributeArray<openvdb::Vec3f, PositionCodec>;
  using VelocityCodec = openvdb::points::TruncateCodec;

  // using VelocityCodec = openvdb::points::NullCodec;
  using velocity_attribute =
      openvdb::points::TypedAttributeArray<openvdb::Vec3f, VelocityCodec>;

  // initialization artument type
  // helps to hint the make_shared type initialization
  // struct init_arg_t {
  // 	init_arg_t(float in_dx,
  // 		const openvdb::Vec3f& in_min,
  // 		const openvdb::Vec3f& in_max) {

  // 		m_dx = in_dx;
  // 		m_domain_vecf_min = in_min;
  // 		m_domain_vecf_max = in_max;
  // 	}

  // 	float m_dx;
  // 	openvdb::Vec3f m_domain_vecf_min, m_domain_vecf_max;
  // };

  // FLIP_vdb(const init_arg_t& in_arg);

  // initialize all grids
  // void init_grids();
  // void init_particles();

  // output the points
  // void write_points(const std::string& fname) const;

  // bool test();
  // void IO(std::string filename_prefix);

  using descriptor_t = openvdb::points::AttributeSet::Descriptor;
  // position attribute descriptor
  //   const descriptor_t::Ptr &pos_descriptor() const {
  //     return m_position_attribute_descriptor;
  //   }

  //   //
  //   const descriptor_t::Ptr &pv_descriptor() const {
  //     return m_pv_attribute_descriptor;
  //   }

  // void add_solid_sdf(openvdb::FloatGrid::Ptr in_solid_sdf);
  // void add_propeller_sdf(openvdb::FloatGrid::Ptr in_propeller_sdf);

  // void take_shapshot();

  // float get_voxel_dx() const { return m_dx; }
  // openvdb::Vec3i get_domain_idx_begin() const { return m_domain_index_begin;
  // } openvdb::Vec3i get_domain_idx_end() const { return m_domain_index_end; }
  // openvdb::points::PointDataGrid::Ptr get_particles() const { return
  // m_particles; } openvdb::FloatGrid::Ptr get_snapshot_liquid_sdf() const {
  // return m_liquid_sdf_snapshot; } openvdb::Vec3fGrid::Ptr
  // get_snapshot_velocity() const { return m_velocity_snapshot; } void
  // set_boundary_volume(openvdb::FloatGrid::Ptr in_boundary_fill_kill_volume) {
  // m_boundary_fill_kill_volume = in_boundary_fill_kill_volume; } void
  // set_boundary_velocity_volume(openvdb::Vec3fGrid::Ptr
  // in_boundary_velocity_volume) { m_boundary_velocity_volume =
  // in_boundary_velocity_volume; } void seed_liquid(openvdb::FloatGrid::Ptr
  // in_sdf, const openvdb::Vec3f& init_vel); int get_framenumber() { return
  // m_framenumber; }

  static void Advect(float dt, float dx,
                     openvdb::points::PointDataGrid::Ptr &particles,
                     openvdb::Vec3fGrid::Ptr &velocity,
                     openvdb::Vec3fGrid::Ptr &velocity_after_p2g,
                     float pic_component, int RK_ORDER);
  static void Advect(float dt, float dx,
                     openvdb::points::PointDataGrid::Ptr &particles,
                     openvdb::Vec3fGrid::Ptr &velocity,
                     openvdb::Vec3fGrid::Ptr &velocity_after_p2g,
                     openvdb::FloatGrid::Ptr &solid_sdf,
                     openvdb::Vec3fGrid::Ptr &solid_vel, float pic_component,
                     int RK_ORDER);
  static void AdvectSheetty(float dt, float dx, float surfacedist,
                            openvdb::points::PointDataGrid::Ptr &particles,
                            openvdb::FloatGrid::Ptr &liquid_sdf,
                            openvdb::Vec3fGrid::Ptr &velocity,
                            openvdb::Vec3fGrid::Ptr &velocity_after_p2g,
                            openvdb::FloatGrid::Ptr &solid_sdf,
                            openvdb::Vec3fGrid::Ptr &solid_vel,
                            float pic_component, int RK_ORDER);
  static void custom_move_points_and_set_flip_vel(
      openvdb::points::PointDataGrid &in_out_points,
      const openvdb::Vec3fGrid &in_velocity_field,
      const openvdb::Vec3fGrid &in_old_velocity, float PIC_component, float dt,
      float dx, int RK_order);
  static void custom_move_points_and_set_flip_vel(
      openvdb::points::PointDataGrid::Ptr in_out_points,
      const openvdb::FloatGrid::Ptr in_liquid_sdf,
      const openvdb::Vec3fGrid::Ptr in_velocity_field,
      const openvdb::Vec3fGrid::Ptr in_velocity_field_to_be_advected,
      const openvdb::Vec3fGrid::Ptr in_old_velocity,
      openvdb::FloatGrid::Ptr in_solid_sdf,
      openvdb::Vec3fGrid::Ptr in_solid_vel, float PIC_component, float dt,
      float surfacedist, int RK_order);
  static void
  update_solid_sdf(std::vector<openvdb::FloatGrid::Ptr> &moving_solids,
                   openvdb::FloatGrid::Ptr &m_solid_sdf,
                   openvdb::points::PointDataGrid::Ptr &particles);
  static void
  particle_to_grid_collect_style(openvdb::points::PointDataGrid::Ptr &particles,
                                 openvdb::Vec3fGrid::Ptr &velocity,
                                 openvdb::Vec3fGrid::Ptr &velocity_after_p2g,
                                 openvdb::Vec3fGrid::Ptr &velocity_weights,
                                 openvdb::FloatGrid::Ptr &liquid_sdf,
                                 openvdb::FloatGrid::Ptr &pushed_out_liquid_sdf,
                                 float dx);
  static void calculate_face_weights(openvdb::Vec3fGrid::Ptr &face_weight,
                                     openvdb::FloatGrid::Ptr &liquid_sdf,
                                     openvdb::FloatGrid::Ptr &solid_sdf);
  
  static void clamp_liquid_phi_in_solids(
      openvdb::FloatGrid::Ptr &liquid_sdf, openvdb::FloatGrid::Ptr &solid_sdf,
      openvdb::FloatGrid::Ptr &pushed_out_liquid_sdf, float dx);

  static void solve_pressure_simd(
      openvdb::FloatGrid::Ptr &liquid_sdf,
      openvdb::FloatGrid::Ptr &pushed_out_liquid_sdf,
      openvdb::FloatGrid::Ptr &rhsgrid, openvdb::FloatGrid::Ptr &curr_pressure,
      openvdb::Vec3fGrid::Ptr &face_weight, openvdb::Vec3fGrid::Ptr &velocity,
      openvdb::Vec3fGrid::Ptr &solid_velocity, float dt, float dx);

  static void apply_pressure_gradient(
      openvdb::FloatGrid::Ptr &liquid_sdf, openvdb::FloatGrid::Ptr &solid_sdf,
      openvdb::FloatGrid::Ptr &pushed_out_liquid_sdf,
      openvdb::FloatGrid::Ptr &pressure, openvdb::Vec3fGrid::Ptr &face_weight,
      openvdb::Vec3fGrid::Ptr &velocity,
      openvdb::Vec3fGrid::Ptr &solid_velocity, float dt, float dx);

  static void field_add_vector(openvdb::Vec3fGrid::Ptr &velocity_field,
                               openvdb::Vec3fGrid::Ptr &face_weight, float x,
                               float y, float z, float dt);
  static void emit_liquid(openvdb::points::PointDataGrid::Ptr &in_out_particles,
                          openvdb::FloatGrid::Ptr &sdf,
                          openvdb::Vec3fGrid::Ptr &vel,
                          openvdb::FloatGrid::Ptr &liquid_sdf, float vx,
                          float vy, float vz);

  static float cfl(openvdb::Vec3fGrid::Ptr &vel);

  static void
  point_integrate_vector(openvdb::points::PointDataGrid::Ptr &in_out_particles,
                         openvdb::Vec3R &dx, std::string channel);

  static void
  reseed_fluid(openvdb::points::PointDataGrid::Ptr &in_out_particles,
               openvdb::FloatGrid::Ptr &liquid_sdf,
               openvdb::Vec3fGrid::Ptr &velocity);
  // private:
  // 	void initialize_attribute_descriptor() {
  // 		auto pnamepair = position_attribute::attributeType();
  // 		m_position_attribute_descriptor =
  // openvdb::points::AttributeSet::Descriptor::create(pnamepair);

  // 		auto vnamepair = velocity_attribute::attributeType();
  // 		m_pv_attribute_descriptor =
  // m_position_attribute_descriptor->duplicateAppend("v", vnamepair);
  // 	}
  // 	void advection(float dt);

  // 	void custom_move_points_and_set_flip_vel(
  // 		openvdb::points::PointDataGrid& in_out_points,
  // 		const openvdb::Vec3fGrid& in_velocity_field,
  // 		const openvdb::Vec3fGrid& in_old_velocity,
  // 		float PIC_component,float dt, int RK_order);

  // 	void extrapolate_velocity(int layer=5);
  // 	void particle_to_grid_reduce_style();
  // 	void particle_to_grid_collect_style();
  // 	void fill_kill_particles();

  // 	bool below_waterline(float in_height);
  // 	bool below_sea_level(const openvdb::Vec3f& P);
  // 	float index_space_sea_level(const openvdb::Coord& xyz) const;
  // 	//void init_velocity_volume();
  // 	void init_boundary_fill_kill_volume();
  // 	//void init_domain_solid_sdf();
  // 	//void update_solid_sdf();

  // 	void calculate_face_weights();
  // 	void clamp_liquid_phi_in_solids();
  // 	void set_solid_velocity();
  // 	void apply_pressure_gradient(float in_dt);
  // 	void solve_pressure_simd(float in_dt);
  // 	void apply_body_force(float dt);
  // 	float cfl();
  // 	float cfl_and_regularize_velocity();
  // 	openvdb::points::PointDataGrid::Ptr narrow_band_particles();
  // 	float m_waterline;

  // 	//length of the voxel
  // 	float m_dx;

  // 	float m_cfl;

  // 	//radius of the particle
  // 	//prefered to be set as 0.5*sqrt(3)*dx
  // 	//so that when one particle is inside a voxel
  // 	//it will make the liquid sdf in the center negative
  // 	float m_particle_radius;

  // 	//signed minimum and maximum voxel index for the simulation
  // 	//it describes the pressure index
  // 	openvdb::Vec3i m_domain_index_begin, m_domain_index_end;

  // 	int m_collide_with_domain;
  // 	/*
  // 	   |<----dx---->|
  // 	   .------------.
  // 	   |            |
  // 	   |            |
  // 	   u     p      |
  // 	   |            |
  // 	   |            |
  // 	   *-----v------.

  // 	   2D illustration of the positions of variables
  // 	   with the same  integer index space coordinate
  // 	   p: 1 pressure,
  // 	      2 point grid voxel position : iter.getCoord().asVec3d()
  // 	      3 liquid sdf, possibly liquid sdf weight
  // 		  4 lattice position of velocity lattice
  // 	   v: physical velocity component in staggered grid
  // 	   u: physical velocity component in staggered grid
  // 	   *: solid signed distance function lattice position
  // 	   .: positions of solid sdf of other voxel

  // 	*/

  // 	//the points grid sort particles in voxels with lattice in center
  // 	//it contains attribute position and velocity
  // 	//position attribute is with respect to voxel center with
  // range(-0.5,0.5)
  // 	//velocity attribute has world-coordinate units in m/s
  // 	openvdb::points::PointDataGrid::Ptr m_particles;

  // 	descriptor_t::Ptr m_position_attribute_descriptor;
  // 	descriptor_t::Ptr m_pv_attribute_descriptor;

  // 	//pressure grid with the same transform as the m_particles
  // 	//in that way, each m_particles voxel coincide with a fluid simulation
  // voxel 	openvdb::FloatGrid::Ptr m_pressure; 	openvdb::FloatGrid::Ptr
  // m_rhsgrid;
  // 	//for each face of the voxel
  // 	//calculate the face weight used for variational pressure solver
  // 	openvdb::Vec3fGrid::Ptr m_face_weight;

  // 	//The degree of freedom index for the pressure
  // 	openvdb::Int32Grid::Ptr m_pressure_dofid;

  // 	//pressure solve related
  // 	tbb::concurrent_vector<openvdb::Index32> m_isolated_cell_dof;

  // 	//staggered grid velocity with the same transform as points
  // 	//the origin of the grid is (0,0,0) in world coordinate
  // 	//that origin is interpreted as the origin of the pressure
  // 	openvdb::Vec3fGrid::Ptr m_velocity;
  // 	openvdb::Vec3fGrid::Ptr m_velocity_update;
  // 	openvdb::Vec3fGrid::Ptr m_velocity_snapshot;
  // 	openvdb::Vec3fGrid::Ptr m_velocity_after_p2g;

  // 	//staggered solid velocity with the same transform as velocity
  // 	openvdb::Vec3fGrid::Ptr m_solid_velocity;

  // 	//used for points->grid transfer, normalizing the weighted velocity
  // 	//three channels, not necessary
  // 	openvdb::Vec3fGrid::Ptr m_velocity_weights;

  // 	//liquid signed distance function calculated by p2g,
  // 	//it has the same transform as the m_particles, pressure
  // 	openvdb::FloatGrid::Ptr m_liquid_sdf;
  // 	openvdb::FloatGrid::Ptr m_liquid_sdf_snapshot;
  // 	openvdb::FloatGrid::Ptr m_pushed_out_liquid_sdf;

  // 	openvdb::FloatGrid::Ptr m_shrinked_liquid_sdf;

  // 	//solid signed distance function
  // 	//the origin of m_solid_sdf is (-dx/2,-dx/2,-dx/2) where dx is the voxel
  // 	//edge length
  // 	openvdb::FloatGrid::Ptr m_solid_sdf;

  // 	openvdb::Vec3fGrid::Ptr m_acceleration_fields;

  // 	//the velocity function used to set the emitted particles and boundary
  // 	openvdb::Vec3fGrid::Ptr m_boundary_velocity_volume;

  // 	//indicate in the fill and kill layer, what proportions are valid region
  // 	//to seed particles, by default it is a box whose top is the sea level 0
  // 	openvdb::FloatGrid::Ptr m_boundary_fill_kill_volume;

  // 	//some solids sdfs that can be translated and have constant velocity
  // 	std::vector<openvdb::FloatGrid::Ptr> m_moving_solids;
  // 	//a propeller is a sdf indicating regions to add constant acceleration
  // 	//this design is not as flexible as adding a force field.
  // 	std::vector<openvdb::FloatGrid::Ptr> m_propeller_sdf;

  // 	//transforms that will be shared
  // 	openvdb::math::Transform::Ptr m_voxel_center_transform,
  // m_voxel_vertex_transform;

  // 	int m_framenumber;
};
