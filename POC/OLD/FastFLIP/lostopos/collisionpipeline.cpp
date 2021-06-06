// ---------------------------------------------------------
//
//  collisionpipeline.cpp
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//
//  Encapsulates all collision detection and resolution functions.
//
// ---------------------------------------------------------
#define NOMINMAX
#include <algorithm>
#include "Timer.h"
#include <collisionpipeline.h>
#include <broadphase.h>
#include <collisionqueries.h>
#include <dynamicsurface.h>
#include <impactzonesolver.h>
#include <runstats.h>
#include <wallclocktime.h>
#include <set>

// ---------------------------------------------------------
// Global externs
// ---------------------------------------------------------


extern bool simplex_verbose;


namespace LosTopos {

	extern RunStats g_stats;

	// ---------------------------------------------------------
	// File-local variables, functions, etc.
	// ---------------------------------------------------------

	namespace {

		// ---------------------------------------------------------
		// Local constants, typedefs, macros
		// ---------------------------------------------------------

		const double IMPULSE_MULTIPLIER = 1.0;

		// ---------------------------------------------------------
		// Static function definitions
		// ---------------------------------------------------------

		// ---------------------------------------------------------
		///
		/// Comparison function for sorting collision candidates
		///
		// ---------------------------------------------------------

		bool CollisionCandidateSetLT(const Vec3st& a, const Vec3st& b)
		{
			if (a[0] < b[0])
			{
				return true;
			}
			else if (a[0] == b[0])
			{
				if (a[1] < b[1])
				{
					return true;
				}
				else if (a[1] == b[1])
				{
					if (a[2] < b[2])
					{
						return true;
					}
				}
			}

			return false;

		}

		class CCollisionCandidateSetLT
		{
		public:
			bool operator () (const Vec3st& a, const Vec3st& b) const
			{
				return CollisionCandidateSetLT(a, b);
			}
		};

	}   // namespace


	// ---------------------------------------------------------
	// Member function definitions
	// ---------------------------------------------------------

	// ---------------------------------------------------------
	///
	/// Constructor
	///
	// ---------------------------------------------------------

	CollisionPipeline::CollisionPipeline(DynamicSurface& surface,
		BroadPhase& broadphase,
		double in_friction_coefficient) :
		m_friction_coefficient(in_friction_coefficient),
		m_surface(surface),
		m_broadphase(broadphase)
	{}



	// ---------------------------------------------------------
	///
	/// Apply an impulse along the given normal to the specified vertices, weighted by the given barycentric coordinates
	///
	// ---------------------------------------------------------

	void CollisionPipeline::apply_impulse(const Vec4d& alphas,
		const Vec4st& vertex_indices,
		double impulse_magnitude,
		const Vec3d& normal,
		double dt)
	{

		size_t e0 = vertex_indices[0];
		size_t e1 = vertex_indices[1];
		size_t e2 = vertex_indices[2];
		size_t e3 = vertex_indices[3];

		Vec3d& v0 = m_surface.m_velocities[e0];
		Vec3d& v1 = m_surface.m_velocities[e1];
		Vec3d& v2 = m_surface.m_velocities[e2];
		Vec3d& v3 = m_surface.m_velocities[e3];

		double s0 = alphas[0];
		double s1 = alphas[1];
		double s2 = alphas[2];
		double s3 = alphas[3];

		Vec3d pre_relative_velocity = s0 * v0 + s1 * v1 + s2 * v2 + s3 * v3;
		Vec3d pre_rv_normal = dot(normal, pre_relative_velocity) * normal;
		Vec3d pre_rv_tangential = pre_relative_velocity - pre_rv_normal;

		//
		// Friction
		//

		Vec3d post_relative_velocity = s0 * v0 + s1 * v1 + s2 * v2 + s3 * v3;
		Vec3d post_rv_normal = dot(normal, post_relative_velocity) * normal;
		double delta_rv_normal = mag(post_rv_normal - pre_rv_normal);
		double friction_impulse = min(m_friction_coefficient * delta_rv_normal, mag(pre_rv_tangential));

		Vec3d tan_collision_normal = -pre_rv_tangential;
		double mag_n = mag(tan_collision_normal);
		if (mag_n > 1e-8)
		{
			tan_collision_normal /= mag_n;
		}
		else
		{
			tan_collision_normal = Vec3d(0);
		}

		//    for (int j = 0; j < 3; j++)
		//    {
		//        double inv_m0 = 1.0 / m_surface.m_masses[e0][j];
		//        double inv_m1 = 1.0 / m_surface.m_masses[e1][j];
		//        double inv_m2 = 1.0 / m_surface.m_masses[e2][j];
		//        double inv_m3 = 1.0 / m_surface.m_masses[e3][j];
		//        
		//        double i = impulse_magnitude / (s0*s0*inv_m0 + s1*s1*inv_m1 + s2*s2*inv_m2 + s3*s3*inv_m3);
		//        
		//        if (i == DynamicSurface::solid_mass() || i == -DynamicSurface::solid_mass())
		//            continue;
		//        
		//        v0[j] += i*s0*inv_m0 * normal[j];
		//        v1[j] += i*s1*inv_m1 * normal[j];
		//        v2[j] += i*s2*inv_m2 * normal[j];
		//        v3[j] += i*s3*inv_m3 * normal[j];
		//
		//        if ( std::abs(i) > 100.0 / dt )
		//        {
		//            std::cout << "big impulse: " << i << " axis = " << j << " impulse magnitude = " << impulse_magnitude << " s = " << s0 << " " << s1 << " " << s2 << " " << s3 << " m = " << inv_m0 << " " << inv_m1 << " " << inv_m2 << " " << inv_m3 << std::endl;
		//            std::cout << "post impulse velocities: " << v0[j] << " " << v1[j] << " " << v2[j] << " " << v3[j] << std::endl;
		//        }
		//        
		//        //
		//        // Friction
		//        //
		//        
		//        double friction_i = friction_impulse / (s0*s0*inv_m0 + s1*s1*inv_m1 + s2*s2*inv_m2 + s3*s3*inv_m3);
		//        
		//        v0[j] += friction_i*s0*inv_m0 * tan_collision_normal[j];
		//        v1[j] += friction_i*s1*inv_m1 * tan_collision_normal[j];
		//        v2[j] += friction_i*s2*inv_m2 * tan_collision_normal[j];
		//        v3[j] += friction_i*s3*inv_m3 * tan_collision_normal[j];
		//    }

		double denom = 0;   // denom = n^T * (\sum w_i M_i^{-1}) * n

		for (int j = 0; j < 3; j++)
		{
			double inv_m0 = 1.0 / m_surface.m_masses[e0][j];
			double inv_m1 = 1.0 / m_surface.m_masses[e1][j];
			double inv_m2 = 1.0 / m_surface.m_masses[e2][j];
			double inv_m3 = 1.0 / m_surface.m_masses[e3][j];

			denom += normal[j] * normal[j] * (s0 * s0 * inv_m0 + s1 * s1 * inv_m1 + s2 * s2 * inv_m2 + s3 * s3 * inv_m3);
		}
		if (denom == 0) std::cout << "denom = 0: impulse_magnitude = " << impulse_magnitude << " vertices = " << e0 << " " << e1 << " " << e2 << " " << e3 << " s = " << s0 << " " << s1 << " " << s2 << " " << s3 << " normal = " << normal << " masses = " << m_surface.m_masses[e0] << "; " << m_surface.m_masses[e1] << "; " << m_surface.m_masses[e2] << "; " << m_surface.m_masses[e3] << std::endl;

		double i = impulse_magnitude / denom;

		for (int j = 0; j < 3; j++)
		{
			double inv_m0 = 1.0 / m_surface.m_masses[e0][j];
			double inv_m1 = 1.0 / m_surface.m_masses[e1][j];
			double inv_m2 = 1.0 / m_surface.m_masses[e2][j];
			double inv_m3 = 1.0 / m_surface.m_masses[e3][j];

			v0[j] += i * s0 * inv_m0 * normal[j];
			v1[j] += i * s1 * inv_m1 * normal[j];
			v2[j] += i * s2 * inv_m2 * normal[j];
			v3[j] += i * s3 * inv_m3 * normal[j];
		}

		// friction not implemented
		assert(m_friction_coefficient == 0);

		m_surface.set_newposition(e0, m_surface.get_position(e0) + dt * m_surface.m_velocities[e0]);
		m_surface.set_newposition(e1, m_surface.get_position(e1) + dt * m_surface.m_velocities[e1]);
		m_surface.set_newposition(e2, m_surface.get_position(e2) + dt * m_surface.m_velocities[e2]);
		m_surface.set_newposition(e3, m_surface.get_position(e3) + dt * m_surface.m_velocities[e3]);

		assert(m_surface.get_newposition(e0) == m_surface.get_newposition(e0));
		assert(m_surface.get_newposition(e1) == m_surface.get_newposition(e1));
		assert(m_surface.get_newposition(e2) == m_surface.get_newposition(e2));
		assert(m_surface.get_newposition(e3) == m_surface.get_newposition(e3));
	}


	// ---------------------------------------------------------
	///
	/// Apply an impulse between two edges 
	///
	// ---------------------------------------------------------

	void CollisionPipeline::apply_edge_edge_impulse(const Collision& collision, double impulse_magnitude, double dt)
	{
		assert(collision.m_is_edge_edge);

		double s0 = collision.m_alphas[0];
		double s1 = collision.m_alphas[1];
		double s2 = collision.m_alphas[2];
		double s3 = collision.m_alphas[3];

		Vec4d alphas;
		alphas[0] = s0;
		alphas[1] = s1;
		alphas[2] = -s2;
		alphas[3] = -s3;

		apply_impulse(alphas, collision.m_vertex_indices, impulse_magnitude, collision.m_normal, dt);


	}


	// ---------------------------------------------------------
	///
	/// Apply an impulse between a point and a triangle
	///
	// ---------------------------------------------------------


	void CollisionPipeline::apply_triangle_point_impulse(const Collision& collision, double impulse_magnitude, double dt)
	{
		assert(!collision.m_is_edge_edge);

		double s0 = collision.m_alphas[0];
		double s1 = collision.m_alphas[1];
		double s2 = collision.m_alphas[2];
		double s3 = collision.m_alphas[3];

		assert(s0 == 1.0);

		Vec4d alphas;
		alphas[0] = s0;
		alphas[1] = -s1;
		alphas[2] = -s2;
		alphas[3] = -s3;

		apply_impulse(alphas, collision.m_vertex_indices, impulse_magnitude, collision.m_normal, dt);

	}


	// ---------------------------------------------------------
	///
	/// Add point-triangle collision candidates for a specified triangle
	///
	// ---------------------------------------------------------

	void CollisionPipeline::add_triangle_candidates(size_t t,
		bool return_solid,
		bool return_dynamic,
		CollisionCandidateSet& collision_candidates)
	{
		Vec3d tmin, tmax;
		m_surface.triangle_continuous_bounds(t, tmin, tmax);

		static std::vector<size_t> candidate_vertices(10);
		candidate_vertices.clear();
		m_broadphase.get_potential_vertex_collisions(tmin, tmax, return_solid, return_dynamic, candidate_vertices);

		for (size_t j = 0; j < candidate_vertices.size(); j++)
		{
			collision_candidates.push_back(Vec3st(t, candidate_vertices[j], 0));
		}

	}

	// ---------------------------------------------------------
	///
	/// Add edge-edge collision candidates for a specified edge
	///
	// ---------------------------------------------------------

	void CollisionPipeline::add_edge_candidates(size_t e,
		bool return_solid,
		bool return_dynamic,
		CollisionCandidateSet& collision_candidates)
	{
		Vec3d emin, emax;
		m_surface.edge_continuous_bounds(e, emin, emax);

		static std::vector<size_t> candidate_edges(10);
		candidate_edges.clear();
		m_broadphase.get_potential_edge_collisions(emin, emax, return_solid, return_dynamic, candidate_edges);

		for (size_t j = 0; j < candidate_edges.size(); j++)
		{
			collision_candidates.push_back(Vec3st(e, candidate_edges[j], 1));
		}
	}

	// ---------------------------------------------------------
	///
	/// Add point-triangle collision candidates for a specified vertex
	///
	// ---------------------------------------------------------

	void CollisionPipeline::add_point_candidates(size_t v,
		bool return_solid,
		bool return_dynamic,
		CollisionCandidateSet& collision_candidates)
	{
		Vec3d vmin, vmax;
		m_surface.vertex_continuous_bounds(v, vmin, vmax);

		static std::vector<size_t> candidate_triangles(10);
		candidate_triangles.clear();
		m_broadphase.get_potential_triangle_collisions(vmin, vmax, return_solid, return_dynamic, candidate_triangles);

		for (size_t j = 0; j < candidate_triangles.size(); j++)
		{
			collision_candidates.push_back(Vec3st(candidate_triangles[j], v, 0));
		}
	}

	// ---------------------------------------------------------
	///
	/// Add collision candidates for a specified vertex and all elements incident on the vertex
	///
	// ---------------------------------------------------------

	void CollisionPipeline::add_point_update_candidates(size_t v,
		CollisionCandidateSet& collision_candidates)
	{

		// Avoid solid-vs-solid tests during sequential impulses phase

		if (m_surface.vertex_is_all_solid(v)) { return; }

		add_point_candidates(v, true, true, collision_candidates);

		std::vector<size_t>& incident_triangles = m_surface.m_mesh.m_vertex_to_triangle_map[v];
		std::vector<size_t>& incident_edges = m_surface.m_mesh.m_vertex_to_edge_map[v];

		for (size_t i = 0; i < incident_triangles.size(); i++)
		{
			add_triangle_candidates(incident_triangles[i], true, true, collision_candidates);
		}

		for (size_t i = 0; i < incident_edges.size(); i++)
		{
			add_edge_candidates(incident_edges[i], true, true, collision_candidates);
		}

	}


	// =========================================================
	//
	// PROXIMITIES
	//
	// =========================================================

	// ---------------------------------------------------------
	///
	/// Apply impulses to all proximal elements in the list of potentially proximal elements
	///
	// ---------------------------------------------------------

	void CollisionPipeline::process_proximity_candidates(double dt,
		CollisionCandidateSet& candidates)
	{

		static const double k = 10.0;

		while (false == candidates.empty())
		{

			Vec3st candidate = candidates.back();
			candidates.pop_back();

			if (candidate[2] == 1)
			{
				// edge-edge

				Vec2st e0 = m_surface.m_mesh.m_edges[candidate[0]];
				Vec2st e1 = m_surface.m_mesh.m_edges[candidate[1]];

				if (e0[0] == e0[1]) { continue; }
				if (e1[0] == e1[1]) { continue; }

				if (e0[0] != e1[0] && e0[0] != e1[1] && e0[1] != e1[0] && e0[1] != e1[1])
				{
					double distance, s0, s2;
					Vec3d normal;

					check_edge_edge_proximity(m_surface.get_position(e0[0]),
						m_surface.get_position(e0[1]),
						m_surface.get_position(e1[0]),
						m_surface.get_position(e1[1]),
						distance, s0, s2, normal);

					if (distance < m_surface.m_proximity_epsilon && distance > 0)
					{
						assert(mag(normal) > 0);

						double relvel = dot(normal,
							s0 * m_surface.m_velocities[e0[0]] +
							(1.0 - s0) * m_surface.m_velocities[e0[1]] -
							s2 * m_surface.m_velocities[e1[0]] -
							(1.0 - s2) * m_surface.m_velocities[e1[1]]);

						Vec3d diff = s0 * m_surface.get_position(e0[0]) +
							(1.0 - s0) * m_surface.get_position(e0[1]) -
							s2 * m_surface.get_position(e1[0]) -
							(1.0 - s2) * m_surface.get_position(e1[1]);

						if (dot(normal, diff) < 0.0)
						{
							continue;
						}

						double d = m_surface.m_proximity_epsilon - distance;

						if (relvel > 0.1 * d / dt)
						{
							continue;
						}

						double impulse1 = max(0.0, 0.1 * d / dt - relvel);

						double impulse2 = dt * k * d;

						double impulse = min(impulse1, impulse2);

						Collision proximity(true,
							Vec4st(e0[0], e0[1], e1[0], e1[1]),
							normal,
							Vec4d(s0, 1.0 - s0, s2, 1.0 - s2),
							dt * relvel);

						apply_edge_edge_impulse(proximity, impulse, dt);

					}
				}

			}
			else
			{
				// point-triangle

				size_t t = candidate[0];
				const Vec3st& tri = m_surface.m_mesh.get_triangle(t);
				size_t v = candidate[1];

				if (tri[0] != v && tri[1] != v && tri[2] != v)
				{
					double distance, s1, s2, s3;
					Vec3d normal;

					check_point_triangle_proximity(m_surface.get_position(v),
						m_surface.get_position(tri[0]),
						m_surface.get_position(tri[1]),
						m_surface.get_position(tri[2]),
						distance, s1, s2, s3, normal);

					if (distance == 0)
					{
						double s1, s2, s3;
						LosTopos::Vec3d normal;
						double rel_disp;

						LosTopos::Vec3d a = m_surface.get_position(tri[0]);
						LosTopos::Vec3d b = m_surface.get_position(tri[1]);
						LosTopos::Vec3d c = m_surface.get_position(tri[2]);
						LosTopos::Vec3d d = m_surface.get_position(v);
						bool col = LosTopos::point_triangle_collision(d, d, 0, a, a, 1, b, b, 2, c, c, 3, s1, s2, s3, normal, rel_disp);
						std::cout << "collision = " << col << std::endl;
						std::cout << "s = " << s1 << " " << s2 << " " << s3 << std::endl;
						std::cout << "normal = " << normal << " rel_disp = " << rel_disp << std::endl;
						LosTopos::Vec3d x1 = a;
						LosTopos::Vec3d x2 = b;
						LosTopos::Vec3d x3 = c;
						LosTopos::Vec3d x0 = d;
						std::cout << "cross = " << cross(x3 - x2, x0 - x2) << std::endl;
						Vec3d dx(x3 - x2);
						double m2 = mag2(dx);
						double s = clamp(dot(x3 - x0, dx) / m2, 0., 1.);
						normal = x0 - (s * x2 + (1 - s) * x3);
						std::cout << "normal = " << normal << " mag = " << mag(normal) << std::endl;
					}


					if (distance < m_surface.m_proximity_epsilon && distance > 0)
					{
						assert(mag(normal) > 0);

						double relvel = dot(normal,
							m_surface.m_velocities[v] -
							(s1 * m_surface.m_velocities[tri[0]] +
								s2 * m_surface.m_velocities[tri[1]] +
								s3 * m_surface.m_velocities[tri[2]]));

						Vec3d diff = m_surface.get_position(v) -
							(s1 * m_surface.get_position(tri[0]) +
								s2 * m_surface.get_position(tri[1]) +
								s3 * m_surface.get_position(tri[2]));

						if (dot(normal, diff) < 0.0)
						{
							continue;
						}

						double d = m_surface.m_proximity_epsilon - distance;

						if (relvel > 0.1 * d / dt)
						{
							continue;
						}

						double impulse1 = max(0.0, 0.1 * d / dt - relvel);

						double impulse2 = dt * k * d;

						double impulse = min(impulse1, impulse2);

						Collision proximity(false,
							Vec4st(v, tri[0], tri[1], tri[2]),
							normal,
							Vec4d(1.0, s1, s2, s3),
							dt * relvel);

						apply_triangle_point_impulse(proximity, impulse, dt);

					}
				}
			}
		}

	}

	// ---------------------------------------------------------
	///
	/// Handle dynamic point vs. solid triangle proximities
	///
	// ---------------------------------------------------------

	void CollisionPipeline::dynamic_point_vs_solid_triangle_proximities(double dt)
	{
		// dynamic point vs solid triangles

		static CollisionCandidateSet point_collision_candidates;
		point_collision_candidates.clear();
		for (size_t i = 0; i < m_surface.get_num_vertices(); ++i)
		{
			if (m_surface.vertex_is_all_solid(i))
			{
				continue;
			}

			// check vs solid triangles
			add_point_candidates(i, true, false, point_collision_candidates);
		}

		process_proximity_candidates(dt, point_collision_candidates);

	}

	// ---------------------------------------------------------
	///
	/// Handle dynamic triangle vs. all points proximities
	///
	// ---------------------------------------------------------

	void CollisionPipeline::dynamic_triangle_vs_all_point_proximities(double dt)
	{

		static CollisionCandidateSet triangle_collision_candidates;
		triangle_collision_candidates.clear();

		for (size_t i = 0; i < m_surface.m_mesh.num_triangles(); ++i)
		{
			if (m_surface.triangle_is_all_solid(i))
			{
				continue;
			}

			// check vs all points

			add_triangle_candidates(i, true, true, triangle_collision_candidates);
		}

		process_proximity_candidates(dt, triangle_collision_candidates);

	}


	// ---------------------------------------------------------
	///
	/// Handle dynamic edge vs. all other edges proximities
	///
	// ---------------------------------------------------------

	void CollisionPipeline::dynamic_edge_vs_all_edge_proximities(double dt)
	{

		static CollisionCandidateSet edge_collision_candidates;
		edge_collision_candidates.clear();

		for (size_t i = 0; i < m_surface.m_mesh.m_edges.size(); ++i)
		{
			if (m_surface.edge_is_all_solid(i))
			{
				continue;
			}

			// check vs all edges
			add_edge_candidates(i, true, true, edge_collision_candidates);
		}

		process_proximity_candidates(dt, edge_collision_candidates);

	}

	// ---------------------------------------------------------
	///
	/// Detect all proximities and apply repulsion forces
	///
	// ---------------------------------------------------------

	void CollisionPipeline::handle_proximities(double dt)
	{

		// dynamic point vs solid triangles

		dynamic_point_vs_solid_triangle_proximities(dt);

		// dynamic triangle vs static points
		// dynamic triangle vs dynamic points

		dynamic_triangle_vs_all_point_proximities(dt);

		// dynamic edge vs static edges
		// dynamic edge vs dynamic edges

		dynamic_edge_vs_all_edge_proximities(dt);

	}


	// =========================================================
	//
	// COLLISIONS
	//
	// =========================================================

	// ---------------------------------------------------------
	///
	/// Detect is there is a collision between two edges (continuous)
	///
	// ---------------------------------------------------------

	bool CollisionPipeline::detect_segment_segment_collision(const Vec3st& candidate, Collision& collision)
	{

		assert(candidate[2] == 1);

		Vec2st e0 = m_surface.m_mesh.m_edges[candidate[0]];
		Vec2st e1 = m_surface.m_mesh.m_edges[candidate[1]];

		if (e0[0] == e0[1]) { return false; }
		if (e1[0] == e1[1]) { return false; }

		if (e0[0] == e1[0] || e0[0] == e1[1] || e0[1] == e1[0] || e0[1] == e1[1])
		{
			return false;
		}

		if (e0[1] < e0[0]) { swap(e0[0], e0[1]); }
		if (e1[1] < e1[0]) { swap(e1[0], e1[1]); }


		if (m_surface.edge_is_all_solid(candidate[0]) && m_surface.edge_is_all_solid(candidate[1]))
		{
			return false;
		}

		double s0, s2, rel_disp;
		Vec3d normal;

		size_t a = e0[0];
		size_t b = e0[1];
		size_t c = e1[0];
		size_t d = e1[1];

		if (segment_segment_collision(m_surface.get_position(a), m_surface.get_newposition(a), a,
			m_surface.get_position(b), m_surface.get_newposition(b), b,
			m_surface.get_position(c), m_surface.get_newposition(c), c,
			m_surface.get_position(d), m_surface.get_newposition(d), d,
			s0, s2, normal, rel_disp))
		{
			collision = Collision(true, Vec4st(a, b, c, d), normal, Vec4d(s0, (1 - s0), s2, (1 - s2)), rel_disp);
			return true;
		}

		return false;

	}

	// ---------------------------------------------------------
	///
	/// Detect is there is a collision between a point and a triangle (continuous)
	///
	// ---------------------------------------------------------


	bool CollisionPipeline::detect_point_triangle_collision(const Vec3st& candidate, Collision& collision)
	{
		assert(candidate[2] == 0);

		size_t t = candidate[0];
		const Vec3st& tri = m_surface.m_mesh.get_triangle(candidate[0]);
		size_t v = candidate[1];

		if (tri[0] == v || tri[1] == v || tri[2] == v)
		{
			return false;
		}


		if (m_surface.triangle_is_all_solid(t) && m_surface.vertex_is_all_solid(v))
		{
			//in old pipeline, all solid is not handled
			//in free air pipeline, all solid are treated as heavy vertices
			//and collisions are treated as normal
			//return false;
		}

		double s1, s2, s3, rel_disp;
		Vec3d normal;
		Vec3st sorted_tri = sort_triangle(tri);
		if (point_triangle_collision(m_surface.get_position(v), m_surface.get_newposition(v), v,
			m_surface.get_position(sorted_tri[0]), m_surface.get_newposition(sorted_tri[0]), sorted_tri[0],
			m_surface.get_position(sorted_tri[1]), m_surface.get_newposition(sorted_tri[1]), sorted_tri[1],
			m_surface.get_position(sorted_tri[2]), m_surface.get_newposition(sorted_tri[2]), sorted_tri[2],
			s1, s2, s3, normal, rel_disp))

		{
			collision = Collision(false, Vec4st(v, sorted_tri[0], sorted_tri[1], sorted_tri[2]), normal, Vec4d(1, s1, s2, s3), rel_disp);
			return true;
		}

		return false;

	}

	// ---------------------------------------------------------
	///
	/// Run collision detection on all given collision candidates, and apply 
	///
	// ---------------------------------------------------------

	void CollisionPipeline::process_collision_candidates(double dt,
		CollisionCandidateSet& candidates,
		bool add_to_new_candidates,
		CollisionCandidateSet& new_candidates,
		ProcessCollisionStatus& status)
	{

		size_t max_iteration = 5 * candidates.size();
		size_t i = 0;

		static const size_t MAX_CANDIDATES = 1000000;

		while (false == candidates.empty() && i++ < max_iteration)
		{

			Vec3st candidate = candidates.back();
			candidates.pop_back();

			if (candidate[2] == 1)
			{
				// edge-edge
				Collision collision;
				if (detect_segment_segment_collision(candidate, collision))
				{

					g_stats.add_to_int("CollisionPipeline::total_num_collisions", 1);

					double relvel = collision.m_relative_displacement / dt;
					double desired_relative_velocity = 0.0;
					double impulse = IMPULSE_MULTIPLIER * (desired_relative_velocity - relvel);
					apply_edge_edge_impulse(collision, impulse, dt);

					status.collision_found = true;

					if (new_candidates.size() > MAX_CANDIDATES)
					{
						status.overflow = true;
					}

					if (!status.overflow && add_to_new_candidates)
					{
						add_point_update_candidates(collision.m_vertex_indices[0], new_candidates);
						add_point_update_candidates(collision.m_vertex_indices[1], new_candidates);
						add_point_update_candidates(collision.m_vertex_indices[2], new_candidates);
						add_point_update_candidates(collision.m_vertex_indices[3], new_candidates);
					}
				}
			}
			else
			{
				// point-triangle
				
				Collision collision;
				if (detect_point_triangle_collision(candidate, collision))
				{
					
					g_stats.add_to_int("CollisionPipeline::total_num_collisions", 1);

					double relvel = collision.m_relative_displacement / dt;
					double desired_relative_velocity = 0.0;
					double impulse = IMPULSE_MULTIPLIER * (desired_relative_velocity - relvel);
					apply_triangle_point_impulse(collision, impulse, dt);
					
					status.collision_found = true;

					if (new_candidates.size() > MAX_CANDIDATES)
					{
						status.overflow = true;
					}

					if (!status.overflow && add_to_new_candidates)
					{
						add_point_update_candidates(collision.m_vertex_indices[0], new_candidates);
						add_point_update_candidates(collision.m_vertex_indices[1], new_candidates);
						add_point_update_candidates(collision.m_vertex_indices[2], new_candidates);
						add_point_update_candidates(collision.m_vertex_indices[3], new_candidates);
					}
				}

			}
		}

		if (m_surface.m_verbose && max_iteration > 0 && i >= max_iteration)
		{
			std::cout << "CollisionPipeline::process_collision_candidates: max_iteration reached" << std::endl;
		}

		status.all_candidates_processed = candidates.empty();

	}

	// ---------------------------------------------------------
	///
	/// Run collision detection on all given collision candidates
	///
	// ---------------------------------------------------------

	void CollisionPipeline::test_collision_candidates(CollisionCandidateSet& candidates,
		std::vector<Collision>& collisions,
		ProcessCollisionStatus& status)
	{

		const size_t MAX_COLLISIONS = 5000;

		while (false == candidates.empty())
		{

			Vec3st candidate = candidates.back();
			candidates.pop_back();

			if (candidate[2] == 1)
			{
				// edge-edge
				Collision collision;
				if (detect_segment_segment_collision(candidate, collision))
				{
					status.collision_found = true;

					collisions.push_back(collision);

					if (collisions.size() > MAX_COLLISIONS)
					{
						status.overflow = true;
						status.all_candidates_processed = false;
						return;
					}
				}

			}
			else
			{
				// point-triangle

				Collision collision;
				if (detect_point_triangle_collision(candidate, collision))
				{
					status.collision_found = true;

					collisions.push_back(collision);

					if (collisions.size() > MAX_COLLISIONS)
					{
						status.overflow = true;
						status.all_candidates_processed = false;
						return;
					}
				}

			}
		}


		status.all_candidates_processed = true;

		assert(status.all_candidates_processed == !status.overflow);

	}

	// ---------------------------------------------------------
	///
	/// Check if any collision exists in the set of candidates.  Stop when the first collision is found.
	///
	// ---------------------------------------------------------

	bool CollisionPipeline::any_collision(CollisionCandidateSet& candidates, Collision& collision)
	{

		CollisionCandidateSet::iterator iter = candidates.begin();

		for (; iter != candidates.end(); ++iter)
		{

			Vec3st candidate = *iter;

			if (candidate[2] == 1)
			{
				// edge-edge
				if (detect_segment_segment_collision(candidate, collision))
				{
					return true;
				}
			}
			else
			{
				// point-triangle
				if (detect_point_triangle_collision(candidate, collision))
				{
					return true;
				}
			}
		}

		return false;

	}


	// ---------------------------------------------------------
	///
	/// Check for collisions between dynamic points and solid triangles
	///
	// ---------------------------------------------------------

	void CollisionPipeline::dynamic_point_vs_solid_triangle_collisions(double dt,
		bool collect_candidates,
		CollisionCandidateSet& update_collision_candidates,
		ProcessCollisionStatus& status)
	{
		// dynamic point vs solid triangles

		for (size_t i = 0; i < m_surface.get_num_vertices(); ++i)
		{
			if (m_surface.vertex_is_all_solid(i))
			{
				continue;
			}

			static CollisionCandidateSet point_collision_candidates;
			point_collision_candidates.clear();

			// check vs solid triangles
			add_point_candidates(i, true, false, point_collision_candidates);

			process_collision_candidates(dt,
				point_collision_candidates,
				collect_candidates,
				update_collision_candidates,
				status);

		}

	}

	// ---------------------------------------------------------
	///
	/// Check for collisions between dynamic triangles and all points
	///
	// ---------------------------------------------------------

	void CollisionPipeline::dynamic_triangle_vs_all_point_collisions(double dt,
		bool collect_candidates,
		CollisionCandidateSet& update_collision_candidates,
		ProcessCollisionStatus& status)
	{

		for (size_t i = 0; i < m_surface.m_mesh.num_triangles(); ++i)
		{
			
			if (m_surface.triangle_is_all_solid(i))
			{
				continue;
			}

			static CollisionCandidateSet triangle_collision_candidates;
			triangle_collision_candidates.clear();

			// check vs all points
			add_triangle_candidates(i, true, true, triangle_collision_candidates);
			
			process_collision_candidates(dt,
				triangle_collision_candidates,
				collect_candidates,
				update_collision_candidates,
				status);
			
		}

	}


	// ---------------------------------------------------------
	///
	/// Check for collisions between dynamic edges and all other edges 
	///
	// ---------------------------------------------------------

	void CollisionPipeline::dynamic_edge_vs_all_edge_collisions(double dt,
		bool collect_candidates,
		CollisionCandidateSet& update_collision_candidates,
		ProcessCollisionStatus& status)
	{




		for (size_t i = 0; i < m_surface.m_mesh.m_edges.size(); ++i)
		{
			if (m_surface.edge_is_all_solid(i))
			{
				continue;
			}


			static CollisionCandidateSet edge_collision_candidates;
			edge_collision_candidates.clear();

			// check vs all edges
			add_edge_candidates(i, true, true, edge_collision_candidates);

			process_collision_candidates(dt,
				edge_collision_candidates,
				collect_candidates,
				update_collision_candidates,
				status);

		}

	}


	// ---------------------------------------------------------
	///
	/// Detect and fix all collisions, sweeping over all mesh primitives a number of times.
	///
	// ---------------------------------------------------------

	bool CollisionPipeline::handle_collisions(double dt)
	{

		bool verbose = m_surface.m_verbose;

		static const int MAX_PASS = 3;

		CollisionCandidateSet update_collision_candidates;

		//m_surface.check_continuous_broad_phase_is_up_to_date();

		for (int pass = 0; pass < MAX_PASS; ++pass)
		{
			// if last time through the loop, fill out the update_collision_candidates array
			// when the loop exits, we will wind down this array

			bool collect_candidates = (pass == (MAX_PASS - 1));

			bool collision_found = false;
			ProcessCollisionStatus status;

			status.overflow = false;
			status.collision_found = false;
			status.all_candidates_processed = false;

			// dynamic point vs solid triangles
			//CSim::TimerMan::timer("Sim.step/BPS.step/advection/st_integrate/handle_collisions/point_solidtri").start();
			dynamic_point_vs_solid_triangle_collisions(dt,
				collect_candidates,
				update_collision_candidates,
				status);
			//CSim::TimerMan::timer("Sim.step/BPS.step/advection/st_integrate/handle_collisions/point_solidtri").stop();
			collision_found |= status.collision_found;
			if (verbose)
			{
				std::cout << "collision pass " << pass << " 1, dynamic_point_vs_solid_triangle_collisions" << std::endl;
			}
			// dynamic triangle vs static points
			// dynamic triangle vs dynamic points
			//CSim::TimerMan::timer("Sim.step/BPS.step/advection/st_integrate/handle_collisions/trivsall").start();
			dynamic_triangle_vs_all_point_collisions(dt,
				collect_candidates,
				update_collision_candidates,
				status);
			//CSim::TimerMan::timer("Sim.step/BPS.step/advection/st_integrate/handle_collisions/trivsall").stop();
			collision_found |= status.collision_found;
			if (verbose)
			{
				std::cout << "collision pass " << pass << " 2, dynamic_triangle_vs_all_point_collisions" << std::endl;
			}
			// dynamic edge vs static edges
			// dynamic edge vs dynamic edges
			//CSim::TimerMan::timer("Sim.step/BPS.step/advection/st_integrate/handle_collisions/ee").start();
			dynamic_edge_vs_all_edge_collisions(dt,
				collect_candidates,
				update_collision_candidates,
				status);
			//CSim::TimerMan::timer("Sim.step/BPS.step/advection/st_integrate/handle_collisions/ee").stop();
			collision_found |= status.collision_found;
			if (verbose)
			{
				std::cout << "collision pass " << pass << " 3, dynamic_edge_vs_all_edge_collisions" << std::endl;
			}
			if (status.overflow)
			{
				if (verbose)
				{
					std::cout << "overflow, returning early" << std::endl;
				}
				return false;
			}

			if (!collision_found)
			{
				if (verbose)
				{
					std::cout << "no collision found this pass, returning early" << std::endl;
				}
				return true;
			}

			if (verbose)
			{
				std::cout << "collision pass " << pass << " completed" << std::endl;
			}

		}
		if (verbose) {
			printf("Find collisions complete\n");
		}
		// Unique-ify the remaining list of candidates
	//    std::sort( update_collision_candidates.begin(), update_collision_candidates.end(), CollisionCandidateSetLT );
	//    CollisionCandidateSet::iterator new_end = std::unique(update_collision_candidates.begin(), update_collision_candidates.end());
	//    update_collision_candidates.erase(new_end, update_collision_candidates.end());
		//CSim::TimerMan::timer("Sim.step/BPS.step/advection/st_integrate/handle_collisions/sort").start();
		std::set<Vec3st, CCollisionCandidateSetLT> candidate_set(update_collision_candidates.begin(), update_collision_candidates.end());
		update_collision_candidates.assign(candidate_set.begin(), candidate_set.end());
		//CSim::TimerMan::timer("Sim.step/BPS.step/advection/st_integrate/handle_collisions/sort").stop();
		//CSim::TimerMan::timer("Sim.step/BPS.step/advection/st_integrate/handle_collisions/process").start();
		// now wind down the update_collision_candidates list
		ProcessCollisionStatus status;
		status.overflow = false;
		process_collision_candidates(dt, update_collision_candidates, true, update_collision_candidates, status);
		//CSim::TimerMan::timer("Sim.step/BPS.step/advection/st_integrate/handle_collisions/process").stop();
		bool ok = status.all_candidates_processed;

		if (m_surface.m_verbose && !ok)
		{
			std::cout << "Didn't resolve all collisions" << std::endl;
			g_stats.add_to_int("CollisionPipeline::handle_collisions:didnt_resolve_all_collisions", 1);
		}

		if (status.overflow)
		{
			ok = false;
			if (m_surface.m_verbose)
			{
				std::cout << "overflowed candidate list" << std::endl;
				g_stats.add_to_int("CollisionPipeline::handle_collisions:overflowed_candidate_list", 1);
			}
		}

		return ok;

	}


	// ---------------------------------------------------------
	///
	/// Check all elements for collisions, returning a vector of all collisions
	///
	// ---------------------------------------------------------

	bool CollisionPipeline::detect_collisions(std::vector<Collision>& collisions)
	{
		//m_surface.check_continuous_broad_phase_is_up_to_date();

		CollisionCandidateSet collision_candidates;

		// dynamic point vs solid triangles

		for (size_t i = 0; i < m_surface.get_num_vertices(); ++i)
		{
			if (m_surface.vertex_is_all_solid(i))
			{
				continue;
			}

			// check vs solid triangles
			add_point_candidates(i, true, false, collision_candidates);
		}

		// dynamic triangles vs all points

		for (size_t i = 0; i < m_surface.m_mesh.num_triangles(); ++i)
		{
			if (m_surface.triangle_is_all_solid(i))
			{
				continue;
			}

			// check vs all points
			add_triangle_candidates(i, true, true, collision_candidates);

		}

		// dynamic edges vs all edges

		for (size_t i = 0; i < m_surface.m_mesh.m_edges.size(); ++i)
		{
			if (m_surface.edge_is_all_solid(i))
			{
				continue;
			}

			// check vs all edges
			add_edge_candidates(i, true, true, collision_candidates);
		}


		//
		// Run narrow phase collision detection on all candidates
		//

		ProcessCollisionStatus status;
		test_collision_candidates(collision_candidates,
			collisions,
			status);

		// Check if all collisions were tested

		if (!status.all_candidates_processed)
		{
			assert(status.overflow);
			return false;
		}

		return true;

	}


	// ---------------------------------------------------------
	///
	/// Detect continuous collisions among elements in the given ImpactZones, and adjacent to the given ImpactZones.
	///
	// ---------------------------------------------------------

	bool CollisionPipeline::detect_new_collisions(const std::vector<ImpactZone> impact_zones, std::vector<Collision>& collisions)
	{
		//m_surface.check_continuous_broad_phase_is_up_to_date();

		std::vector<size_t> zone_vertices;
		std::vector<size_t> zone_edges;
		std::vector<size_t> zone_triangles;

		// Get all vertices in the impact zone

		for (size_t i = 0; i < impact_zones.size(); ++i)
		{
			for (size_t j = 0; j < impact_zones[i].m_collisions.size(); ++j)
			{
				add_unique(zone_vertices, impact_zones[i].m_collisions[j].m_vertex_indices[0]);
				add_unique(zone_vertices, impact_zones[i].m_collisions[j].m_vertex_indices[1]);
				add_unique(zone_vertices, impact_zones[i].m_collisions[j].m_vertex_indices[2]);
				add_unique(zone_vertices, impact_zones[i].m_collisions[j].m_vertex_indices[3]);
			}
		}

		// Get all triangles in the impact zone

		const NonDestructiveTriMesh& mesh = m_surface.m_mesh;

		for (size_t i = 0; i < zone_vertices.size(); ++i)
		{
			for (size_t j = 0; j < mesh.m_vertex_to_triangle_map[zone_vertices[i]].size(); ++j)
			{
				add_unique(zone_triangles, mesh.m_vertex_to_triangle_map[zone_vertices[i]][j]);
			}
		}

		// Get all edges in the impact zone

		for (size_t i = 0; i < zone_vertices.size(); ++i)
		{
			for (size_t j = 0; j < mesh.m_vertex_to_edge_map[zone_vertices[i]].size(); ++j)
			{
				add_unique(zone_edges, mesh.m_vertex_to_edge_map[zone_vertices[i]][j]);
			}
		}

		CollisionCandidateSet collision_candidates;


		// Check dynamic point vs all triangles

		for (size_t j = 0; j < zone_vertices.size(); ++j)
		{
			size_t vertex_index = zone_vertices[j];

			// check vs all triangles
			add_point_candidates(vertex_index, true, true, collision_candidates);

		}


		// dynamic triangles vs all points

		for (size_t j = 0; j < zone_triangles.size(); ++j)
		{
			size_t triangle_index = zone_triangles[j];

			// check vs all points
			add_triangle_candidates(triangle_index, true, true, collision_candidates);

		}


		// dynamic edges vs all edges

		for (size_t j = 0; j < zone_edges.size(); ++j)
		{
			size_t edge_index = zone_edges[j];

			// check vs all edges
			add_edge_candidates(edge_index, true, true, collision_candidates);
		}


		//
		// Run narrow phase collision detection on all candidates
		//

		ProcessCollisionStatus status;
		status.overflow = false;

		test_collision_candidates(collision_candidates,
			collisions,
			status);

		// Check if all collisions were tested

		if (!status.all_candidates_processed || status.overflow)
		{
			return false;
		}

		return true;

	}


	// ---------------------------------------------------------
	///
	/// Detect collisions between elements of the given edge and triangle
	///
	// ---------------------------------------------------------

	void CollisionPipeline::detect_collisions(size_t edge_index,
		size_t triangle_index,
		std::vector<Collision>& collisions)

	{

		size_t e0 = m_surface.m_mesh.m_edges[edge_index][0];
		size_t e1 = m_surface.m_mesh.m_edges[edge_index][1];
		if (e1 < e0) { swap(e0, e1); }

		Vec3st tri = m_surface.m_mesh.get_triangle(triangle_index);
		tri = sort_triangle(tri);

		size_t t0 = tri[0];
		size_t t1 = tri[1];
		size_t t2 = tri[2];

		double s0, s1, s2, s3, rel_disp;
		Vec3d normal;


		simplex_verbose = true;

		// edge vs triangle edge 0

		if (segment_segment_collision(m_surface.get_position(e0), m_surface.get_newposition(e0), e0,
			m_surface.get_position(e1), m_surface.get_newposition(e1), e1,
			m_surface.get_position(t0), m_surface.get_newposition(t0), t0,
			m_surface.get_position(t1), m_surface.get_newposition(t1), t1,
			s0, s2, normal, rel_disp))

		{
			Collision new_collision(true, Vec4st(e0, e1, t0, t1), normal, Vec4d(s0, (1 - s0), s2, (1 - s2)), rel_disp);
			collisions.push_back(new_collision);

			size_t edge1 = m_surface.m_mesh.get_edge_index(t0, t1);
			assert(edge1 < m_surface.m_mesh.m_edges.size());
			Vec3st check_candidate(edge_index, edge1, 1);
			Collision check_collision;
			bool check_hit = detect_segment_segment_collision(check_candidate, check_collision);
			assert(check_hit);

		}

		// edge vs triangle edge 1

		if (segment_segment_collision(m_surface.get_position(e0), m_surface.get_newposition(e0), e0,
			m_surface.get_position(e1), m_surface.get_newposition(e1), e1,
			m_surface.get_position(t1), m_surface.get_newposition(t1), t1,
			m_surface.get_position(t2), m_surface.get_newposition(t2), t2,
			s0, s2, normal, rel_disp))

		{
			Collision new_collision(true, Vec4st(e0, e1, t1, t2), normal, Vec4d(s0, (1 - s0), s2, (1 - s2)), rel_disp);
			collisions.push_back(new_collision);

			size_t edge1 = m_surface.m_mesh.get_edge_index(t1, t2);
			assert(edge1 < m_surface.m_mesh.m_edges.size());
			Vec3st check_candidate(edge_index, edge1, 1);
			Collision check_collision;
			bool check_hit = detect_segment_segment_collision(check_candidate, check_collision);
			assert(check_hit);

		}


		// edge vs triangle edge 2


		if (segment_segment_collision(m_surface.get_position(e0), m_surface.get_newposition(e0), e0,
			m_surface.get_position(e1), m_surface.get_newposition(e1), e1,
			m_surface.get_position(t0), m_surface.get_newposition(t0), t0,
			m_surface.get_position(t2), m_surface.get_newposition(t2), t2,
			s0, s2, normal, rel_disp))

		{
			Collision new_collision(true, Vec4st(e0, e1, t2, t0), normal, Vec4d(s0, (1 - s0), s2, (1 - s2)), rel_disp);
			collisions.push_back(new_collision);

			size_t edge1 = m_surface.m_mesh.get_edge_index(t2, t0);
			assert(edge1 < m_surface.m_mesh.m_edges.size());
			Vec3st check_candidate(edge_index, edge1, 1);
			Collision check_collision;
			bool check_hit = detect_segment_segment_collision(check_candidate, check_collision);
			assert(check_hit);

		}

		// edge point 0 vs triangle

		if (point_triangle_collision(m_surface.get_position(e0), m_surface.get_newposition(e0), e0,
			m_surface.get_position(t0), m_surface.get_newposition(t0), t0,
			m_surface.get_position(t1), m_surface.get_newposition(t1), t1,
			m_surface.get_position(t2), m_surface.get_newposition(t2), t2,
			s1, s2, s3, normal, rel_disp))
		{
			Collision new_collision(false, Vec4st(e0, t0, t1, t2), normal, Vec4d(1.0, s1, s2, s3), rel_disp);
			collisions.push_back(new_collision);
		}

		// edge point 1 vs triangle

		if (point_triangle_collision(m_surface.get_position(e1), m_surface.get_newposition(e1), e1,
			m_surface.get_position(t0), m_surface.get_newposition(t0), t0,
			m_surface.get_position(t1), m_surface.get_newposition(t1), t1,
			m_surface.get_position(t2), m_surface.get_newposition(t2), t2,
			s1, s2, s3, normal, rel_disp))
		{
			Collision new_collision(false, Vec4st(e1, t0, t1, t2), normal, Vec4d(1.0, s1, s2, s3), rel_disp);
			collisions.push_back(new_collision);
		}



		simplex_verbose = false;

		// ------

		for (size_t i = 0; i < collisions.size(); ++i)
		{
			Collision& coll = collisions[i];

			std::cout << "\n ======== Collision: is_edge_edge: " << coll.m_is_edge_edge << ", indices: " << coll.m_vertex_indices << std::endl;

			if (coll.m_is_edge_edge)
			{

				size_t edge0 = m_surface.m_mesh.get_edge_index(coll.m_vertex_indices[0], coll.m_vertex_indices[1]);
				std::cout << "edge0: " << edge0 << std::endl;

				size_t edge1 = m_surface.m_mesh.get_edge_index(coll.m_vertex_indices[2], coll.m_vertex_indices[3]);
				std::cout << "edge1: " << edge1 << std::endl;

				Vec3st check_candidate(edge0, edge1, 1);
				Collision check_collision;
				bool check_hit = detect_segment_segment_collision(check_candidate, check_collision);
				assert(check_hit);


				CollisionCandidateSet collision_candidates0;
				add_edge_candidates(edge0, true, true, collision_candidates0);
				bool edge1_found = false;
				for (size_t j = 0; j < collision_candidates0.size(); ++j)
				{
					if (collision_candidates0[j][0] == edge1 || collision_candidates0[j][1] == edge1)
					{
						edge1_found = true;
					}
				}

				if (!edge1_found) { std::cout << "broadphase didn't find edge " << edge1 << std::endl; }

				CollisionCandidateSet collision_candidates1;
				add_edge_candidates(edge1, true, true, collision_candidates1);
				bool edge0_found = false;
				for (size_t j = 0; j < collision_candidates1.size(); ++j)
				{
					if (collision_candidates1[j][0] == edge0 || collision_candidates1[j][1] == edge0)
					{
						edge0_found = true;
					}
				}

				if (!edge0_found) { std::cout << "broadphase didn't find edge " << edge0 << std::endl; }

			}
			else
			{
				size_t vert = coll.m_vertex_indices[0];
				size_t collision_tri = m_surface.m_mesh.get_triangle_index(coll.m_vertex_indices[1], coll.m_vertex_indices[2], coll.m_vertex_indices[3]);

				CollisionCandidateSet v_collision_candidates;
				add_point_candidates(vert, true, true, v_collision_candidates);
				bool tri_found = false;
				for (size_t j = 0; j < v_collision_candidates.size(); ++j)
				{
					if (v_collision_candidates[j][0] == collision_tri || v_collision_candidates[j][1] == collision_tri)
					{
						tri_found = true;
					}
				}

				if (!tri_found) { std::cout << "broadphase didn't find tri " << collision_tri << std::endl; }

				CollisionCandidateSet t_collision_candidates;
				add_triangle_candidates(collision_tri, true, true, t_collision_candidates);
				bool vert_found = false;
				for (size_t j = 0; j < t_collision_candidates.size(); ++j)
				{
					if (t_collision_candidates[j][0] == vert || t_collision_candidates[j][1] == vert)
					{
						vert_found = true;
					}
				}

				if (!vert_found) { std::cout << "broadphase didn't find vertex " << vert << std::endl; }

			}


		}


	}

	// ---------------------------------------------------------
	///
	/// Determine if the vertices specified by the Collision object are still in collision
	///
	// ---------------------------------------------------------

	bool CollisionPipeline::check_if_collision_persists(const Collision& collision)
	{
		const Vec4st& vs = collision.m_vertex_indices;


		if (collision.m_is_edge_edge)
		{
			return segment_segment_collision(m_surface.get_position(vs[0]), m_surface.get_newposition(vs[0]), vs[0],
				m_surface.get_position(vs[1]), m_surface.get_newposition(vs[1]), vs[1],
				m_surface.get_position(vs[2]), m_surface.get_newposition(vs[2]), vs[2],
				m_surface.get_position(vs[3]), m_surface.get_newposition(vs[3]), vs[3]);

		}
		else
		{
			return point_triangle_collision(m_surface.get_position(vs[0]), m_surface.get_newposition(vs[0]), vs[0],
				m_surface.get_position(vs[1]), m_surface.get_newposition(vs[1]), vs[1],
				m_surface.get_position(vs[2]), m_surface.get_newposition(vs[2]), vs[2],
				m_surface.get_position(vs[3]), m_surface.get_newposition(vs[3]), vs[3]);

		}

	}

}

