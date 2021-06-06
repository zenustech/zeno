// ---------------------------------------------------------
//
//  dynamicsurface.cpp
//  Tyson Brochu 2008
//  Christopher Batty, Fang Da 2014
//
//  A triangle mesh with associated vertex locations and  masses.  Query functions for getting geometry info.
//
// ---------------------------------------------------------

// ---------------------------------------------------------
// Includes
// ---------------------------------------------------------
#define NOMINMAX
#include <dynamicsurface.h>
#include "Timer.h"
#include <broadphasegrid.h>
#include <cassert>
#include <ccd_wrapper.h>
#include <collisionpipeline.h>
#include <collisionqueries.h>
#include <ctime>
#include <impactzonesolver.h>
#include <iomesh.h>
#include <lapack_wrapper.h>
#include <mat.h>
#include <queue>
#include <runstats.h>
#include <vec.h>
#include <vector>
#include <wallclocktime.h>
#include <algorithm>

// ---------------------------------------------------------
// Local constants, typedefs, macros
// ---------------------------------------------------------

// ---------------------------------------------------------
//  Extern globals
// ---------------------------------------------------------

namespace LosTopos {

extern RunStats g_stats;

// ---------------------------------------------------------
// Static function definitions
// ---------------------------------------------------------

// ---------------------------------------------------------
// Member function definitions
// ---------------------------------------------------------

// ---------------------------------------------------------
///
/// DynamicSurface constructor.  Copy triangles and vertex locations.
///
// ---------------------------------------------------------

DynamicSurface::DynamicSurface( const std::vector<Vec3d>& vertex_positions, 
                               const std::vector<Vec3st>& triangles,
                               const std::vector<Vec2i>& labels,
                               const std::vector<Vec3d>& masses,
                               double in_proximity_epsilon,
                               double in_friction_coefficient,
                               bool in_collision_safety,
                               bool in_verbose ) :
m_proximity_epsilon( in_proximity_epsilon ),
m_verbose( in_verbose ),   
m_collision_safety( in_collision_safety ),
m_masses( masses ), 
m_mesh(), 
m_broad_phase( new BroadPhaseGrid() ),
m_collision_pipeline( NULL ),    // allocated and initialized in the constructor body
m_aabb_padding( max( in_proximity_epsilon, 1e-7 ) ),
//m_feature_edge_angle_threshold(M_PI/6),
m_feature_edge_angle_threshold(M_PI),    //&&&& ignore edge features
pm_positions(vertex_positions),
pm_newpositions(vertex_positions),
pm_velocities(vertex_positions.size(),Vec3d(0,0,0)),
m_velocities(vertex_positions.size())
{
    
    if ( m_verbose )
    {
        std::cout << "constructing dynamic surface" << std::endl;
    }
    
    // if masses not provided, set all to 1.0
    if ( m_masses.size() == 0 )
    {
        m_masses.resize( get_num_vertices(), Vec3d(1.0, 1.0, 1.0) );
    }

    assert(triangles.size() == labels.size());

    m_mesh.set_num_vertices( get_num_vertices() );   
    m_mesh.replace_all_triangles( triangles, labels );
    

    // Some compilers worry about using "this" in the initialization list, so initialize it here
    m_collision_pipeline = new CollisionPipeline( *this, *m_broad_phase, in_friction_coefficient );
    
    if ( m_verbose )
    {
        std::cout << "constructed dynamic surface" << std::endl;
    }
    
}


// ---------------------------------------------------------
///
/// Destructor. Frees memory allocated by DynamicSurface for the broad phase and collision pipeline objects.
///
// ---------------------------------------------------------

DynamicSurface::~DynamicSurface()
{
    delete m_broad_phase;
    delete m_collision_pipeline;
}

double DynamicSurface::solid_mass() { return  10; }// std::numeric_limits<double>::infinity(); }
// ---------------------------------------------------------
///
/// Compute the unsigned distance to the surface.
///
// ---------------------------------------------------------

double DynamicSurface::distance_to_surface( const Vec3d& p, size_t& closest_triangle ) const
{
    
    double padding = m_aabb_padding;
    double min_distance = BIG_DOUBLE;
    
    while ( min_distance == BIG_DOUBLE )
    {
        
        Vec3d xmin( p - Vec3d( padding ) );
        Vec3d xmax( p + Vec3d( padding ) );
        
        std::vector<size_t> nearby_triangles;   
        
        m_broad_phase->get_potential_triangle_collisions( xmin, xmax, true, true, nearby_triangles );
        
        for ( size_t j = 0; j < nearby_triangles.size(); ++j )
        {
            const Vec3st& tri = m_mesh.get_triangle( nearby_triangles[j] );
            
            if ( tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2] ) { continue; }
            
            double curr_distance;
            check_point_triangle_proximity( p, get_position(tri[0]), get_position(tri[1]), get_position(tri[2]), curr_distance );
            if ( curr_distance < padding )
            {   
                min_distance = min( min_distance, curr_distance );
                closest_triangle = nearby_triangles[j];
            }
        }
        
        padding *= 2.0;
        
    }
    
    return min_distance;
    
}


// --------------------------------------------------------
///
/// Break up the triangle mesh into connected components, determine surface IDs for all vertices.
///
// --------------------------------------------------------

void DynamicSurface::partition_surfaces( std::vector<size_t>& surface_ids, std::vector< std::vector< size_t> >& surfaces ) const
{
    
    static const size_t UNASSIGNED = (size_t) ~0;
    
    surfaces.clear();
    
    surface_ids.clear();
    surface_ids.resize( get_num_vertices(), UNASSIGNED );
    
    size_t curr_surface = 0;
    
    while ( true )
    { 
        size_t next_unassigned_vertex;
        for ( next_unassigned_vertex = 0; next_unassigned_vertex < surface_ids.size(); ++next_unassigned_vertex )
        {
            if ( m_mesh.m_vertex_to_edge_map[next_unassigned_vertex].empty() ) { continue; }
            
            if ( surface_ids[next_unassigned_vertex] == UNASSIGNED )
            {
                break;
            }
        }
        
        if ( next_unassigned_vertex == surface_ids.size() )
        {
            break;
        }
        
        std::queue<size_t> open;
        open.push( next_unassigned_vertex );
        
        std::vector<size_t> surface_vertices;
        
        while ( false == open.empty() )
        {
            size_t vertex_index = open.front();
            open.pop();
            
            if ( m_mesh.m_vertex_to_edge_map[vertex_index].empty() ) { continue; }
            
            if ( surface_ids[vertex_index] != UNASSIGNED )
            {
                assert( surface_ids[vertex_index] == curr_surface );
                continue;
            }
            
            surface_ids[vertex_index] = curr_surface;
            surface_vertices.push_back( vertex_index );
            
            const std::vector<size_t>& incident_edges = m_mesh.m_vertex_to_edge_map[vertex_index];
            
            for( size_t i = 0; i < incident_edges.size(); ++i )
            {
                size_t adjacent_vertex = m_mesh.m_edges[ incident_edges[i] ][0];
                if ( adjacent_vertex == vertex_index ) { adjacent_vertex = m_mesh.m_edges[ incident_edges[i] ][1]; }
                
                if ( surface_ids[adjacent_vertex] == UNASSIGNED )
                {
                    open.push( adjacent_vertex );
                }
                else
                {
                    assert( surface_ids[adjacent_vertex] == curr_surface );
                }
                
            } 
        }
        
        surfaces.push_back( surface_vertices );
        
        ++curr_surface;
        
    }
    
    //
    // assert all vertices are assigned and share volume IDs with their neighbours
    //
    
    for ( size_t i = 0; i < surface_ids.size(); ++i )
    {
        if ( m_mesh.m_vertex_to_edge_map[i].empty() ) { continue; }
        
        assert( surface_ids[i] != UNASSIGNED );
        
        const std::vector<size_t>& incident_edges = m_mesh.m_vertex_to_edge_map[i];    
        for( size_t j = 0; j < incident_edges.size(); ++j )
        {
            size_t adjacent_vertex = m_mesh.m_edges[ incident_edges[j] ][0];
            if ( adjacent_vertex == i ) { adjacent_vertex = m_mesh.m_edges[ incident_edges[j] ][1]; }
            assert( surface_ids[adjacent_vertex] == surface_ids[i] );         
        } 
        
    }
    
}

// --------------------------------------------------------
///
/// Compute all vertex normals (unweighted average).
///
// --------------------------------------------------------

void DynamicSurface::get_all_vertex_normals( std::vector<Vec3d>& normals ) const
{
    normals.resize( get_num_vertices() );
    for ( size_t i = 0; i < get_num_vertices(); ++i )
    {
        normals[i] = get_vertex_normal(i);
    }
}


// ---------------------------------------------------------
///
/// Run intersection detection against all triangles
///
// ---------------------------------------------------------

void DynamicSurface::get_triangle_intersections( const Vec3d& segment_point_a, 
                                                const Vec3d& segment_point_b,
                                                std::vector<double>& hit_ss,
                                                std::vector<size_t>& hit_triangles,
                                                bool verbose ) const
{
    Vec3d aabb_low, aabb_high;
    minmax( segment_point_a, segment_point_b, aabb_low, aabb_high );
    
    //TODO Use some kind of line-drawing algorithm to find a sequence of acceleration grid cells that overlap/enclose 
    //the segment, to avoid having to search massive chunks of the mesh.
    
    std::vector<size_t> overlapping_triangles;
    m_broad_phase->get_potential_triangle_collisions( aabb_low, aabb_high, true, true, overlapping_triangles );
    
    for ( size_t i = 0; i < overlapping_triangles.size(); ++i )
    {
        const Vec3st& tri = m_mesh.get_triangle( overlapping_triangles[i] );
        
        Vec3st t = tri;
        //Vec3st t = sort_triangle( tri );
        //assert( t[0] < t[1] && t[0] < t[2] && t[1] < t[2] );
        
        const Vec3d& v0 = get_position( t[0] );
        const Vec3d& v1 = get_position( t[1] );
        const Vec3d& v2 = get_position( t[2] );      
        
        size_t dummy_index = get_num_vertices();
        
        double bary1, bary2, bary3;
        Vec3d normal;
        double sa, sb;
        
        bool hit = segment_triangle_intersection(segment_point_a, dummy_index, 
                                                 segment_point_b, dummy_index+1,
                                                 v0, t[0],
                                                 v1, t[1],
                                                 v2, t[2],
                                                 sa, sb, bary1, bary2, bary3,
                                                 false, verbose );
        
        if ( hit )
        {
            hit_ss.push_back( sb );
            hit_triangles.push_back( overlapping_triangles[i] );
        }         
        
    }
    
}

// ---------------------------------------------------------
///
/// Run intersection detection against all triangles and return the number of hits.
///
// ---------------------------------------------------------

size_t DynamicSurface::get_number_of_triangle_intersections( const Vec3d& segment_point_a, 
                                                            const Vec3d& segment_point_b ) const
{
    int num_hits = 0;
    int num_misses = 0;
    Vec3d aabb_low, aabb_high;
    minmax( segment_point_a, segment_point_b, aabb_low, aabb_high );
    
    std::vector<size_t> overlapping_triangles;
    m_broad_phase->get_potential_triangle_collisions( aabb_low, aabb_high, true, true, overlapping_triangles );
    
    for ( size_t i = 0; i < overlapping_triangles.size(); ++i )
    {
        const Vec3st& tri = m_mesh.get_triangle( overlapping_triangles[i] );
        
        Vec3st t = tri;
        //Vec3st t = sort_triangle( tri );
        //assert( t[0] < t[1] && t[0] < t[2] && t[1] < t[2] );
        
        const Vec3d& v0 = get_position( t[0] );
        const Vec3d& v1 = get_position( t[1] );
        const Vec3d& v2 = get_position( t[2] );      
        
        size_t dummy_index = get_num_vertices();
        static const bool degenerate_counts_as_hit = true;
        
        bool hit = segment_triangle_intersection( segment_point_a, dummy_index,
                                                 segment_point_b, dummy_index + 1, 
                                                 v0, t[0],
                                                 v1, t[1],
                                                 v2, t[2],   
                                                 degenerate_counts_as_hit );
        
        if ( hit )
        {
            ++num_hits;
        }         
        else
        {
            ++num_misses;
        }
    }
    
    return num_hits;
    
}


// ---------------------------------------------------------
///
/// Compute rank of the quadric metric tensor at a vertex
///
// ---------------------------------------------------------

unsigned int DynamicSurface::vertex_primary_space_rank( size_t v, int region ) const
{     
   if ( m_mesh.m_vertex_to_triangle_map[v].empty() )     { return 0; }

   const std::vector<size_t>& incident_triangles = m_mesh.m_vertex_to_triangle_map[v];
   
   //Check how many labels we have
   std::set<int> labelset;
   for(size_t i = 0; i < incident_triangles.size(); ++i) {
      Vec2i label = m_mesh.get_triangle_label(incident_triangles[i]);
      labelset.insert(label[0]);
      labelset.insert(label[1]);
   }
   
   //If manifold, just do the easy/cheap case.
   if(labelset.size() <= 2) {
      return compute_rank_from_triangles(incident_triangles);
   }
   else if(region != -1) {
      //if requesting rank for just a specific manifold component...

      //collect all the triangles with the relevant label
      std::vector<size_t> cur_tri_set;
      for(size_t i = 0; i < incident_triangles.size(); ++i) {
         Vec2i label = m_mesh.get_triangle_label(incident_triangles[i]);
         if(label[0] == region || label[1] == region)
            cur_tri_set.push_back(incident_triangles[i]);
      }
      return compute_rank_from_triangles(cur_tri_set);
   }
   else {
      //Otherwise, visit each set of the manifold regions separately, take the max. 
      //This seems to work somewhat better than doing all at once in the non-manifold case.
      unsigned int max_rank = 0;
      for(std::set<int>::iterator it = labelset.begin(); it != labelset.end(); ++it) {
         int cur_label = *it;

         //collect all the triangles with the relevant label
         std::vector<size_t> cur_tri_set;
         for(size_t i = 0; i < incident_triangles.size(); ++i) {
            Vec2i label = m_mesh.get_triangle_label(incident_triangles[i]);
            if(label[0] == cur_label || label[1] == cur_label)
               cur_tri_set.push_back(incident_triangles[i]);
         }

         unsigned int rank = compute_rank_from_triangles(cur_tri_set);
         max_rank = max(rank, max_rank);
      }
   
      return max_rank;
   }
}

unsigned int DynamicSurface::compute_rank_from_triangles(const std::vector<size_t>& triangles) const {
   Mat33d A(0,0,0,0,0,0,0,0,0);

   for ( size_t i = 0; i < triangles.size(); ++i )
   {
      size_t triangle_index = triangles[i];
      Vec3d normal = get_triangle_normal(triangle_index);
      double w = get_triangle_area(triangle_index);

      A(0,0) += normal[0] * w * normal[0];
      A(1,0) += normal[1] * w * normal[0];
      A(2,0) += normal[2] * w * normal[0];

      A(0,1) += normal[0] * w * normal[1];
      A(1,1) += normal[1] * w * normal[1];
      A(2,1) += normal[2] * w * normal[1];

      A(0,2) += normal[0] * w * normal[2];
      A(1,2) += normal[1] * w * normal[2];
      A(2,2) += normal[2] * w * normal[2];
   }

   // get eigen decomposition
   double eigenvalues[3];
   double work[9];
   int info = ~0, n = 3, lwork = 9;
   LAPACK::get_eigen_decomposition( &n, A.a, &n, eigenvalues, work, &lwork, &info );

   if ( info != 0 )
   {
      if ( m_verbose )
      {
         std::cout << "Eigen decomposition failed.  Incident triangles: " << std::endl;
         for ( size_t i = 0; i < triangles.size(); ++i )
         {
            size_t triangle_index = triangles[i];
            Vec3d normal = get_triangle_normal(triangle_index);
            double w = get_triangle_area(triangle_index);

            std::cout << "normal: ( " << normal << " )    ";  
            std::cout << "area: " << w << std::endl;
         }
      }
      return 4;
   }

   // compute rank of primary space
   unsigned int rank = 0;
   for ( unsigned int i = 0; i < 3; ++i )
   {
      if ( eigenvalues[i] > G_EIGENVALUE_RANK_RATIO * eigenvalues[2] )
      {
         ++rank;
      }
   }
   return rank;
}


/// Look at all triangle pairs and get the smallest angle, ignoring regions.
double DynamicSurface::get_largest_dihedral(size_t edge) const {
   const std::vector<size_t>& tri_list = m_mesh.m_edge_to_triangle_map[edge];

   //consider all triangle pairs
   size_t v0 = m_mesh.m_edges[edge][0];
   size_t v1 = m_mesh.m_edges[edge][1];

   double largest_angle = 0;
   for(size_t i = 0; i < tri_list.size(); ++i) {
      size_t tri_id0 = tri_list[i];
      Vec3d norm0 = get_triangle_normal(tri_id0);
      for(size_t j = i+1; j < tri_list.size(); ++j) {
         size_t tri_id1 = tri_list[j];
         Vec3d norm1 = get_triangle_normal(tri_id1);
         
         //possibly flip one normal so the tris are oriented in a matching way, to get the right dihedral angle.
         if (m_mesh.oriented(v0, v1, m_mesh.get_triangle(tri_id0)) != m_mesh.oriented(v1, v0, m_mesh.get_triangle(tri_id1))) {
            norm1 = -norm1;
         }

         double angle = acos(dot(norm0,norm1));
         largest_angle = std::max(largest_angle,angle);
      }
   }

   return largest_angle;

}

/// Look at all triangle pairs and get the smallest angle, ignoring regions.
double DynamicSurface::get_largest_dihedral(size_t edge, const std::vector<Vec3d> & cached_normals) const {
   const std::vector<size_t>& tri_list = m_mesh.m_edge_to_triangle_map[edge];

   //consider all triangle pairs
   size_t v0 = m_mesh.m_edges[edge][0];
   size_t v1 = m_mesh.m_edges[edge][1];

   double largest_angle = 0;
   for(size_t i = 0; i < tri_list.size(); ++i) {
      size_t tri_id0 = tri_list[i];
      Vec3d norm0 = cached_normals[tri_id0];
      for(size_t j = i+1; j < tri_list.size(); ++j) {
         size_t tri_id1 = tri_list[j];
         Vec3d norm1 = cached_normals[tri_id1];

         //possibly flip one normal so the tris are oriented in a matching way, to get the right dihedral angle.
         if (m_mesh.oriented(v0, v1, m_mesh.get_triangle(tri_id0)) != m_mesh.oriented(v1, v0, m_mesh.get_triangle(tri_id1))) {
            norm1 = -norm1;
         }
         double angle = acos(dot(norm0,norm1));
         largest_angle = std::max(largest_angle,angle);
      }
   }

   return largest_angle;

}


// ---------------------------------------------------------
///
/// Determine whether a point is inside the volume defined by the surface.
/// Uses raycasting to find the first intersection, and then the triangle normal
/// is used to determine inside/outside. (This can readily be extended to the multi-material case.)
/// Return -1 if we can't make a safe determination
// ---------------------------------------------------------
int DynamicSurface::test_region_via_ray_and_normal(const Vec3d& p, const Vec3d& ray_end) {

   std::vector<double> hit_ss;
   std::vector<size_t> hit_tris;
   get_triangle_intersections(p, ray_end, hit_ss, hit_tris);
   
   int first_hit = -1;
   double near_dist = 2; //result is between 0 and 1.

   bool good_hit = false;
   for(unsigned int i = 0; i < hit_ss.size(); ++i) {
      if(hit_ss[i] < near_dist && !m_mesh.triangle_is_deleted(hit_tris[i])) {
         first_hit = i;
         near_dist = hit_ss[i];
         good_hit = true;
      }
   }

   if(!good_hit) return 0; //assume no hits means outside (region 0).

   // get the normal of this triangle, check its orientation relative to the ray
   const Vec3st& t = m_mesh.m_tris[ hit_tris[first_hit] ];
   const Vec3d& v0 = pm_positions[ t[0] ];
   const Vec3d& v1 = pm_positions[ t[1] ];
   const Vec3d& v2 = pm_positions[ t[2] ];     
   Vec3d tri_normal = -cross(v2-v0, v1-v0);
   
   const double eps = 1e-6;
   if(mag(tri_normal) > eps)
      normalize(tri_normal);
   else
      return -1;
   
   Vec2i labels = m_mesh.get_triangle_label(hit_tris[first_hit]);
   
   Vec3d vec_dir = ray_end-p;
   normalize(vec_dir);

   if(dot(tri_normal, vec_dir) > eps)
      return labels[1];
   else if(dot(tri_normal, vec_dir) < -eps)
      return labels[0];
   else
      return -1; //uncertain case...
}


int DynamicSurface::get_region_containing_point( const Vec3d& p )
{
   //TODO Take a distance as input? To avoid the hard-coded
   //ray distance of 1000.
   double raydist = 100;
   //
   // The point is inside if the dot product between the normal of the first
   // triangle intersection and the ray direction is positive.
   // We use voting (for the moment) to enhance robustness, in the absence of a geometrically
   // exact test.
   std::map<int,int> region_counts;

   // shoot a ray in the positive-x direction
   Vec3d ray_end( p + Vec3d( raydist, 0, 0 ) );
   int region = test_region_via_ray_and_normal(p, ray_end);
   if(region != -1)
      region_counts[region]++;

   // negative x
   ray_end = p - Vec3d( raydist, 0, 0 );
   region = test_region_via_ray_and_normal(p, ray_end);
   if(region != -1)
      region_counts[region]++;

   // positive y
   ray_end = p + Vec3d( 0, raydist, 0 );
   region = test_region_via_ray_and_normal(p, ray_end);
   if(region != -1)
      region_counts[region]++;

   // negative y
   ray_end = p - Vec3d( 0, raydist, 0 );
   region = test_region_via_ray_and_normal(p, ray_end);
   if(region != -1)
      region_counts[region]++;

   // positive z
   ray_end = p + Vec3d( 0, 0, raydist );
   region = test_region_via_ray_and_normal(p, ray_end);
   if(region != -1)
      region_counts[region]++;

   // negative z 
   ray_end = p - Vec3d( 0, 0, raydist );
   region = test_region_via_ray_and_normal(p, ray_end);
   if(region != -1)
      region_counts[region]++;

   int max_val = 0;
   int max_ind = -1;
   int total = 0;
   std::map<int,int>::iterator it = region_counts.begin();
   for(;it != region_counts.end(); ++it) {
      if(it->second >= max_val) {
         max_ind = it->first;
         max_val = it->second;
      }
      total += it->second;
   }
   double consistency = (double)max_val / (double)total;
   if(consistency < 0.9 || total < 4) {
      //slightly dubious case, so cast a few more random rays for good measure.
      //std::cout << "Winning region score: " << max_val << " out of 6.\n";
      std::map<int,int>::iterator it = region_counts.begin();
      /*for(;it != region_counts.end(); ++it) {
         std::cout << "   " << it->first  << " got " << it->second << std::endl;
      }*/
      static int seed = 0;
      //throw another 7 rays in random directions
      for(int i = 0; i < 7; ++i) {
         float x_dir = randhashf(seed, -1, 1); ++seed;
         float y_dir = randhashf(seed, -1, 1); ++seed;
         float z_dir = randhashf(seed, -1, 1); ++seed;
         Vec3d direction(x_dir,y_dir,z_dir);
         if(mag(direction) > 0.00001)
            normalize(direction);
         ray_end = p + raydist*direction;
         region = test_region_via_ray_and_normal(p, ray_end);
         if(region != -1)
            region_counts[region]++;
      }

      it = region_counts.begin();
      int max_val = 0;
      int max_ind = -1;
      int total = 0;
      for(;it != region_counts.end(); ++it) {
         if(it->second > max_val) {
            max_ind = it->first;
            max_val = it->second;
         }
         total += it->second;
      }
      /*std::cout << "Updated region scores: " << (double)max_val << " out of 11.\n";
      it = region_counts.begin();
      for(;it != region_counts.end(); ++it) {
         std::cout << "   " << it->first  << " got " << it->second << std::endl;
      }*/
      consistency = (double)max_val / (double)total;
      if(consistency < 0.9) std::cout << "Warning: Consistency is only " << consistency << " out of " << total << " rays.\n";
      assert(consistency > 0.6); //if we didn't even hit 60% consistently the same, this labeling is pretty dubious
   }
   
   

   return max_ind;

}

// ---------------------------------------------------------
///
/// Advance mesh by one time step 
///
// ---------------------------------------------------------

void DynamicSurface::integrate( double desired_dt, double& actual_dt )
{     
    
    if ( m_collision_safety )
    {
      std::cout << "Checking collisions before integration.\n";
      assert_mesh_is_intersection_free( false );
      
    }
    std::cout << "Integrating\n";
    static const bool DEGEN_DOES_NOT_COUNT = false;   
    static const bool USE_NEW_POSITIONS = true;
    
    if ( m_verbose ) 
    {
        std::cout << "---------------------- Los Topos: integration and collision handling --------------------" << std::endl;
    }
    
    double start_time = get_time_in_seconds();
    
    double curr_dt = desired_dt;
    bool success = false;
    
    const std::vector<Vec3d> saved_predicted_positions = get_newpositions();
    
    while ( !success )
    {
        if (m_verbose) {
            printf("Integrating dt:%f\n", curr_dt);
        }
        
        m_velocities.resize( get_num_vertices() );
        for(size_t i = 0; i < get_num_vertices(); i++)
        {
            m_velocities[i] = ( get_newposition(i) - get_position(i) ) / curr_dt;  
        }
        
        // Handle proximities
        //CSim::TimerMan::timer("Sim.step/BPS.step/advection/st_integrate/handle_proximities").start();
        if ( m_collision_safety )
        {
            m_collision_pipeline->handle_proximities( curr_dt );
            if (m_verbose) {
                printf("handle_proximities Done\n");
            }
            //check_continuous_broad_phase_is_up_to_date();
        }
        
        //CSim::TimerMan::timer("Sim.step/BPS.step/advection/st_integrate/handle_proximities").stop();
        if ( m_collision_safety )
        {        
            
            // Handle continuous collisions
            bool all_collisions_handled = false;
            //CSim::TimerMan::timer("Sim.step/BPS.step/advection/st_integrate/handle_collisions").start();
            all_collisions_handled = m_collision_pipeline->handle_collisions( curr_dt );
            //CSim::TimerMan::timer("Sim.step/BPS.step/advection/st_integrate/handle_collisions").stop();
            if (m_verbose) {
                printf("handle_collisions Done\n");
            }
            // failsafe impact zones 
            
            ImpactZoneSolver impactZoneSolver( *this );
            
            bool solver_ok = all_collisions_handled;
            
            if ( !solver_ok )
            {
                //if ( m_verbose ) 
                { std::cout << "IIZ" << std::endl; }
                solver_ok = impactZoneSolver.inelastic_impact_zones( curr_dt );            
            }
            
            if ( !solver_ok )
            {
                //if ( m_verbose ) 
                { std::cout << "RIZ" << std::endl; }
                // punt to rigid impact zones
                solver_ok = impactZoneSolver.rigid_impact_zones( curr_dt );
            }  
            
            if ( !solver_ok )
            {
                // back up and try again:
                
                curr_dt = 0.5 * curr_dt;
                for ( size_t i = 0; i < get_num_vertices(); ++i )
                {
                    set_newposition(i, get_position(i) + 0.5 * (saved_predicted_positions[i] - get_position(i)) ) ;
                }
                
                continue;      
            }
            
            
            // verify intersection-free predicted mesh
            std::vector<Intersection> intersections;
            get_intersections( DEGEN_DOES_NOT_COUNT, USE_NEW_POSITIONS, intersections );
            
            if ( !intersections.empty() )
            {
                std::cout << "Intersection in predicted mesh." << std::endl;
                
                if ( all_collisions_handled )
                {
                    std::cout << "Intersection in predicted mesh but handle collisions returned ok." << std::endl;
                    std::cout << "Using dt = " << curr_dt << std::endl;
                    assert( false );
                }
                
                if ( m_verbose )
                {
                    std::cout << "Intersection in predicted mesh, cutting timestep." << std::endl;
                }
                
                // back up and try again:
                
                curr_dt = 0.5 * curr_dt;
                for ( size_t i = 0; i < get_num_vertices(); ++i )
                {
                    set_newposition( i, get_position(i) + 0.5 * ( saved_predicted_positions[i] - get_position(i) ) );
                }
                
                continue;      
                
            }                 
            
        }
        
        // Set m_positions
        set_positions_to_newpositions();
        if ( m_collision_safety )
        {
            assert_mesh_is_intersection_free( DEGEN_DOES_NOT_COUNT );
        }
        
        actual_dt = curr_dt;
        
        success = true;
        
    }
    
    double end_time = get_time_in_seconds();
    
    static unsigned int step = 0;
    g_stats.add_per_frame_double( "DynamicSurface:integration_time_per_timestep", step, end_time - start_time );
    ++step;
    std::cout << "Done integrating\n";
    
}

// ---------------------------------------------------------
///
/// Construct static acceleration structure
///
// ---------------------------------------------------------

void DynamicSurface::rebuild_static_broad_phase()
{
    assert( m_collision_safety );
    if(m_verbose)
      std::cout << "Rebuilding broad phase\n";
    m_broad_phase->update_broad_phase( *this, false );
    
    if(m_verbose)
      std::cout << "Done rebuilding broad phase\n";
}

// ---------------------------------------------------------
///
/// Construct continuous acceleration structure
///
// ---------------------------------------------------------

void DynamicSurface::rebuild_continuous_broad_phase()
{
    assert( m_collision_safety );
    m_broad_phase->update_broad_phase( *this, true );
}


// ---------------------------------------------------------
///
/// Update the broadphase elements incident to the given vertex
///
// ---------------------------------------------------------

void DynamicSurface::update_static_broad_phase( size_t vertex_index )
{
    const std::vector<size_t>& incident_tris = m_mesh.m_vertex_to_triangle_map[ vertex_index ];
    const std::vector<size_t>& incident_edges = m_mesh.m_vertex_to_edge_map[ vertex_index ];
    
    Vec3d low, high;
    vertex_static_bounds( vertex_index, low, high );
    m_broad_phase->update_vertex( vertex_index, low, high, vertex_is_all_solid(vertex_index) );
    
    for ( size_t t = 0; t < incident_tris.size(); ++t )
    {
        triangle_static_bounds( incident_tris[t], low, high );
        m_broad_phase->update_triangle( incident_tris[t], low, high, triangle_is_all_solid(incident_tris[t]) );
    }
    
    for ( size_t e = 0; e < incident_edges.size(); ++e )
    {
        edge_static_bounds( incident_edges[e], low, high );
        m_broad_phase->update_edge( incident_edges[e], low, high, edge_is_all_solid(incident_edges[e]) );
    }
    
}


// ---------------------------------------------------------
///
/// Update the broadphase elements incident to the given vertex, using current and predicted vertex positions
///
// ---------------------------------------------------------

void DynamicSurface::update_continuous_broad_phase( size_t vertex_index )
{
    assert( m_collision_safety );
    
    const std::vector<size_t>& incident_tris = m_mesh.m_vertex_to_triangle_map[ vertex_index ];
    const std::vector<size_t>& incident_edges = m_mesh.m_vertex_to_edge_map[ vertex_index ];
    
    Vec3d low, high;
    vertex_continuous_bounds( vertex_index, low, high );
    m_broad_phase->update_vertex( vertex_index, low, high, vertex_is_all_solid(vertex_index) );
    
    for ( size_t t = 0; t < incident_tris.size(); ++t )
    {
        triangle_continuous_bounds( incident_tris[t], low, high );
        m_broad_phase->update_triangle( incident_tris[t], low, high, triangle_is_all_solid(incident_tris[t]) );
    }
    
    for ( size_t e = 0; e < incident_edges.size(); ++e )
    {
        edge_continuous_bounds( incident_edges[e], low, high );
        m_broad_phase->update_edge( incident_edges[e], low, high, edge_is_all_solid(incident_edges[e]) );
    }
}


// ---------------------------------------------------------
///
/// Compute the (padded) AABB of a vertex
///
// ---------------------------------------------------------

void DynamicSurface::vertex_static_bounds(size_t v, Vec3d &xmin, Vec3d &xmax) const
{
    if ( m_mesh.m_vertex_to_triangle_map[v].empty() )
    {
        xmin = Vec3d(m_aabb_padding);
        xmax = -Vec3d(m_aabb_padding);
    }
    else
    {
        xmin = get_position(v) - Vec3d(m_aabb_padding);
        xmax = get_position(v) + Vec3d(m_aabb_padding);
    }
}

// ---------------------------------------------------------
///
/// Compute the AABB of an edge
///
// ---------------------------------------------------------

void DynamicSurface::edge_static_bounds(size_t e, Vec3d &xmin, Vec3d &xmax) const
{
    const Vec2st& edge = m_mesh.m_edges[e];
    if ( edge[0] == edge[1] )
    {
        xmin = Vec3d(m_aabb_padding);
        xmax = -Vec3d(m_aabb_padding); 
    }
    else
    {            
        minmax( get_position(edge[0]), get_position(edge[1]), xmin, xmax);
        xmin -= Vec3d(m_aabb_padding);
        xmax += Vec3d(m_aabb_padding);
    }
}

// ---------------------------------------------------------
///
/// Compute the AABB of a triangle
///
// ---------------------------------------------------------

void DynamicSurface::triangle_static_bounds(size_t t, Vec3d &xmin, Vec3d &xmax) const
{
    const Vec3st& tri = m_mesh.get_triangle(t);  
    if ( tri[0] == tri[1] )
    {
        xmin = Vec3d(m_aabb_padding);
        xmax = -Vec3d(m_aabb_padding);
    }
    else
    {      
        minmax(get_position(tri[0]), get_position(tri[1]), get_position(tri[2]), xmin, xmax);
        xmin -= Vec3d(m_aabb_padding);
        xmax += Vec3d(m_aabb_padding);
    }
}

// ---------------------------------------------------------
///
/// Compute the AABB of a continuous vertex
///
// ---------------------------------------------------------

void DynamicSurface::vertex_continuous_bounds(size_t v, Vec3d &xmin, Vec3d &xmax) const
{
    if ( m_mesh.m_vertex_to_triangle_map[v].empty() )
    {
        xmin = Vec3d(m_aabb_padding);
        xmax = -Vec3d(m_aabb_padding);
    }
    else
    {
        minmax( get_position(v), get_newposition(v), xmin, xmax);
        xmin -= Vec3d(m_aabb_padding);
        xmax += Vec3d(m_aabb_padding);
    }
}

// ---------------------------------------------------------
///
/// Compute the AABB of a continuous edge
///
// ---------------------------------------------------------

void DynamicSurface::edge_continuous_bounds(size_t e, Vec3d &xmin, Vec3d &xmax) const
{
    const Vec2st& edge = m_mesh.m_edges[e];   
    if ( edge[0] == edge[1] )
    {
        xmin = Vec3d(m_aabb_padding);
        xmax = -Vec3d(m_aabb_padding);
    }
    else
    {      
        minmax(get_position(edge[0]), get_newposition(edge[0]), 
               get_position(edge[1]), get_newposition(edge[1]), 
               xmin, xmax);
        xmin -= Vec3d(m_aabb_padding);
        xmax += Vec3d(m_aabb_padding);
    }
}

// ---------------------------------------------------------
///
/// Compute the AABB of a continuous triangle
///
// ---------------------------------------------------------

void DynamicSurface::triangle_continuous_bounds(size_t t, Vec3d &xmin, Vec3d &xmax) const
{
    const Vec3st& tri = m_mesh.get_triangle(t);
    if ( tri[0] == tri[1] )
    {
        xmin = Vec3d(m_aabb_padding);
        xmax = -Vec3d(m_aabb_padding);
    }
    else
    {
        minmax(get_position(tri[0]), get_newposition(tri[0]), 
               get_position(tri[1]), get_newposition(tri[1]), 
               get_position(tri[2]), get_newposition(tri[2]), 
               xmin, xmax);
        
        xmin -= Vec3d(m_aabb_padding);
        xmax += Vec3d(m_aabb_padding);
    }
}


// ---------------------------------------------------------
///
/// Check two axis-aligned bounding boxes for intersection
///
// ---------------------------------------------------------

static bool aabbs_intersect( const Vec3d& a_xmin, const Vec3d& a_xmax, const Vec3d& b_xmin, const Vec3d& b_xmax )
{
    if ( (a_xmin[0] <= b_xmax[0] && a_xmin[1] <= b_xmax[1] && a_xmin[2] <= b_xmax[2]) &&
        (a_xmax[0] >= b_xmin[0] && a_xmax[1] >= b_xmin[1] && a_xmax[2] >= b_xmin[2]) )
    {
        return true;
    }
    
    return false;
}


// ---------------------------------------------------------
///
/// Caution: slow!
/// Check the consistency of the broad phase by comparing against the N^2 broadphase.
///
// ---------------------------------------------------------

void DynamicSurface::check_static_broad_phase_is_up_to_date() const
{
    
    // Verify by running against the n^2 broad phase
    
    //
    // vertex vs triangle
    //
    
    for ( size_t i = 0; i < get_num_vertices(); ++i )
    {
        if ( m_mesh.vertex_is_deleted(i) ) { continue; }
        
        // First, accelerated broad phase
        
        Vec3d aabb_low, aabb_high;
        vertex_static_bounds( i, aabb_low, aabb_high );
        
        std::vector<size_t> overlapping_triangles;
        m_broad_phase->get_potential_triangle_collisions(aabb_low, aabb_high, true, true, overlapping_triangles); 
        
        // filter deleted triangles
        for ( int k = 0; k < (int)overlapping_triangles.size(); ++k )
        {
            if ( m_mesh.triangle_is_deleted( overlapping_triangles[k] ) )
            {
                overlapping_triangles.erase( overlapping_triangles.begin() + k );
                --k;
            }
        }
        
        // Second, brute force check
        
        std::vector<size_t> brute_force_overlapping_triangles;
        
        for ( size_t j = 0; j < m_mesh.num_triangles(); ++j )
        {
            if ( m_mesh.triangle_is_deleted(j) ) { continue; }
            
            Vec3d tri_aabb_low, tri_aabb_high;
            triangle_static_bounds( j, tri_aabb_low, tri_aabb_high );
            
            if ( aabbs_intersect( aabb_low, aabb_high, tri_aabb_low, tri_aabb_high ) )
            {
                brute_force_overlapping_triangles.push_back( j );            
            }
        }
        
        assert( overlapping_triangles.size() == brute_force_overlapping_triangles.size() );
        
        std::sort(overlapping_triangles.begin(), overlapping_triangles.end());
        std::sort(brute_force_overlapping_triangles.begin(), brute_force_overlapping_triangles.end());
        
        for ( size_t k = 0; k < overlapping_triangles.size(); ++k )
        {
            assert( overlapping_triangles[k] == brute_force_overlapping_triangles[k] );
        }
        
    }
    
    //
    // edge vs edge
    //
    
    for ( size_t i = 0; i < m_mesh.m_edges.size(); ++i )
    {
        if ( m_mesh.edge_is_deleted(i) ) { continue; }
        
        // Accelerated broad phase
        
        Vec3d aabb_low, aabb_high;
        edge_static_bounds( i, aabb_low, aabb_high );
        
        std::vector<size_t> overlapping_edges;
        m_broad_phase->get_potential_edge_collisions( aabb_low, aabb_high, true, true, overlapping_edges );
        
        // filter deleted edges
        for ( int k = 0; k < (int)overlapping_edges.size(); ++k )
        {
            if ( m_mesh.edge_is_deleted( overlapping_edges[k] ) )
            {
                overlapping_edges.erase( overlapping_edges.begin() + k );
                --k;
            }
        }
        
        // Brute force
        std::vector<size_t> brute_force_overlapping_edges;
        for ( size_t j = 0; j < m_mesh.m_edges.size(); ++j )
        {
            if ( m_mesh.edge_is_deleted(j) ) { continue; }
            
            Vec3d edge_aabb_low, edge_aabb_high;
            edge_static_bounds( j, edge_aabb_low, edge_aabb_high );
            
            if ( aabbs_intersect( aabb_low, aabb_high, edge_aabb_low, edge_aabb_high ) )
            {
                brute_force_overlapping_edges.push_back( j );
            }
        }
        
        if ( overlapping_edges.size() != brute_force_overlapping_edges.size() )
        {
            
            std::cout << "edge " << i << ": " << m_mesh.m_edges[i] << std::endl;
            std::cout << "overlapping_edges.size(): " << overlapping_edges.size() << std::endl;
            for ( size_t k = 0; k < overlapping_edges.size(); ++k )
            {
                std::cout << k << ": " << overlapping_edges[k] << std::endl;
            }
            
            std::cout << "brute_force_overlapping_edges.size(): " << brute_force_overlapping_edges.size() << std::endl;
            for ( size_t k = 0; k < brute_force_overlapping_edges.size(); ++k )
            {
                std::cout << k << ": " << brute_force_overlapping_edges[k] << std::endl;
            }
            
        }
        
        assert( overlapping_edges.size() == brute_force_overlapping_edges.size() );
        
        std::sort( overlapping_edges.begin(), overlapping_edges.end() );
        std::sort( brute_force_overlapping_edges.begin(), brute_force_overlapping_edges.end() );
        
        for ( size_t k = 0; k < overlapping_edges.size(); ++k )
        {
            assert( overlapping_edges[k] == brute_force_overlapping_edges[k] );
        }
    }
    
    //
    // triangle vs vertex
    //
    
    for ( size_t i = 0; i < m_mesh.num_triangles(); ++i )
    {
        if ( m_mesh.triangle_is_deleted(i) ) { continue; }
        
        // Accelerated broad phase
        
        Vec3d aabb_low, aabb_high;
        triangle_static_bounds( i, aabb_low, aabb_high );
        
        std::vector<size_t> overlapping_vertices;
        m_broad_phase->get_potential_vertex_collisions( aabb_low, aabb_high, true, true, overlapping_vertices );
        
        // filter deleted vertices
        for ( int k = 0; k < (int)overlapping_vertices.size(); ++k )
        {
            if ( m_mesh.vertex_is_deleted( overlapping_vertices[k] ) )
            {
                overlapping_vertices.erase( overlapping_vertices.begin() + k );
                --k;
            }
        }
        
        // Brute force
        std::vector<size_t> brute_force_overlapping_vertices;
        for ( size_t j = 0; j < get_num_vertices(); ++j )
        {
            if ( m_mesh.vertex_is_deleted(j) ) { continue; }
            
            Vec3d vertex_aabb_low, vertex_aabb_high;
            vertex_static_bounds( j, vertex_aabb_low, vertex_aabb_high );
            
            if ( aabbs_intersect( aabb_low, aabb_high, vertex_aabb_low, vertex_aabb_high ) )
            {
                brute_force_overlapping_vertices.push_back( j );
            }
        }
        
        if ( overlapping_vertices.size() != brute_force_overlapping_vertices.size() )
        {
            std::cout << "triangle " << i << ": " << m_mesh.get_triangle(i) << std::endl;
            std::cout << "overlapping_vertices.size(): " << overlapping_vertices.size() << std::endl;
            for ( size_t k = 0; k < overlapping_vertices.size(); ++k )
            {
                std::cout << k << ": " << overlapping_vertices[k] << " --- ";
                std::cout << "is deleted: " << m_mesh.vertex_is_deleted( overlapping_vertices[k] ) << std::endl;
            }
            
            std::cout << "brute_force_overlapping_vertices.size(): " << brute_force_overlapping_vertices.size() << std::endl;
            for ( size_t k = 0; k < brute_force_overlapping_vertices.size(); ++k )
            {
                std::cout << k << ": " << brute_force_overlapping_vertices[k] << " --- ";
                std::cout << "is deleted: " << m_mesh.vertex_is_deleted( brute_force_overlapping_vertices[k] ) << std::endl;
            }
        }
        
        assert( overlapping_vertices.size() == brute_force_overlapping_vertices.size() );
        
        std::sort( overlapping_vertices.begin(), overlapping_vertices.end() );
        std::sort( brute_force_overlapping_vertices.begin(), brute_force_overlapping_vertices.end() );
        
        for ( size_t k = 0; k < overlapping_vertices.size(); ++k )
        {
            assert( overlapping_vertices[k] == brute_force_overlapping_vertices[k] );
        }
        
    }
    
    
}


// ---------------------------------------------------------
///
/// Caution: slow!
/// Check the consistency of the broad phase by comparing against the N^2 broadphase.  Checks using current and predicted vertex 
/// positions.
///
// ---------------------------------------------------------

void DynamicSurface::check_continuous_broad_phase_is_up_to_date() const
{
    
    // Verify by running against the n^2 broad phase
    
    //
    // vertex vs triangle
    //
    
    for ( size_t i = 0; i < get_num_vertices(); ++i )
    {
        if ( m_mesh.vertex_is_deleted(i) ) { continue; }
        
        // First, accelerated broad phase
        
        Vec3d aabb_low, aabb_high;
        vertex_continuous_bounds( i, aabb_low, aabb_high );
        
        std::vector<size_t> overlapping_triangles;
        m_broad_phase->get_potential_triangle_collisions(aabb_low, aabb_high, true, true, overlapping_triangles); 
        
        // filter deleted triangles
        for ( int k = 0; k < (int)overlapping_triangles.size(); ++k )
        {
            if ( m_mesh.triangle_is_deleted( overlapping_triangles[k] ) )
            {
                overlapping_triangles.erase( overlapping_triangles.begin() + k );
                --k;
            }
        }
        
        // Second, brute force check
        
        std::vector<size_t> brute_force_overlapping_triangles;
        
        for ( size_t j = 0; j < m_mesh.num_triangles(); ++j )
        {
            if ( m_mesh.triangle_is_deleted(j) ) { continue; }
            
            Vec3d tri_aabb_low, tri_aabb_high;
            triangle_continuous_bounds( j, tri_aabb_low, tri_aabb_high );
            
            if ( aabbs_intersect( aabb_low, aabb_high, tri_aabb_low, tri_aabb_high ) )
            {
                brute_force_overlapping_triangles.push_back( j );            
            }
        }
        
        assert( overlapping_triangles.size() == brute_force_overlapping_triangles.size() );
        
        std::sort(overlapping_triangles.begin(), overlapping_triangles.end());
        std::sort(brute_force_overlapping_triangles.begin(), brute_force_overlapping_triangles.end());
        
        for ( size_t k = 0; k < overlapping_triangles.size(); ++k )
        {
            assert( overlapping_triangles[k] == brute_force_overlapping_triangles[k] );
        }
        
    }
    
    //
    // edge vs edge
    //
    
    for ( size_t i = 0; i < m_mesh.m_edges.size(); ++i )
    {
        if ( m_mesh.edge_is_deleted(i) ) { continue; }
        
        // Accelerated broad phase
        
        Vec3d aabb_low, aabb_high;
        edge_continuous_bounds( i, aabb_low, aabb_high );
        
        std::vector<size_t> overlapping_edges;
        m_broad_phase->get_potential_edge_collisions( aabb_low, aabb_high, true, true, overlapping_edges );
        
        // filter deleted edges
        for ( int k = 0; k < (int)overlapping_edges.size(); ++k )
        {
            if ( m_mesh.edge_is_deleted( overlapping_edges[k] ) )
            {
                overlapping_edges.erase( overlapping_edges.begin() + k );
                --k;
            }
        }
        
        // Brute force
        std::vector<size_t> brute_force_overlapping_edges;
        for ( size_t j = 0; j < m_mesh.m_edges.size(); ++j )
        {
            if ( m_mesh.edge_is_deleted(j) ) { continue; }
            
            Vec3d edge_aabb_low, edge_aabb_high;
            edge_continuous_bounds( j, edge_aabb_low, edge_aabb_high );
            
            if ( aabbs_intersect( aabb_low, aabb_high, edge_aabb_low, edge_aabb_high ) )
            {
                brute_force_overlapping_edges.push_back( j );
            }
        }
        
        if ( overlapping_edges.size() != brute_force_overlapping_edges.size() )
        {
            
            std::cout << "edge " << i << ": " << m_mesh.m_edges[i] << std::endl;
            std::cout << "overlapping_edges.size(): " << overlapping_edges.size() << std::endl;
            for ( size_t k = 0; k < overlapping_edges.size(); ++k )
            {
                std::cout << k << ": " << overlapping_edges[k] << std::endl;
            }
            
            std::cout << "brute_force_overlapping_edges.size(): " << brute_force_overlapping_edges.size() << std::endl;
            for ( size_t k = 0; k < brute_force_overlapping_edges.size(); ++k )
            {
                std::cout << k << ": " << brute_force_overlapping_edges[k] << std::endl;
            }
            
        }
        
        assert( overlapping_edges.size() == brute_force_overlapping_edges.size() );
        
        std::sort( overlapping_edges.begin(), overlapping_edges.end() );
        std::sort( brute_force_overlapping_edges.begin(), brute_force_overlapping_edges.end() );
        
        for ( size_t k = 0; k < overlapping_edges.size(); ++k )
        {
            assert( overlapping_edges[k] == brute_force_overlapping_edges[k] );
        }
    }
    
    //
    // triangle vs vertex
    //
    
    for ( size_t i = 0; i < m_mesh.num_triangles(); ++i )
    {
        if ( m_mesh.triangle_is_deleted(i) ) { continue; }
        
        // Accelerated broad phase
        
        Vec3d aabb_low, aabb_high;
        triangle_continuous_bounds( i, aabb_low, aabb_high );
        
        std::vector<size_t> overlapping_vertices;
        m_broad_phase->get_potential_vertex_collisions( aabb_low, aabb_high, true, true, overlapping_vertices );
        
        // filter deleted vertices
        for ( int k = 0; k < (int)overlapping_vertices.size(); ++k )
        {
            if ( m_mesh.vertex_is_deleted( overlapping_vertices[k] ) )
            {
                overlapping_vertices.erase( overlapping_vertices.begin() + k );
                --k;
            }
        }
        
        // Brute force
        std::vector<size_t> brute_force_overlapping_vertices;
        for ( size_t j = 0; j < get_num_vertices(); ++j )
        {
            if ( m_mesh.vertex_is_deleted(j) ) { continue; }
            
            Vec3d vertex_aabb_low, vertex_aabb_high;
            vertex_continuous_bounds( j, vertex_aabb_low, vertex_aabb_high );
            
            if ( aabbs_intersect( aabb_low, aabb_high, vertex_aabb_low, vertex_aabb_high ) )
            {
                brute_force_overlapping_vertices.push_back( j );
            }
        }
        
        if ( overlapping_vertices.size() != brute_force_overlapping_vertices.size() )
        {
            std::cout << "triangle " << i << ": " << m_mesh.get_triangle(i) << std::endl;
            std::cout << "overlapping_vertices.size(): " << overlapping_vertices.size() << std::endl;
            for ( size_t k = 0; k < overlapping_vertices.size(); ++k )
            {
                std::cout << k << ": " << overlapping_vertices[k] << " --- ";
                std::cout << "is deleted: " << m_mesh.vertex_is_deleted( overlapping_vertices[k] ) << std::endl;
            }
            
            std::cout << "brute_force_overlapping_vertices.size(): " << brute_force_overlapping_vertices.size() << std::endl;
            for ( size_t k = 0; k < brute_force_overlapping_vertices.size(); ++k )
            {
                std::cout << k << ": " << brute_force_overlapping_vertices[k] << " --- ";
                std::cout << "is deleted: " << m_mesh.vertex_is_deleted( brute_force_overlapping_vertices[k] ) << std::endl;
                
                Vec3d lo, hi;
                bool is_solid = vertex_is_all_solid( brute_force_overlapping_vertices[k] );
                m_broad_phase->get_vertex_aabb( brute_force_overlapping_vertices[k], is_solid, lo, hi );
                std::cout << "AABB: " << lo << " - " << hi << std::endl;
                std::cout << "x: " << pm_positions[brute_force_overlapping_vertices[k]] << ", new_x: " << pm_newpositions[brute_force_overlapping_vertices[k]] << std::endl;
                
                bool query_overlaps_broadphase_aabb = aabbs_intersect( aabb_low, aabb_high, lo, hi );
                std::cout << "query_overlaps_broadphase_aabb: " << query_overlaps_broadphase_aabb << std::endl;
                
                
                BroadPhaseGrid* grid_bf = static_cast<BroadPhaseGrid*>(m_broad_phase);
                
                const std::vector<Vec3st>& cells = grid_bf->m_dynamic_vertex_grid.m_elementidxs[ brute_force_overlapping_vertices[k] ];
                std::cout << "cells: " << std::endl;
                for ( size_t m = 0; m < cells.size(); ++m )
                {
                    std::cout << cells[m] << std::endl;
                }
                
            }
        }
        
        assert( overlapping_vertices.size() == brute_force_overlapping_vertices.size() );
        
        std::sort( overlapping_vertices.begin(), overlapping_vertices.end() );
        std::sort( brute_force_overlapping_vertices.begin(), brute_force_overlapping_vertices.end() );
        
        for ( size_t k = 0; k < overlapping_vertices.size(); ++k )
        {
            assert( overlapping_vertices[k] == brute_force_overlapping_vertices[k] );
        }
        
    }
    
}


// --------------------------------------------------------
///
/// Check a triangle (by index) vs all other triangles for any kind of intersection
///
// --------------------------------------------------------

bool DynamicSurface::check_triangle_vs_all_triangles_for_intersection( size_t tri_index  )
{
    return check_triangle_vs_all_triangles_for_intersection( m_mesh.get_triangle(tri_index) );
}

// --------------------------------------------------------
///
/// Check a triangle vs all other triangles for any kind of intersection
///
// --------------------------------------------------------

bool DynamicSurface::check_triangle_vs_all_triangles_for_intersection( const Vec3st& tri )
{
    bool any_intersection = false;
    
    static std::vector<size_t> overlapping_triangles(20);
    overlapping_triangles.clear();

    Vec3d low, high;
    
    minmax( get_position(tri[0]), get_position(tri[1]), low, high );
    low -= Vec3d(m_aabb_padding);
    high += Vec3d(m_aabb_padding);
    
    m_broad_phase->get_potential_triangle_collisions( low, high, true, true, overlapping_triangles );
    
    for ( size_t i = 0; i < overlapping_triangles.size(); ++i )
    {
        
        const Vec3st& curr_tri = m_mesh.get_triangle( overlapping_triangles[i] );
        bool result = check_edge_triangle_intersection_by_index( tri[0], tri[1],
                                                                curr_tri[0], curr_tri[1], curr_tri[2],
                                                                get_positions(),
                                                                false );
        
        if ( result )
        {
            check_edge_triangle_intersection_by_index( tri[0], tri[1],
                                                      curr_tri[0], curr_tri[1], curr_tri[2],
                                                      get_positions(),
                                                      true );
            
            any_intersection = true;
        }
    }
    
    minmax( get_position(tri[1]), get_position(tri[2]), low, high );
    low -= Vec3d(m_aabb_padding);
    high += Vec3d(m_aabb_padding);
    
    overlapping_triangles.clear();
    m_broad_phase->get_potential_triangle_collisions( low, high, true, true,  overlapping_triangles );
    
    for ( size_t i = 0; i < overlapping_triangles.size(); ++i )
    {
        const Vec3st& curr_tri = m_mesh.get_triangle( overlapping_triangles[i] );
        
        bool result = check_edge_triangle_intersection_by_index( tri[1], tri[2],
                                                                curr_tri[0], curr_tri[1], curr_tri[2],
                                                                get_positions(),
                                                                false );
        
        if ( result )
        {
            check_edge_triangle_intersection_by_index( tri[1], tri[2],
                                                      curr_tri[0], curr_tri[1], curr_tri[2],
                                                      get_positions(),
                                                      true );
            
            any_intersection = true;
        }
    }
    
    minmax( get_position(tri[2]), get_position(tri[0]), low, high );
    low -= Vec3d(m_aabb_padding);
    high += Vec3d(m_aabb_padding);
    
    overlapping_triangles.clear();
    m_broad_phase->get_potential_triangle_collisions( low, high, true, true, overlapping_triangles );
    
    for ( size_t i = 0; i < overlapping_triangles.size(); ++i )
    {
        const Vec3st& curr_tri = m_mesh.get_triangle( overlapping_triangles[i] );
        
        bool result = check_edge_triangle_intersection_by_index( tri[2], tri[0],
                                                                curr_tri[0], curr_tri[1], curr_tri[2],
                                                                get_positions(),
                                                                false );
        
        if ( result )
        {
            check_edge_triangle_intersection_by_index( tri[2], tri[0],
                                                      curr_tri[0], curr_tri[1], curr_tri[2],
                                                      get_positions(),
                                                      true );
            
            any_intersection = true;         
        }
    }
    
    //
    // edges
    //
    
    minmax( get_position(tri[0]), get_position(tri[1]), get_position(tri[2]), low, high );
    low -= Vec3d(m_aabb_padding);
    high += Vec3d(m_aabb_padding);
    
    static std::vector<size_t> overlapping_edges(10);
    overlapping_edges.clear();
    m_broad_phase->get_potential_edge_collisions( low, high, true, true, overlapping_edges );
    
    for ( size_t i = 0; i < overlapping_edges.size(); ++i )
    {
        
        bool result = check_edge_triangle_intersection_by_index(m_mesh.m_edges[overlapping_edges[i]][0], 
                                                                m_mesh.m_edges[overlapping_edges[i]][1], 
                                                                tri[0], tri[1], tri[2],
                                                                get_positions(),
                                                                false );
        
        if ( result )
        {
            check_edge_triangle_intersection_by_index(m_mesh.m_edges[overlapping_edges[i]][0], 
                                                      m_mesh.m_edges[overlapping_edges[i]][1], 
                                                      tri[0], tri[1], tri[2],
                                                      get_positions(),
                                                      true );
            
            any_intersection = true;         
        }
    }
    
    return any_intersection;
}


// ---------------------------------------------------------
///
/// Detect all edge-triangle intersections.
///
// ---------------------------------------------------------

void DynamicSurface::get_intersections( bool degeneracy_counts_as_intersection, 
                                       bool use_new_positions, 
                                       std::vector<Intersection>& intersections )
{
    
    //#pragma omp parallel for schedule(guided)
    for ( int i = 0; i < (int)m_mesh.num_triangles(); ++i )
    {
        
       std::vector<size_t> edge_candidates(50);

        bool get_solid_edges = !triangle_is_all_solid(i);
        edge_candidates.clear();
        Vec3d low, high;
        triangle_static_bounds( i, low, high );       
        m_broad_phase->get_potential_edge_collisions( low, high, get_solid_edges, true, edge_candidates );
        
        const Vec3st& triangle = m_mesh.get_triangle(i);
        
        //skip deleted triangles
        if(m_mesh.triangle_is_deleted(i)) continue;
        
        for ( size_t j = 0; j < edge_candidates.size(); ++j )
        {
          
            const Vec2st& edge = m_mesh.m_edges[ edge_candidates[j] ];
            
            if(m_mesh.edge_is_deleted( edge_candidates[j] ) ) continue;

            if (    edge[0] == triangle[0] || edge[0] == triangle[1] || edge[0] == triangle[2] 
                || edge[1] == triangle[0] || edge[1] == triangle[1] || edge[1] == triangle[2] )
            {
                continue;
            }

            assert( !triangle_is_all_solid( i ) || !edge_is_all_solid( edge_candidates[j] ) );
            
            const Vec3d& e0 = use_new_positions ? get_newposition(edge[0]) : get_position(edge[0]);
            const Vec3d& e1 = use_new_positions ? get_newposition(edge[1]) : get_position(edge[1]);
            const Vec3d& t0 = use_new_positions ? get_newposition(triangle[0]) : get_position(triangle[0]);
            const Vec3d& t1 = use_new_positions ? get_newposition(triangle[1]) : get_position(triangle[1]);
            const Vec3d& t2 = use_new_positions ? get_newposition(triangle[2]) : get_position(triangle[2]);
            
            if ( segment_triangle_intersection( e0, edge[0], 
                                               e1, edge[1],
                                               t0, triangle[0], 
                                               t1, triangle[1], 
                                               t2, triangle[2], 
                                               degeneracy_counts_as_intersection, m_verbose ) )
            {
                std::cout << "intersection: " << edge << " vs " << triangle << std::endl;
                std::cout << "e0: " << e0 << std::endl;
                std::cout << "e1: " << e1 << std::endl;
                std::cout << "t0: " << t0 << std::endl;
                std::cout << "t1: " << t1 << std::endl;
                std::cout << "t2: " << t2 << std::endl;            
                //#pragma omp critical 
                {
                  intersections.push_back( Intersection( edge_candidates[j], i ) );
                }
            }
            
        }
        
    }
    
}

// ---------------------------------------------------------
///
/// Fire an assert if any edge is intersecting any triangles
///
// ---------------------------------------------------------

void DynamicSurface::assert_mesh_is_intersection_free( bool degeneracy_counts_as_intersection )
{
    
    std::vector<Intersection> intersections;
    get_intersections( degeneracy_counts_as_intersection, false, intersections );
    
    for ( size_t i = 0; i < intersections.size(); ++i )
    {
        
        const Vec3st& triangle = m_mesh.get_triangle( intersections[i].m_triangle_index );
        const Vec2st& edge = m_mesh.m_edges[ intersections[i].m_edge_index ];
        
        std::cout << "Intersection!  Triangle " << triangle << " vs edge " << edge << std::endl;
        
        segment_triangle_intersection( get_position(edge[0]), edge[0], 
                                      get_position(edge[1]), edge[1],
                                      get_position(triangle[0]), triangle[0],
                                      get_position(triangle[1]), triangle[1], 
                                      get_position(triangle[2]), triangle[2],
                                      true, true );
        
        assert( false );
        
    }
    
}


// ---------------------------------------------------------
///
/// Using m_newpositions as the geometry, fire an assert if any edge is intersecting any triangles.
/// This is a useful debugging tool, as it will detect any missed collisions before the mesh is advected
/// into an intersecting state.
///
// ---------------------------------------------------------

void DynamicSurface::assert_predicted_mesh_is_intersection_free( bool degeneracy_counts_as_intersection )
{
    
    std::vector<Intersection> intersections;
    get_intersections( degeneracy_counts_as_intersection, true, intersections );
    
    for ( size_t i = 0; i < intersections.size(); ++i )
    {
        
        const Vec3st& triangle = m_mesh.get_triangle( intersections[i].m_triangle_index );
        const Vec2st& edge = m_mesh.m_edges[ intersections[i].m_edge_index ];
        
        std::cout << "Intersection!  Triangle " << triangle << " vs edge " << edge << std::endl;
        
        segment_triangle_intersection(get_position(edge[0]), edge[0], 
                                      get_position(edge[1]), edge[1],
                                      get_position(triangle[0]), triangle[0],
                                      get_position(triangle[1]), triangle[1], 
                                      get_position(triangle[2]), triangle[2],
                                      true, true );
        
        const Vec3d& ea = get_position(edge[0]);
        const Vec3d& eb = get_position(edge[1]);
        const Vec3d& ta = get_position(triangle[0]);
        const Vec3d& tb = get_position(triangle[1]);
        const Vec3d& tc = get_position(triangle[2]);
        
       
        std::vector<Collision> check_collisions;
        m_collision_pipeline->detect_collisions( check_collisions );
        std::cout << "number of collisions detected: " << check_collisions.size() << std::endl;
        
        for ( size_t c = 0; c < check_collisions.size(); ++c )
        {
            const Collision& collision = check_collisions[c];
            std::cout << "Collision " << c << ": " << std::endl;
            if ( collision.m_is_edge_edge )
            {
                std::cout << "edge-edge: ";
            }
            else
            {
                std::cout << "point-triangle: ";
            }
            std::cout << collision.m_vertex_indices << std::endl;
            
        }
        
        std::cout << "-----\n edge-triangle check using m_positions:" << std::endl;
        
        bool result = segment_triangle_intersection( get_position(edge[0]), edge[0], 
                                                    get_position(edge[1]), edge[1],
                                                    get_position(triangle[0]), triangle[0], 
                                                    get_position(triangle[1]), triangle[1],
                                                    get_position(triangle[2]), triangle[2],
                                                    degeneracy_counts_as_intersection, 
                                                    m_verbose );
        
        std::cout << "result: " << result << std::endl;
        
        std::cout << "-----\n edge-triangle check using new m_positions" << std::endl;
        
        result = segment_triangle_intersection( get_newposition(edge[0]), edge[0], 
                                               get_newposition(edge[1]), edge[1],
                                               get_newposition(triangle[0]), triangle[0], 
                                               get_newposition(triangle[1]), triangle[1],
                                               get_newposition(triangle[2]), triangle[2],
                                               degeneracy_counts_as_intersection, 
                                               m_verbose );
        
        std::cout << "result: " << result << std::endl;
        
        const Vec3d& ea_new = get_newposition(edge[0]);
        const Vec3d& eb_new = get_newposition(edge[1]);
        const Vec3d& ta_new = get_newposition(triangle[0]);
        const Vec3d& tb_new = get_newposition(triangle[1]);
        const Vec3d& tc_new = get_newposition(triangle[2]);
        
        std::cout.precision(20);
        
        std::cout << "old: (edge0 edge1 tri0 tri1 tri2 )" << std::endl;
        
        std::cout << "Vec3d ea( " << ea[0] << ", " << ea[1] << ", " << ea[2] << ");" << std::endl;
        std::cout << "Vec3d eb( " << eb[0] << ", " << eb[1] << ", " << eb[2] << ");" << std::endl;            
        std::cout << "Vec3d ta( " << ta[0] << ", " << ta[1] << ", " << ta[2] << ");" << std::endl;
        std::cout << "Vec3d tb( " << tb[0] << ", " << tb[1] << ", " << tb[2] << ");" << std::endl;            
        std::cout << "Vec3d tc( " << tc[0] << ", " << tc[1] << ", " << tc[2] << ");" << std::endl;
        
        std::cout << "Vec3d ea_new( " << ea_new[0] << ", " << ea_new[1] << ", " << ea_new[2] << ");" << std::endl;
        std::cout << "Vec3d eb_new( " << eb_new[0] << ", " << eb_new[1] << ", " << eb_new[2] << ");" << std::endl;            
        std::cout << "Vec3d ta_new( " << ta_new[0] << ", " << ta_new[1] << ", " << ta_new[2] << ");" << std::endl;
        std::cout << "Vec3d tb_new( " << tb_new[0] << ", " << tb_new[1] << ", " << tb_new[2] << ");" << std::endl;            
        std::cout << "Vec3d tc_new( " << tc_new[0] << ", " << tc_new[1] << ", " << tc_new[2] << ");" << std::endl;
        
        std::vector<double> possible_times;
        
        Vec3d normal;
        
        std::cout << "-----" << std::endl;
        
        assert( !segment_segment_collision(ea, ea_new, edge[0], eb, eb_new, edge[1], 
                                           ta, ta_new, triangle[0], tb, tb_new, triangle[1] ) );
        
        std::cout << "-----" << std::endl;
        
        assert( !segment_segment_collision(ea, ea_new, edge[0], eb, eb_new, edge[1], 
                                           tb, tb_new, triangle[1], tc, tc_new, triangle[2] ) );
        
        std::cout << "-----" << std::endl;
        
        assert( !segment_segment_collision(ea, ea_new, edge[0], eb, eb_new, edge[1], 
                                           ta, ta_new, triangle[0], tc, tc_new, triangle[2] ) );
        
        std::cout << "-----" << std::endl;
        
        assert( !point_triangle_collision(ea, ea_new, edge[0], ta, ta_new, triangle[0], 
                                          tb, tb_new, triangle[1], tc, tc_new, triangle[2] ) );
        
        std::cout << "-----" << std::endl;
        
        assert( !point_triangle_collision(eb, eb_new, edge[1], ta, ta_new, triangle[0], 
                                          tb, tb_new, triangle[1], tc, tc_new, triangle[2] ) );
        
        m_verbose = false;
        
        std::cout << "no collisions detected" << std::endl;
        
        assert( false );
        
    }
    
}

}



