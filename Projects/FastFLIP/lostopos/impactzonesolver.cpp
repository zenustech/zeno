// ---------------------------------------------------------
//
//  impactzonesolver.cpp
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//
//  Encapsulates two impact zone solvers: inelastic impact zones, and rigid impact zones.
//
// ---------------------------------------------------------

#include <collisionpipeline.h>
#include <dynamicsurface.h>
#include <impactzonesolver.h>
#include <krylov_solvers.h>
#include <mat.h>
#include <sparse_matrix.h>
#include <runstats.h>

namespace LosTopos {

namespace {
    
    // ---------------------------------------------------------
    ///
    /// Combine impact zones which have overlapping vertex stencils
    ///
    // ---------------------------------------------------------
    
    void merge_impact_zones( std::vector<ImpactZone>& new_impact_zones, std::vector<ImpactZone>& master_impact_zones )
    {
        
        bool merge_ocurred = true;
        
        for ( size_t i = 0; i < master_impact_zones.size(); ++i )
        {
            master_impact_zones[i].m_all_solved = true;
        }
        
        for ( size_t i = 0; i < new_impact_zones.size(); ++i )
        {
            new_impact_zones[i].m_all_solved = false;
        }
        
        
        while ( merge_ocurred )
        {
            
            merge_ocurred = false;
            
            for ( size_t i = 0; i < new_impact_zones.size(); ++i )
            {
                bool i_is_disjoint = true;
                
                for ( size_t j = 0; j < master_impact_zones.size(); ++j )
                {
                    // check if impact zone i and j share any vertices
                    
                    if ( master_impact_zones[j].share_vertices( new_impact_zones[i] ) )
                    {
                        
                        bool found_new_collision = false;
                        
                        // steal all of j's collisions
                        for ( size_t c = 0; c < new_impact_zones[i].m_collisions.size(); ++c )
                        {
                            
                            bool same_collision_exists = false;
                            
                            for ( size_t m = 0; m < master_impact_zones[j].m_collisions.size(); ++m )
                            {                 
                                if ( master_impact_zones[j].m_collisions[m].same_vertices( new_impact_zones[i].m_collisions[c] ) )
                                {
                                    
                                    same_collision_exists = true;
                                    break;
                                }
                                
                            }
                            
                            if ( !same_collision_exists )
                            {
                                master_impact_zones[j].m_collisions.push_back( new_impact_zones[i].m_collisions[c] );
                                found_new_collision = true;
                            }
                        }
                        
                        // did we find any collisions in zone i that zone j didn't already have?
                        if ( found_new_collision )
                        {
                            master_impact_zones[j].m_all_solved &= new_impact_zones[i].m_all_solved;
                        }
                        
                        merge_ocurred = true;
                        i_is_disjoint = false;
                        break;
                    }
                    
                }     // end for(j)
                
                if ( i_is_disjoint )
                {
                    // copy the impact zone
                    
                    ImpactZone new_zone;
                    for ( size_t c = 0; c < new_impact_zones[i].m_collisions.size(); ++c )
                    {
                        new_zone.m_collisions.push_back( new_impact_zones[i].m_collisions[c] );
                    }
                    
                    new_zone.m_all_solved = new_impact_zones[i].m_all_solved;
                    
                    master_impact_zones.push_back( new_zone );
                }
            }     // end for(i)
            
            new_impact_zones = master_impact_zones;
            master_impact_zones.clear();
            
        }  // while
        
        master_impact_zones = new_impact_zones;
        
    }
    
    // ---------------------------------------------------------
    ///
    /// Helper function: multiply transpose(A) * D * B
    ///
    // ---------------------------------------------------------
    
    void AtDB(const SparseMatrixDynamicCSR &A, const double* diagD, const SparseMatrixDynamicCSR &B, SparseMatrixDynamicCSR &C)
    {
        assert(A.m==B.m);
        C.resize(A.n, B.n);
        C.set_zero();
        for(int k=0; k<A.m; ++k)
        {
            const DynamicSparseVector& r = A.row[k];
            
            for( DynamicSparseVector::const_iterator p=r.begin(); p != r.end(); ++p )
            {
                int i = p->index;
                double multiplier = p->value * diagD[k];
                C.add_sparse_row( i, B.row[k], multiplier );
            }
        }
    }
    
}  // unnamed namespace 


// ---------------------------------------------------------
///
/// Constructor
///
// ---------------------------------------------------------

ImpactZoneSolver::ImpactZoneSolver( DynamicSurface& surface) :
m_surface( surface ),
m_rigid_zone_infinite_mass( 1000.0 )
{}


// ---------------------------------------------------------
///
/// Iteratively project out relative normal velocities for a set of collisions in an impact zone until all collisions are solved.
///
// ---------------------------------------------------------

bool ImpactZoneSolver::iterated_inelastic_projection( ImpactZone& iz, double dt )
{
    assert( m_surface.m_masses.size() == m_surface.get_num_vertices() );
    
    static const unsigned int MAX_PROJECTION_ITERATIONS = 20;
    
    for ( unsigned int i = 0; i < MAX_PROJECTION_ITERATIONS; ++i )
    {
        bool success = inelastic_projection( iz );
        
        if ( !success )
        {
            if ( m_surface.m_verbose ) { std::cout << "failure in inelastic projection" << std::endl; }
            return false;
        }
        
        bool collision_still_exists = false;
        
        for ( size_t c = 0; c < iz.m_collisions.size(); ++c )
        {
            
            // run collision detection on this pair again
            
            Collision& collision = iz.m_collisions[c];
            const Vec4st& vs = collision.m_vertex_indices;
            
            m_surface.set_newposition( vs[0], m_surface.get_position(vs[0]) + dt * m_surface.m_velocities[vs[0]]);
            m_surface.set_newposition( vs[1], m_surface.get_position(vs[1]) + dt * m_surface.m_velocities[vs[1]]);
            m_surface.set_newposition( vs[2], m_surface.get_position(vs[2]) + dt * m_surface.m_velocities[vs[2]]);
            m_surface.set_newposition( vs[3], m_surface.get_position(vs[3]) + dt * m_surface.m_velocities[vs[3]]);         
            
            if ( m_surface.m_verbose ) { std::cout << "checking collision " << vs << std::endl; }
            
            if ( collision.m_is_edge_edge )
            {
                
                double s0, s2, rel_disp;
                Vec3d normal;
                
                assert( vs[0] < vs[1] && vs[2] < vs[3] );       // should have been sorted by original collision detection
                
                if ( segment_segment_collision( m_surface.get_position(vs[0]), m_surface.get_newposition(vs[0]), vs[0],
                                               m_surface.get_position(vs[1]), m_surface.get_newposition(vs[1]), vs[1],
                                               m_surface.get_position(vs[2]), m_surface.get_newposition(vs[2]), vs[2],
                                               m_surface.get_position(vs[3]), m_surface.get_newposition(vs[3]), vs[3],
                                               s0, s2,
                                               normal,
                                               rel_disp ) )               
                {
                    collision.m_normal = normal;
                    collision.m_alphas = Vec4d( -s0, -(1-s0), s2, (1-s2) );
                    collision.m_relative_displacement = rel_disp;
                    collision_still_exists = true;
                }
                
            }
            else
            {
                
                double s1, s2, s3, rel_disp;
                Vec3d normal;
                
                assert( vs[1] < vs[2] && vs[2] < vs[3] && vs[1] < vs[3] );    // should have been sorted by original collision detection
                
                if ( point_triangle_collision( m_surface.get_position(vs[0]), m_surface.get_newposition(vs[0]), vs[0],
                                              m_surface.get_position(vs[1]), m_surface.get_newposition(vs[1]), vs[1],
                                              m_surface.get_position(vs[2]), m_surface.get_newposition(vs[2]), vs[2],
                                              m_surface.get_position(vs[3]), m_surface.get_newposition(vs[3]), vs[3],
                                              s1, s2, s3,
                                              normal,
                                              rel_disp ) )                                 
                {
                    collision.m_normal = normal;
                    collision.m_alphas = Vec4d( 1, -s1, -s2, -s3 );
                    collision.m_relative_displacement = rel_disp;
                    collision_still_exists = true;
                }
                
            }
            
        } // for collisions
        
        if ( false == collision_still_exists )  
        {
            return true; 
        }
        
    } // for iterations
    
    if ( m_surface.m_verbose ) 
    { 
        std::cout << "reached max iterations for this zone" << std::endl; 
    }
    
    return false;
    
}


// ---------------------------------------------------------
///
/// Project out relative normal velocities for a set of collisions in an impact zone.
///
// ---------------------------------------------------------

bool ImpactZoneSolver::inelastic_projection( const ImpactZone& iz )
{
    
    if ( m_surface.m_verbose )
    {
        std::cout << " ----- using sparse solver " << std::endl;
    }
    
    const size_t k = iz.m_collisions.size();    // notation from [Harmon et al 2008]: k == number of collisions
    
    std::vector<size_t> zone_vertices;
    iz.get_all_vertices( zone_vertices );
    
    const size_t n = zone_vertices.size();       // n == number of distinct colliding vertices
    
    if ( m_surface.m_verbose ) { std::cout << "GCT: " << 3*n << "x" << k << std::endl; }
    
    SparseMatrixDynamicCSR GCT( to_int(3*n), to_int(k) );
    GCT.set_zero();
    
    // construct matrix grad C transpose
    for ( int i = 0; i < to_int(k); ++i )
    {
        // set col i
        const Collision& coll = iz.m_collisions[i];
        
        for ( unsigned int v = 0; v < 4; ++v )
        {
            // block row j ( == block column j of grad C )
            size_t j = coll.m_vertex_indices[v];
            
            std::vector<size_t>::iterator zone_vertex_iter = find( zone_vertices.begin(), zone_vertices.end(), j );
            
            assert( zone_vertex_iter != zone_vertices.end() );
            
            int mat_j = to_int( zone_vertex_iter - zone_vertices.begin() );
            
            GCT(mat_j*3, i) = coll.m_alphas[v] * coll.m_normal[0];
            GCT(mat_j*3+1, i) = coll.m_alphas[v] * coll.m_normal[1];
            GCT(mat_j*3+2, i) = coll.m_alphas[v] * coll.m_normal[2];
            
        }
    }
    
    Array1d inv_masses;
    inv_masses.reserve(3*(unsigned long)n);
    Array1d column_velocities;
    column_velocities.reserve(3*(unsigned long)n);
    
    for ( size_t i = 0; i < n; ++i )
    {
        
        inv_masses.push_back( 1.0 / m_surface.m_masses[zone_vertices[i]][0] );
        inv_masses.push_back( 1.0 / m_surface.m_masses[zone_vertices[i]][1] );
        inv_masses.push_back( 1.0 / m_surface.m_masses[zone_vertices[i]][2] );
        
        column_velocities.push_back( m_surface.m_velocities[zone_vertices[i]][0] );
        column_velocities.push_back( m_surface.m_velocities[zone_vertices[i]][1] );
        column_velocities.push_back( m_surface.m_velocities[zone_vertices[i]][2] );
    }
    
    //
    // minimize | M^(-1/2) * GC^T x - M^(1/2) * v |^2
    //
    
    // solution vector
    Array1d x((unsigned long)k);
    
    KrylovSolverStatus solver_result;
    
    // normal equations: GC * M^(-1) GCT * x = GC * v
    //                   A * x = b
    
    SparseMatrixDynamicCSR A( to_int(k), to_int(k) );
    A.set_zero();
    AtDB( GCT, inv_masses.data, GCT, A ); 
    
    Array1d b((unsigned long)k);
    GCT.apply_transpose( column_velocities.data, b.data );   
    
    if ( m_surface.m_verbose )  { std::cout << "system built" << std::endl; }
    
    MINRES_CR_Solver solver;   
    SparseMatrixStaticCSR solver_matrix( A );    // convert dynamic to static
    solver.max_iterations = 1000;
    solver_result = solver.solve( solver_matrix, b.data, x.data ); 
    
    if ( solver_result != KRYLOV_CONVERGED )
    {
        if ( m_surface.m_verbose )
        {
            std::cout << "CR solver failed: ";      
            if ( solver_result == KRYLOV_BREAKDOWN )
            {
                std::cout << "KRYLOV_BREAKDOWN" << std::endl;
            }
            else
            {
                std::cout << "KRYLOV_EXCEEDED_MAX_ITERATIONS" << std::endl;
            }
            
            double residual_norm = BLAS::abs_max(solver.r);
            std::cout << "residual_norm: " << residual_norm << std::endl;
            
        }
        
        return false;          
    } 
    
    // apply impulses 
    Array1d applied_impulses(3*(unsigned long)n);
    GCT.apply( x.data, applied_impulses.data );
    
    static const double IMPULSE_MULTIPLIER = 0.8;
    
    for ( size_t i = 0; i < applied_impulses.size(); ++i )
    {
        column_velocities[(unsigned long)i] -= IMPULSE_MULTIPLIER * inv_masses[(unsigned long)i] * applied_impulses[(unsigned long)i];      
    }
    
    for ( size_t i = 0; i < n; ++i )
    {
        m_surface.m_velocities[zone_vertices[i]][0] = column_velocities[3*(unsigned long)i];
        m_surface.m_velocities[zone_vertices[i]][1] = column_velocities[3*(unsigned long)i + 1];
        m_surface.m_velocities[zone_vertices[i]][2] = column_velocities[3*(unsigned long)i + 2];      
    }
    
    
    return true;
    
}


// ---------------------------------------------------------
///
/// Handle all collisions simultaneously by iteratively solving individual impact zones until no new collisions are detected.
///
// ---------------------------------------------------------

bool ImpactZoneSolver::inelastic_impact_zones(double dt)
{
    
    // copy
    std::vector<Vec3d> old_velocities = m_surface.m_velocities;
    
    std::vector<ImpactZone> impact_zones;
    
    bool finished_detecting_collisions = false;
    
    std::vector<Collision> total_collisions;
    finished_detecting_collisions = m_surface.m_collision_pipeline->detect_collisions(total_collisions);
    
    while ( false == total_collisions.empty() )
    {      
        // insert each new collision constraint into its own impact zone
        std::vector<ImpactZone> new_impact_zones;
        for ( size_t i = 0; i < total_collisions.size(); ++i )
        {
            ImpactZone new_zone;
            new_zone.m_collisions.push_back( total_collisions[i] );
            new_impact_zones.push_back( new_zone );
        }
        
        // now we have one zone for each collision
        assert( new_impact_zones.size() == total_collisions.size() );
        
        // merge all impact zones that share vertices
        merge_impact_zones( new_impact_zones, impact_zones );
        
        // remove impact zones which have been solved
        for ( int i = 0; i < (int) impact_zones.size(); ++i )
        {
            if ( impact_zones[i].m_all_solved ) 
            {
                impact_zones.erase( impact_zones.begin() + i );
                --i;
            }
        }
        
        for ( int i = 0; i < (int) impact_zones.size(); ++i )
        {
            assert( false == impact_zones[i].m_all_solved );
        }            
        
        bool all_zones_solved_ok = true;
        
        // for each impact zone
        for ( size_t i = 0; i < impact_zones.size(); ++i )
        {
            
            // reset impact zone to pre-response m_velocities
            for ( size_t j = 0; j < impact_zones[i].m_collisions.size(); ++j )
            {
                const Vec4st& vs = impact_zones[i].m_collisions[j].m_vertex_indices;            
                m_surface.m_velocities[vs[0]] = old_velocities[vs[0]];
                m_surface.m_velocities[vs[1]] = old_velocities[vs[1]];
                m_surface.m_velocities[vs[2]] = old_velocities[vs[2]];
                m_surface.m_velocities[vs[3]] = old_velocities[vs[3]]; 
            }
            
            // apply inelastic projection
            
            all_zones_solved_ok &= iterated_inelastic_projection( impact_zones[i], dt );
            
            // reset predicted positions
            for ( size_t j = 0; j < impact_zones[i].m_collisions.size(); ++j )
            {
                const Vec4st& vs = impact_zones[i].m_collisions[j].m_vertex_indices;            
                
                m_surface.set_newposition( vs[0], m_surface.get_position(vs[0]) + dt * m_surface.m_velocities[vs[0]] );
                m_surface.set_newposition( vs[1], m_surface.get_position(vs[1]) + dt * m_surface.m_velocities[vs[1]] );
                m_surface.set_newposition( vs[2], m_surface.get_position(vs[2]) + dt * m_surface.m_velocities[vs[2]] );
                m_surface.set_newposition( vs[3], m_surface.get_position(vs[3]) + dt * m_surface.m_velocities[vs[3]] );
                
            } 
            
        }  // for IZs
        
        
        if ( false == all_zones_solved_ok )
        {
            if ( m_surface.m_verbose ) 
            { 
                std::cout << "at least one impact zone had a solver problem" << std::endl; 
            }
            
            return false;
        }
        
        total_collisions.clear();
        
        if ( !finished_detecting_collisions )
        {
            if ( m_surface.m_verbose ) { std::cout << "attempting to finish global collision detection" << std::endl; }
            finished_detecting_collisions = m_surface.m_collision_pipeline->detect_collisions( total_collisions );
            impact_zones.clear();
        }
        else
        {
            bool detect_ok = m_surface.m_collision_pipeline->detect_new_collisions( impact_zones, total_collisions );
            if ( !detect_ok )
            {
                return false;
            }
        }
        
    }
    
    return true;
    
}



// ---------------------------------------------------------
///
///  Rigid Impact Zones, as described in [Bridson, Fedkiw, Anderson 2002].
///
// ---------------------------------------------------------
extern RunStats g_stats;

bool ImpactZoneSolver::rigid_impact_zones(double dt)
{
    
    
    g_stats.add_to_int( "ImpactZoneSolver:rigid_impact_zones", 1 );
    
    // copy
    std::vector<Vec3d> old_velocities = m_surface.m_velocities;
    
    std::vector<ImpactZone> impact_zones;
    
    bool finished_detecting_collisions = false;
    
    std::vector<Collision> total_collisions;
    finished_detecting_collisions = m_surface.m_collision_pipeline->detect_collisions(total_collisions);
    
    while ( false == total_collisions.empty() )
    {      
        // insert each new collision constraint into its own impact zone
        std::vector<ImpactZone> new_impact_zones;
        for ( size_t i = 0; i < total_collisions.size(); ++i )
        {
            ImpactZone new_zone;
            new_zone.m_collisions.push_back( total_collisions[i] );
            new_impact_zones.push_back( new_zone );
            
        }  // for loop over total_collisions
        
        assert( new_impact_zones.size() == total_collisions.size() );
        
        // merge all impact zones that share vertices
        merge_impact_zones( new_impact_zones, impact_zones );
        
        for ( int i = 0; i < (int) impact_zones.size(); ++i )
        {
            if ( impact_zones[i].m_all_solved ) 
            {
                impact_zones[i].m_all_solved = false;
                impact_zones.erase( impact_zones.begin() + i );
                --i;
            }
        }
        
        for ( int i = 0; i < (int) impact_zones.size(); ++i )
        {
            assert( false == impact_zones[i].m_all_solved );
        }            
        
        // for each impact zone
        for ( size_t i = 0; i < impact_zones.size(); ++i )
        {
            
            std::vector<size_t> zone_vertices;
            impact_zones[i].get_all_vertices( zone_vertices );
            bool rigid_motion_ok = calculate_rigid_motion(dt, zone_vertices);
            
            if ( !rigid_motion_ok )
            {
                std::cout << "rigid impact zone fails" << std::endl;
                return false;
            }
            
        }  
        
        total_collisions.clear();
        
        if ( !finished_detecting_collisions )
        {
            finished_detecting_collisions = m_surface.m_collision_pipeline->detect_collisions( total_collisions );
            impact_zones.clear();
        }
        else
        {
            bool detect_ok = m_surface.m_collision_pipeline->detect_new_collisions( impact_zones, total_collisions );
            std::cout << "new collisions detected: " << total_collisions.size() << std::endl;
            
            if ( !detect_ok )
            {
                return false;
            }
            
        }
        
    }
    
    return true;
}


// ---------------------------------------------------------
///
/// Compute the best-fit rigid motion for the set of moving vertices
///
// ---------------------------------------------------------

bool ImpactZoneSolver::calculate_rigid_motion(double dt, std::vector<size_t>& vs)
{
    Vec3d xcm(0,0,0);
    Vec3d vcm(0,0,0);
    double mass = 0;
    
    for(size_t i = 0; i < vs.size(); i++)
    {
        size_t idx = vs[i];
        
        double m = (m_surface.m_masses[idx][0] + m_surface.m_masses[idx][1] + m_surface.m_masses[idx][2]) / 3.0;
        
        if ( m_surface.vertex_is_any_solid(idx) )
        {
            m = m_rigid_zone_infinite_mass;
        }
        
        assert( m != DynamicSurface::solid_mass());
        
        mass += m;
        
        m_surface.m_velocities[idx] = ( m_surface.get_newposition(idx) - m_surface.get_position(idx) ) / dt;
        
        xcm += m * m_surface.get_position(idx);
        vcm += m * m_surface.m_velocities[idx];
    }
    
    
    double min_dist_t0 = 1e+30;
    double min_dist_t1 = 1e+30;
    for(size_t i = 0; i < vs.size(); i++)
    {
        for(size_t j = i+1; j < vs.size(); j++)
        {
            min_dist_t0 = min( min_dist_t0, dist( m_surface.get_position(vs[i]), m_surface.get_position(vs[j]) ) );
            min_dist_t1 = min( min_dist_t1, dist( m_surface.get_newposition(vs[i]), m_surface.get_newposition(vs[j]) ) );
        }
    }
    
    assert( mass > 0 );
    
    xcm /= mass;
    vcm /= mass;
    
    Vec3d L(0,0,0);
    
    for(size_t i = 0; i < vs.size(); i++)
    {
        size_t idx = vs[i];
        
        double m = (m_surface.m_masses[idx][0] + m_surface.m_masses[idx][1] + m_surface.m_masses[idx][2]) / 3.0;
        
        if ( m_surface.vertex_is_any_solid(idx) )
        {
            m = m_rigid_zone_infinite_mass;
        }
        
        assert( m != DynamicSurface::solid_mass());
        
        Vec3d xdiff = m_surface.get_position(idx) - xcm;
        Vec3d vdiff = m_surface.m_velocities[idx] - vcm;
        
        L += m * cross(xdiff, vdiff);
    }
    
    Mat33d I(0,0,0,0,0,0,0,0,0);
    
    for(size_t i = 0; i < vs.size(); i++)
    {
        size_t idx = vs[i];
        double m = (m_surface.m_masses[idx][0] + m_surface.m_masses[idx][1] + m_surface.m_masses[idx][2]) / 3.0;
        
        if ( m_surface.vertex_is_any_solid(idx) )
        {
            m = m_rigid_zone_infinite_mass;
        }
        
        assert( m != DynamicSurface::solid_mass());
        
        Vec3d xdiff = m_surface.get_position(idx) - xcm;
        Mat33d tens = outer(-xdiff, xdiff);
        
        double d = mag2(xdiff);
        tens(0,0) += d;
        tens(1,1) += d;
        tens(2,2) += d;
        
        I += m * tens;
    }
    
    double det = determinant(I);
    assert( det != 0 );   
    Vec3d w = inverse(I) * L;
    double wmag = mag(w);
    
    if ( wmag == 0 )
    {
        return false;
    }
    
    assert( wmag > 0 );
    
    Vec3d wnorm = w/wmag;
    
    double cosdtw = cos(dt * wmag);
    Vec3d sindtww = sin(dt * wmag) * wnorm;
    
    Vec3d xrigid = xcm + dt * vcm;
    
    double max_velocity_mag = -1.0;
    
    for(size_t i = 0; i < vs.size(); i++)
    {
        size_t idx = vs[i];
        
        Vec3d xdiff = m_surface.get_position(idx) - xcm;
        Vec3d xf = dot(xdiff, wnorm) * wnorm;
        Vec3d xr = xdiff - xf;
        
        m_surface.set_newposition( idx, xrigid + xf + cosdtw * xr + cross(sindtww, xr) );
        
        m_surface.m_velocities[idx] = ( m_surface.get_newposition(idx) - m_surface.get_position(idx) ) / dt;
        
        max_velocity_mag = max( max_velocity_mag, mag( m_surface.m_velocities[idx] ) );
        
    }
    
    min_dist_t1 = 1e+30;
    for(size_t i = 0; i < vs.size(); i++)
    {
        for(size_t j = i+1; j < vs.size(); j++)
        {
            min_dist_t1 = min( min_dist_t1, dist( m_surface.get_newposition(vs[i]), m_surface.get_newposition(vs[j]) ) );
        }
    }
    
    return true;
    
}

}

