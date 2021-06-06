// ---------------------------------------------------------
//
//  trianglequality.cpp
//  Tyson Brochu 2011
//  Christopher Batty, Fang Da 2014
//
//  Functions for getting various triangle mesh measures.
//
// ---------------------------------------------------------

#include <trianglequality.h>
#include <limits>
#include <surftrack.h>
#include <set>

#include "lapack_wrapper.h"

// ---------------------------------------------------------
///
/// Determine the "mixed" voronoi and barycentric area of the vertex within the given triangle.
///
// ---------------------------------------------------------

namespace LosTopos {

double mixed_area( size_t vertex_index, size_t triangle_index, const SurfTrack& surf )
{
    const Vec3st& tri = surf.m_mesh.get_triangle(triangle_index);
    
    Vec2st opposite_edge;
    if ( vertex_index == tri[0] )
    {
        opposite_edge = Vec2st( tri[1], tri[2] );
    }
    else if ( vertex_index == tri[1] )
    {
        opposite_edge = Vec2st( tri[2], tri[0] );
    }
    else
    {
        opposite_edge = Vec2st( tri[0], tri[1] );
    }
    
    const Vec3d& a = surf.get_position(vertex_index);
    const Vec3d& b = surf.get_position(opposite_edge[0]);
    const Vec3d& c = surf.get_position(opposite_edge[1]);
    
    bool obtuse_triangle = ( ( dot(b-a, c-a) < 0.0 ) || ( dot(a-b, c-b) < 0.0 ) || ( dot(a-c, b-c) < 0.0 ) );
    
    if ( obtuse_triangle )
    {
        //std::cout << "obtuse_triangle " << triangle_index << ": " << tri << std::endl;
        
        if ( dot(b-a, c-a) < 0.0 )
        {
            // obtuse at a
            return 0.5 * surf.get_triangle_area( triangle_index );
        }
        else
        {
            // obtuse somewhere else
            return 0.25 * surf.get_triangle_area( triangle_index );
        }
    }
    else
    {
        // not obtuse, use voronoi area
        
        double cross_c = mag( cross( a-c, b-c ) );      
        double cot_c = dot( a-c, b-c) / cross_c;      
        
        double cross_b = mag( cross( a-b, c-b ) );      
        double cot_b = dot( a-b, c-b) / cross_b;      
        
        return 1.0 / 8.0 * (mag2(b-a) * cot_c + mag2(c-a) * cot_b);
    }
    
}

// ---------------------------------------------------------
///
/// Get Kappa * n, the surface normal multiplied by mean curvature at the specified vertex.
///
// ---------------------------------------------------------

void vertex_mean_curvature_normal( size_t vertex_index, const SurfTrack& surf, Vec3d& out, double& weight_sum )
{
    Vec3d mean_curvature_normal( 0, 0, 0 );
    weight_sum = 0;
    
    double edge_length_sum = 0.0;
    
    for ( size_t i = 0; i < surf.m_mesh.m_vertex_to_edge_map[vertex_index].size(); ++i )
    {
        size_t e = surf.m_mesh.m_vertex_to_edge_map[vertex_index][i];
        const Vec2st& curr_edge = surf.m_mesh.m_edges[e];
        Vec3d edge_vector;
        if ( curr_edge[0] == vertex_index )
        {
            edge_vector = surf.get_position( curr_edge[1] ) - surf.get_position( vertex_index );
        }
        else
        {
            assert( curr_edge[1] == vertex_index );
            edge_vector = surf.get_position( curr_edge[0] ) - surf.get_position( vertex_index );
        }
        
        edge_length_sum += mag( edge_vector );
        
        if ( surf.m_mesh.m_edge_to_triangle_map[e].size() != 2 )
        {
            // TODO: properly handle more than 2 incident triangles
            out = Vec3d(0,0,0);
            return;
        }
        
        size_t tri0 = surf.m_mesh.m_edge_to_triangle_map[e][0];
        size_t tri1 = surf.m_mesh.m_edge_to_triangle_map[e][1];
        
        size_t third_vertex_0 = surf.m_mesh.get_third_vertex( curr_edge[0], curr_edge[1], surf.m_mesh.get_triangle(tri0) );
        size_t third_vertex_1 = surf.m_mesh.get_third_vertex( curr_edge[0], curr_edge[1], surf.m_mesh.get_triangle(tri1) );
        
        Vec3d v00 = surf.get_position( curr_edge[0] ) - surf.get_position( third_vertex_0 );
        Vec3d v10 = surf.get_position( curr_edge[1] ) - surf.get_position( third_vertex_0 );
        
        double cross_0 = mag( cross( v00, v10 ) );
        if ( cross_0 < 1e-10 )
        {
            continue;
        }
        double cot_0 = dot(v00, v10) / cross_0;
        
        Vec3d v01 = surf.get_position( curr_edge[0] ) - surf.get_position( third_vertex_1 );
        Vec3d v11 = surf.get_position( curr_edge[1] ) - surf.get_position( third_vertex_1 );
        
        double cross_1 = mag( cross( v01, v11 ) );
        if ( cross_1 < 1e-10 )
        {
            continue;
        }
        
        double cot_1 = dot(v01, v11) / cross_1;
        
        double weight = cot_0 + cot_1;
        weight_sum += weight;
        
        mean_curvature_normal += weight * edge_vector;
        
    }
    
    double vertex_area = 0.0;
    for ( size_t i = 0; i < surf.m_mesh.m_vertex_to_triangle_map[vertex_index].size(); ++i )
    {
        vertex_area += mixed_area( vertex_index, surf.m_mesh.m_vertex_to_triangle_map[vertex_index][i], surf );
    }
    
    double coeff = 1.0 / (2.0 * vertex_area);
    
    weight_sum *= coeff;
    
    out = coeff * mean_curvature_normal;
    
}


// ---------------------------------------------------------
///
/// Return an estimate for mean curvature at the given vertex, computed using the Kappa * n estimate above.
///
// ---------------------------------------------------------

double unsigned_vertex_mean_curvature( size_t vertex_index, const SurfTrack& surf )
{
    Vec3d mc_normal;
    double weight_sum;
    
    vertex_mean_curvature_normal( vertex_index, surf, mc_normal, weight_sum );
    
    return mag( mc_normal );
}


// ---------------------------------------------------------
///
/// Return an estimate for curvature by computing the minimum radius of a sphere defined by the edge neighbourhood around a vertex,
/// and taking the reciprocal.
///
// ---------------------------------------------------------

double inv_min_radius_curvature( const SurfTrack& surf, size_t vertex )
{
    
    Vec3d normal = surf.get_vertex_normal( vertex );
    
    //   double min_radius = BIG_DOUBLE;
    
    double inv_min_radius = -BIG_DOUBLE;
    
    for ( size_t i = 0; i < surf.m_mesh.m_vertex_to_edge_map[vertex].size(); ++i )
    {
        size_t edge_index = surf.m_mesh.m_vertex_to_edge_map[vertex][i];
        
        assert( edge_index < surf.m_mesh.m_edges.size() );
        
        const Vec2st& edge = surf.m_mesh.m_edges[ edge_index ];
        
        Vec3d P;
        if ( edge[0] == vertex )
        {
            P = surf.get_position( edge[1] ) - surf.get_position( vertex );
        }
        else
        {
            P = surf.get_position( edge[0] ) - surf.get_position( vertex );
        }
        
        //      double radius = 0.5 * dot( P, P ) / dot( normal, P );
        //      min_radius = min( min_radius, radius );
        
        double inv_radius = 2.0 * dot( normal, P ) / dot( P, P );
        inv_min_radius = max( inv_min_radius, inv_radius );
        
    }
    
    return inv_min_radius;
    
}


// ---------------------------------------------------------
///
/// Compute curvature at a vertex using quadric fitting
///
// ---------------------------------------------------------

double estimated_max_curvature(const SurfTrack& surf, size_t vertex) {
    
    Vec3d normal = surf.get_vertex_normal(vertex);

    Vec3d seed(0,0,0);
    if(abs(normal[0]) < 0.5)
        seed[0] = 1.0;
    else
        seed[1] = 1.0;
    Vec3d u = cross(normal, seed);
    normalize(u);
    Vec3d v = cross(normal, u);

    std::set<size_t> twoneighbors;
    twoneighbors.insert(vertex);
    for(int i=0; i<2; i++)
    {
        std::set<size_t> toadd;
        for(std::set<size_t>::iterator it = twoneighbors.begin(); it != twoneighbors.end(); ++it)
        {
            size_t curVert = *it;
            const std::vector<size_t>& nbrEdges = surf.m_mesh.m_vertex_to_edge_map[curVert];
            for(unsigned int nbrIdx = 0; nbrIdx < nbrEdges.size(); ++nbrIdx) 
            {
                size_t edge_id = nbrEdges[nbrIdx];
                Vec2st edge = surf.m_mesh.m_edges[edge_id];
                size_t fv = edge[0];
                if(fv == curVert)
                    toadd.insert(edge[1]);
                else
                    toadd.insert(fv);
            }
        }
        for(std::set<size_t>::iterator it = toadd.begin(); it != toadd.end(); ++it)
            twoneighbors.insert(*it);
    }

    
    int numneighbors = (int)twoneighbors.size();
    if(numneighbors < 6)
        return 0;

    //a u^2 + b v^2 + c uv + d u + e v + f

    std::vector<double> Mat(numneighbors*6);
    std::vector<double> rhs(numneighbors);
    Vec3d centpt = surf.get_position(vertex);
    int row=0;
    for(std::set<size_t>::iterator it = twoneighbors.begin(); it != twoneighbors.end(); ++it)
    {
        size_t curVert = *it;
        Vec3d adjpt = surf.get_position(curVert);
        Vec3d diff;
        for(int i=0; i<3; i++)
            diff[i] = adjpt[i]-centpt[i];
        double uval = dot(diff,u);
        double vval = dot(diff,v);
        rhs[row] = dot(diff,normal);
        /*Mat[row*numneighbors] = uval*uval;
        Mat[row*numneighbors+1] = vval*vval;
        Mat[row*numneighbors+2] = uval*vval;
        Mat[row*numneighbors+3] = uval;
        Mat[row*numneighbors+4] = vval;
        Mat[row*numneighbors+5] = 1;*/
        Mat[0*numneighbors + row] = uval*uval;
        Mat[1*numneighbors + row] = vval*vval;
        Mat[2*numneighbors + row] = uval*vval;
        Mat[3*numneighbors + row] = uval;
        Mat[4*numneighbors + row] = vval;
        Mat[5*numneighbors + row] = 1;
        row++;
    }
    assert(row == numneighbors);

    int info;
    int rank;
    double rcond = 0;
    LAPACK::simple_least_squares_svd(numneighbors, 6, 1, &Mat[0], numneighbors, &rhs[0], info, rcond, rank);
    std::vector<double> coeffs(6);
    for(int id = 0; id < 6; ++id) coeffs[id] = rhs[id];
  
    double areaElt = sqrt(1+coeffs[3]*coeffs[3]+coeffs[4]*coeffs[4]);
    Vec3d newnormal = (normal - coeffs[3]*u - coeffs[4]*v) / areaElt;

    double E = 1 + coeffs[3]*coeffs[3];
    double F = coeffs[3]*coeffs[4];
    double G = 1 + coeffs[4]*coeffs[4];

    //double fac = normal.dot(newnormal);
    double L = 2*coeffs[0]/areaElt;
    double M = coeffs[2]/areaElt;
    double N = 2*coeffs[1]/areaElt;

    double det = E*G-F*F;
    if(abs(det) < 1e-6)
        return 0;

    Mat22d H(L, M, M, N);
    
    //LosTopos stores things in column major rather than row major, I believe, hence the transposing
    Mat33d U(u[0], v[0], normal[0], 
        u[1], v[1], normal[1], 
        u[2], v[2], normal[2]); 
    U = U.transpose();

    Mat32d J(1, 0, coeffs[3], 
             0, 1, coeffs[4]);


    J = U*J;
    
    Mat22d shapeOperator = 1/areaElt * inverse(J.transpose()*J) * H;

    //now do an eigensolve on the shape operator to find the principal curvatures/directions
    int n = 2, lwork = 10;
    double eigenvalues[2];
    double work[10];
    LAPACK::get_eigen_decomposition( &n, shapeOperator.a, &n, eigenvalues, work, &lwork, &info ); 
    double max_curvature = max(std::fabs(eigenvalues[0]), std::fabs(eigenvalues[1]));
    
    return max_curvature;
    
}

// ---------------------------------------------------------
///
/// Compute curvatures at all vertices using inv_min_radius_curvature.
///
// ---------------------------------------------------------

void compute_vertex_curvatures( const SurfTrack& surf, std::vector<double>& vertex_curvatures )
{
    
    vertex_curvatures.resize( surf.get_num_vertices() );
    
    for ( size_t i = 0; i < surf.get_num_vertices(); ++i )
    {
        
        if ( surf.m_mesh.m_is_boundary_vertex[i] ) 
        { 
            vertex_curvatures[i] = 1.0;
            continue; 
        }
        
        vertex_curvatures[i] = inv_min_radius_curvature( surf, i );
    }   
}


#define USE_INV_MIN_RADIUS

// ---------------------------------------------------------
///
/// Get the length of the specified edge, scaled by an estimate of curvature at each of the vertices.
///
// ---------------------------------------------------------

double get_curvature_scaled_length(const SurfTrack& surf, 
                                   size_t vertex_a, 
                                   size_t vertex_b, 
                                   double min_curvature_multiplier,
                                   double max_curvature_multiplier,
                                   double rest_curvature )
{
    
    assert( vertex_a < surf.get_num_vertices() );
    assert( vertex_b < surf.get_num_vertices() );
    
    double length = dist(  surf.get_position( vertex_a ), surf.get_position( vertex_b ) );
    
    //std::cout << "\n\nTrue length: " << length << std::endl;
#ifdef USE_INV_MIN_RADIUS
    double curv_a = std::fabs( inv_min_radius_curvature( surf, vertex_a ) );
#else
    double curv_a = unsigned_vertex_mean_curvature( vertex_a, surf );
#endif
    
    

    curv_a /= rest_curvature;
    curv_a = std::max( min_curvature_multiplier, curv_a );
    curv_a = std::min( max_curvature_multiplier, curv_a );
    
    //std::cout << "Curv a: " << curv_a << std::endl;

#ifdef USE_INV_MIN_RADIUS
    double curv_b = std::fabs( inv_min_radius_curvature( surf, vertex_b ) );
#else
    double curv_b = unsigned_vertex_mean_curvature( vertex_b, m_surf );
#endif
    
    curv_b /= rest_curvature;
    curv_b = std::max( min_curvature_multiplier, curv_b );
    curv_b = std::min( max_curvature_multiplier, curv_b );
    
    //std::cout << "Curv b: " << curv_b << std::endl;

    length *= 0.5 * ( curv_a + curv_b );
    
    //std::cout << "Multiplier: " << 0.5*(curv_a+curv_b) << std::endl;

    return length;
    
}


// ---------------------------------------------------------
///
/// Get the length of the specified edge, scaled by an estimate of curvature at each of the vertices.
///
// ---------------------------------------------------------

double get_edge_curvature(const SurfTrack& surf, 
  size_t vertex_a, 
  size_t vertex_b)
{

  assert( vertex_a < surf.get_num_vertices() );
  assert( vertex_b < surf.get_num_vertices() );

#ifdef USE_INV_MIN_RADIUS
  double curv_a = estimated_max_curvature( surf, vertex_a );
  //double curv_a = std::fabs( inv_min_radius_curvature( surf, vertex_a ) );
#else
  double curv_a = unsigned_vertex_mean_curvature( vertex_a, surf );
#endif

#ifdef USE_INV_MIN_RADIUS
  double curv_b = estimated_max_curvature( surf, vertex_b );
  //double curv_b = std::fabs( inv_min_radius_curvature( surf, vertex_b ) );
#else
  double curv_b = unsigned_vertex_mean_curvature( vertex_b, surf );
#endif

  return max(curv_a, curv_b);//0.5 * ( curv_a + curv_b );

}


// ---------------------------------------------------------
///
/// Return the minimum triangle area in the specified surface.
///
// ---------------------------------------------------------

double min_triangle_area( const SurfTrack& surf )
{
    double min_area = BIG_DOUBLE;
    for ( size_t i = 0; i < surf.m_mesh.num_triangles(); ++i )
    {
        if ( surf.m_mesh.triangle_is_deleted(i) ) { continue; }
        if ( surf.triangle_is_all_solid(i) ) { continue; }
        
        double area = surf.get_triangle_area(i);
        min_area = std::min( area, min_area );
    }
    
    return min_area;
    
}


// ---------------------------------------------------------
///
/// Return the minimun triangle angle in the specified surface.
///
// ---------------------------------------------------------

double min_triangle_angle( const SurfTrack& surf )
{
    double min_angle = BIG_DOUBLE;
    for ( size_t i = 0; i < surf.m_mesh.num_triangles(); ++i )
    {
        if ( surf.m_mesh.triangle_is_deleted(i) ) { continue; }
        
        const Vec3d& a = surf.get_position( surf.m_mesh.get_triangle(i)[0] );
        const Vec3d& b = surf.get_position( surf.m_mesh.get_triangle(i)[1] );
        const Vec3d& c = surf.get_position( surf.m_mesh.get_triangle(i)[2] );
        
        double curr_min_angle = min_triangle_angle( a, b, c );
        
        min_angle = std::min( curr_min_angle, min_angle );
    }
    
    return min_angle;
    
}


// ---------------------------------------------------------
///
/// Return the maximum triangle angle in the specified surface.
///
// ---------------------------------------------------------

double max_triangle_angle( const SurfTrack& surf )
{
    double max_angle = -BIG_DOUBLE;
    
    for ( size_t i = 0; i < surf.m_mesh.num_triangles(); ++i )
    {
        if ( surf.m_mesh.triangle_is_deleted(i) ) { continue; }
        
        const Vec3d& a = surf.get_position( surf.m_mesh.get_triangle(i)[0] );
        const Vec3d& b = surf.get_position( surf.m_mesh.get_triangle(i)[1] );
        const Vec3d& c = surf.get_position( surf.m_mesh.get_triangle(i)[2] );
        
        double curr_max_angle = max_triangle_angle( a, b, c );
        
        max_angle = std::max( curr_max_angle, max_angle );
    }
    
    return max_angle;
}   


// ---------------------------------------------------------
///
/// Count the number of triangle angles below the given threshold.
///
// ---------------------------------------------------------

size_t num_angles_below_threshold( const SurfTrack& surf, double low_threshold )
{
    size_t num_small_angles = 0;
    
    for ( size_t i = 0; i < surf.m_mesh.num_triangles(); ++i )
    {
        if ( surf.m_mesh.triangle_is_deleted(i) ) { continue; }
        
        const Vec3d& a = surf.get_position( surf.m_mesh.get_triangle(i)[0] );
        const Vec3d& b = surf.get_position( surf.m_mesh.get_triangle(i)[1] );
        const Vec3d& c = surf.get_position( surf.m_mesh.get_triangle(i)[2] );
        
        double angle_a, angle_b, angle_c;
        triangle_angles( a, b, c, angle_a, angle_b, angle_c );
        
        if ( angle_a < low_threshold ) { ++num_small_angles; }
        if ( angle_b < low_threshold ) { ++num_small_angles; }
        if ( angle_c < low_threshold ) { ++num_small_angles; }
    }
    
    return num_small_angles;
    
}

// ---------------------------------------------------------
///
/// Count the number of triangle angles above the given threshold.
///
// ---------------------------------------------------------

size_t num_angles_above_threshold( const SurfTrack& surf, double high_threshold )
{
    size_t num_large_angles = 0;
    
    for ( size_t i = 0; i < surf.m_mesh.num_triangles(); ++i )
    {
        if ( surf.m_mesh.triangle_is_deleted(i) ) { continue; }
        
        const Vec3d& a = surf.get_position( surf.m_mesh.get_triangle(i)[0] );
        const Vec3d& b = surf.get_position( surf.m_mesh.get_triangle(i)[1] );
        const Vec3d& c = surf.get_position( surf.m_mesh.get_triangle(i)[2] );
        
        double angle_a, angle_b, angle_c;
        triangle_angles( a, b, c, angle_a, angle_b, angle_c );
        
        if ( angle_a > high_threshold ) { ++num_large_angles; }
        if ( angle_b > high_threshold ) { ++num_large_angles; }
        if ( angle_c > high_threshold ) { ++num_large_angles; }
    }
    
    return num_large_angles;
    
}


// ---------------------------------------------------------
///
/// Compute the aspect ratio of the given triangle
///
// ---------------------------------------------------------

double triangle_aspect_ratio( const SurfTrack& surf, size_t triangle_index )
{
    const Vec3st& tri = surf.m_mesh.get_triangle(triangle_index);
    assert( tri[0] != tri[1] );
    return triangle_aspect_ratio( surf.get_position(tri[0]), surf.get_position(tri[1]), surf.get_position(tri[2]) );   
}

// ---------------------------------------------------------
///
/// Find the smallest triangle aspect ratio in the given mesh
///
// ---------------------------------------------------------

double min_triangle_aspect_ratio( const SurfTrack& surf, size_t& output_triangle_index )
{
    double min_ratio = std::numeric_limits<double>::max();
    output_triangle_index = (size_t)~0;
    
    for ( size_t i = 0; i < surf.m_mesh.num_triangles(); ++i )
    {
        double a_ratio = triangle_aspect_ratio( surf, i );
        if ( a_ratio < min_ratio )
        {
            output_triangle_index = i;
            min_ratio = a_ratio;
        }
    }
    return min_ratio;
}


// ---------------------------------------------------------
///
/// Find the greatest triangle aspect ratio in the given mesh
///
// ---------------------------------------------------------

double max_triangle_aspect_ratio( const SurfTrack& surf, size_t& output_triangle_index )
{
    double max_ratio = -1.0;
    output_triangle_index = (size_t)~0;
    
    for ( size_t i = 0; i < surf.m_mesh.num_triangles(); ++i )
    {
        double a_ratio = triangle_aspect_ratio( surf, i );
        if ( a_ratio > max_ratio )
        {
            output_triangle_index = i;
            max_ratio = a_ratio;
        }
    }
    return max_ratio;
}

}
