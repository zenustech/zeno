
#include <rootparitycollisiontest.h>
#include <cstdlib>
namespace LosTopos {
namespace rootparity 
{
    
    namespace   // unnamed namespace for local functions
    {
        
        ///
        /// Local helper functions
        ///
        
        template<unsigned int N, class T>
        inline void make_vector( const Vec<N,double>& in, Vec<N,T>& out );
        
        template<class T>
        void point_triangle_collision_function(const Vec3d &d_x0,    const Vec3d &d_x1,    const Vec3d &d_x2,    const Vec3d &d_x3,
                                               const Vec3d &d_x0new, const Vec3d &d_x1new, const Vec3d &d_x2new, const Vec3d &d_x3new,
                                               bool d_t, bool d_u, bool d_v,
                                               Vec<3,T>& out );
        
        template<class T>
        void edge_edge_collision_function(const Vec3d &d_x0,    const Vec3d &d_x1,    const Vec3d &d_x2,    const Vec3d &d_x3,
                                          const Vec3d &d_x0new, const Vec3d &d_x1new, const Vec3d &d_x2new, const Vec3d &d_x3new,
                                          bool d_t, bool d_u, bool d_v,
                                          Vec<3,T>& out );
        
        
        template<class T>
        void get_quad_vertices(const Vec3d &d_x0old, const Vec3d &d_x1old, const Vec3d &d_x2old, const Vec3d &d_x3old,
                               const Vec3d &d_x0new, const Vec3d &d_x1new, const Vec3d &d_x2new, const Vec3d &d_x3new,
                               bool is_edge_edge, 
                               const Vec4b& ts, const Vec4b& us, const Vec4b& vs, 
                               Vec<3,T>& q0, Vec<3,T>& q1, Vec<3,T>& q2, Vec<3,T>& q3 );
        
        
        template<class T>
        void get_triangle_vertices(const Vec3d &d_x0old, const Vec3d &d_x1old, const Vec3d &d_x2old, const Vec3d &d_x3old,
                                   const Vec3d &d_x0new, const Vec3d &d_x1new, const Vec3d &d_x2new, const Vec3d &d_x3new,
                                   bool is_edge_edge, 
                                   const Vec3b& ts, const Vec3b& us, const Vec3b& vs, 
                                   Vec<3,T>& q0, Vec<3,T>& q1, Vec<3,T>& q2 );
        
        template<class T>
        static T orientation2d(const Vec<2,T>& x0,
                               const Vec<2,T>& x1,
                               const Vec<2,T>& x2);
        
        
        template<class T>
        T orientation3d(const Vec<3,T>& x0,
                        const Vec<3,T>& x1,
                        const Vec<3,T>& x2,
                        const Vec<3,T>& x3);
        
        int expansion_simplex_intersection1d(int k,
                                             const expansion& x0,
                                             const expansion& x1,
                                             const expansion& x2,
                                             double* out_alpha0, 
                                             double* out_alpha1, 
                                             double* out_alpha2);
        
        int expansion_simplex_intersection2d(int k,
                                             const Vec2e& x0,
                                             const Vec2e& x1,
                                             const Vec2e& x2,
                                             double* out_alpha0, 
                                             double* out_alpha1, 
                                             double* out_alpha2 );
        
        int expansion_simplex_intersection2d(int k,
                                             const Vec2e& x0,
                                             const Vec2e& x1,
                                             const Vec2e& x2,
                                             const Vec2e& x3,
                                             double* out_alpha0, 
                                             double* out_alpha1, 
                                             double* out_alpha2,
                                             double* out_alpha3);
        
        bool expansion_simplex_intersection3d(int k,
                                              const Vec3e& x0,
                                              const Vec3e& x1,
                                              const Vec3e& x2,
                                              const Vec3e& x3,
                                              double* alpha0, 
                                              double* alpha1, 
                                              double* alpha2,
                                              double* alpha3);
        
        int degenerate_point_tetrahedron_intersection(const Vec3e& x0,
                                                      const Vec3e& x1,
                                                      const Vec3e& x2,
                                                      const Vec3e& x3,
                                                      const Vec3e& x4,                                      
                                                      double* alpha0, 
                                                      double* alpha1, 
                                                      double* alpha2,
                                                      double* alpha3,
                                                      double* alpha4);
        
        int degenerate_point_tetrahedron_intersection(const Vec3Interval& x0,
                                                      const Vec3Interval& x1,
                                                      const Vec3Interval& x2,
                                                      const Vec3Interval& x3,
                                                      const Vec3Interval& x4,                                      
                                                      double* out_alpha0, 
                                                      double* out_alpha1, 
                                                      double* out_alpha2,
                                                      double* out_alpha3,
                                                      double* out_alpha4);
        
        template<class T>
        int point_tetrahedron_intersection(const Vec<3,T>& x0,
                                           const Vec<3,T>& x1,
                                           const Vec<3,T>& x2,
                                           const Vec<3,T>& x3,
                                           const Vec<3,T>& x4,                                      
                                           double* out_alpha0, 
                                           double* out_alpha1, 
                                           double* out_alpha2,
                                           double* out_alpha3,
                                           double* out_alpha4 );
        
        int degenerate_edge_triangle_intersection(const Vec3e& x0,
                                                  const Vec3e& x1,
                                                  const Vec3e& x2,
                                                  const Vec3e& x3,
                                                  const Vec3e& x4,                                      
                                                  double* alpha0, 
                                                  double* alpha1, 
                                                  double* alpha2,
                                                  double* alpha3,
                                                  double* alpha4);
        
        int degenerate_edge_triangle_intersection(const Vec3Interval& x0,
                                                  const Vec3Interval& x1,
                                                  const Vec3Interval& x2,
                                                  const Vec3Interval& x3,
                                                  const Vec3Interval& x4,                                      
                                                  double* out_alpha0, 
                                                  double* out_alpha1, 
                                                  double* out_alpha2,
                                                  double* out_alpha3,
                                                  double* out_alpha4);
        
        template<class T>
        int edge_triangle_intersection(const Vec<3,T>& x0,
                                       const Vec<3,T>& x1,
                                       const Vec<3,T>& x2,
                                       const Vec<3,T>& x3,
                                       const Vec<3,T>& x4,                                      
                                       double* out_alpha0, 
                                       double* out_alpha1, 
                                       double* out_alpha2,
                                       double* out_alpha3,
                                       double* out_alpha4 );
        
        bool edge_triangle_intersection(const Vec3d &d_x0old, const Vec3d &d_x1old, const Vec3d &d_x2old, const Vec3d &d_x3old,
                                        const Vec3d &d_x0new, const Vec3d &d_x1new, const Vec3d &d_x2new, const Vec3d &d_x3new,
                                        bool is_edge_edge,
                                        const Vec3b& ts, const Vec3b& us, const Vec3b& vs,
                                        const Vec3d& d_s0, const Vec3d& d_s1,
                                        double* alphas );
        
        
        template<class T>   
        T plane_dist( const Vec<3,T>& x, const Vec<3,T>& q, const Vec<3,T>& r, const Vec<3,T>& p );
        
        template<class T>
        void implicit_surface_function( const Vec<3,T>& x, const Vec<3,T>& q0, const Vec<3,T>& q1, const Vec<3,T>& q2, const Vec<3,T>& q3, T& out );
        
        
        bool test_with_triangles(const Vec3d &d_x0old, const Vec3d &d_x1old, const Vec3d &d_x2old, const Vec3d &d_x3old,
                                 const Vec3d &d_x0new, const Vec3d &d_x1new, const Vec3d &d_x2new, const Vec3d &d_x3new,
                                 bool is_edge_edge,
                                 const Vec4b& ts, const Vec4b& us, const Vec4b& vs,
                                 const Vec3d& s0, const Vec3d& s1,
                                 bool use_positive_triangles, bool& edge_hit );
        
        
        //
        // Local function definitions
        //
        
        // --------------------------------------------------------
        
        template<unsigned int N, class T>
        inline void make_vector( const Vec<N,double>& in, Vec<N,T>& out )
        {
            for ( unsigned int i = 0; i < N; ++i )
            {
                create_from_double( in[i], out[i] );
            }
        }
        
        
        // --------------------------------------------------------
        
        template<class T>
        inline void edge_edge_collision_function(const Vec3d &d_x0,    const Vec3d &d_x1,    const Vec3d &d_x2,    const Vec3d &d_x3,
                                                 const Vec3d &d_x0new, const Vec3d &d_x1new, const Vec3d &d_x2new, const Vec3d &d_x3new,
                                                 bool b_t, bool b_u, bool b_v,
                                                 Vec<3,T>& out )
        {
            Vec<3,T> x0, x1, x2, x3;
            make_vector( d_x0, x0 );
            make_vector( d_x1, x1 );
            make_vector( d_x2, x2 );
            make_vector( d_x3, x3 );
            
            out = x0;
            out -= x2;
            
            if ( b_t )
            {
                
                Vec<3,T> x0new, x1new, x2new, x3new;
                make_vector( d_x0new, x0new );
                make_vector( d_x1new, x1new );
                make_vector( d_x2new, x2new );
                make_vector( d_x3new, x3new );
                
                out += x0new;
                out -= x0;
                out -= x2new;
                out += x2;
                
                if ( b_u )
                {
                    out += x0;
                    out -= x0new;
                    out += x1new;
                    out -= x1;
                }
                
                if ( b_v )
                {
                    out += x2new;
                    out -= x2;
                    out -= x3new;
                    out += x3;
                }
            }
            
            if ( b_u )
            {
                out += x1;
                out -= x0;
            }
            
            if ( b_v )
            {
                out += x2;
                out -= x3;
            }
            
        }
        
        /// --------------------------------------------------------
        
        inline void add( Vec3d& a, const Vec3d& b )
        {
            a[0] += b[0];
            a[1] += b[1];
            a[2] += b[2];
        }
        
        /// --------------------------------------------------------
        
        inline void sub( Vec3d& a, const Vec3d& b )
        {
            a[0] -= b[0];
            a[1] -= b[1];
            a[2] -= b[2];
        }
        
        // --------------------------------------------------------
        
        inline void edge_edge_collision_function(const Vec3d &d_x0,    const Vec3d &d_x1,    const Vec3d &d_x2,    const Vec3d &d_x3,
                                                 const Vec3d &d_x0new, const Vec3d &d_x1new, const Vec3d &d_x2new, const Vec3d &d_x3new,
                                                 bool b_t, bool b_u, bool b_v,
                                                 Vec<3,IntervalType>& out )
        {
            
            Vec3d out_lower( -d_x0[0], -d_x0[1], -d_x0[2] );
            add( out_lower, d_x2 );
            
            Vec3d out_upper( d_x0[0], d_x0[1], d_x0[2] );
            sub( out_upper, d_x2 );
            
            if ( b_t )
            {
                
                sub( out_lower, d_x0new );
                add( out_lower, d_x0 );
                add( out_lower, d_x2new );
                sub( out_lower, d_x2 );
                
                add( out_upper, d_x0new );
                sub( out_upper, d_x0 );
                sub( out_upper, d_x2new );
                add( out_upper, d_x2 );
                
                if ( b_u )
                {
                    sub( out_lower, d_x0 );
                    add( out_lower, d_x0new );
                    sub( out_lower, d_x1new );
                    add( out_lower, d_x1 );
                    
                    add( out_upper, d_x0 );
                    sub( out_upper, d_x0new );
                    add( out_upper, d_x1new );
                    sub( out_upper, d_x1 );
                }
                
                if ( b_v )
                {
                    sub( out_lower, d_x2new );
                    add( out_lower, d_x2 );
                    add( out_lower, d_x3new );
                    sub( out_lower, d_x3 );
                    
                    add( out_upper, d_x2new );
                    sub( out_upper, d_x2 );
                    sub( out_upper, d_x3new );
                    add( out_upper, d_x3 );
                }
            }
            
            if ( b_u )
            {
                sub( out_lower, d_x1 );
                add( out_lower, d_x0 );
                
                add( out_upper, d_x1 );
                sub( out_upper, d_x0 );
            }
            
            if ( b_v )
            {
                sub( out_lower, d_x2 );
                add( out_lower, d_x3 );
                
                add( out_upper, d_x2 );
                sub( out_upper, d_x3 );
            }
            
            out[0].v[0] = out_lower[0];
            out[1].v[0] = out_lower[1];
            out[2].v[0] = out_lower[2];
            
            out[0].v[1] = out_upper[0];
            out[1].v[1] = out_upper[1];
            out[2].v[1] = out_upper[2];
            
        }
        
        
        
        /// --------------------------------------------------------
        
        template<class T>
        void point_triangle_collision_function(const Vec3d &d_x0,    const Vec3d &d_x1,    const Vec3d &d_x2,    const Vec3d &d_x3,
                                               const Vec3d &d_x0new, const Vec3d &d_x1new, const Vec3d &d_x2new, const Vec3d &d_x3new,
                                               bool b_t, bool b_u, bool b_v,
                                               Vec<3,T>& out )
        {
            Vec<3,T> x0, x1, x2, x3; 
            make_vector( d_x0, x0 );
            make_vector( d_x1, x1 );
            make_vector( d_x2, x2 );
            make_vector( d_x3, x3 );
            
            out = x0;
            out -= x1;
            
            if ( b_t )
            {
                Vec<3,T> x0new, x1new, x2new, x3new;
                make_vector( d_x0new, x0new );
                make_vector( d_x1new, x1new );
                make_vector( d_x2new, x2new );
                make_vector( d_x3new, x3new );
                
                out += x0new;
                out -= x0;
                out -= x1new;
                out += x1;
                
                if ( b_u )
                {
                    out += x1new;
                    out -= x1;
                    out -= x2new;
                    out += x2;
                }
                
                if ( b_v )
                {
                    out += x1new;
                    out -= x1;
                    out -= x3new;
                    out += x3;
                }
            }
            
            if ( b_u )
            {
                out += x1;
                out -= x2;
            }
            
            if ( b_v )
            {
                out += x1;
                out -= x3;
            }
            
        }
        
        /// --------------------------------------------------------
        
        
        void point_triangle_collision_function(const Vec3d &d_x0,    const Vec3d &d_x1,    const Vec3d &d_x2,    const Vec3d &d_x3,
                                               const Vec3d &d_x0new, const Vec3d &d_x1new, const Vec3d &d_x2new, const Vec3d &d_x3new,
                                               bool b_t, bool b_u, bool b_v,
                                               Vec<3,IntervalType>& out )
        {
            
            Vec3d out_lower( -d_x0[0], -d_x0[1], -d_x0[2] );
            Vec3d out_upper( d_x0[0], d_x0[1], d_x0[2] );
            
            add( out_lower, d_x1 );
            sub( out_upper, d_x1 );
            
            if ( b_t )
            {
                sub( out_lower, d_x0new );
                add( out_lower, d_x0 );
                add( out_lower, d_x1new );
                sub( out_lower, d_x1 );
                
                add( out_upper, d_x0new );
                sub( out_upper, d_x0 );
                sub( out_upper, d_x1new );
                add( out_upper, d_x1 );
                
                if ( b_u )
                {
                    sub( out_lower, d_x1new );
                    add( out_lower, d_x1 );
                    add( out_lower, d_x2new );
                    sub( out_lower, d_x2 );
                    
                    add( out_upper, d_x1new );
                    sub( out_upper, d_x1 );
                    sub( out_upper, d_x2new );
                    add( out_upper, d_x2 );
                }
                
                if ( b_v )
                {
                    sub( out_lower, d_x1new );
                    add( out_lower, d_x1 );
                    add( out_lower, d_x3new );
                    sub( out_lower, d_x3 );
                    
                    add( out_upper, d_x1new );
                    sub( out_upper, d_x1 );
                    sub( out_upper, d_x3new );
                    add( out_upper, d_x3 );
                }
            }
            
            if ( b_u )
            {
                sub( out_lower, d_x1 );
                add( out_lower, d_x2 );
                add( out_upper, d_x1 );
                sub( out_upper, d_x2 );
            }
            
            if ( b_v )
            {
                sub( out_lower, d_x1 );
                add( out_lower, d_x3 );
                
                add( out_upper, d_x1 );
                sub( out_upper, d_x3 );
            }
            
            out[0].v[0] = out_lower[0];
            out[1].v[0] = out_lower[1];
            out[2].v[0] = out_lower[2];
            
            out[0].v[1] = out_upper[0];
            out[1].v[1] = out_upper[1];
            out[2].v[1] = out_upper[2];
            
        }
        
        
        
        // --------------------------------------------------------
        
        template<class T>
        void get_quad_vertices(const Vec3d &d_x0old, const Vec3d &d_x1old, const Vec3d &d_x2old, const Vec3d &d_x3old,
                               const Vec3d &d_x0new, const Vec3d &d_x1new, const Vec3d &d_x2new, const Vec3d &d_x3new,
                               bool is_edge_edge, 
                               const Vec4b& ts, const Vec4b& us, const Vec4b& vs, 
                               Vec<3,T>& q0, Vec<3,T>& q1, Vec<3,T>& q2, Vec<3,T>& q3 )
        {
            T::begin_special_arithmetic();
            if ( is_edge_edge )
            {
                edge_edge_collision_function( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, ts[0], us[0], vs[0], q0 );
                edge_edge_collision_function( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, ts[1], us[1], vs[1], q1 );
                edge_edge_collision_function( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, ts[2], us[2], vs[2], q2 );
                edge_edge_collision_function( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, ts[3], us[3], vs[3], q3 );      
            }
            else
            {
                point_triangle_collision_function( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, ts[0], us[0], vs[0], q0 );
                point_triangle_collision_function( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, ts[1], us[1], vs[1], q1 );
                point_triangle_collision_function( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, ts[2], us[2], vs[2], q2 );
                point_triangle_collision_function( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, ts[3], us[3], vs[3], q3 );      
            }
            T::end_special_arithmetic();
        }
        
        // --------------------------------------------------------
        
        template<class T>
        void get_triangle_vertices(const Vec3d &d_x0old, const Vec3d &d_x1old, const Vec3d &d_x2old, const Vec3d &d_x3old,
                                   const Vec3d &d_x0new, const Vec3d &d_x1new, const Vec3d &d_x2new, const Vec3d &d_x3new,
                                   bool is_edge_edge, 
                                   const Vec3b& ts, const Vec3b& us, const Vec3b& vs, 
                                   Vec<3,T>& q0, Vec<3,T>& q1, Vec<3,T>& q2 )
        {
            T::begin_special_arithmetic();
            if ( is_edge_edge )
            {
                edge_edge_collision_function( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, ts[0], us[0], vs[0], q0 );
                edge_edge_collision_function( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, ts[1], us[1], vs[1], q1 );
                edge_edge_collision_function( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, ts[2], us[2], vs[2], q2 );
            }
            else
            {
                point_triangle_collision_function( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, ts[0], us[0], vs[0], q0 );
                point_triangle_collision_function( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, ts[1], us[1], vs[1], q1 );
                point_triangle_collision_function( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, ts[2], us[2], vs[2], q2 );
            }
            T::end_special_arithmetic();
        }
        
        
        
        // ----------------------------------------
        
        template<class T>
        static T orientation2d(const Vec<2,T>& x0,
                               const Vec<2,T>& x1,
                               const Vec<2,T>& x2)
        {
            return x0[0]*x1[1] + -(x0[0])*x2[1]
            + x1[0]*x2[1] + -(x1[0])*x0[1]
            + x2[0]*x0[1] + -(x2[0])*x1[1];
        }
        
        // ----------------------------------------
        
        inline expansion orientation2d(const Vec<2,expansion>& x0,
                                       const Vec<2,expansion>& x1,
                                       const Vec<2,expansion>& x2)
        {
            expansion prod;
            
            expansion result;
            
            multiply( x0[0], x1[1], prod );
            add( result, prod, result );
            
            multiply( x0[0], x2[1], prod );
            subtract( result, prod, result );
            
            multiply( x1[0], x2[1], prod );
            add( result, prod, result );
            
            multiply( x1[0], x0[1], prod );
            subtract( result, prod, result );
            
            multiply( x2[0], x0[1], prod );
            add( result, prod, result );
            
            multiply( x2[0], x1[1], prod );
            subtract( result, prod, result );
            
            return result;
        }
        
        // --------------------------------------------------------
        
        template<class T>
        void orientation3d(const Vec<3,T>& x0,
                           const Vec<3,T>& x1,
                           const Vec<3,T>& x2,
                           const Vec<3,T>& x3,
                           T& result )
        {
            // avoid recomputing common factors
            T x00x11 = x0[0] * x1[1];
            T x00x21 = x0[0] * x2[1];
            T x00x31 = x0[0] * x3[1];
            
            T x10x01 = x1[0] * x0[1];
            T x10x21 = x1[0] * x2[1];
            T x10x31 = x1[0] * x3[1];
            
            T x20x01 = x2[0] * x0[1];
            T x20x11 = x2[0] * x1[1];
            T x20x31 = x2[0] * x3[1];
            
            T x30x01 = x3[0] * x0[1];
            T x30x11 = x3[0] * x1[1];
            T x30x21 = x3[0] * x2[1];
            
            result = ( x00x11 * ( x2[2] - x3[2] ) )
            + ( x00x21 * ( x3[2] - x1[2] ) )
            + ( x00x31 * ( x1[2] - x2[2] ) )
            
            + ( x10x01 * ( x3[2] - x2[2] ) )
            + ( x10x21 * ( x0[2] - x3[2] ) )
            + ( x10x31 * ( x2[2] - x0[2] ) )
            
            + ( x20x01 * ( x1[2]-  x3[2] ) )
            + ( x20x11 * ( x3[2] - x0[2] ) )
            + ( x20x31 * ( x0[2] - x1[2] ) )
            
            + ( x30x01 * ( x2[2] - x1[2] ) )
            + ( x30x11 * ( x0[2] - x2[2] ) )
            + ( x30x21 * ( x1[2] - x0[2] ) );
        }
        
        
        // --------------------------------------------------------
        
        inline void orientation3d(const Vec<3,expansion>& x0,
                                  const Vec<3,expansion>& x1,
                                  const Vec<3,expansion>& x2,
                                  const Vec<3,expansion>& x3,
                                  expansion& result )
        {
            // avoid recomputing common factors
            expansion x00x11;
            multiply( x0[0], x1[1], x00x11 );
            
            expansion x00x21;
            multiply( x0[0], x2[1], x00x21 );
            
            expansion x00x31;
            multiply( x0[0], x3[1], x00x31 );
            
            expansion x10x01;
            multiply( x1[0], x0[1], x10x01 );
            
            expansion x10x21;
            multiply( x1[0], x2[1], x10x21 );
            
            expansion x10x31;
            multiply( x1[0], x3[1], x10x31 );
            
            expansion x20x01;
            multiply( x2[0], x0[1], x20x01 );
            
            expansion x20x11;
            multiply( x2[0], x1[1], x20x11 );
            
            expansion x20x31;
            multiply( x2[0], x3[1], x20x31 );
            
            expansion x30x01;
            multiply( x3[0], x0[1], x30x01 );
            
            expansion x30x11;
            multiply( x3[0], x1[1], x30x11 );
            
            expansion x30x21;
            multiply( x3[0], x2[1], x30x21 );
            
            
            expansion diff;         
            subtract( x2[2], x3[2], diff );
            multiply( x00x11, diff, result );         
            
            expansion prod;
            subtract( x3[2], x1[2], diff );
            multiply( x00x21, diff, prod );
            add( result, prod, result );
            
            subtract( x1[2], x2[2], diff );
            multiply( x00x31, diff, prod );
            add( result, prod, result );
            
            subtract( x3[2], x2[2], diff );
            multiply( x10x01, diff, prod );
            add( result, prod, result );
            
            subtract( x0[2], x3[2], diff );
            multiply( x10x21, diff, prod );
            add( result, prod, result );
            
            subtract( x2[2], x0[2], diff );         
            multiply( x10x31, diff, prod );
            add( result, prod, result );
            
            subtract( x1[2], x3[2], diff );
            multiply( x20x01, diff, prod );
            add( result, prod, result );
            
            subtract( x3[2], x0[2], diff );         
            multiply( x20x11, diff, prod );
            add( result, prod, result );
            
            subtract( x0[2], x1[2], diff );         
            multiply( x20x31, diff, prod );
            add( result, prod, result );
            
            subtract( x2[2], x1[2], diff );      
            multiply( x30x01, diff, prod );
            add( result, prod, result );
            
            subtract( x0[2], x2[2], diff );         
            multiply( x30x11, diff, prod );
            add( result, prod, result );
            
            subtract( x1[2], x0[2], diff );         
            multiply( x30x21, diff, prod );
            add( result, prod, result );
            
        }
        
        
        // ----------------------------------------
        
        int expansion_simplex_intersection1d(int k,
                                             const expansion& x0,
                                             const expansion& x1,
                                             const expansion& x2,
                                             double* out_alpha0, 
                                             double* out_alpha1, 
                                             double* out_alpha2)
        {
            assert( k == 1 );
            assert(out_alpha0 != NULL && out_alpha1 != NULL && out_alpha2 != NULL);
            
            if(out_alpha0 == NULL || out_alpha1 == NULL || out_alpha2 == NULL) //prevent null ptr warning.
               return -1;                                                      //should never occur

            assert(fegetround()== FE_TONEAREST);
            
            if( sign(x1-x2) < 0 )
            {
                if( sign(x0-x1) < 0) return 0;
                else if ( sign(x0-x2) > 0 ) return 0;
                *out_alpha0=1;
                *out_alpha1=(x2.estimate()-x0.estimate())/(x2.estimate()-x1.estimate());
                *out_alpha2=(x0.estimate()-x1.estimate())/(x2.estimate()-x1.estimate());
                return 1;
            }
            else if( sign(x1-x2) > 0 )
            {
                if( sign(x0-x2) < 0 ) return 0;
                else if( sign(x0-x1) > 0) return 0;
                *out_alpha0=1;
                *out_alpha1=(x2.estimate()-x0.estimate())/(x2.estimate()-x1.estimate());
                *out_alpha2=(x0.estimate()-x1.estimate())/(x2.estimate()-x1.estimate());
                return 1;
            }
            else
            { 
                // x1 == x2 
                if( sign(x0-x1) != 0 ) return 0;
                *out_alpha0=1;
                *out_alpha1=0.5;
                *out_alpha2=0.5;
                return 1;
            }
            
        }
        
        
        // ----------------------------------------
        
        int
        expansion_simplex_intersection2d(int k,
                                         const Vec2e& x0,
                                         const Vec2e& x1,
                                         const Vec2e& x2,
                                         double* out_alpha0, 
                                         double* out_alpha1, 
                                         double* out_alpha2 )
        {
            assert(k==1);
            assert(fegetround()== FE_TONEAREST);
            
            // try projecting each coordinate out in turn
            
            double ax0, ax1, ax2;
            if(!expansion_simplex_intersection1d(1, x0[1], x1[1], x2[1], &ax0, &ax1, &ax2)) return 0;
            
            double ay0, ay1, ay2;
            if(!expansion_simplex_intersection1d(1, x0[0], x1[0], x2[0], &ay0, &ay1, &ay2)) return 0;
            
            // decide which solution is more accurate for barycentric coordinates
            double checkx=std::fabs(-ax0*x0[0].estimate()+ax1*x1[0].estimate()+ax2*x2[0].estimate())
            +std::fabs(-ax0*x0[1].estimate()+ax1*x1[1].estimate()+ax2*x2[1].estimate());
            
            double checky=std::fabs(-ay0*x0[0].estimate()+ay1*x1[0].estimate()+ay2*x2[0].estimate())
            +std::fabs(-ay0*x0[1].estimate()+ay1*x1[1].estimate()+ay2*x2[1].estimate());
            
            if(checkx<=checky)
            {
                *out_alpha0=ax0;
                *out_alpha1=ax1;
                *out_alpha2=ax2;
            }
            else
            {
                *out_alpha0=ay0;
                *out_alpha1=ay1;
                *out_alpha2=ay2;
            }
            return 1;
            
        }
        
        
        // ----------------------------------------
        
        int expansion_simplex_intersection2d(int k,
                                             const Vec2e& x0,
                                             const Vec2e& x1,
                                             const Vec2e& x2,
                                             const Vec2e& x3,
                                             double* out_alpha0, 
                                             double* out_alpha1, 
                                             double* out_alpha2,
                                             double* out_alpha3)
        {
            assert(1<=k && k<=3);
            assert(fegetround()== FE_TONEAREST);
            
            switch(k)
            {
                case 1: // point vs. triangle
                {
                    
                    expansion alpha1 = -orientation2d(x0, x2, x3);
                    expansion alpha2 =  orientation2d(x0, x1, x3);
                    if(certainly_opposite_sign(alpha1, alpha2)) return 0;
                    
                    expansion alpha3 = -orientation2d(x0, x1, x2);
                    if(certainly_opposite_sign(alpha1, alpha3)) return 0;
                    if(certainly_opposite_sign(alpha2, alpha3)) return 0;
                    
                    double sum2 = alpha1.estimate() + alpha2.estimate() + alpha3.estimate();
                    
                    if(sum2)
                    { 
                        *out_alpha0=1;
                        *out_alpha1 = alpha1.estimate() / sum2;
                        *out_alpha2 = alpha2.estimate() / sum2;
                        *out_alpha3 = alpha3.estimate() / sum2;
                        return 1;
                    }
                    else
                    { 
                        // triangle is degenerate and point lies on same line
                        if(expansion_simplex_intersection2d(1, x0, x1, x2, out_alpha0, out_alpha1, out_alpha2))
                        {
                            *out_alpha3=0;
                            return 1;
                        }
                        if(expansion_simplex_intersection2d(1, x0, x1, x3, out_alpha0, out_alpha1, out_alpha3))
                        {
                            *out_alpha2=0;
                            return 1;
                        }
                        if(expansion_simplex_intersection2d(1, x0, x2, x3, out_alpha0, out_alpha2, out_alpha3))
                        {
                            *out_alpha1=0;
                            return 1;
                        }
                        return 0;
                    }
                    
                    
                }
                    
                case 2: // segment vs. segment
                {
                    expansion alpha0 = orientation2d(x1, x2, x3);
                    expansion alpha1 = -orientation2d(x0, x2, x3);
                    if( certainly_opposite_sign(alpha0, alpha1) ) return 0;
                    
                    expansion alpha2 = orientation2d(x0, x1, x3);
                    expansion alpha3 = -orientation2d(x0, x1, x2);
                    if( certainly_opposite_sign(alpha2, alpha3) ) return 0;
                    
                    double sum1, sum2;
                    sum1= alpha0.estimate() + alpha1.estimate();
                    sum2= alpha2.estimate() + alpha3.estimate();
                    
                    if(sum1 && sum2){
                        *out_alpha0 = alpha0.estimate() / sum1;
                        *out_alpha1 = alpha1.estimate() / sum1;
                        *out_alpha2 = alpha2.estimate() / sum2;
                        *out_alpha3 = alpha3.estimate() / sum2;
                        return 1;
                    }
                    else
                    { 
                        // degenerate: segments lie on the same line
                        if(expansion_simplex_intersection2d(1, x0, x2, x3, out_alpha0, out_alpha2, out_alpha3)){
                            *out_alpha1=0;
                            return 1;
                        }
                        if(expansion_simplex_intersection2d(1, x1, x2, x3, out_alpha1, out_alpha2, out_alpha3)){
                            *out_alpha0=0;
                            return 1;
                        }
                        if(expansion_simplex_intersection2d(1, x2, x0, x1, out_alpha2, out_alpha0, out_alpha1)){
                            *out_alpha3=0;
                            return 1;
                        }
                        if(expansion_simplex_intersection2d(1, x3, x0, x1, out_alpha3, out_alpha0, out_alpha1)){
                            *out_alpha2=0;
                            return 1;
                        }
                        return 0;
                    }
                    
                } 
                    break;
                    
                default:
                    assert(0);
                    return -1; // should never get here
            }
            
        }
        
        // --------------------------------------------------------
        // degenerate test in 3d - assumes four points lie on the same plane
        
        bool expansion_simplex_intersection3d(int k,
                                              const Vec3e& x0,
                                              const Vec3e& x1,
                                              const Vec3e& x2,
                                              const Vec3e& x3,
                                              double* alpha0, 
                                              double* alpha1, 
                                              double* alpha2,
                                              double* )
        {
            assert(k<=2);
            assert(fegetround()== FE_TONEAREST);
            
            // try projecting each coordinate out in turn
            
            double ax0, ax1, ax2, ax3;
            if( !expansion_simplex_intersection2d(k, Vec2e(x0[1],x0[2]), Vec2e(x1[1],x1[2]), Vec2e(x2[1],x2[2]), Vec2e(x3[1],x3[2]), &ax0, &ax1, &ax2,&ax3) )
            {
                return 0;
            }
            
            double ay0, ay1, ay2, ay3;
            Vec2e p0( x0[0], x0[2] );
            Vec2e p1( x1[0], x1[2] );
            Vec2e p2( x2[0], x2[2] );
            Vec2e p3( x3[0], x3[2] );
            if ( !expansion_simplex_intersection2d(k, p0, p1, p2, p3, &ay0, &ay1, &ay2, &ay3) )
            {
                return 0;
            }
            
            double az0, az1, az2, az3;
            if( !expansion_simplex_intersection2d(k, Vec2e(x0[0],x0[1]), Vec2e(x1[0],x1[1]), Vec2e(x2[0],x2[1]), Vec2e(x3[0],x3[1]), &az0, &az1, &az2, &az3) )
            {
                return 0;
            }
            
            // decide which solution is more accurate for barycentric coordinates
            double checkx, checky, checkz;
            Vec3d estx0( x0[0].estimate(), x0[1].estimate(), x0[2].estimate() );
            Vec3d estx1( x1[0].estimate(), x1[1].estimate(), x1[2].estimate() );
            Vec3d estx2( x2[0].estimate(), x2[1].estimate(), x2[2].estimate() );
            Vec3d estx3( x3[0].estimate(), x3[1].estimate(), x3[2].estimate() );
            
            if(k==1)
            {
                checkx=std::fabs(-ax0*estx0[0]+ax1*estx1[0]+ax2*estx2[0]+ax3*estx3[0])
                +std::fabs(-ax0*estx0[1]+ax1*estx1[1]+ax2*estx2[1]+ax3*estx3[1])
                +std::fabs(-ax0*estx0[2]+ax1*estx1[2]+ax2*estx2[2]+ax3*estx3[2]);
                checky=std::fabs(-ay0*estx0[0]+ay1*estx1[0]+ay2*estx2[0]+ay3*estx3[0])
                +std::fabs(-ay0*estx0[1]+ay1*estx1[1]+ay2*estx2[1]+ay3*estx3[1])
                +std::fabs(-ay0*estx0[2]+ay1*estx1[2]+ay2*estx2[2]+ay3*estx3[2]);
                checkz=std::fabs(-az0*estx0[0]+az1*estx1[0]+az2*estx2[0]+az3*estx3[0])
                +std::fabs(-az0*estx0[1]+az1*estx1[1]+az2*estx2[1]+az3*estx3[1])
                +std::fabs(-az0*estx0[2]+az1*estx1[2]+az2*estx2[2]+az3*estx3[2]);
            }
            else
            {
                checkx=std::fabs(-ax0*estx0[0]-ax1*estx1[0]+ax2*estx2[0]+ax3*estx3[0])
                +std::fabs(-ax0*estx0[1]-ax1*estx1[1]+ax2*estx2[1]+ax3*estx3[1])
                +std::fabs(-ax0*estx0[2]-ax1*estx1[2]+ax2*estx2[2]+ax3*estx3[2]);
                checky=std::fabs(-ay0*estx0[0]-ay1*estx1[0]+ay2*estx2[0]+ay3*estx3[0])
                +std::fabs(-ay0*estx0[1]-ay1*estx1[1]+ay2*estx2[1]+ay3*estx3[1])
                +std::fabs(-ay0*estx0[2]-ay1*estx1[2]+ay2*estx2[2]+ay3*estx3[2]);
                checkz=std::fabs(-az0*estx0[0]-az1*estx1[0]+az2*estx2[0]+az3*estx3[0])
                +std::fabs(-az0*estx0[1]-az1*estx1[1]+az2*estx2[1]+az3*estx3[1])
                +std::fabs(-az0*estx0[2]-az1*estx1[2]+az2*estx2[2]+az3*estx3[2]);
            }
            if(checkx<=checky && checkx<=checkz)
            {
                *alpha0=ax0;
                *alpha1=ax1;
                *alpha2=ax2;
                *alpha2=ax3;
            }
            else if(checky<=checkz)
            {
                *alpha0=ay0;
                *alpha1=ay1;
                *alpha2=ay2;
                *alpha2=ay3;
            }
            else
            {
                *alpha0=az0;
                *alpha1=az1;
                *alpha2=az2;
                *alpha2=az3;
            }
            
            return 1;
            
        }
        
        
        // --------------------------------------------------------
        
        int degenerate_point_tetrahedron_intersection(const Vec3e& x0,
                                                      const Vec3e& x1,
                                                      const Vec3e& x2,
                                                      const Vec3e& x3,
                                                      const Vec3e& x4,                                      
                                                      double* alpha0, 
                                                      double* alpha1, 
                                                      double* alpha2,
                                                      double* alpha3,
                                                      double* alpha4)
        {
            
            assert(fegetround()== FE_TONEAREST);
            
            // degenerate: point and tetrahedron in same plane
            if (expansion_simplex_intersection3d(1, x0, x2, x3, x4, alpha0, alpha2, alpha3, alpha4))
            {
                *alpha1=0;
                return 1;
            }
            if(expansion_simplex_intersection3d(1, x0, x1, x3, x4, alpha0, alpha1, alpha3, alpha4))
            {
                *alpha2=0;
                return 1;
            }
            if(expansion_simplex_intersection3d(1, x0, x1, x2, x4, alpha0, alpha1, alpha2, alpha4))
            {
                *alpha3=0;
                return 1;
            }
            if(expansion_simplex_intersection3d(1, x0, x1, x2, x3, alpha0, alpha1, alpha2, alpha3))
            {
                *alpha4=0;
                return 1;
            }
            return 0;
        }
        
        
        // --------------------------------------------------------
        
        int degenerate_point_tetrahedron_intersection(const Vec3Interval& ,
                                                      const Vec3Interval& ,
                                                      const Vec3Interval& ,
                                                      const Vec3Interval& ,
                                                      const Vec3Interval& ,                                      
                                                      double* , 
                                                      double* , 
                                                      double* ,
                                                      double* ,
                                                      double* )
        {
            return -1;
        }
        
        
        // --------------------------------------------------------
        
        template<class T>
        int point_tetrahedron_intersection(const Vec<3,T>& x0,
                                           const Vec<3,T>& x1,
                                           const Vec<3,T>& x2,
                                           const Vec<3,T>& x3,
                                           const Vec<3,T>& x4,                                      
                                           double* out_alpha0, 
                                           double* out_alpha1, 
                                           double* out_alpha2,
                                           double* out_alpha3,
                                           double* out_alpha4 )
        {
            
            T::begin_special_arithmetic();
            
            T alpha1;
            orientation3d(x0, x2, x3, x4, alpha1);
            alpha1 = -alpha1;
            
            T alpha2;
            orientation3d(x0, x1, x3, x4, alpha2);
            
            if( certainly_opposite_sign(alpha1, alpha2) )
            {
                T::end_special_arithmetic();
                return 0;
            }
            
            T alpha3;
            orientation3d(x0, x1, x2, x4, alpha3);
            alpha3 = -alpha3;
            
            if( certainly_opposite_sign(alpha1, alpha3) )
            {
                T::end_special_arithmetic();
                return 0;
            }
            
            if( certainly_opposite_sign(alpha2, alpha3) ) 
            {
                T::end_special_arithmetic();
                return 0;
            }
            
            T alpha4;
            orientation3d(x0, x1, x2, x3, alpha4);
            
            T::end_special_arithmetic();
            
            if( certainly_opposite_sign(alpha1, alpha4) ) return 0;
            if( certainly_opposite_sign(alpha2, alpha4) ) return 0;         
            if( certainly_opposite_sign(alpha3, alpha4) ) return 0;
            
            if ( alpha1.indefinite_sign() || alpha2.indefinite_sign() || alpha3.indefinite_sign() || alpha4.indefinite_sign() )
            {
                // degenerate
                return -1;
            }
            
            double sum2 = alpha1.estimate() + alpha2.estimate() + alpha3.estimate() + alpha4.estimate();
            
            if(sum2)
            {
                *out_alpha0 = 1;
                *out_alpha1 = alpha1.estimate() / sum2;
                *out_alpha2 = alpha2.estimate() / sum2;
                *out_alpha3 = alpha3.estimate() / sum2;
                *out_alpha4 = alpha4.estimate() / sum2;
                return 1;
            }
            else
            { 
                // If T is IntervalType, returns -1
                // If T is expansion, computes exact intersection by projecting to lower dimensions.
                return degenerate_point_tetrahedron_intersection( x0, x1, x2, x3, x4, out_alpha0, out_alpha1, out_alpha2, out_alpha3, out_alpha4 );
            }
            
        }
        
        // --------------------------------------------------------
        
        int degenerate_edge_triangle_intersection(const Vec3e& x0,
                                                  const Vec3e& x1,
                                                  const Vec3e& x2,
                                                  const Vec3e& x3,
                                                  const Vec3e& x4,                                      
                                                  double* alpha0, 
                                                  double* alpha1, 
                                                  double* alpha2,
                                                  double* alpha3,
                                                  double* alpha4)
        {
            
            // degenerate: segment and triangle in same plane
            if(expansion_simplex_intersection3d(1, x1, x2, x3, x4, alpha1, alpha2, alpha3, alpha4)){
                *alpha0=0;
                return 1;
            }
            if(expansion_simplex_intersection3d(1, x0, x2, x3, x4, alpha0, alpha2, alpha3, alpha4)){
                *alpha1=0;
                return 1;
            }
            if(expansion_simplex_intersection3d(2, x0, x1, x3, x4, alpha0, alpha1, alpha3, alpha4)){
                *alpha2=0;
                return 1;
            }
            if(expansion_simplex_intersection3d(2, x0, x1, x2, x4, alpha0, alpha1, alpha2, alpha4)){
                *alpha3=0;
                return 1;
            }
            if(expansion_simplex_intersection3d(2, x0, x1, x2, x3, alpha0, alpha1, alpha2, alpha3)){
                *alpha4=0;
                return 1;
            }
            return 0;      
        }
        
        
        // --------------------------------------------------------
        
        int degenerate_edge_triangle_intersection(const Vec3Interval& ,
                                                  const Vec3Interval& ,
                                                  const Vec3Interval& ,
                                                  const Vec3Interval& ,
                                                  const Vec3Interval& ,                                      
                                                  double* , 
                                                  double* , 
                                                  double* ,
                                                  double* ,
                                                  double* )
        {
            return -1;
        }
        
        
        
        // --------------------------------------------------------
        
        template<class T>
        int edge_triangle_intersection(const Vec<3,T>& x0,
                                       const Vec<3,T>& x1,
                                       const Vec<3,T>& x2,
                                       const Vec<3,T>& x3,
                                       const Vec<3,T>& x4,                                      
                                       double* out_alpha0, 
                                       double* out_alpha1, 
                                       double* out_alpha2,
                                       double* out_alpha3,
                                       double* out_alpha4 )
        {
            
            T::begin_special_arithmetic();
            
            T alpha0;
            orientation3d(x1, x2, x3, x4,alpha0);
            
            T alpha1;
            orientation3d(x0, x2, x3, x4,alpha1);
            alpha1 = -alpha1;
            
            if( certainly_opposite_sign(alpha0, alpha1) )
            {
                T::end_special_arithmetic();
                return 0;
            }
            
            T alpha2;
            orientation3d(x0, x1, x3, x4, alpha2);
            
            T alpha3;
            orientation3d(x0, x1, x2, x4, alpha3);
            alpha3 = -alpha3;
            
            if( certainly_opposite_sign(alpha2, alpha3) )
            {
                T::end_special_arithmetic();
                return 0;
            }
            
            T alpha4;
            orientation3d(x0, x1, x2, x3, alpha4);
            
            if( certainly_opposite_sign(alpha2, alpha4) ) 
            {
                T::end_special_arithmetic();
                return 0;         
            }
            
            if( certainly_opposite_sign(alpha3, alpha4) )
            {
                T::end_special_arithmetic();
                return 0;                  
            }
            
            T::end_special_arithmetic();
            
            if ( alpha0.indefinite_sign() || alpha1.indefinite_sign() || alpha2.indefinite_sign() || alpha3.indefinite_sign() || alpha4.indefinite_sign() )
            {
                // degenerate
                return -1;
            }
            
            double sum1 = alpha0.estimate() + alpha1.estimate();
            double sum2 = alpha2.estimate() + alpha3.estimate() + alpha4.estimate();
            
            if(sum1 && sum2)
            {
                *out_alpha0 = alpha0.estimate() / sum1;
                *out_alpha1 = alpha1.estimate() / sum1;
                *out_alpha2 = alpha2.estimate() / sum2;
                *out_alpha3 = alpha3.estimate() / sum2;
                *out_alpha4 = alpha4.estimate() / sum2;
                return 1;
            }
            else
            { 
                // If T is IntervalType, returns -1
                // If T is expansion, computes exact intersection by projecting to lower dimensions.
                return degenerate_edge_triangle_intersection( x0, x1, x2, x3, x4, out_alpha0, out_alpha1, out_alpha2, out_alpha3, out_alpha4 );
            }
            
        }
        
        // ----------------------------------------
        
        bool edge_triangle_intersection(const Vec3d &d_x0old, const Vec3d &d_x1old, const Vec3d &d_x2old, const Vec3d &d_x3old,
                                        const Vec3d &d_x0new, const Vec3d &d_x1new, const Vec3d &d_x2new, const Vec3d &d_x3new,
                                        bool is_edge_edge,
                                        const Vec3b& ts, const Vec3b& us, const Vec3b& vs,
                                        const Vec3d& d_s0, const Vec3d& d_s1,
                                        double* alphas )
        {
            // TODO: These should be cached already
            Vec3Interval t0, t1, t2;
            get_triangle_vertices( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, is_edge_edge, ts, us, vs, t0, t1, t2 );
            
            Vec3Interval s0, s1;
            make_vector( d_s0, s0 );
            make_vector( d_s1, s1 );
            
            int result = edge_triangle_intersection( s0, s1, t0, t1, t2, &alphas[0], &alphas[1], &alphas[2], &alphas[3], &alphas[4] );
            
            // If degenerate, repeat test using expansion arithmetric
            
            if ( result < 0 )
            {            
                // TODO: These might be cached already
                Vec3e exp_t0, exp_t1, exp_t2;
                get_triangle_vertices( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, is_edge_edge, ts, us, vs, exp_t0, exp_t1, exp_t2 );
                
                Vec3e exp_s0, exp_s1;
                make_vector( d_s0, exp_s0 );
                make_vector( d_s1, exp_s1 );
                
                result = edge_triangle_intersection( exp_s0, exp_s1, exp_t0, exp_t1, exp_t2, &alphas[0], &alphas[1], &alphas[2], &alphas[3], &alphas[4] );
            }
            
            if ( result == 0 )
            {
                return false;
            }
            
            return true;
            
        }
        
        
        // ----------------------------------------
        
        template<class T>   
        T plane_dist( const Vec<3,T>& x, const Vec<3,T>& q, const Vec<3,T>& r, const Vec<3,T>& p )
        {
            return dot( x-p, cross( q-p, r-p ) );
        }
        
        // ----------------------------------------
        
        template<class T>
        void implicit_surface_function( const Vec<3,T>& x, const Vec<3,T>& q0, const Vec<3,T>& q1, const Vec<3,T>& q2, const Vec<3,T>& q3, T& out )
        {
            T::begin_special_arithmetic();
            
            T g012 = plane_dist( x, q0, q1, q2 );
            T g132 = plane_dist( x, q1, q3, q2 );   
            T h12 = g012 * g132;
            T g013 = plane_dist( x, q0, q1, q3 );
            T g032 = plane_dist( x, q0, q3, q2 );   
            T h03 = g013 * g032;   
            out = h12 - h03;
            
            T::end_special_arithmetic();  
        }
        
        // ----------------------------------------
        
        bool test_with_triangles(const Vec3d &d_x0old, const Vec3d &d_x1old, const Vec3d &d_x2old, const Vec3d &d_x3old,
                                 const Vec3d &d_x0new, const Vec3d &d_x1new, const Vec3d &d_x2new, const Vec3d &d_x3new,
                                 bool is_edge_edge,
                                 const Vec4b& ts, const Vec4b& us, const Vec4b& vs,
                                 const Vec3d& s0, const Vec3d& s1,
                                 bool use_positive_triangles, bool& edge_hit )
        {
            // determine which two triangles are on the positive side of f
            
            // first try evaluating sign using interval arithmetic
            
            int sign_h12 = -2;
            
            {
                Vec3Interval q0, q1, q2, q3;
                
                // TODO: These should be cached already
                get_quad_vertices( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, is_edge_edge, ts, us, vs, q0, q1, q2, q3 );
                
                IntervalType::begin_special_arithmetic();
                
                Vec3Interval x = IntervalType(0.5) * ( q0 + q3 );
                IntervalType g012 = plane_dist( x, q0, q1, q2 );
                IntervalType g132 = plane_dist( x, q1, q3, q2 );   
                IntervalType h12 = g012 * g132;
                
                IntervalType::end_special_arithmetic();
                
                if ( h12.is_certainly_negative() ) { sign_h12 = -1; }
                if ( h12.is_certainly_zero() )     { sign_h12 = 0; }
                if ( h12.is_certainly_positive() ) { sign_h12 = 1; }      
            }
            
            if ( sign_h12 == -2 )
            {
                // not sure about sign - compute using expansion
                
                Vec3e eq0, eq1, eq2, eq3;
                
                // TODO: These might be cached already
                get_quad_vertices( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, is_edge_edge, ts, us, vs, eq0, eq1, eq2, eq3 );      
                
                Vec3e x = expansion( 0.5 ) * ( eq0 + eq3 );
                
                expansion g012 = plane_dist( x, eq0, eq1, eq2 );
                expansion g132 = plane_dist( x, eq1, eq3, eq2 );   
                expansion h12 = g012 * g132;
                sign_h12 = sign( h12 );
            }
            
            if ( sign_h12 == 0 ) 
            { 
                // use either pair of triangles
                sign_h12 = 1;
            }
            
            if ( sign_h12 > 0 )
            {
                // positive side: 013, 032
                // negative side: 012, 132
                
                if ( use_positive_triangles )
                {
                    double b[5] = { 0.5, 0.5, 0.5, 0.5, 0.5 };
                    
                    Vec3b t013( ts[0], ts[1], ts[3] );
                    Vec3b u013( us[0], us[1], us[3] );
                    Vec3b v013( vs[0], vs[1], vs[3] );
                    bool hit_a = edge_triangle_intersection( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, is_edge_edge, t013, u013, v013, s0, s1, b );
                    
                    if ( b[2] == 0.0 || b[3] == 0.0 || b[4] == 0.0 ) { edge_hit = true; }
                    if ( b[2] == 1.0 || b[3] == 1.0 || b[4] == 1.0 ) { edge_hit = true; }
                    
                    Vec3b t032( ts[0], ts[3], ts[2] );
                    Vec3b u032( us[0], us[3], us[2] );
                    Vec3b v032( vs[0], vs[3], vs[2] );
                    bool hit_b = edge_triangle_intersection( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, is_edge_edge, t032, u032, v032, s0, s1, b );
                    if ( b[2] == 0.0 || b[3] == 0.0 || b[4] == 0.0 ) { edge_hit = true; }
                    if ( b[2] == 1.0 || b[3] == 1.0 || b[4] == 1.0 ) { edge_hit = true; }
                    
                    return hit_a || hit_b;
                }
                else
                {
                    double b[5] = { 0.5, 0.5, 0.5, 0.5, 0.5 };
                    
                    Vec3b t012( ts[0], ts[1], ts[2] );
                    Vec3b u012( us[0], us[1], us[2] );
                    Vec3b v012( vs[0], vs[1], vs[2] );
                    bool hit_a = edge_triangle_intersection( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, is_edge_edge, t012, u012, v012, s0, s1, b );
                    
                    if ( b[2] == 0.0 || b[3] == 0.0 || b[4] == 0.0 ) { edge_hit = true; }
                    if ( b[2] == 1.0 || b[3] == 1.0 || b[4] == 1.0 ) { edge_hit = true; }
                    
                    Vec3b t132( ts[1], ts[3], ts[2] );
                    Vec3b u132( us[1], us[3], us[2] );
                    Vec3b v132( vs[1], vs[3], vs[2] );
                    bool hit_b = edge_triangle_intersection( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, is_edge_edge, t132, u132, v132, s0, s1, b );
                    if ( b[2] == 0.0 || b[3] == 0.0 || b[4] == 0.0 ) { edge_hit = true; }
                    if ( b[2] == 1.0 || b[3] == 1.0 || b[4] == 1.0 ) { edge_hit = true; }
                    
                    return hit_a || hit_b;
                }
            }
            else
            {   
                // positive side: 012, 132
                // negative side: 013, 032
                
                if ( use_positive_triangles )
                {
                    double b[5] = { 0.5, 0.5, 0.5, 0.5, 0.5 };
                    
                    Vec3b t012( ts[0], ts[1], ts[2] );
                    Vec3b u012( us[0], us[1], us[2] );
                    Vec3b v012( vs[0], vs[1], vs[2] );
                    bool hit_a = edge_triangle_intersection( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, is_edge_edge, t012, u012, v012, s0, s1, b );
                    
                    if ( b[2] == 0.0 || b[3] == 0.0 || b[4] == 0.0 ) { edge_hit = true; }
                    if ( b[2] == 1.0 || b[3] == 1.0 || b[4] == 1.0 ) { edge_hit = true; }  
                    
                    Vec3b t132( ts[1], ts[3], ts[2] );
                    Vec3b u132( us[1], us[3], us[2] );
                    Vec3b v132( vs[1], vs[3], vs[2] );
                    bool hit_b = edge_triangle_intersection( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, is_edge_edge, t132, u132, v132, s0, s1, b );
                    
                    if ( b[2] == 0.0 || b[3] == 0.0 || b[4] == 0.0 ) { edge_hit = true; }
                    if ( b[2] == 1.0 || b[3] == 1.0 || b[4] == 1.0 ) { edge_hit = true; }  
                    
                    return hit_a || hit_b;
                }
                else
                {      
                    double b[5] = { 0.5, 0.5, 0.5, 0.5, 0.5 };
                    Vec3b t013( ts[0], ts[1], ts[3] );
                    Vec3b u013( us[0], us[1], us[3] );
                    Vec3b v013( vs[0], vs[1], vs[3] );
                    bool hit_a = edge_triangle_intersection( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, is_edge_edge, t013, u013, v013, s0, s1, b );
                    
                    if ( b[2] == 0.0 || b[3] == 0.0 || b[4] == 0.0 ) { edge_hit = true; }
                    if ( b[2] == 1.0 || b[3] == 1.0 || b[4] == 1.0 ) { edge_hit = true; } 
                    
                    Vec3b t032( ts[0], ts[3], ts[2] );
                    Vec3b u032( us[0], us[3], us[2] );
                    Vec3b v032( vs[0], vs[3], vs[2] );
                    bool hit_b = edge_triangle_intersection( d_x0old, d_x1old, d_x2old, d_x3old, d_x0new, d_x1new, d_x2new, d_x3new, is_edge_edge, t032, u032, v032, s0, s1, b );
                    
                    if ( b[2] == 0.0 || b[3] == 0.0 || b[4] == 0.0 ) { edge_hit = true; }
                    if ( b[2] == 1.0 || b[3] == 1.0 || b[4] == 1.0 ) { edge_hit = true; }   
                    
                    return hit_a || hit_b;
                }
            }
            
            
        }
        
        
        /// --------------------------------------------------------
        
        inline int plane_sign( const Vec3d& n, const Vec3Interval& x )
        {
            
            IntervalType dist = IntervalType(n[0])*x[0] + IntervalType(n[1])*x[1] + IntervalType(n[2])*x[2];      
            
            if ( dist.is_certainly_negative() ) { return -1; }
            else if ( dist.is_certainly_positive() ) { return 1; }
            return 0;
        }
        
        /// --------------------------------------------------------
        
        inline int quick_1_plane_sign( const Vec3i& normal, const Vec3Interval& x )
        {
            
            IntervalType dist( 0, 0 );
            
            if ( normal[0] < 0 ) { dist -= x[0]; }
            else if ( normal[0] > 0 ) { dist += x[0]; }
            
            if ( normal[1] < 0 ) { dist -= x[1]; }
            else if ( normal[1] > 0 ) { dist += x[1]; }
            
            if ( normal[2] < 0 ) { dist -= x[2]; }
            else if ( normal[2] > 0 ) { dist += x[2]; }
            
            if ( dist.is_certainly_negative() ) { return -1; }
            else if ( dist.is_certainly_positive() ) { return 1; }
            return 0;
        } 
        
    }  // end unnamed namespace for local helper functions
    
    
    //
    // Member function definitions
    //
    
    // ----------------------------------------
    ///
    /// Determine if the given point is inside the tetrahedron given by tet_indices
    ///
    // ----------------------------------------
    
    bool RootParityCollisionTest::point_tetrahedron_intersection( const Vec4ui& quad, const Vec4b& ts, const Vec4b& us, const Vec4b& vs, const Vec3d& d_x )
    {
        
        Vec3Interval x;
        make_vector( d_x, x );
        
        double s[5];
        int result = rootparity::point_tetrahedron_intersection(x, 
                                                                m_interval_hex_vertices[quad[0]], 
                                                                m_interval_hex_vertices[quad[1]],
                                                                m_interval_hex_vertices[quad[2]],
                                                                m_interval_hex_vertices[quad[3]],
                                                                &s[0], &s[1], &s[2], &s[3], &s[4] );
        
        // If degenerate, repeat test using expansion arithmetric
        
        if ( result < 0 )
        {
            
            // Check if the expansion vertices have been computed already
            
            for ( unsigned int i = 0; i < 4; ++i )
            {
                if ( !m_expansion_hex_vertices_computed[ quad[i] ] )
                {
                    if ( m_is_edge_edge )
                    {
                        edge_edge_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, ts[i], us[i], vs[i], m_expansion_hex_vertices[ quad[i] ] );
                    }
                    else
                    {
                        point_triangle_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, ts[i], us[i], vs[i], m_expansion_hex_vertices[ quad[i] ] );
                    }
                    
                    m_expansion_hex_vertices_computed[ quad[i] ] = true;
                }            
            }
            
            // now run the test in expansion arithmetic
            
            Vec3e exp_x;
            make_vector( d_x, exp_x );
            
            result = rootparity::point_tetrahedron_intersection(exp_x, 
                                                                m_expansion_hex_vertices[ quad[0] ], 
                                                                m_expansion_hex_vertices[ quad[1] ], 
                                                                m_expansion_hex_vertices[ quad[2] ], 
                                                                m_expansion_hex_vertices[ quad[3] ], 
                                                                &s[0], &s[1], &s[2], &s[3], &s[4] );
        }
        
        if ( result == 0 )
        {
            return false;
        }
        
        return true;
        
    }
    
    // ----------------------------------------
    ///
    /// Determine if the given segment intersects the triangle
    ///
    // ----------------------------------------
    
    bool RootParityCollisionTest::edge_triangle_intersection( const Vec3ui& triangle,
                                                             const Vec3b& ts, const Vec3b& us, const Vec3b& vs,
                                                             const Vec3d& d_s0, const Vec3d& d_s1,
                                                             double* alphas )
    {
        
        const Vec3Interval& t0 = m_interval_hex_vertices[triangle[0]];
        const Vec3Interval& t1 = m_interval_hex_vertices[triangle[1]];
        const Vec3Interval& t2 = m_interval_hex_vertices[triangle[2]];
        
        Vec3Interval s0, s1;
        make_vector( d_s0, s0 );
        make_vector( d_s1, s1 );
        
        int result = rootparity::edge_triangle_intersection( s0, s1, t0, t1, t2, &alphas[0], &alphas[1], &alphas[2], &alphas[3], &alphas[4] );
        
        // If degenerate, repeat test using expansion arithmetric
        
        if ( result < 0 )
        {
            Vec3e exp_t0, exp_t1, exp_t2;
            get_triangle_vertices( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, m_is_edge_edge, ts, us, vs, exp_t0, exp_t1, exp_t2 );
            
            Vec3e exp_s0, exp_s1;
            make_vector( d_s0, exp_s0 );
            make_vector( d_s1, exp_s1 );
            
            result = rootparity::edge_triangle_intersection( exp_s0, exp_s1, exp_t0, exp_t1, exp_t2, &alphas[0], &alphas[1], &alphas[2], &alphas[3], &alphas[4] );
        }
        
        if ( result == 0 )
        {
            return false;
        }
        
        return true;
        
    }
    
    
    // ----------------------------------------
    ///
    /// Compute the sign of the implicit surface function which is zero on the bilinear patch defined by quad.
    ///
    // ----------------------------------------
    
    int RootParityCollisionTest::implicit_surface_function_sign( const Vec4ui& quad, const Vec4b& ts, const Vec4b& us, const Vec4b& vs, const Vec3d& d_x )
    {
        
        // first try evaluating sign using interval arithmetic
        
        {
            const Vec3Interval& q0 = m_interval_hex_vertices[quad[0]];
            const Vec3Interval& q1 = m_interval_hex_vertices[quad[1]];
            const Vec3Interval& q2 = m_interval_hex_vertices[quad[2]];
            const Vec3Interval& q3 = m_interval_hex_vertices[quad[3]];
            
            Vec3Interval x;
            make_vector( d_x, x );
            
            IntervalType f;
            implicit_surface_function( x, q0, q1, q2, q3, f );      
            
            if ( f.is_certainly_negative() ) { return -1; }
            if ( f.is_certainly_zero() )     { return 0; }
            if ( f.is_certainly_positive() ) { return 1; }      
        }
        
        // not sure about sign - compute using expansion
        
        for ( unsigned int i = 0; i < 4; ++i )
        {
            if ( !m_expansion_hex_vertices_computed[ quad[i] ] )
            {
                if ( m_is_edge_edge )
                {
                    edge_edge_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, ts[i], us[i], vs[i], m_expansion_hex_vertices[ quad[i] ] );
                }
                else
                {
                    point_triangle_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, ts[i], us[i], vs[i], m_expansion_hex_vertices[ quad[i] ] );
                }
                
                m_expansion_hex_vertices_computed[ quad[i] ] = true;
            }            
        }
        
        Vec3e x;
        make_vector( d_x, x );
        expansion ef;
        
        implicit_surface_function( x, m_expansion_hex_vertices[ quad[0] ], m_expansion_hex_vertices[ quad[1] ], m_expansion_hex_vertices[ quad[2] ], m_expansion_hex_vertices[ quad[3] ], ef );
        
        return sign( ef );
        
    }
    
    
    // ----------------------------------------
    ///
    /// Test the segment s0-s1 against the bilinear patch defined by quad to determine whether there is an even or odd number of 
    /// intersections. Returns true if odd.
    ///
    // ----------------------------------------
    
    bool RootParityCollisionTest::ray_quad_intersection_parity( const Vec4ui& quad, 
                                                               const Vec4b& ts, const Vec4b& us, const Vec4b& vs,
                                                               const Vec3d& ray_origin, const Vec3d& ray_head, 
                                                               bool& edge_hit, bool& origin_on_surface )
    {
        
        bool point_in_tet0 = point_tetrahedron_intersection( quad, ts, us, vs, ray_origin );
        
        if ( point_in_tet0 )
        {
            
            int sign0 = implicit_surface_function_sign( quad, ts, us, vs, ray_origin );
            
            if ( sign0 == 0 )
            {
                origin_on_surface = true;
                return false;
            }
            
            // s0 is inside the tet, s1 is outside
            
            if ( sign0 > 0 )
            {
                // use the triangles on the negative side of f()
                return test_with_triangles( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, m_is_edge_edge, ts, us, vs, ray_origin, ray_head, false, edge_hit );
                
            }
            else
            {
                // use the triangles on the positive side of f()
                return test_with_triangles( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, m_is_edge_edge, ts, us, vs, ray_origin, ray_head, true, edge_hit );
            }
            
        }
        else
        {
            // s0 is outside the tet
            
            // both outside         
            // use either set of triangles
            
            bool hit_a, hit_b;
            
            {
                double b[5] = { 0.5, 0.5, 0.5, 0.5, 0.5 };
                
                Vec3ui triangle013( quad[0], quad[1], quad[3] );
                Vec3b t013( ts[0], ts[1], ts[3] );
                Vec3b u013( us[0], us[1], us[3] );
                Vec3b v013( vs[0], vs[1], vs[3] );
                hit_a = edge_triangle_intersection( triangle013, t013, u013, v013, ray_origin, ray_head, b );
                
                if ( b[2] == 0.0 || b[3] == 0.0 || b[4] == 0.0 ) { edge_hit = true; }
                if ( b[2] == 1.0 || b[3] == 1.0 || b[4] == 1.0 ) { edge_hit = true; }                 
            }
            
            
            {
                double b[5] = { 0.5, 0.5, 0.5, 0.5, 0.5 };
                
                Vec3ui triangle032( quad[0], quad[3], quad[2] );
                Vec3b t032( ts[0], ts[3], ts[2] );
                Vec3b u032( us[0], us[3], us[2] );
                Vec3b v032( vs[0], vs[3], vs[2] );
                hit_b = edge_triangle_intersection( triangle032, t032, u032, v032, ray_origin, ray_head, b );
                
                if ( b[2] == 0.0 || b[3] == 0.0 || b[4] == 0.0 ) { edge_hit = true; }
                if ( b[2] == 1.0 || b[3] == 1.0 || b[4] == 1.0 ) { edge_hit = true; } 
            }
            
            return hit_a ^ hit_b;
        }
        
        assert( !"Should not get here" );
        
        return true;
        
    }
    
    
    // --------------------------------------------------------
    ///
    /// Determine the parity of the number of intersections between a ray from the origin and the generalized prism made up 
    /// of f(G) where G = the vertices of the domain boundary.
    ///
    // --------------------------------------------------------
    
    bool RootParityCollisionTest::ray_prism_parity_test()
    {
        
        Vec3d test_ray( m_ray );
        
        double ray_len = mag( test_ray );
        
        std::vector<Vec3ui> tris(2);
        tris[0] = Vec3ui( 0, 1, 2 );
        tris[1] = Vec3ui( 3, 4, 5 );
        
        std::vector<Vec4ui> quads(3);
        quads[0] = Vec4ui( 0, 1, 3, 4 );
        quads[1] = Vec4ui( 1, 2, 4, 5 );
        quads[2] = Vec4ui( 0, 2, 3, 5 );
        
        const bool vertex_ts[6] = { 0, 0, 0, 1, 1, 1 };
        const bool vertex_us[6] = { 0, 1, 0, 0, 1, 0 };
        const bool vertex_vs[6] = { 0, 0, 1, 0, 0, 1 };      
        
        // for debugging purposes, store the result of each hit test
        std::vector<bool> tri_hits( 2, false );
        std::vector<bool> quad_hits( 3, false );
        
        bool good_hit = false;
        unsigned int num_tries = 0;
        
        while (!good_hit && num_tries++ < 10 )
        {
            good_hit = true;
            
            // ray-cast against each tri and each quad
            
            for ( unsigned int i = 0; i < tris.size(); ++i )
            {
                const Vec3ui& t = tris[i];
                double bary[5] = { 0.5, 0.5, 0.5, 0.5, 0.5 };
                
                Vec3b ts( vertex_ts[t[0]], vertex_ts[t[1]], vertex_ts[t[2]] );
                Vec3b us( vertex_us[t[0]], vertex_us[t[1]], vertex_us[t[2]] );
                Vec3b vs( vertex_vs[t[0]], vertex_vs[t[1]], vertex_vs[t[2]] );
                
                tri_hits[i] = rootparity::edge_triangle_intersection( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, m_is_edge_edge, ts, us, vs, Vec3d(0,0,0), test_ray, bary );
                
                if ( tri_hits[i] )
                {
                    if ( bary[2] == 0.0 || bary[3] == 0.0 || bary[4] == 0.0 ) { good_hit = false; }
                    if ( bary[2] == 1.0 || bary[3] == 1.0 || bary[4] == 1.0 ) { good_hit = false; }
                    if ( bary[0] == 1.0 ) 
                    {
                        // origin is on surface                        
                        return true;
                    }
                }
            }
            
            for ( unsigned int i = 0; i < quads.size(); ++i )
            {
                const Vec4ui& q = quads[i];
                bool edge_hit = false;
                bool origin_on_surface = false;
                
                Vec4b ts( vertex_ts[q[0]], vertex_ts[q[1]], vertex_ts[q[2]], vertex_ts[q[3]] );
                Vec4b us( vertex_us[q[0]], vertex_us[q[1]], vertex_us[q[2]], vertex_us[q[3]] );
                Vec4b vs( vertex_vs[q[0]], vertex_vs[q[1]], vertex_vs[q[2]], vertex_vs[q[3]] );
                
                quad_hits[i] = ray_quad_intersection_parity( q, ts, us, vs, Vec3d(0,0,0), test_ray, edge_hit, origin_on_surface );
                
                if ( edge_hit ) 
                { 
                    good_hit = false; 
                    break;
                }
                
                if ( origin_on_surface ) 
                { 
                    return true; 
                }
            }
            
            // check if any hit was not good
            if ( !good_hit )
            {
                double r = rand() / (double)RAND_MAX * 2.0 * M_PI;
                test_ray[0] = cos(r) * ray_len;
                test_ray[1] = -sin(r) * ray_len;
            }
        }
        
        if ( !good_hit ) { return true; }
        
        unsigned int num_hits = 0;
        
        for ( unsigned int i = 0; i < tri_hits.size(); ++i )
        {
            if ( tri_hits[i] ) { ++num_hits; }
        }
        
        for ( unsigned int i = 0; i < quad_hits.size(); ++i )
        {
            if ( quad_hits[i] ) { ++num_hits; }
        }
        
        return ( num_hits % 2 == 1 );
        
    }
    
    // --------------------------------------------------------
    ///
    /// Determine the parity of the number of intersections between a ray from the origin and the generalized hexahedron made
    /// up of f(G) where G = the vertices of the domain boundary (corners of the unit cube).
    ///
    // --------------------------------------------------------
    
    bool RootParityCollisionTest::ray_hex_parity_test( )
    {
        
        const bool vertex_ts[8] = { 0, 1, 1, 0, 0, 1, 1, 0 };
        const bool vertex_us[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
        const bool vertex_vs[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };      
        
        Vec3d test_ray( m_ray );
        
        double ray_len = mag( test_ray );
        
        bool good_hit = false;
        unsigned int num_tries = 0;
        
        // bilinear patch faces of the mapped hex
        std::vector<Vec4ui> quads(6);
        quads[0] = Vec4ui( 0, 1, 3, 2 );
        quads[1] = Vec4ui( 0, 1, 4, 5 );
        quads[2] = Vec4ui( 1, 2, 5, 6 );
        quads[3] = Vec4ui( 2, 3, 6, 7 );
        quads[4] = Vec4ui( 3, 0, 7, 4 );
        quads[5] = Vec4ui( 4, 5, 7, 6 );
        
        // for debugging purposes, store the result of each hit test
        std::vector<bool> hits( 6, false );
        
        // (t,u,v) coordinates of each vertex on the hexahedron
        
        
        while (!good_hit && num_tries++ < 10 )
        {
            good_hit = true;
            bool any_origin_on_surface = false;
            
            // ray-cast against each quad
            
            for ( int i = 0; i < 6; ++i )
            {
                const Vec4ui& quad = quads[i];
                
                Vec4b ts( vertex_ts[quad[0]], vertex_ts[quad[1]], vertex_ts[quad[2]], vertex_ts[quad[3]] );
                Vec4b us( vertex_us[quad[0]], vertex_us[quad[1]], vertex_us[quad[2]], vertex_us[quad[3]] );
                Vec4b vs( vertex_vs[quad[0]], vertex_vs[quad[1]], vertex_vs[quad[2]], vertex_vs[quad[3]] );
                bool edge_hit = false;
                bool origin_on_surface = false;
                
                hits[i] = ray_quad_intersection_parity( quad, ts, us, vs, Vec3d(0,0,0), test_ray, edge_hit, origin_on_surface );
                
                if ( edge_hit ) 
                { 
                    good_hit = false; 
                }
                
                if ( origin_on_surface )
                {
                    any_origin_on_surface = true;
                }            
            }
            
            
            if ( any_origin_on_surface )
            {
                return true;
            }
            
            // check if any hit was not okay
            if ( !good_hit )
            {
                double r = rand() / (double)RAND_MAX * 2.0 * M_PI;
                test_ray[0] =  cos(r) * ray_len;
                test_ray[1] = -sin(r) * ray_len;
            }
        }
        
        if ( !good_hit ) { return true; }
        
        unsigned int num_hits = 0;
        for ( unsigned int i = 0; i < hits.size(); ++i )
        {
            if ( hits[i] ) { ++num_hits; }
        }
        
        return ( num_hits % 2 == 1 );
        
    }
    
    
    /// --------------------------------------------------------
    ///
    /// For each triangle, form the plane it lies on, and determine if all m_interval_hex_vertices are on one side of the plane.
    ///
    /// --------------------------------------------------------   
    
    bool RootParityCollisionTest::plane_culling( const std::vector<Vec3ui>& triangles, const std::vector<Vec3d>& boundary_vertices )
    {
        
        std::size_t num_triangles = triangles.size();
        std::size_t num_boundary_vertices = boundary_vertices.size();
        
        for ( unsigned int i = 0; i < num_triangles; ++i )
        {
            const Vec3ui& t = triangles[i];
            Vec3d normal = cross( boundary_vertices[t[1]] - boundary_vertices[t[0]], boundary_vertices[t[2]] - boundary_vertices[t[0]] );
            
            if ( mag(normal) == 0.0 ) { continue; }
            
            normal = normalized(normal);
            
            IntervalType::begin_special_arithmetic();         
            
            const int sgn = plane_sign( normal, m_interval_hex_vertices[0] );
            
            IntervalType::end_special_arithmetic();
            
            if ( sgn == 0 )
            {
                continue;
            }
            
            bool all_same_side = true;
            
            for ( unsigned int v = 1; v < num_boundary_vertices; ++v )
            {
                IntervalType::begin_special_arithmetic();         
                
                const int this_plane_sign = plane_sign( normal, m_interval_hex_vertices[v] );
                
                IntervalType::end_special_arithmetic();
                
                if ( this_plane_sign == 0 || this_plane_sign != sgn )
                {
                    all_same_side = false;
                    break;
                }
            }
            
            if ( all_same_side )
            {
                return true;
            }
        }
        
        IntervalType::end_special_arithmetic();
        return false;
        
    }
    
    /// --------------------------------------------------------
    ///
    /// Take a set of planes defined by the mapped domain boundary, and determine if all m_interval_hex_vertices on one side of
    /// any plane.
    ///
    /// --------------------------------------------------------
    
    bool RootParityCollisionTest::edge_edge_interval_plane_culling()
    {
        
        std::vector<Vec3d> hex_vertices(8);
        for ( unsigned int i = 0; i < 8; ++i )
        {
            hex_vertices[i][0] = m_interval_hex_vertices[i][0].estimate();
            hex_vertices[i][1] = m_interval_hex_vertices[i][1].estimate();
            hex_vertices[i][2] = m_interval_hex_vertices[i][2].estimate();
        }
        
        std::vector<Vec3ui> triangles(12);
        triangles[0] = Vec3ui(0,1,3);
        triangles[1] = Vec3ui(0,3,2);
        triangles[2] = Vec3ui(0,1,4);
        triangles[3] = Vec3ui(0,4,5);
        triangles[4] = Vec3ui(1,2,5);
        triangles[5] = Vec3ui(1,5,6);
        triangles[6] = Vec3ui(2,3,6);
        triangles[7] = Vec3ui(2,6,7);
        triangles[8] = Vec3ui(3,0,7);
        triangles[9] = Vec3ui(3,7,4);
        triangles[10] = Vec3ui(4,5,7);
        triangles[11] = Vec3ui(4,7,6);
        
        return plane_culling( triangles, hex_vertices );
        
    }
    
    /// --------------------------------------------------------
    ///
    /// Take a set of planes defined by the mapped domain boundary, and determine if all m_interval_hex_vertices lie on one side of
    /// any plane.
    ///
    /// --------------------------------------------------------
    
    bool RootParityCollisionTest::point_triangle_interval_plane_culling()
    {
        
        std::vector<Vec3d> hex_vertices(6);
        for ( unsigned int i = 0; i < 6; ++i )
        {
            hex_vertices[i][0] = m_interval_hex_vertices[i][0].estimate();
            hex_vertices[i][1] = m_interval_hex_vertices[i][1].estimate();
            hex_vertices[i][2] = m_interval_hex_vertices[i][2].estimate();
        }
        
        std::vector<Vec3ui> triangles(8);
        triangles[0] = Vec3ui(0,1,2);
        triangles[1] = Vec3ui(3,4,5);
        triangles[2] = Vec3ui(0,1,3);
        triangles[3] = Vec3ui(0,3,4);
        triangles[4] = Vec3ui(1,2,4);
        triangles[5] = Vec3ui(1,4,5);
        triangles[6] = Vec3ui(0,2,3);
        triangles[7] = Vec3ui(0,2,5);
        
        return plane_culling( triangles, hex_vertices );
    }
    
    
    /// --------------------------------------------------------
    ///
    /// Take a fixed set of planes and determine if all m_interval_hex_vertices lie on one side of any plane.
    /// 
    /// --------------------------------------------------------
    
    bool RootParityCollisionTest::fixed_plane_culling( unsigned int num_hex_vertices )
    {
        
        for ( int i = -1; i < 1; ++i )
        {
            for ( int j = -1; j < 1; ++j )
            {
                for ( int k = -1; k < 1; ++k )
                {
                    
                    Vec3i normal( i, k, j );
                    
                    const int plane_sign = quick_1_plane_sign( normal, m_interval_hex_vertices[0] );
                    
                    if ( plane_sign == 0 )
                    {
                        continue;
                    }
                    
                    bool all_same_side = true;
                    
                    for ( unsigned int v = 1; v < num_hex_vertices; ++v )
                    {
                        const int this_plane_sign = quick_1_plane_sign( normal, m_interval_hex_vertices[v] );
                        
                        if ( this_plane_sign == 0 || this_plane_sign != plane_sign )
                        {
                            all_same_side = false;
                            break;
                        }
                    }
                    
                    if ( all_same_side ) 
                    {
                        return true;
                    }
                    
                }
            }
        }
        
        
        return false;
        
    }
    
    
    
    /// --------------------------------------------------------
    ///
    /// Run edge-edge continuous collision detection
    ///
    /// --------------------------------------------------------
    
    bool RootParityCollisionTest::edge_edge_collision( )
    {
        
        static const bool vertex_ts[8] = { 0, 1, 1, 0, 0, 1, 1, 0 };
        static const bool vertex_us[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
        static const bool vertex_vs[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };      
        
        // Get the transformed corners of the domain boundary in interval representation
        
        IntervalType::begin_special_arithmetic();
        edge_edge_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, 
                                     vertex_ts[0], vertex_us[0], vertex_vs[0], m_interval_hex_vertices[0] );
        edge_edge_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, 
                                     vertex_ts[1], vertex_us[1], vertex_vs[1], m_interval_hex_vertices[1] );
        edge_edge_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, 
                                     vertex_ts[2], vertex_us[2], vertex_vs[2], m_interval_hex_vertices[2] );
        edge_edge_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, 
                                     vertex_ts[3], vertex_us[3], vertex_vs[3], m_interval_hex_vertices[3] );      
        edge_edge_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, 
                                     vertex_ts[4], vertex_us[4], vertex_vs[4], m_interval_hex_vertices[4] );
        edge_edge_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, 
                                     vertex_ts[5], vertex_us[5], vertex_vs[5], m_interval_hex_vertices[5] );
        edge_edge_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, 
                                     vertex_ts[6], vertex_us[6], vertex_vs[6], m_interval_hex_vertices[6] );
        edge_edge_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, 
                                     vertex_ts[7], vertex_us[7], vertex_vs[7], m_interval_hex_vertices[7] );      
        
        // Plane culling: check if all corners are on one side of the plane passing through the origin
        
        bool plane_culled = fixed_plane_culling(8);      
        
        IntervalType::end_special_arithmetic();
        
        bool hex_plane_culled = false;
        
        if ( !plane_culled )
        {
            hex_plane_culled = edge_edge_interval_plane_culling();
        }
        
        bool safe_parity = false;
        
        
        if ( !plane_culled && !hex_plane_culled )
        {
            
            // Cast ray from origin against boundary image (6 quads)
            
            Vec3d xmin( 1e30 ), xmax( -1e30 );
            
            for ( unsigned int i = 0; i < 8; ++i )
            {
                Vec2d interval0 = m_interval_hex_vertices[i][0].get_actual_interval();
                Vec2d interval1 = m_interval_hex_vertices[i][1].get_actual_interval();
                Vec2d interval2 = m_interval_hex_vertices[i][2].get_actual_interval();
                
                xmin[0] = std::min( xmin[0], interval0[0]  );
                xmin[1] = std::min( xmin[1], interval1[0]  );
                xmin[2] = std::min( xmin[2], interval2[0]  );
                
                xmax[0] = std::max( xmax[0], interval0[1]  );
                xmax[1] = std::max( xmax[1], interval1[1]  );
                xmax[2] = std::max( xmax[2], interval2[1]  );
            }
            
            const double ray_len = std::max( mag(xmin), mag(xmax) ) + 10.0;
            
            m_ray = Vec3d( ray_len, ray_len, 0 );
            
            safe_parity = ray_hex_parity_test( );
            
        }      
        
        return safe_parity;
        
    }
    
    
    /// ----------------------------------------
    ///
    /// Run point-triangle continuous collision detection
    ///
    /// ----------------------------------------
    
    bool RootParityCollisionTest::point_triangle_collision( )
    {
        static const bool vertex_ts[6] = { 0, 0, 0, 1, 1, 1 };
        static const bool vertex_us[6] = { 0, 1, 0, 0, 1, 0 };
        static const bool vertex_vs[6] = { 0, 0, 1, 0, 0, 1 };      
        
        // Get the transformed corners of the domain boundary in interval representation
        
        IntervalType::begin_special_arithmetic();
        point_triangle_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, 
                                          vertex_ts[0], vertex_us[0], vertex_vs[0], m_interval_hex_vertices[0] );
        point_triangle_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, 
                                          vertex_ts[1], vertex_us[1], vertex_vs[1], m_interval_hex_vertices[1] );
        point_triangle_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, 
                                          vertex_ts[2], vertex_us[2], vertex_vs[2], m_interval_hex_vertices[2] );
        point_triangle_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, 
                                          vertex_ts[3], vertex_us[3], vertex_vs[3], m_interval_hex_vertices[3] );
        point_triangle_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, 
                                          vertex_ts[4], vertex_us[4], vertex_vs[4], m_interval_hex_vertices[4] );
        point_triangle_collision_function( m_x0old, m_x1old, m_x2old, m_x3old, m_x0new, m_x1new, m_x2new, m_x3new, 
                                          vertex_ts[5], vertex_us[5], vertex_vs[5], m_interval_hex_vertices[5] );
        
        // Plane culling: check if all corners are on one side of the plane passing through the origin
        
        bool plane_culled = fixed_plane_culling(6);      
        
        IntervalType::end_special_arithmetic();
        
        if ( plane_culled )
        {
            return false;
        }
        
        bool prism_plane_culled = point_triangle_interval_plane_culling();
        
        if ( prism_plane_culled )
        {
            return false;
        }
        
        // Cast ray from origin against boundary image (3 quads + 2 triangles)
        
        Vec3d xmin( 1e30 ), xmax( -1e30 );
        
        for ( unsigned int i = 0; i < 6; ++i )
        {
            Vec2d interval0 = m_interval_hex_vertices[i][0].get_actual_interval();
            Vec2d interval1 = m_interval_hex_vertices[i][1].get_actual_interval();
            Vec2d interval2 = m_interval_hex_vertices[i][2].get_actual_interval();
            
            xmin[0] = std::min( xmin[0], interval0[0]  );
            xmin[1] = std::min( xmin[1], interval1[0]  );
            xmin[2] = std::min( xmin[2], interval2[0]  );
            
            xmax[0] = std::max( xmax[0], interval0[1]  );
            xmax[1] = std::max( xmax[1], interval1[1]  );
            xmax[2] = std::max( xmax[2], interval2[1]  );
        }
        
        const double ray_len = std::max( mag(xmin), mag(xmax) ) + 10.0;
        
        m_ray = Vec3d( ray_len, ray_len, 0 );
        
        bool safe_parity = ray_prism_parity_test( );
        
        return safe_parity;
        
    }
    
}  // namespace rootparity 
} // namespace LosTopos


