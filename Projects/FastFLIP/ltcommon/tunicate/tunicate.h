#ifndef TUNICATE_H
#define TUNICATE_H

// Released into the public domain by Robert Bridson, 2009.

// Warning: this code hasn't been formally proven to be correct, and
// almost certainly *will* fail in cases of overflow or underflow.

// Note: all routines (except in 1d) internally change the floating point
// rounding mode, and set it to the usual "round-to-nearest" on exit. If the
// calling code relies on it being something else, you must arrange this
// yourself.

#ifdef __cplusplus
extern "C" {
#endif
    
    //==============================================================================
    // The orientation routines take, in n dimensions, n+1 vectors
    // describing the vertices of a simplex, and return the associated
    // determinant (related to the signed volume of the simplex).
    // The sign of this determinant gives the orientation of the simplex.
    // The sign is determined exactly, with zero indicating a degenerate simplex.
    // The magnitude of the result is only approximate.
    
    // This one is a little silly, but useful in generality.
    // It returns the length of interval x0-x1.
    double
    orientation1d(const double* x0,
                  const double* x1);
    
    // This returns twice the signed area of the triangle x0-x1-x2.
    double
    orientation2d(const double* x0,
                  const double* x1,
                  const double* x2);
    
    void
    interval_orientation2d(const double* x0,
                           const double* x1,
                           const double* x2,
                           double* lower,
                           double* upper);
    
    // This returns six times the signed volume of the tetrahedron x0-x1-x2-x3.
    double
    orientation3d(const double* x0,
                  const double* x1,
                  const double* x2,
                  const double* x3);
    
    void
    interval_orientation3d(const double* x0,
                           const double* x1,
                           const double* x2,
                           const double* x3,
                           double* lower,
                           double* upper);
    
    // This returns 24 times the signed hypervolume of the simplex x0-x1-x2-x3-x4.
    // x0, ..., x4 are 3d vectors; time0, ..., time4 should be either 0 or 1,
    // providing the fourth coordinate.
    double
    orientation_time3d(const double* x0, int time0,
                       const double* x1, int time1,
                       const double* x2, int time2,
                       const double* x3, int time3,
                       const double* x4, int time4);
    
    // This returns 24 times the signed hypervolume of the simplex x0-x1-x2-x3-x4.
    double
    orientation4d(const double* x0,
                  const double* x1,
                  const double* x2,
                  const double* x3,
                  const double* x4);
    
    //==============================================================================
    // The sos_orientation routines are similar to the orientation routines, only
    // are guaranteed to never return zero: Edelsbrunner and Mucke's Simulation
    // of Simplicity (SoS) procudure is used to perturb the results to positive or
    // negative in a consistent way (returning plus or minus the minimum double
    // instead of zero). The crucial requirement is that a distinct integer
    // priority be given to each vertex, say the index in a mesh. Each coordinate
    // of each vertex is symbolically perturbed by an infinitesimal positibe
    // amount, with the perturbation of the i'th coordinate of the j'th point
    // being unassailably larger than that of i'th coordinates points less than j,
    // and the i+1'st or greater coordinates of all points.
    
    // This one is a little silly, but useful in generality.
    // It returns the length of interval x0-x1.
    double
    sos_orientation1d(int priority0, const double* x0,
                      int priority1, const double* x1);
    
    // This returns twice the signed area of the triangle x0-x1-x2.
    double
    sos_orientation2d(int priority0, const double* x0,
                      int priority1, const double* x1,
                      int priority2, const double* x2);
    
    // This returns six times the signed volume of the tetrahedron x0-x1-x2-x3.
    double
    sos_orientation3d(int priority0, const double* x0,
                      int priority1, const double* x1,
                      int priority2, const double* x2,
                      int priority3, const double* x3);
    
    // This returns 24 times the signed hypervolume of the simplex x0-x1-x2-x3-x4.
    double
    sos_orientation4d(int priority0, const double* x0,
                      int priority1, const double* x1,
                      int priority2, const double* x2,
                      int priority3, const double* x3,
                      int priority4, const double* x4);
    
    //==============================================================================
    // Simplex-simplex intersection tests, in n dimensions, take n+2 points as the
    // vertices of the two simplices along with an integer k (1<=k<=n+1) specifying
    // the number of vertices in the first simplex: the first simplex uses vertices
    // 0, 1, ..., k-1, and the second simplex uses the rest. The routines return 0
    // for no intersection, and 1 for intersection (including degenerate cases).
    // If there is an intersection, approximations to the barycentric coordinates
    // (alpha) of the intersection are also determined, with the sum of the first k
    // alpha values approximately equal to 1, the sum of the remainder also
    // approximately equal to 1, and the point of intersection approximately equal
    // to the associated weighted sum of either simplice's vertices. If there is no
    // intersection, some or all of the alpha values may be overwritten with
    // intermediate values of no use.
    
    int
    simplex_intersection1d(int k,
                           const double* x0,
                           const double* x1,
                           const double* x2,
                           double* alpha0, 
                           double* alpha1, 
                           double* alpha2);
    
    int
    simplex_intersection2d(int k,
                           const double* x0,
                           const double* x1,
                           const double* x2,
                           const double* x3,
                           double* alpha0, 
                           double* alpha1, 
                           double* alpha2,
                           double* alpha3);
    
    int
    simplex_intersection3d(int k,
                           const double* x0,
                           const double* x1,
                           const double* x2,
                           const double* x3,
                           const double* x4,
                           double* alpha0, 
                           double* alpha1, 
                           double* alpha2,
                           double* alpha3,
                           double* alpha4);
    
    int
    simplex_intersection_time3d(int k,
                                const double* x0, int time0,
                                const double* x1, int time1,
                                const double* x2, int time2,
                                const double* x3, int time3,
                                const double* x4, int time4,
                                const double* x5, int time5,
                                double* alpha0, 
                                double* alpha1, 
                                double* alpha2,
                                double* alpha3,
                                double* alpha4,
                                double* alpha5);
    
    int
    simplex_intersection4d(int k,
                           const double* x0,
                           const double* x1,
                           const double* x2,
                           const double* x3,
                           const double* x4,
                           const double* x5,
                           double* alpha0, 
                           double* alpha1, 
                           double* alpha2,
                           double* alpha3,
                           double* alpha4,
                           double* alpha5);
    
    //==============================================================================
    // The SoS simplex intersection tests, in n dimensions, take n+2 points as the
    // vertices of the two simplices along with an integer k (1<=k<=n+1) specifying
    // the number of vertices in the first simplex: the first simplex uses vertices
    // 0, 1, ..., k-1, and the second simplex uses the rest. The routines return 0
    // for no intersection, and 1 for intersection---with SoS perturbation as above
    // to handle degenerate situations. If there is an intersection, approximations
    // to the barycentric coordinates (alpha) of the intersection are also
    // determined, with the sum of the first k alpha values approximately equal
    // to 1, the sum of the remainder also approximately equal to 1, and the
    // point of intersection approximately equal to the associated weighted sum of
    // either simplice's vertices. If there is no intersection, some or all of
    // the alpha values may be overwritten with intermediate values of no use.
    
    int
    sos_simplex_intersection1d(int k,
                               int priority0, const double* x0,
                               int priority1, const double* x1,
                               int priority2, const double* x2,
                               double* alpha0, 
                               double* alpha1, 
                               double* alpha2);
    
    int
    sos_simplex_intersection2d(int k,
                               int priority0, const double* x0,
                               int priority1, const double* x1,
                               int priority2, const double* x2,
                               int priority3, const double* x3,
                               double* alpha0, 
                               double* alpha1, 
                               double* alpha2,
                               double* alpha3);
    
    int
    sos_simplex_intersection3d(int k,
                               int priority0, const double* x0,
                               int priority1, const double* x1,
                               int priority2, const double* x2,
                               int priority3, const double* x3,
                               int priority4, const double* x4,
                               double* alpha0, 
                               double* alpha1, 
                               double* alpha2,
                               double* alpha3,
                               double* alpha4);
    
    int
    sos_simplex_intersection4d(int k,
                               int priority0, const double* x0,
                               int priority1, const double* x1,
                               int priority2, const double* x2,
                               int priority3, const double* x3,
                               int priority4, const double* x4,
                               int priority5, const double* x5,
                               double* alpha0, 
                               double* alpha1, 
                               double* alpha2,
                               double* alpha3,
                               double* alpha4,
                               double* alpha5);
    
#ifdef __cplusplus
} // end of extern "C" block
#endif

#endif
