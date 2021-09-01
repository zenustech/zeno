// Voro++, a 3D cell-based Voronoi library
//
// Author   : Chris H. Rycroft (LBL / UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

/** \file v_compute.hh
 * \brief Header file for the 2D voro_compute template and related classes. */

#ifndef VOROPP_V_COMPUTE_2D_HH
#define VOROPP_V_COMPUTE_2D_HH

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
using namespace std;

#include "config.hh"
#include "worklist_2d.hh"
#include "cell_2d.hh"
#include "cell_nc_2d.hh"

namespace voro {

/** \brief Structure for holding information about a particle.
 *
 * This small structure holds information about a single particle, and is used
 * by several of the routines in the voro_compute template for passing
 * information by reference between functions. */
struct particle_record_2d {
	/** The index of the block that the particle is within. */
	int ij;
	/** The number of particle within its block. */
	int l;
	/** The x-index of the block. */
	int di;
	/** The y-index of the block. */
	int dj;
};

/** \brief Template for carrying out Voronoi cell computations. */
template <class c_class_2d>
class voro_compute_2d {
	public:
		/** A reference to the container class on which to carry out*/
		c_class_2d &con;
		/** The size of an internal computational block in the x
		 * direction. */
		const double boxx;
		/** The size of an internal computational block in the y
		 * direction. */
		const double boxy;
		/** The inverse box length in the x direction, set to
		 * nx/(bx-ax). */
		const double xsp;
		/** The inverse box length in the y direction, set to
		 * ny/(by-ay). */
		const double ysp;
		/** The number of boxes in the x direction for the searching mask. */
		const int hx;
		/** The number of boxes in the y direction for the searching mask. */
		const int hy;
		/** A constant, set to the value of hx multiplied by hy, which
		 * is used in the routines which step through mask boxes in
		 * sequence. */
		const int hxy;
		/** The number of floating point entries to store for each
		 * particle. */
		const int ps;
		/** This array holds the numerical IDs of each particle in each
		 * computational box. */
		int **id;
		/** A two dimensional array holding particle positions. For the
		 * derived container_poly class, this also holds particle
		 * radii. */
		double **p;
		/** An array holding the number of particles within each
		 * computational box of the container. */
		int *co;
		voro_compute_2d(c_class_2d &con_,int hx_,int hy_);
		/** The class destructor frees the dynamically allocated memory
		 * for the mask and queue. */
		~voro_compute_2d() {
			delete [] qu;
			delete [] mask;
		}
		template<class v_cell_2d>
		bool compute_cell(v_cell_2d &c,int ij,int s,int ci,int cj);
		void find_voronoi_cell(double x,double y,int ci,int cj,int ij,particle_record_2d &w,double &mrs);
	private:
		/** A constant set to boxx*boxx+boxy*boxy+boxz*boxz, which is
		 * frequently used in the computation. */
		const double bxsq;
		/** This sets the current value being used to mark tested blocks
		 * in the mask. */
		unsigned int mv;
		/** The current size of the search list. */
		int qu_size;
		/** A pointer to the array of worklists. */
		const unsigned int *wl;
		/** An pointer to the array holding the minimum distances
		 * associated with the worklists. */
		double *mrad;
		/** This array is used during the cell computation to determine
		 * which blocks have been considered. */
		unsigned int *mask;
		/** An array is used to store the queue of blocks to test
		 * during the Voronoi cell computation. */
		int *qu;
		/** A pointer to the end of the queue array, used to determine
		 * when the queue is full. */
		int *qu_l;
		template<class v_cell_2d>
		inline bool corner_test(v_cell_2d &c,double xl,double yl,double xh,double yh);
		template<class v_cell_2d>
		inline bool edge_x_test(v_cell_2d &c,double xl,double y0,double y1);
		template<class v_cell_2d>
		inline bool edge_y_test(v_cell_2d &c,double x0,double yl,double x1);
		bool compute_min_max_radius(int di,int dj,double fx,double fy,double gx,double gy,double& crs,double mrs);
		bool compute_min_radius(int di,int dj,double fx,double fy,double mrs);
		inline void add_to_mask(int ei,int ej,int *&qu_e);
		inline void scan_bits_mask_add(unsigned int q,unsigned int *mij,int ei,int ej,int *&qu_e);
		inline void scan_all(int ij,double x,double y,int di,int dj,particle_record_2d &w,double &mrs);
		void add_list_memory(int*& qu_s,int*& qu_e);
		/** Resets the mask in cases where the mask counter wraps
		 * around. */
		inline void reset_mask() {
			for(unsigned int *mp=mask;mp<mask+hxy;mp++) *mp=0;
		}
};

}

#endif
