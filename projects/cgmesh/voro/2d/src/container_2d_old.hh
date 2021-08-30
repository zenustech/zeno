/** \file container_2d.hh
 * \brief Header file for the container_2d class. */

#ifndef VOROPP_CONTAINER_2D_HH
#define VOROPP_CONTAINER_2D_HH

#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace std;

#include "config.hh"
#include "cell_2d.hh"

class voropp_loop_2d;

/** \brief A class representing the whole 2D simulation region.
 *
 * The container class represents the whole simulation region. The
 * container constructor sets up the geometry and periodicity, and divides
 * the geometry into rectangular grid of blocks, each of which handles the
 * particles in a particular area. Routines exist for putting in particles,
 * importing particles from standard input, and carrying out Voronoi
 * calculations. */
class container_2d {
	public:
		/** The minimum x coordinate of the container. */
		const double ax;
		/** The maximum x coordinate of the container. */
		const double bx;
		/** The minimum y coordinate of the container. */
		const double ay;
		/** The maximum y coordinate of the container. */
		const double by;
		/** The box length in the x direction, set to (bx-ax)/nx. */
		const double boxx;
		/** The box length in the y direction, set to (by-ay)/ny. */
		const double boxy;
		/** The inverse box length in the x direction. */
		const double xsp;
		/** The inverse box length in the y direction. */
		const double ysp;
		/** The number of boxes in the x direction. */
		const int nx;
		/** The number of boxes in the y direction. */
		const int ny;
		/** A constant, set to the value of nx multiplied by ny, which
		 * is used in the routines which step through boxes in
		 * sequence. */
		const int nxy;
		/** A boolean value that determines if the x coordinate in
		 * periodic or not. */
		const bool xperiodic;
		/** A boolean value that determines if the y coordinate in
		 * periodic or not. */
		const bool yperiodic;
		/** This array holds the number of particles within each
		 * computational box of the container. */
		int *co;
		/** This array holds the maximum amount of particle memory for
		 * each computational box of the container. If the number of
		 * particles in a particular box ever approaches this limit,
		 * more is allocated using the add_particle_memory() function.
		 */
		int *mem;
		/** This array holds the numerical IDs of each particle in each
		 * computational box. */
		int **id;
		/** A two dimensional array holding particle positions. For the
		 * derived container_poly class, this also holds particle
		 * radii. */
		double **p;
		container_2d(double xa,double xb,double ya,double yb,int xn,int yn,bool xper,bool yper,int memi);
		~container_2d();
		void import(FILE *fp=stdin);
		/** Imports a list of particles from a file.
		 * \param[in] filename the file to read from. */
		inline void import(const char *filename) {
			FILE *fp(voropp_safe_fopen(filename,"r"));
			import(fp);
			fclose(fp);
		}
		void draw_particles(FILE *fp=stdout);
		/** Dumps all the particle positions and IDs to a file.
		 * \param[in] filename the file to write to. */
		inline void draw_particles(const char *filename) {
			FILE *fp(voropp_safe_fopen(filename,"w"));
			draw_particles(fp);
			fclose(fp);
		}
		void draw_particles_pov(FILE *fp=stdout);
		/** Dumps all the particles positions in POV-Ray format.
		 * \param[in] filename the file to write to. */
		inline void draw_particles_pov(const char *filename) {
			FILE *fp(voropp_safe_fopen(filename,"w"));
			draw_particles_pov(fp);
			fclose(fp);
		}
		void draw_cells_gnuplot(FILE *fp=stdout);
		/** Computes the Voronoi cells for all particles and saves the
		 * output in gnuplot format.
		 * \param[in] filename the file to write to. */
		inline void draw_cells_gnuplot(const char *filename) {
			FILE *fp(voropp_safe_fopen(filename,"w"));
			draw_cells_gnuplot(fp);
			fclose(fp);
		}
		void draw_cells_pov(FILE *fp=stdout);
		/** Computes the Voronoi cells for all particles and saves the
		 * output in POV-Ray format.
		 * \param[in] filename the file to write to. */
		inline void draw_cells_pov(const char *filename) {
			FILE *fp(voropp_safe_fopen(filename,"w"));
			draw_cells_pov(fp);
			fclose(fp);
		}
		void print_custom(const char *format,FILE *fp=stdout);
		/** Computes the Voronoi cells for all particles in the
		 * container, and for each cell, outputs a line containing
		 * custom information about the cell structure. The output
		 * format is specified using an input string with control
		 * sequences similar to the standard C printf() routine.
		 * \param[in] format the format of the output lines, using
		 *                   control sequences to denote the different
		 *                   cell statistics.
		 * \param[in] filename the file to write to. */
		inline void print_custom(const char *format,const char *filename) {
			FILE *fp(voropp_safe_fopen(filename,"w"));
			print_custom(format,fp);
			fclose(fp);
		}
		double sum_cell_areas();
		void compute_all_cells();
		/** An overloaded version of the compute_cell_sphere routine,
		 * that sets up the x and y variables.
		 *\param[in,out] c a reference to a voronoicell object.
		 * \param[in] (i,j) the coordinates of the block that the test
		 *                  particle is in.
		 * \param[in] ij the index of the block that the test particle
		 *               is in, set to i+nx*j.
		 * \param[in] s the index of the particle within the test
		 *              block.
		 * \return False if the Voronoi cell was completely removed
		 * during the computation and has zero volume, true otherwise.
		 */
		inline bool compute_cell_sphere(voronoicell_2d &c,int i,int j,int ij,int s) {
			double x=p[ij][2*s],y=p[ij][2*s+1];
			return compute_cell_sphere(c,i,j,ij,s,x,y);
		}
		bool compute_cell_sphere(voronoicell_2d	&c,int i,int j,int ij,int s,double x,double y);
		bool initialize_voronoicell(voronoicell_2d &c,double x,double y);
		void put(int n,double x,double y);
		void clear();
	private:
		inline bool put_locate_block(int &ij,double &x,double &y);
		inline bool put_remap(int &ij,double &x,double &y);
		void add_particle_memory(int i);
		/** Custom int function, that gives consistent stepping for
		 * negative numbers. With normal int, we have
		 * (-1.5,-0.5,0.5,1.5) -> (-1,0,0,1). With this routine, we
		 * have (-1.5,-0.5,0.5,1.5) -> (-2,-1,0,1). */
		inline int step_int(double a) {return a<0?int(a)-1:int(a);}
		/** Custom modulo function, that gives consistent stepping for
		 * negative numbers. */
		inline int step_mod(int a,int b) {return a>=0?a%b:b-1-(b-1-a)%b;}
		/** Custom integer division function, that gives consistent
		 * stepping for negative numbers. */
		inline int step_div(int a,int b) {return a>=0?a/b:-1+(a+1)/b;}
		friend class voropp_loop_2d;
};

/** \brief A class to handle loops on regions of the container handling
 * non-periodic and periodic boundary conditions.
 *
 * Many of the container routines require scanning over a rectangular sub-grid
 * of blocks, and the routines for handling this are stored in the
 * voropp_loop_2d class. A voropp_loop_2d class can first be initialized to
 * either calculate the subgrid which is within a distance r of a vector
 * (vx,vy,vz), or a subgrid corresponding to a rectangular box. The routine
 * inc() can then be successively called to step through all the blocks within
 * this subgrid.
 */
class voropp_loop_2d {
	public:
		voropp_loop_2d(container_2d &con);
		int init(double vx,double vy,double r,double &px,double &py);
		int init(double xmin,double xmax,double ymin,double ymax,double &px,double &py);
		int inc(double &px,double &py);
		/** The current block index in the x direction, referencing a
		 * real cell in the range 0 to nx-1. */
		int ip;
		/** The current block index in the y direction, referencing a
		 * real cell in the range 0 to ny-1. */
		int jp;
	private:
		const double boxx,boxy,xsp,ysp,ax,ay;
		const int nx,ny,nxy;
		const bool xperiodic,yperiodic;
		double apx,apy;
		int i,j,ai,bi,aj,bj,s;
		int aip,ajp,inc1;
		/** Custom modulo function, that gives consistent stepping for
		 * negative numbers. */
		inline int step_mod(int a,int b) {return a>=0?a%b:b-1-(b-1-a)%b;}
		/** Custom integer division function, that gives consistent
		 * stepping for negative numbers. */
		inline int step_div(int a,int b) {return a>=0?a/b:-1+(a+1)/b;}
		/** Custom int function, that gives consistent stepping for
		 * negative numbers. With normal int, we have
		 * (-1.5,-0.5,0.5,1.5) -> (-1,0,0,1). With this routine, we
		 * have (-1.5,-0.5,0.5,1.5) -> (-2,-1,0,1). */
		inline int step_int(double a) {return a<0?int(a)-1:int(a);}
};

#endif
