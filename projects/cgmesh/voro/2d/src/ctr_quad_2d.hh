#ifndef CONTAINER_QUAD_2D_HH
#define CONTAINER_QUAD_2D_HH

#include "cell_2d.hh"
#include "config.hh"
#include "common.hh"

namespace voro {

const int qt_max=6;

class container_quad_2d;

class quadtree {
	public:
		container_quad_2d &parent;
		const double cx;
		const double cy;
		const double lx;
		const double ly;
		const int ps;
		int *id;
		double *p;
		int co;
		unsigned int mask;
		quadtree *qsw;
		quadtree *qse;
		quadtree *qnw;
		quadtree *qne;
		quadtree **nei;
		int nco;
		quadtree(double cx_,double cy_,double lx_,double ly_,container_quad_2d &parent_);
		~quadtree();
		void put(int i,double x,double y);
		void split();
		void draw_particles(FILE *fp=stdout);
		void draw_cross(FILE *fp=stdout);
		void setup_neighbors();
		void draw_neighbors(FILE *fp=stdout);
		void draw_cells_gnuplot(FILE *fp=stdout);
		inline void quick_put(int i,double x,double y) {
			id[co]=i;
			p[ps*co]=x;
			p[1+ps*co++]=y;
		}
		inline void add_neighbor(quadtree *qt) {
			if(nco==nmax) add_neighbor_memory();
			nei[nco++]=qt;
		}
		inline void bound(double &xlo,double &xhi,double &ylo,double &yhi) {
			xlo=cx-lx;xhi=cx+lx;
			ylo=cy-ly;yhi=cy+ly;
		}
		double sum_cell_areas();
		void compute_all_cells();
		bool compute_cell(voronoicell_2d &c,int j);
		void reset_mask();
	protected:
		int nmax;
		inline bool corner_test(voronoicell_2d &c,double xl,double yl,double xh,double yh);
		inline bool edge_x_test(voronoicell_2d &c,double xl,double y0,double y1);
		inline bool edge_y_test(voronoicell_2d &c,double x0,double yl,double x1);
		void we_neighbors(quadtree *qw,quadtree *qe);
		void ns_neighbors(quadtree *qs,quadtree *qn);
		void add_neighbor_memory();
};

class container_quad_2d : public quadtree {
	public:
		using quadtree::draw_particles;
		using quadtree::draw_neighbors;
		using quadtree::draw_cells_gnuplot;
		/** The minimum x coordinate of the container. */
		const double ax;
		/** The maximum x coordinate of the container. */
		const double bx;
		/** The minimum y coordinate of the container. */
		const double ay;
		/** The maximum y coordinate of the container. */
		const double by;
		unsigned int bmask;
		container_quad_2d(double ax_,double bx_,double ay_,double by_);
		inline void draw_particles(const char* filename) {
			FILE *fp=safe_fopen(filename,"w");
			draw_particles(fp);
			fclose(fp);
		}
		inline void draw_neighbors(const char* filename) {
			FILE *fp=safe_fopen(filename,"w");
			draw_neighbors(fp);
			fclose(fp);
		}
		inline void draw_cells_gnuplot(const char* filename) {
			FILE *fp=safe_fopen(filename,"w");
			draw_cells_gnuplot(fp);
			fclose(fp);
		}
		void draw_quadtree(FILE *fp=stdout);
		inline void draw_quadtree(const char* filename) {
			FILE *fp=safe_fopen(filename,"w");
			draw_quadtree(fp);
			fclose(fp);
		}
		inline void initialize_voronoicell(voronoicell_2d &c,double x,double y) {
			c.init(ax-x,bx-x,ay-y,by-y);
		}
};

}

#endif
