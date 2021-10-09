// Voro++, a 3D cell-based Voronoi library
//
// Author   : Chris H. Rycroft (LBL / UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

/** \file c_loops_2d.hh
 * \brief Header file for the 2D loop classes. */

#ifndef VOROPP_C_LOOPS_2D_HH
#define VOROPP_C_LOOPS_2D_HH

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
using namespace std;

#include "config.hh"

namespace voro {

/** A type associated with a c_loop_subset_2d class, determining what type of
 * geometrical region to loop over. */
enum c_loop_subset_mode_2d {
	circle,
	rectangle,
	no_check
};

/** \brief A class for storing ordering information when particles are added to
 * a container.
 *
 * When particles are added to a container class, they are sorted into an
 * internal computational grid of blocks. The particle_order class provides a
 * mechanism for remembering which block particles were sorted into. The import
 * and put routines in the container class have variants that also take a
 * particle_order class. Each time they are called, they will store the block
 * that the particle was sorted into, plus the position of the particle within
 * the block. The particle_order class can used by the c_loop_order class to
 * specifically loop over the particles that have their information stored
 * within it. */
class particle_order {
	public:
		/** A pointer to the array holding the ordering. */
		int *o;
		/** A pointer to the next position in the ordering array in
		 * which to store an entry. */
		int *op;
		/** The current memory allocation for the class, set to the
		 * number of entries which can be stored. */
		int size;
		/** The particle_order constructor allocates memory to store the
		 * ordering information.
		 * \param[in] init_size the initial amount of memory to
		 *                      allocate. */
		particle_order(int init_size=init_ordering_size)
			: o(new int[init_size<<1]),op(o),size(init_size) {}
		/** The particle_order destructor frees the dynamically allocated
		 * memory used to store the ordering information. */
		~particle_order() {
			delete [] o;
		}
		/** Adds a record to the order, corresponding to the memory
		 * address of where a particle was placed into the container.
		 * \param[in] ijk the block into which the particle was placed.
		 * \param[in] q the position within the block where the
		 * 		particle was placed. */
		inline void add(int ijk,int q) {
			if(op==o+size) add_ordering_memory();
			*(op++)=ijk;*(op++)=q;
		}
	private:
		void add_ordering_memory();
};

/** \brief Base class for looping over particles in a container.
 *
 * This class forms the base of all classes that can loop over a subset of
 * particles in a contaner in some order. When initialized, it stores constants
 * about the corresponding container geometry. It also contains a number of
 * routines for interrogating which particle currently being considered by the
 * loop, which are common between all of the derived classes. */
class c_loop_base_2d {
	public:
		/** The number of blocks in the x direction. */
		const int nx;
		/** The number of blocks in the y direction. */
		const int ny;
		/** A constant, set to the value of nx multiplied by ny, which
		 * is used in the routines that step through blocks in
		 * sequence. */
		const int nxy;
		/** The number of floating point numbers per particle in the
		 * associated container data structure. */
		const int ps;
		/** A pointer to the particle position information in the
		 * associated container data structure. */
		double **p;
		/** A pointer to the particle ID information in the associated
		 * container data structure. */
		int **id;
		/** A pointer to the particle counts in the associated
		 * container data structure. */
		int *co;
		/** The current x-index of the block under consideration by the
		 * loop. */
		int i;
		/** The current y-index of the block under consideration by the
		 * loop. */
		int j;
		/** The current index of the block under consideration by the
		 * loop. */
		int ij;
		/** The index of the particle under consideration within the current
		 * block. */
		int q;
		/** The constructor copies several necessary constants from the
		 * base container class.
		 * \param[in] con the container class to use. */
		template<class c_class_2d>
		c_loop_base_2d(c_class_2d &con) : nx(con.nx), ny(con.ny), nxy(con.nxy),
						  ps(con.ps), p(con.p), id(con.id),
						  co(con.co) {}
		/** Returns the position vector of the particle currently being
		 * considered by the loop.
		 * \param[out] (x,y) the position vector of the particle. */
		inline void pos(double &x,double &y) {
			double *pp=p[ij]+ps*q;
			x=*(pp++);y=*pp;
		}
		/** Returns the ID, position vector, and radius of the particle
		 * currently being considered by the loop.
		 * \param[out] pid the particle ID.
		 * \param[out] (x,y) the position vector of the particle.
		 * \param[out] r the radius of the particle. If no radius
		 * 		 information is available the default radius
		 * 		 value is returned. */
		inline void pos(int &pid,double &x,double &y,double &r) {
			pid=id[ij][q];
			double *pp=p[ij]+ps*q;
			x=*(pp++);y=*pp;
			r=ps==2?default_radius_2d:*(++pp);
		}
		/** Returns the x position of the particle currently being
		 * considered by the loop. */
		inline double x() {return p[ij][ps*q];}
		/** Returns the y position of the particle currently being
		 * considered by the loop. */
		inline double y() {return p[ij][ps*q+1];}
		/** Returns the ID of the particle currently being considered
		 * by the loop. */
		inline int pid() {return id[ij][q];}
};

/** \brief Class for looping over all of the particles in a container.
 *
 * This is one of the simplest loop classes, that scans the computational
 * blocks in order, and scans all the particles within each block in order. */
class c_loop_all_2d : public c_loop_base_2d {
	public:
		/** The constructor copies several necessary constants from the
		 * base container class.
		 * \param[in] con the container class to use. */
		template<class c_class_2d>
		c_loop_all_2d(c_class_2d &con) : c_loop_base_2d(con) {}
		/** Sets the class to consider the first particle.
		 * \return True if there is any particle to consider, false
		 * otherwise. */
		inline bool start() {
			i=j=ij=q=0;
			while(co[ij]==0) if(!next_block()) return false;
			return true;
		}
		/** Finds the next particle to test.
		 * \return True if there is another particle, false if no more
		 * particles are available. */
		inline bool inc() {
			q++;
			if(q>=co[ij]) {
				q=0;
				do {
					if(!next_block()) return false;
				} while(co[ij]==0);
			}
			return true;
		}
	private:
		/** Updates the internal variables to find the next
		 * computational block with any particles.
		 * \return True if another block is found, false if there are
		 * no more blocks. */
		inline bool next_block() {
			ij++;
			i++;
			if(i==nx) {
				i=0;j++;
				if(j==ny) return false;
			}
			return true;
		}
};

/** \brief Class for looping over a subset of particles in a container.
 *
 * This class can loop over a subset of particles in a certain geometrical
 * region within the container. The class can be set up to loop over a
 * rectangle or circle. It can also rectangular group of internal computational
 * blocks. */
class c_loop_subset_2d : public c_loop_base_2d {
	public:
		/** The current mode of operation, determining whether tests
		 * should be applied to particles to ensure they are within a
		 * certain geometrical object. */
		c_loop_subset_mode_2d mode;
		/** The constructor copies several necessary constants from the
		 * base container class.
		 * \param[in] con the container class to use. */
		template<class c_class_2d>
		c_loop_subset_2d(c_class_2d &con) : c_loop_base_2d(con), ax(con.ax), ay(con.ay),
			sx(con.bx-ax), sy(con.by-ay), xsp(con.xsp), ysp(con.ysp),
			xperiodic(con.xperiodic), yperiodic(con.yperiodic) {}
		void setup_circle(double vx,double vy,double r,bool bounds_test=true);
		void setup_box(double xmin,double xmax,double ymin,double ymax,bool bounds_test=true);
		void setup_intbox(int ai_,int bi_,int aj_,int bj_);
		bool start();
		/** Finds the next particle to test.
		 * \return True if there is another particle, false if no more
		 * particles are available. */
		inline bool inc() {
			do {
				q++;
				while(q>=co[ij]) {q=0;if(!next_block()) return false;}
			} while(mode!=no_check&&out_of_bounds());
			return true;
		}
	private:
		const double ax,ay,sx,sy,xsp,ysp;
		const bool xperiodic,yperiodic;
		double px,py,apx,apy;
		double v0,v1,v2,v3;
		int ai,bi,aj,bj;
		int ci,cj,di,dj,inc1;
		inline int step_mod(int a,int b) {return a>=0?a%b:b-1-(b-1-a)%b;}
		inline int step_div(int a,int b) {return a>=0?a/b:-1+(a+1)/b;}
		inline int step_int(double a) {return a<0?int(a)-1:int(a);}
		void setup_common();
		bool next_block();
		bool out_of_bounds();
};

/** \brief Class for looping over all of the particles specified in a
 * pre-assembled particle_order class.
 *
 * The particle_order class can be used to create a specific order of particles
 * within the container. This class can then loop over these particles in this
 * order. The class is particularly useful in cases where the ordering of the
 * output must match the ordering of particles as they were inserted into the
 * container. */
class c_loop_order_2d : public c_loop_base_2d {
	public:
		/** A reference to the ordering class to use. */
		particle_order &vo;
		/** A pointer to the current position in the ordering class. */
		int *cp;
		/** A pointer to the end position in the ordering class. */
		int *op;
		/** The constructor copies several necessary constants from the
		 * base class, and sets up a reference to the ordering class to
		 * use.
		 * \param[in] con the container class to use.
		 * \param[in] vo_ the ordering class to use. */
		template<class c_class_2d>
		c_loop_order_2d(c_class_2d &con,particle_order &vo_)
		: c_loop_base_2d(con), vo(vo_), nx(con.nx) {}
		/** Sets the class to consider the first particle.
		 * \return True if there is any particle to consider, false
		 * otherwise. */
		inline bool start() {
			cp=vo.o;op=vo.op;
			if(cp!=op) {
				ij=*(cp++);decode();
				q=*(cp++);
				return true;
			} else return false;
		}
		/** Finds the next particle to test.
		 * \return True if there is another particle, false if no more
		 * particles are available. */
		inline bool inc() {
			if(cp==op) return false;
			ij=*(cp++);decode();
			q=*(cp++);
			return true;
		}
	private:
		/** The number of computational blocks in the x direction. */
		const int nx;
		/** Takes the current block index and computes indices in the
		 * x, y directions. */
		inline void decode() {
			j=ij/nx;
			i=ij-j*nx;
		}
};

}

#endif
