// Voro++, a cell-based Voronoi library
//
// Authors  : Chris H. Rycroft (LBL / UC Berkeley)
//            Cody Robert Dance (UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

/** \file cell_nc_2d.hh
 * \brief Header file for the non-convex 2D Voronoi classes. */

#ifndef VORO_CELL_NC_2D_HH
#define VORO_CELL_NC_2D_HH

#include "cell_2d.hh"

namespace voro {

class voronoicell_nonconvex_base_2d : public voronoicell_base_2d {
	public:
		using voronoicell_base_2d::nplane;
		bool exclude;
		bool nonconvex;
		template<class vc_class>
		inline bool nplane_base(vc_class &vc,double x,double y,double rs,int p_id) {
			return exclude?(nonconvex?
				(-x*reg[3]+y*reg[2]<-tolerance&&x*reg[5]-y*reg[4]<-tolerance)||nplane_nonconvex(vc,x,y,rs,p_id):
				-x*reg[3]+y*reg[2]<-tolerance||x*reg[5]-y*reg[4]<-tolerance||nplane(vc,x,y,rs,p_id)):
				nplane(vc,x,y,rs,p_id);
		}
		template<class vc_class>
		bool nplane_nonconvex(vc_class &vc,double x,double y,double rs,int p_id);
		void init_nonconvex_base(double xmin,double xmax,double ymin,double ymax,double wx0,double wy0,double wx1,double wy1);
	private:
		double reg[6];
		inline int face(double xmin,double xmax,double ymin,double ymax,double &wx,double &wy);
};

class voronoicell_nonconvex_2d : public voronoicell_nonconvex_base_2d {
	public:
		inline bool nplane(double x,double y,double rs,int p_id) {
			return nplane_base(*this,x,y,rs,0);
		}
		inline bool nplane(double x,double y,int p_id) {
			double rs=x*x+y*y;
			return nplane(x,y,rs,0);
		}
		inline bool plane(double x,double y,double rs) {
			return nplane(x,y,rs,0);
		}
		inline bool plane(double x,double y) {
			double rs=x*x+y*y;
			return nplane(x,y,rs,0);
		}
		inline void init(double xmin,double xmax,double ymin,double ymax) {
			nonconvex=exclude=false;
			init_base(xmin,xmax,ymin,ymax);
		}
		inline void init_nonconvex(double xmin,double xmax,double ymin,double ymax,double wx0,double wy0,double wx1,double wy1) {
			init_nonconvex_base(xmin,xmax,ymin,ymax,wx0,wy0,wx1,wy1);
		}
	private:
		inline void n_add_memory_vertices() {}
		inline void n_copy(int a,int b) {}
		inline void n_set(int a,int id) {}
		friend class voronoicell_base_2d;
};

class voronoicell_nonconvex_neighbor_2d : public voronoicell_nonconvex_base_2d {
	public:
		int *ne;
		voronoicell_nonconvex_neighbor_2d() : ne(new int[init_vertices]) {}
		~voronoicell_nonconvex_neighbor_2d() {delete [] ne;}
		inline bool nplane(double x,double y,double rs,int p_id) {
			return nplane_base(*this,x,y,rs,p_id);
		}
		inline bool nplane(double x,double y,int p_id) {
			double rs=x*x+y*y;
			return nplane(x,y,rs,p_id);
		}
		inline bool plane(double x,double y,double rs) {
			return nplane(x,y,rs,0);
		}
		inline bool plane(double x,double y) {
			double rs=x*x+y*y;
			return nplane(x,y,rs,0);
		}
		void init(double xmin,double xmax,double ymin,double ymax);
		void init_nonconvex(double xmin,double xmax,double ymin,double ymax,double wx0,double wy0,double wx1,double wy1);
		virtual void neighbors(vector<int> &v);
	private:
		inline void n_add_memory_vertices() {
			int *nne=new int[current_vertices],
			    *nee=ne+(current_vertices>>1),
			    *nep=ne,*nnep=nne;
			while(nep<nee) *(nnep++)=*(nep++);
			delete [] ne;ne=nne;
		}
		inline void n_copy(int a,int b) {ne[a]=ne[b];}
		inline void n_set(int a,int id) {ne[a]=id;}
		friend class voronoicell_base_2d;
};

}
#endif
