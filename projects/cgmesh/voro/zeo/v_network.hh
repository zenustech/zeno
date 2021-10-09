#ifndef ZEOPP_V_NETWORK_HH
#define ZEOPP_V_NETWORK_HH

#include <vector>

#include "voro++.hh"
using namespace voro;

const int init_network_edge_memory=4;
const int init_network_vertex_memory=64;
const int max_network_vertex_memory=65536;

struct block {
	double dis;
	double e;
	inline void first(double v,double d) {e=v>0?v:0;dis=d;}
	inline void add(double v,double d) {
		if(v<0) e=0;
		else if(v<e) {e=v;dis=d;}
	}
	inline void print(FILE *fp) {fprintf(fp," %g %g",e,dis);}
};

class voronoi_network {
	public:
		const double bx;
		const double bxy;
		const double by;
		const double bxz;
		const double byz;
		const double bz;
		const int nx;
		const int ny;
		const int nz;
		const int nxyz;
		const double xsp,ysp,zsp;
		const double net_tol;
		double **pts;
		int **idmem;
		int *ptsc;
		int *ptsmem;
		int **ed;
		int **ne;
		block **raded;
		unsigned int **pered;
		int edc,edmem;
		int *nu;
		int *nec;
		int *numem;
		int *reg;
		int *regp;
		int *vmap;
		int map_mem;
		template<class c_class>
		voronoi_network(c_class &c,double net_tol_=tolerance);
		~voronoi_network();
		void print_network(FILE *fp=stdout,bool reverse_remove=false);
		inline void print_network(const char* filename,bool reverse_remove=false) {
			FILE *fp(safe_fopen(filename,"w"));
			print_network(fp);
			fclose(fp);
		}
		void draw_network(FILE *fp=stdout);
		inline void draw_network(const char* filename) {
			FILE *fp(safe_fopen(filename,"w"));
			draw_network(fp);
			fclose(fp);
		}
		template<class v_cell>
		inline void add_to_network(v_cell &c,int idn,double x,double y,double z,double rad,int *&cmap) {
			cmap=new int[4*c.p];
			add_to_network_internal(c,idn,x,y,z,rad,cmap);
		}
		template<class v_cell>
		inline void add_to_network_rectangular(v_cell &c,int idn,double x,double y,double z,double rad,int *&cmap) {
			cmap=new int[4*c.p];
			add_to_network_rectangular_internal(c,idn,x,y,z,rad,cmap);
		}
		template<class v_cell>
		inline void add_to_network(v_cell &c,int idn,double x,double y,double z,double rad) {
			if(c.p>map_mem) add_mapping_memory(c.p);
			add_to_network_internal(c,idn,x,y,z,rad,vmap);
		}
		template<class v_cell>
		inline void add_to_network_rectangular(v_cell &c,int idn,double x,double y,double z,double rad) {
			if(c.p>map_mem) add_mapping_memory(c.p);
			add_to_network_rectangular_internal(c,idn,x,y,z,rad,vmap);
		}

		void clear_network();
	private:
		inline int step_div(int a,int b);
		inline int step_int(double a);
		inline void add_neighbor(int k,int idn);
		void add_particular_vertex_memory(int l);
		void add_edge_network_memory();
		void add_network_memory(int l);
		void add_mapping_memory(int pmem);
		inline unsigned int pack_periodicity(int i,int j,int k);
		inline void unpack_periodicity(unsigned int pa,int &i,int &j,int &k);
		template<class v_cell>
		void add_edges_to_network(v_cell &c,double x,double y,double z,double rad,int *cmap);
		int not_already_there(int k,int j,unsigned int cper);
		bool search_previous(double gx,double gy,double x,double y,double z,int &ijk,int &q,int &ci,int &cj,int &ck);
		bool safe_search_previous_rect(double x,double y,double z,int &ijk,int &q,int &ci,int &cj,int &ck);
		bool search_previous_rect(double x,double y,double z,int &ijk,int &q,int &ci,int &cj,int &ck);
		template<class v_cell>
		void add_to_network_internal(v_cell &c,int idn,double x,double y,double z,double rad,int *cmap);
		template<class v_cell>
		void add_to_network_rectangular_internal(v_cell &c,int idn,double x,double y,double z,double rad,int *cmap);
};

#endif
