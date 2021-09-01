#ifndef VOROPP_V_CONNECT_HH
#define VOROPP_V_CONNECT_HH
//To be used with voro++ to get full connectivity information
using namespace std;
#include "voro++_2d.hh"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
namespace voro{

class v_connect{
	public:
		double minx,maxx,miny,maxy;
		int nx,ny;
		vector<int> vid;
		vector<double> vpos;
		vector<char> vbd;
		int bd;
		//# of generators
		int ng;
		//middle generator id
		int mid;
		//current size of vertl array/2
		int current_vertices;
		//number of vertices
		int nv;
		//for all i>=degenerate_vertices, the ith vertex is degenerate
		int degenerate_vertices;
		//current size of ed_to_vert array
		int current_edges;
		//number of edges
		int ne;
		//maps a generator's id to the place it was recieved in the input file
		//i.e. if the first particle in the input file has id 20, then mp[20]=0;
		int *mp;
		//vertl[2*i]=x coordinate of ith vertex
		double *vertl;
		//vert_to_gen[i]= vector containing list of generators that ith vertex is touching
		vector<int> *vert_to_gen;
		//vert_to_ed[i]= vector containing list of edges that the i'th vertex is a member of
		vector<int> *vert_to_ed;
		//vert_on_bd[i].size()==0 if vertex i is not on boundary. if vertex i is on boundary it is a size 2 vector of the generators that define the part of the boundary it is on
		vector<int> *vert_on_bd;
		//gen_to_vert[i]= list of vertices that ith generatos is touching in cc order
		vector<int> *gen_to_vert;
		//gen_to_edge[i]= list of edges that the ith generator is touching in cc order (- edge number means ~edge number with reverse orientation
		vector<int> *gen_to_ed;
		//gen_to_gen_e[i]= list of neighbors of ith generator through edges in cc order
		vector<int> *gen_to_gen_e;
		//gen_to_gen_v[i]= list of neighbors of ith generator through degenerate vertices in cc order
		vector<int> *gen_to_gen_v;
		//ed_to_vert[2*i(+1)]=the vertex with the lower(higher) id constituting edge i
		int *ed_to_vert;
		//ed_to_gen[2*i(+1)]=the generator with the left-hand(right-hand) orientation touching the edge
		vector<int> *ed_to_gen;
		//ed_on_bd[i].size()==0 if edge i is not on the boundary. if it is, ed_on_bd[i] is a 2 element list of the generators that define that part of the boundary that it is on
		vector<int> *ed_on_bd;
		//vertex_is_generator[i]=(-1 if ith vertex is not a generator)(j if ith vertex is generator j)
		int *vertex_is_generator;
		//see above
		int *generator_is_vertex;

		v_connect(){
			bd=-1;
			nv=0;
			ne=0;
			//ng initialized during import routine.
			minx=large_number;
			maxx=-minx;
			miny=minx;
			maxy=maxx;
			current_vertices=init_vertices;
			current_edges=init_vertices;
			vertl=new double[2*current_vertices];
			vert_to_gen= new vector<int>[current_vertices];
			vertex_is_generator=new int[current_vertices];
			for(int i=0;i<current_vertices;i++) vertex_is_generator[i]=-1;
		}

		~v_connect(){
			delete[] vertl;
			delete[] vert_to_gen;
			delete[] vert_to_ed;
			delete[] vert_on_bd;
			delete[] gen_to_vert;
			delete[] gen_to_ed;
			delete[] gen_to_gen_e;
			delete[] gen_to_gen_v;
			delete[] ed_to_vert;
			delete[] ed_to_gen;
			delete[] ed_on_bd;
			delete[] mp;
		}

		void import(FILE *fp=stdin);

		void import(const char* filename) {
			FILE *fp=safe_fopen(filename, "r");
			import(fp);
			fclose(fp);
		}

		vector<int> groom_vertexg_help(double x, double y,double vx, double vy, vector<int> &g);
		vector<int> groom_vertexg_help2(double x,double y,double vx,double vy,vector<int> &g);
		inline void groom_vertexg(voronoicell_nonconvex_neighbor_2d &c){
			double x=vpos[2*mp[c.my_id]],y=vpos[2*mp[c.my_id]+1];
			for(int i=0;i<c.p;i++){
				c.vertexg[i]=groom_vertexg_help(x ,y,(c.pts[2*i]*.5)+x,(c.pts[2*i+1]*.5)+y,c.vertexg[i]);
			}
		}
		inline void add_memory_vector(vector<int>* &old,int size){
			vector<int> *newv=new vector<int>[size];
			for(int i=0;i<(size>>1);i++){
				newv[i]=old[i];
			}
			delete [] old;
			old=newv;
		}
		inline void add_memory_array(double* &old,int size){
			double *newa= new double[2*size];
			for(int i=0;i<(size);i++){
				newa[i]=old[i];
			}
			delete [] old;
			old=newa;
		}
		inline void add_memory_array(int* &old,int size){
			int *newa=new int[2*size];
			for(int i=0;i<size;i++){
				newa[i]=old[i];
			}
			delete [] old;
			old=newa;
		}
		inline void add_memory_table(unsigned int* &old,int size){
			unsigned int* newt=new unsigned int[((size*size)/32)+1];
			for(int i=0;i<((((size>>1)*(size>>1))/32)+1);i++){
				newt[i]=old[i];
			}
			delete [] old;
			old=newt;
		}
		//return true if vector contains the two elements
		inline bool contains_two(vector<int> &a,int b, int c){
			int i=0,j=0;
			for(int k=0;k<a.size();k++){
				if(a[k]==b){
					i=1;
					break;
				}
			}for(int k=0;k<a.size();k++){
				if(a[k]==c){
					j=1;
					break;
				}
			}
			if(i==1 && j==1) return true;
			else return false;
		}
		//returns true if a vector contains the element
		inline bool contains(vector<int> &a,int b){
			for(int i=0;i<a.size();i++){
				if(a[i]==b) return true;
			}
			return false;
		}
		//given a three element vector, returns an element != b or c
		inline int not_these_two(vector<int> &a,int b, int c){
			int d=-1;
			for(int i=0;i<a.size();i++){
				if(a[i]!=b && a[i]!=c){
					d=a[i];
					break;
				}
			}
			return d;
		}
		//returns two elements the vectors have in common in g1,g2
		inline void two_in_common(vector<int> a,vector<int> b,int &g1,int &g2){
			g1=g2=-1;
			for(int i=0;i<a.size();i++){
				for(int k=0;k<b.size();k++){
					if(a[i]==b[k]){
						if(g1==-1) g1=a[i];
						else if(a[i]==g1) continue;
						else g2=a[i];
					}
				}
			}
		}
		//if a and b share an element, returns it in g1, if not returns -1
		inline void one_in_common(vector<int> a,vector<int> b,int &g1){
			g1=-1;
			for(int i=0;i<a.size();i++){
				for(int j=0;j<b.size();j++){
					if(a[i]==b[j]){
						g1=a[i];
						return;
					}
				}
			}
		}
		//returns true iff a and b contain the same elements
		inline bool contain_same_elements(vector<int> a,vector<int> b){
			for(int i=0;i<a.size();i++){
				if(!contains(b,a[i])) return false;
			}
			for(int i=0;i<b.size();i++){
				if(!contains(a,b[i])) return false;
			}
			return true;
		}
		//returns any element in a !=b
		inline int not_this_one(vector<int> a, int b){
			for(int i=0;i<a.size();i++){
				if(a[i]!=b) return a[i];
			}
			return -1;
		}
		//returns true if the elements of a are a subset of elements of b
		inline bool subset(vector<int> a,vector<int> b){
			for(int i=0;i<a.size();i++){
				if(!contains(b,a[i])) return false;
			}
			return true;
		}
		inline double cross_product(double x1,double y1,double x2,double y2){
			return ((x1*y2)-(y1*x2));
		}
		inline double dot_product(double x1, double y1, double x2, double y2){
			return ((x1*x2)+(y1*y2));
		}
		void arrange_cc_x_to_gen(vector<int> &list,double cx,double cy);
		void arrange_cc_gen_to_vert(vector<int> &list,double cx,double cy);
		void arrange_cc_gen_to_ed(vector<int> &list);
		void arrange_cc_vert_to_ed(vector<int> &list,double cx,double cy,int id);
		void assemble_vertex();
		void assemble_gen_ed();
		void assemble_boundary();
		void draw_gnu(FILE *fp=stdout);
		inline void draw_gnu(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			draw_gnu(fp);
			fclose(fp);
		}
		void draw_vtg_gnu(FILE *fp=stdout);
		inline void draw_vtg_gnu(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			draw_vtg_gnu(fp);
			fclose(fp);
		}
		void draw_gen_gen(FILE *fp=stdout);
		inline void draw_gen_gen(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			draw_gen_gen(fp);
			fclose(fp);
		}
		void label_vertices(FILE *fp=stdout);
		inline void label_vertices(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			label_vertices(fp);
			fclose(fp);
		}
		void label_generators(FILE *fp=stdout);
		inline void label_generators(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			label_generators(fp);
			fclose(fp);
		}
		void label_edges(FILE *fp=stdout);
		inline void label_edges(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			label_edges(fp);
			fclose(fp);
		}
		void label_centroids(FILE *fp=stdout);
		inline void label_centroids(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			label_centroids(fp);
			fclose(fp);
		}
		void print_gen_to_ed_table(FILE *fp=stdout);
		inline void print_gen_to_ed_table(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			print_gen_to_ed_table(fp);
			fclose(fp);
		}
		void print_gen_to_vert_table(FILE *fp=stdout);
		inline void print_gen_to_vert_table(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			print_gen_to_vert_table(fp);
			fclose(fp);
		}
		void print_vert_to_gen_table(FILE *fp=stdout);
		inline void print_vert_to_gen_table(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			print_vert_to_gen_table(fp);
			fclose(fp);
		}
		void print_ed_to_gen_table(FILE *fp=stdout);
		inline void print_ed_to_gen_table(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			print_ed_to_gen_table(fp);
			fclose(fp);
		}
		void print_vert_to_ed_table(FILE *fp=stdout);
		inline void print_vert_to_ed_table(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			print_vert_to_ed_table(fp);
			fclose(fp);
		}
		void print_vert_boundary(FILE *fp=stdout);
		inline void print_vert_boundary(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			print_vert_boundary(fp);
			fclose(fp);
		}
		void print_ed_boundary(FILE *fp=stdout);
		inline void print_ed_boundary(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			print_ed_boundary(fp);
			fclose(fp);
		}
		void add_memory_vertices();
		void ascii_output(FILE *fp=stdout);
		inline void ascii_output(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			ascii_output(fp);
			fclose(fp);
		}
		double signed_area(int g);
		void centroid(int g,double &x,double &y);
		void lloyds(double epsilon);
		void draw_median_mesh(FILE *fp=stdout);
		inline void draw_median_mesh(const char *filename){
			FILE *fp=safe_fopen(filename,"w");
			draw_median_mesh(fp);
			fclose(fp);
		}
		void draw_closest_generator(FILE *fp,double x,double y);
		inline void draw_closest_generator(const char *filename,double x,double y){
			FILE *fp=safe_fopen(filename,"w");
			draw_closest_generator(fp,x,y);
			fclose(fp);
		}
};

}
#endif
