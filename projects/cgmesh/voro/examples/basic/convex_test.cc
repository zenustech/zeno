// Single Voronoi cell example code
//
// Author   : Chris H. Rycroft (Harvard SEAS / LBL)
// Email    : chr@alum.mit.edu
// Date     : February 16th 2014

#include "voro++.hh"
using namespace voro;

// This function returns a random floating point number between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
//	double x,y,z,rsq,r;
	voronoicell v;

	v.init_l_shape();
	v.draw_gnuplot(0,0,0,"single_cell.gnu");
	int lp=-1,ls=-1;
	double l=1e-20,u=1e-20;
	//bool suc=v.search_upward(-1,3,0,0.5,lp,ls,l,u);
	v.plane(-1,3,0,0.5);
	v.draw_gnuplot(0,0,0,"single_cell2.gnu");
	v.plane(-1,3,0.4,0.53);
	v.plane(-1,3,-0.4,0.54);
	puts("cr");
	v.check_relations();
	v.check_duplicates();
	puts("fi");
//	v.plane(-1,3,-0.2,0.54);
	bool suc=true;

	printf("%s lp=%d ls=%d l=%g u=%g up=%d\n",suc?"True":"False",lp,ls,l,u,v.up);

	v.draw_gnuplot(0,0,0,"single_cell3.gnu");
}
