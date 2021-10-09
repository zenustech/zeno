// Single Voronoi cell example code
//
// Author   : Chris H. Rycroft (Harvard University / LBL)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

#include "voro++.hh"
using namespace voro;

// This function returns a random floating point number between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	double x,y,rsq,r;
	voronoicell v;

	// Initialize the Voronoi cell to be a square of side length 2 in the xy-plane. Set the
	// cell to be 1 high in the z-direction.
	v.init(-1,1,-1,1,-0.5,0.5);

	// Cut the cell by 250 random planes which are all a distance 1 away
	// from the origin, to make an approximation to a sphere
	for(int i=0;i<25;i++) {
		x=2*rnd()-1;
		y=2*rnd()-1;
		rsq=x*x+y*y;
		if(rsq>0.01&&rsq<1) {
			r=1/sqrt(rsq);x*=r;y*=r;
			v.plane(x,y,0,1);
		}
	}

	// Output the Voronoi cell to a file, in the gnuplot format
	v.draw_gnuplot(0,0,0,"single_cell_2d.gnu");
}
