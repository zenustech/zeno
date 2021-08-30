#include "voro++_2d.hh"
#include <cmath>
using namespace voro;

// This function returns a random floating point number between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

const double tpi=8*atan(1.0);

int main() {
	int i=0;
	double x,y,t;

	// Initialize the container class to be the unit square, with
	// non-periodic boundary conditions. Divide it into a 6 by 6 grid, with
	// an initial memory allocation of 16 particles per grid square.
	container_boundary_2d con(-1,1,-1,1,6,6,false,false,8);
	
	// Create outer circle, tracing in the positive sense (i.e. anticlockwise)
	con.start_boundary();
	for(t=0.01*tpi;t<tpi;t+=0.02*tpi,i++) con.put(i,0.95*cos(t),0.95*sin(t));
	con.end_boundary();
	
	// Create inner circle. Since this is a hole, the points must trace around
	// the circle in the opposite sense (i.e. clockwise) 
	con.start_boundary();
	for(t=0;t<tpi;t+=0.02*tpi,i++) con.put(i,0.6*cos(t),0.6*sin(-t));
	con.end_boundary();

	// Add random points
	while(i<500) {
		x=-1+2*rnd();
		y=-1+2*rnd();
		if(con.point_inside(x,y)) {con.put(i,x,y);i++;}
	}

	// Output
	con.draw_boundary_gnuplot("annulus.bd");
	con.draw_particles("annulus.par");
	con.setup();
	con.draw_cells_gnuplot("annulus.gnu");

}
