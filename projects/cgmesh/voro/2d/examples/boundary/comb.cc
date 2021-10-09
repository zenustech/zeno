#include "voro++_2d.hh"
using namespace voro;

// This function returns a random floating point number between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	int i;
	double x,y;

	// Initialize the container class to be the unit square, with
	// non-periodic boundary conditions. Divide it into a 6 by 6 grid, with
	// an initial memory allocation of 16 particles per grid square.
	container_boundary_2d con(-1,1,-1,1,6,6,false,false,8);
	
	// Create comb-like domain
	con.start_boundary();
	con.put(0,-0.9,-0.9);
	con.put(1,0.9,-0.9);
	con.put(2,0.9,0.9);
	i=3;
	for(x=0.8;x>-0.9;x-=0.2) {
		con.put(i,x,-0.7);i++;
		con.put(i,x-0.1,0.9);i++;
	}
	con.end_boundary();

	// Add random points
	while(i<200) {
		x=-1+2*rnd();
		y=-1+2*rnd();
		if(con.point_inside(x,y)) {con.put(i,x,y);i++;}
	}

	con.draw_boundary_gnuplot("comb.bd");
	con.draw_particles("comb.par");

	con.setup();
	con.draw_cells_gnuplot("comb.gnu");

	// Sum the Voronoi cell areas and compare to the container area
//	double carea=1,varea=con.sum_cell_areas();
//	printf("Total container area    : %g\n"
//	       "Total Voronoi cell area : %g\n"
//	       "Difference              : %g\n",carea,varea,varea-carea);
}
