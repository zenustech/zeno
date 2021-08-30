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
	
	// Add 1000 random points to the container
	con.start_boundary();
	con.put(0,-0.8,-0.8);
	con.put(1,0.8,-0.8);
	con.put(2,0.8,0.8);
	con.put(3,0.1,0.75);
	con.put(4,0,-0.3);
	con.put(5,-0.1,0.95);
	con.put(6,-0.8,0.8);
	con.put(7,-0.799,-0.6);
	con.end_boundary();
	
	for(i=0;i<100;i++) {
		x=-1+2*rnd();
		y=-1+2*rnd();
		if(con.point_inside(x,y)) con.put(i+8,x,y);
	}

	con.draw_boundary_gnuplot("container_bd.gnu");
	con.draw_particles("container_bd.par");

	con.setup();
	con.draw_cells_gnuplot("container_bd_v.gnu");

	// Sum the Voronoi cell areas and compare to the container area
//	double carea=1,varea=con.sum_cell_areas();
//	printf("Total container area    : %g\n"
//	       "Total Voronoi cell area : %g\n"
//	       "Difference              : %g\n",carea,varea,varea-carea);
}
