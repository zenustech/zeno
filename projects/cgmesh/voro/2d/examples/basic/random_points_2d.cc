#include "voro++_2d.hh"
using namespace voro;

// This function returns a random floating point number between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	int i;double x,y;

	// Initialize the container class to be the unit square, with
	// non-periodic boundary conditions. Divide it into a 10 by 10 grid,
	// with an initial memory allocation of 16 particles per grid square.
	container_2d con(0,1,0,1,10,10,false,false,16);
	
	// Add 1000 random points to the container
	for(i=0;i<1000;i++) {
		x=rnd();
		y=rnd();
		con.put(i,x,y);
	}

	// Output the particle positions to a file
	con.draw_particles("random_points_2d.par");

	// Output the Voronoi cells to a file, in the gnuplot format
	con.draw_cells_gnuplot("random_points_2d.gnu");
	
	// Sum the Voronoi cell areas and compare to the container area
	double carea=1,varea=con.sum_cell_areas();
	printf("Total container area    : %g\n"
	       "Total Voronoi cell area : %g\n"
	       "Difference              : %g\n",carea,varea,varea-carea);
}
