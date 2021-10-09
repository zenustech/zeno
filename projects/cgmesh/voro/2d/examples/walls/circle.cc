#include "voro++_2d.hh"
using namespace voro;

const double pi=3.1415926535897932384626433832795;
const double radius=0.7;

// This function returns a random floating point number between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	int i;double x,y;

	// Initialize the container class to be the unit square, with
	// non-periodic boundary conditions. Divide it into a 6 by 6 grid, with
	// an initial memory allocation of 16 particles per grid square.
	container_2d con(-1,1,-1,1,10,10,false,false,16);

	// Add circular wall object
	wall_circle_2d wc(0,0,radius);
	//con.add_wall(wc);

	// Add 1000 random points to the container
	for(i=0;i<1000;i++) {
		x=2*rnd()-1;
		y=2*rnd()-1;
		if(x*x+y*y<radius*radius) con.put(i,x,y);
	}

	// Output the particle positions to a file
	con.draw_particles("circle.par");

	// Output the Voronoi cells to a file, in the gnuplot format
	con.draw_cells_gnuplot("circle.gnu");

	con.print_custom("%i %q %a %n","circle.vol");

	// Sum the Voronoi cell areas and compare to the circle area
	double carea=pi*radius*radius,varea=con.sum_cell_areas();
	printf("Total circle area       : %g\n"
	       "Total Voronoi cell area : %g\n"
	       "Difference              : %g\n",carea,varea,varea-carea);
}
