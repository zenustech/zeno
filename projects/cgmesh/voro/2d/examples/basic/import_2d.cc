#include "voro++_2d.hh"
using namespace voro;

int main() {

	// Initialize the container class to be the unit square, with
	// non-periodic boundary conditions. Divide it into a 6 by 6 grid, with
	// an initial memory allocation of 16 particles per grid square.
	container_2d con(0,1,0,1,6,6,false,false,16);

	// Import the spiral data set
	con.import("particles_spiral");

	// Do a custom computation on the Voronoi cells, printing the IDs,
	// positions, and Voronoi cell areas to a file
	con.print_custom("%i %x %y %a","particles_spiral.out");
}
