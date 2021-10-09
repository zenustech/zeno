#include "voro++_2d.hh"
using namespace voro;

int main() {
	int i;
	char buffer[64];

	// Initialize the container class to be the unit square, with
	// non-periodic boundary conditions. Divide it into a 6 by 6 grid, with
	// an initial memory allocation of 16 particles per grid square.
	container_2d con(0,1,0,1,6,6,false,false,16);

	// Import the spiral data set, and only save those particles that are
	// within the container bounds 
	con.import("particles_spiral");
	sprintf(buffer,"lloyd_output/lloyd_p.%d",0);
	con.draw_particles(buffer);

	// Carry out sixty four iterations of Lloyd's algorithm
	for(i=0;i<256;i++) {
		con.clear();
		con.import(buffer);
		sprintf(buffer,"lloyd_output/lloyd_v.%d",i);
		con.draw_cells_gnuplot(buffer);
		sprintf(buffer,"lloyd_output/lloyd_p.%d",i+1);
		con.print_custom("%i %C",buffer);
	}

	// Draw the final Voronoi cells
	con.draw_cells_gnuplot("lloyd_output/lloyd_v.256");
}
