// File import example code
//
// Author   : Chris H. Rycroft (Harvard University / LBL)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

#include "voro++.cc"
using namespace voro;

// Set up constants for the container geometry
const double x_min=-50000,x_max=50000;
const double y_min=-50000,y_max=50000;
const double z_min=-50000,z_max=50000;

// Set up the number of blocks that the container is divided into
const int n_x=6,n_y=6,n_z=6;

int main() {
    FILE * outputFile;
    int id;
    double x,y,z;
    voronoicell c;
    // Create a container with the geometry given above, and make it
	// non-periodic in each of the three coordinates. Allocate space for
	// eight particles within each computational block
	container con(x_min,x_max,y_min,y_max,z_min,z_max,n_x,n_y,n_z,
			false,false,false,8);

	//Randomly add particles into the container
	con.import("pointlist.txt");

	// Save the Voronoi network of all the particles to text files
	// in gnuplot and POV-Ray formats
   /*	con.draw_cells_gnuplot("pack_ten_cube.gnu");
	con.draw_cells_pov("pack_ten_cube_v.pov");

	// Output the particles in POV-Ray format
	con.draw_particles_pov("pack_ten_cube_p.pov");*/

	outputFile = fopen("MESH2","w");
	printf("STARING FIRST LOOP");
	c_loop_all cl(con);
	//FIRST LOOP START
	if(cl.start()) do if(con.compute_cell(c,cl)) {
		cl.pos(x,y,z);id=cl.pid();
		printf("%d %e %e %e\n", id,x,y,z);
		fflush(stdout);
		fprintf(outputFile,"%d %e %e %e\n", id,x,y,z);
	} while (cl.inc());
	fclose(outputFile);
}
