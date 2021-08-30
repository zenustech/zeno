// Voronoi calculation example code
//
// Author   : Chris H. Rycroft (Harvard University / LBL)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

#include "voro++.hh"
using namespace voro;

// Set up constants for the container geometry
const double x_min=-1,x_max=1;
const double y_min=-1,y_max=1;
const double z_min=-1,z_max=1;
const double cvol=(x_max-x_min)*(y_max-y_min)*(x_max-x_min);

// Set up the number of blocks that the container is divided into
const int n_x=6,n_y=6,n_z=6;

// Set the number of particles that are going to be randomly introduced
const int particles=20;

// This function returns a random double between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	int i;
	double x,y,z;

	// Create a container with the geometry given above, and make it
	// non-periodic in each of the three coordinates. Allocate space for
	// eight particles within each computational block
	container_periodic_poly con(2,0.5,2,0,0,2,n_x,n_y,n_z,8);

	// Randomly add particles into the container
	for(i=0;i<4;i++) {
		x=x_min+rnd()*(x_max-x_min);
		y=y_min+rnd()*(y_max-y_min);
		z=z_min+rnd()*(z_max-z_min);
		con.put(i,x,y,0,1);
	}

	// Output the particle positions in gnuplot format
	con.draw_particles("ghost_test_p.gnu");

	// Output the Voronoi cells in gnuplot format
	con.draw_cells_gnuplot("ghost_test_v.gnu");

	// Open file for test ghost cell
	FILE *fp=safe_fopen("ghost_test_c.gnu","w");
	voronoicell c;

	// Compute a range of ghost cells
//	for(y=-3.5;y<3.5;y+=0.05) if(con.compute_ghost_cell(c,1,y,0,1))
//		c.draw_gnuplot(1,y,0,fp);

	// Compute a single ghost cell
	if(con.compute_ghost_cell(c,1.56,0.67,0,1)) c.draw_gnuplot(1.56,0.67,0,fp);
	
	// Close ghost cell file
	fclose(fp);

	// Draw the domain
	con.draw_domain_gnuplot("ghost_test_d.gnu");
}
