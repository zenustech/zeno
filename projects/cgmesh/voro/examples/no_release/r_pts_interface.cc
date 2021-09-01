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
const double cvol=(x_max-x_min)*(y_max-y_min)*(z_max-z_min);

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
	container con(x_min,x_max,y_min,y_max,z_min,z_max,n_x,n_y,n_z,
			false,false,false,8);

	// Randomly add particles into the container
	for(i=0;i<particles;i++) {
		x=x_min+rnd()*(x_max-x_min);
		y=y_min+rnd()*(y_max-y_min);
		z=z_min+rnd()*(z_max-z_min);
		con.put(i,x,y,z);
	}

	// Use the C++ interface to do custom computations
	double cx,cy,cz,vvol=0;
	voronoicell c;
	c_loop_all cl(con);
	if(cl.start()) do if(con.compute_cell(c,cl)) {

		// Get particle position and ID
		cl.pos(x,y,z);i=cl.pid();

		// Calculate centroid
		c.centroid(cx,cy,cz);
		printf("Particle %2d at (% .3f,% .3f,% .3f), centroid at (% .3f,% .3f,% .3f)\n",
		       i,x,y,z,x+cx,y+cy,z+cz);	

		// Calculate volume and sum it
		vvol+=c.volume();

		// Do other calculations and store the information
		// ...
		// ...
		// ...

	} while(cl.inc());	

	// Print the volume check
	printf("\nContainer volume : %g\n"
	       "Voronoi volume   : %g\n"
	       "Difference       : %g\n",cvol,vvol,vvol-cvol);

	// Output the particle positions in gnuplot format
	con.draw_particles("random_points_p.gnu");

	// Output the Voronoi cells in gnuplot format
	con.draw_cells_gnuplot("random_points_v.gnu");
}
