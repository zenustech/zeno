// Voronoi calculation example code
//
// Author   : Chris H. Rycroft (Harvard University / LBL)
// Email    : chr@alum.mit.edu
// Date     : April 14th 2013

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
const int particles=40;

// This function returns a random double between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	int i;
	double x,y,z;
	voronoicell_neighbor c;
	std::vector<int> neigh;

	// Create a container with the geometry given above, and make it
	// non-periodic in each of the three coordinates. Allocate space for
	// eight particles within each computational block
	container con(x_min,x_max,y_min,y_max,z_min,z_max,n_x,n_y,n_z,
			true,true,true,8);

	// Randomly add particles into the container
	for(i=0;i<particles;i++) {
		x=x_min+rnd()*(x_max-x_min);
		y=y_min+rnd()*(y_max-y_min);
		z=z_min+rnd()*(z_max-z_min);
		con.put(i,x,y,z);
	}

	// Output the particle positions in gnuplot format
	con.draw_particles("random_points_p.gnu");

	// Output the Voronoi cells in gnuplot format
	con.draw_cells_gnuplot("random_points_v.gnu");

	// Loop over all of the particles and compute their Voronoi cells
	c_loop_all cl(con);
	if(cl.start()) do if(con.compute_cell(c,cl)) {
		cl.pos(x,y,z);i=cl.pid();
		c.neighbors(neigh);

		// Print out the information about neighbors
		printf("Particle %2d at (% 1.3f,% 1.3f,% 1.3f):",i,x,y,z);
		for(unsigned int j=0;j<neigh.size();j++) printf(" %d",neigh[j]);
		puts("");
	} while (cl.inc());
}
