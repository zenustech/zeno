#include "voro++.hh"
using namespace voro;

// Set up constants for the container geometry
const double bx=10;
const double by=10;
const double bz=10;
const double bxy=0;
const double bxz=5;
const double byz=0;

// Set up the number of blocks that the container is divided
// into
const int n_x=3,n_y=3,n_z=3;

// Set the number of particles to add
const int particles=20;

// This function returns a random double between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	int i;
	double x,y,z;

	// Create a container with the geometry given above, and make it
	// non-periodic in each of the three coordinates. Allocate space for
	// eight particles within each computational block.	
        container_periodic con(bx,bxy,by,bxz,byz,bz,n_x,n_y,n_z,8);

	// Add particles into the container at random positions
	for(i=0;i<particles;i++) {
		x=bx*rnd();
		y=by*rnd();
		z=bz*rnd();
		con.put(i,x,y,z);
	}


        // Output volume
        double vvol=con.sum_cell_volumes();
        printf("Container volume : %g\n"
	       "Voronoi volume   : %g\n",bx*by*bz,vvol);

        // Output particle positions, Voronoi cells, and the domain
        con.draw_particles("particles_periodic.gnu");
        con.draw_cells_gnuplot("cells_periodic.gnu");
	con.draw_domain_gnuplot("domain_periodic.gnu");

}
