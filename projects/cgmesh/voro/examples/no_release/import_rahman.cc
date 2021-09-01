// Voronoi method to generate nanocrystalline grain boundaries
// Oct 18, 2011
#include "voro++.hh"
using namespace voro;

// Box geometry
const double x_min=-10,x_max=10;
const double y_min=-10,y_max=10;
const double z_min=-10,z_max=10;
const double cvol=(x_max-x_min)*(y_max-y_min)*(x_max-x_min);

// Number of blocks that the Box is divided into
const int n_x=5,n_y=5,n_z=4;

// Total no of particles

const int particles=10000;

// Function for random double between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	int i;
	double x,y,z;
    	
	// Creating Box and allcating 100 particles within each block
	
	container con(x_min,x_max,y_min,y_max,z_min,z_max,n_x,n_y,n_z,
			false,false,false,100);

	// Set up particle order class
	particle_order po;

	// Add particles into the Box	
	for(i=1;i<particles;i++) {
		x=x_min+rnd()*(x_max-x_min);
		y=y_min+rnd()*(y_max-y_min);
		z=z_min+rnd()*(z_max-z_min);
		con.put(po,i,x,y,z);
	}

	// Setup an ordered loop class
	c_loop_order cl(con,po);

	// Customize output for LAMMPS, preserving ordering
	FILE *fp=safe_fopen("lammps_input","w");
	con.print_custom(cl,"%i 1 %x %y %z",fp);
	fclose(fp);
}
