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

	double vo[400],ar[400],tvo,tar;
	for(i=0;i<400;i++) vo[i]=0;
	for(i=0;i<400;i++) ar[i]=0;

	voronoicell c;
	c_loop_all cl(con);
	if(cl.start()) do if(con.compute_cell(c,cl)) {
		for(i=0;i<400;i++) {
			c.minkowski(i*0.005,tvo,tar);
			vo[i]+=tvo;ar[i]+=tar;
		}
	} while(cl.inc());


	for(i=0;i<400;i++) printf("%g %g %g\n",i*0.005,vo[i],ar[i]);
}
