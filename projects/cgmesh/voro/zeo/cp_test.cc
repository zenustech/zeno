// Parallelepiped calculation example code
//
// Author   : Chris H. Rycroft (LBL / UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

#include "voro++.hh"
using namespace voro;

// Set up 
const double bx=4;
const double bxy=0,by=2;
const double bxz=1,byz=0,bz=3;

// Set up the number of blocks that the container is divided into
const int nx=5,ny=5,nz=5;

// Set the number of particles that are going to be randomly introduced
const int particles=20;

// This function returns a random double between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	int i;
	double x,y,z;

	// Create a parallelepiped with  
	container_periodic con(bx,bxy,by,bxz,byz,bz,nx,ny,nz,8);

	// Randomly add particles into the container
	for(i=0;i<particles;i++) {
		x=rnd()*bx;
		y=rnd()*by;
		z=rnd()*bz;
		con.put(i,x,y,z);
	}

	// Output the particle positions in gnuplot format
	con.draw_particles("cp_test_p.gnu");

	// Output the Voronoi cells in gnuplot format
	con.draw_cells_gnuplot("cp_test_v.gnu");

	// Output the domain outline
	con.draw_domain_gnuplot("cp_test_d.gnu");
}
