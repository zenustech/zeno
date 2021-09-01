#include <ctime>

#include "voro++_2d.hh"
using namespace voro;

// Set up the number of blocks that the container is divided into. If the
// preprocessor variable NNN hasn't been passed to the code, then initialize it
// to a good value. Otherwise, use the value that has been passed.
#ifndef NNN
#define NNN 26
#endif

// Set the number of particles that are going to be randomly introduced
const int particles=100000;

// This function returns a random floating point number between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	clock_t start,end;
	int i;double x,y;

	// Initialize the container class to be the unit square, with
	// non-periodic boundary conditions. Divide it into a 6 by 6 grid, with
	// an initial memory allocation of 16 particles per grid square.
	container_2d con(0,1,0,1,NNN,NNN,false,false,16);

	//Randomly add particles into the container
	for(i=0;i<particles;i++) {
		x=rnd();
		y=rnd();
		con.put(i,x,y);
	}

	// Store the initial clock time
	start=clock();

	// Carry out a dummy computation of all cells in the entire container
	con.compute_all_cells();

	// Calculate the elapsed time and print it
	end=clock();
	double runtime=double(end-start)/CLOCKS_PER_SEC;
	printf("%g\n",runtime);
}
