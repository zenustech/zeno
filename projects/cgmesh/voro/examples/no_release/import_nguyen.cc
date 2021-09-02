// File import example code
//
// Author   : Chris H. Rycroft (Harvard University / LBL)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

#include "voro++.hh"
using namespace voro;

#include <vector>
using namespace std;

// Set up constants for the container geometry
const double x_min=-5,x_max=5;
const double y_min=-5,y_max=5;
const double z_min=-5,z_max=5;

// Set up the number of blocks that the container is divided into
const int n_x=6,n_y=6,n_z=6;

int main() {
	
	// Construct container
	container con(-5,5,-5,5,0,10,6,6,6,false,false,false,8);

	// Import particles
	con.import("../basic/pack_ten_cube");

	// Loop over all the particles and compute the Voronoi cell for each
	unsigned int i;
	int id;
	double x,y,z;
	vector<double> vd;
	voronoicell c;
	c_loop_all cl(con);
	if(cl.start()) do if(con.compute_cell(c,cl)) {

		// Get particle position and ID
		cl.pos(x,y,z);id=cl.pid();

		// Get face areas
		c.face_areas(vd);

		// Output information (additional diagnostics could be done
		// here)
		printf("ID %d (%.3f,%.3f,%.3f) :",id,x,y,z);
		for(i=0;i<vd.size();i++) printf(" %.3f",vd[i]);
		puts("");
	} while (cl.inc());
}
