// Irregular packing example code
//
// Author   : Chris H. Rycroft (Harvard University / LBL)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

#include "voro++.hh"
using namespace voro;

#include <vector>
using namespace std;

// Set up constants for the container geometry
const double x_min=-15,x_max=15;
const double y_min=-7,y_max=7;
const double z_min=-15,z_max=15;

// Golden ratio constants
const double Phi=0.5*(1+sqrt(5.0));
const double phi=0.5*(1-sqrt(5.0));

// Set up the number of blocks that the container is divided
// into.
const int n_x=8,n_y=8,n_z=8;

// ID for dodecahedron faces
const int wid=-10;

// Create a wall class that, whenever called, will replace the Voronoi cell
// with a prescribed shape, in this case a dodecahedron
class wall_initial_shape : public wall {
	public:
		wall_initial_shape() {

			// Create a dodecahedron with neighbor information all
			// set to -10
			v.init(-2,2,-2,2,-2,2);
			v.nplane(0,Phi,1,wid);v.nplane(0,-Phi,1,wid);v.nplane(0,Phi,-1,wid);
			v.nplane(0,-Phi,-1,wid);v.nplane(1,0,Phi,wid);v.nplane(-1,0,Phi,wid);
			v.nplane(1,0,-Phi,wid);v.nplane(-1,0,-Phi,wid);v.nplane(Phi,1,0,wid);
			v.nplane(-Phi,1,0,wid);v.nplane(Phi,-1,0,wid);v.nplane(-Phi,-1,0,wid);
		};
		bool point_inside(double x,double y,double z) {return true;}
		bool cut_cell(voronoicell &c,double x,double y,double z) {

			// Just ignore this case
			return true;
		}
		bool cut_cell(voronoicell_neighbor &c,double x,double y,double z) {

			// Set the cell to be equal to the dodecahedron
			c=v;
			return true;
		}
	private:
		voronoicell_neighbor v;
};

// Determines whether any of the sides in the neighbor information are from the
// initial dodecahedron
bool has_dodec_sides(vector<int> &vi) {
	for(unsigned int i=0;i<vi.size();i++) if(vi[i]==wid) return true;
	return false;
}

int main() {

	// Create a container with the geometry given above. This is bigger
	// than the particle packing itself.
	container con(x_min,x_max,y_min,y_max,z_min,z_max,n_x,n_y,n_z,
			false,false,false,8);

	// Create the "initial shape" wall class and add it to the container
	wall_initial_shape(wis);
	con.add_wall(wis);

	// Import the irregular particle packing
	con.import("pack_semicircle");

	// Open files to save the "inside" and "outside" particles
	FILE *finside=safe_fopen("finite_sys_in.pov","w"),
	     *foutside=safe_fopen("finite_sys_out.pov","w");

	// Loop over all particles
	double x,y,z;
	vector<int> vi;
	voronoicell_neighbor c;
	c_loop_all cl(con);
	if(cl.start()) do {

		// Get particle position
		cl.pos(x,y,z);

		// Remove half of the particles to see a cross-section 
		if(y<0) continue;

		if(con.compute_cell(c,cl)) {

			// Get the neighboring IDs of all the faces
			c.neighbors(vi);

			// Depending on whether any of the faces are
			// from the original dodecahedron, print to
			// the "inside" or "outside" file
			fprintf(has_dodec_sides(vi)?foutside:finside,
				"sphere{<%g,%g,%g>,s}\n",x,y,z);
		}
	} while(cl.inc());
	
	// Close the output files
	fclose(finside);
	fclose(foutside);
}
