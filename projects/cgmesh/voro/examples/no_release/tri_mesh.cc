// Voronoi calculation example code
//
// Author   : Chris H. Rycroft (Harvard University / LBL)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

#include "voro++.hh"
using namespace voro;

#include <vector>
using namespace std;

// Set up constants for the container geometry
const double x_min=-1,x_max=1;
const double y_min=-1,y_max=1;
const double z_min=-1,z_max=1;

// Set up the number of blocks that the container is divided into
const int n_x=6,n_y=6,n_z=6;

// Set the number of particles that are going to be randomly introduced
const int particles=1;

// This function returns a random double between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	int i,j,id,nv;
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

	// Sum up the volumes, and check that this matches the container volume
	c_loop_all cl(con);
	vector<int> f_vert;
	vector<double> v;
	voronoicell c;	
	if(cl.start()) do if(con.compute_cell(c,cl)) {
		cl.pos(x,y,z);id=cl.pid();
		printf("Particle %d:\n",id);

		// Gather information about the computed Voronoi cell
		c.face_vertices(f_vert);
		c.vertices(x,y,z,v);

		// Print vertex positions
		for(i=0;i<v.size();i+=3) printf("Vertex %d : (%g,%g,%g)\n",i/3,v[i],v[i+1],v[i+2]);
		puts("");

		// Loop over all faces of the Voronoi cell
		j=0;
		while(j<f_vert.size()) {

			// Number of vertices in this face
			nv=f_vert[j];

			// Print triangles
			for(i=2;i<nv;i++)
				printf("Triangle : (%d,%d,%d)\n",f_vert[j+1],f_vert[j+i],f_vert[j+i+1]);

			// Move j to point at the next face
			j+=nv+1;
		}
		puts("");
	} while (cl.inc());

	// Output the particle positions in gnuplot format
	con.draw_particles("random_points_p.gnu");

	// Output the Voronoi cells in gnuplot format
	con.draw_cells_gnuplot("random_points_v.gnu");
}
