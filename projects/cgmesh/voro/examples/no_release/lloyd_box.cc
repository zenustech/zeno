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
const double boxl=1;

// Set up the number of blocks that the container is divided into
const int bl=10;

// Set the number of particles that are going to be randomly introduced
const int particles=4000;

// Set the number of Voronoi faces to bin
const int nface=40;

// This function returns a random double between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	int i,l;
	double x,y,z,r,dx,dy,dz;
	int faces[nface],*fp;
	double p[3*particles];

	// Create a container with the geometry given above, and make it
	// non-periodic in each of the three coordinates. Allocate space for
	// eight particles within each computational block
	container con(-boxl,boxl,-boxl,boxl,-boxl,boxl,bl,bl,bl,false,false,false,8);

	// Randomly add particles into the container
	for(i=0;i<particles;i++) {
		x=boxl*(2*rnd()-1);
		y=boxl*(2*rnd()-1);
		z=boxl*(2*rnd()-1);
		con.put(i,x,y,z);
	}

	for(l=0;l<=200;l++) {
		c_loop_all vl(con);
		voronoicell c;
		for(fp=faces;fp<faces+nface;fp++) *fp=0;
		if(vl.start()) do if(con.compute_cell(c,vl)) {
			vl.pos(i,x,y,z,r);
			c.centroid(dx,dy,dz);
			p[3*i]=x+dx;
			p[3*i+1]=y+dy;
			p[3*i+2]=z+dz;

			i=c.number_of_faces()-4;
			if(i<0) i=0;if(i>=nface) i=nface-1;
			faces[i]++;
		} while (vl.inc());
		con.clear();
		for(i=0;i<particles;i++) con.put(i,p[3*i],p[3*i+1],p[3*i+2]);
		printf("%d",l);
		for(fp=faces;fp<faces+nface;fp++) printf(" %d",*fp);
		puts("");
	}

	// Output the particle positions in gnuplot format
	con.draw_particles("sphere_mesh_p.gnu");

	// Output the Voronoi cells in gnuplot format
	con.draw_cells_gnuplot("sphere_mesh_v.gnu");

	// Output the neighbor mesh in gnuplot format
	FILE *ff=safe_fopen("sphere_mesh.net","w");
	vector<int> vi;
	voronoicell_neighbor c;
	c_loop_all vl(con);
	if(vl.start()) do if(con.compute_cell(c,vl)) {
		i=vl.pid();
		c.neighbors(vi);
		for(l=0;l<(signed int) vi.size();l++) if(vi[l]>i)
			fprintf(ff,"%g %g %g\n%g %g %g\n\n\n",
				p[3*i],p[3*i+1],p[3*i+2],
				p[3*vi[l]],p[3*vi[l]+1],p[3*vi[l]+2]);
	} while (vl.inc());
	fclose(ff);
}
