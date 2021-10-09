// Voronoi method to generate nanocrystalline grain boundaries
// Nov 18, 2011

#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace std;

#include "voro++.hh"
using namespace voro;

const double pi=3.1415926535897932384626433832795;

// Box geometry
const double x_min=0,x_max=81;
const double y_min=0,y_max=81;
const double z_min=0,z_max=81;
const double cvol=(x_max-x_min)*(y_max-y_min)*(x_max-x_min);

// Total number of particles
const int particles=24;

// Lattice size
const double h=4;

// Function for random double between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	int i,j=0,c_id;
	double x,y,z,x1,y1,z1,x2,y2,z2,x3,y3,z3,xt,yt,zt,rx,ry,rz;
	double theta,cth,sth,r,v,xc,yc,zc,vx,vy,vz;
    	
	// Create the container class
	container con(x_min,x_max,y_min,y_max,z_min,z_max,10,10,10,
			false,false,false,8);

	// Add generators to the box
	for(i=1;i<=particles;i++) {
		x=x_min+rnd()*(x_max-x_min);
		y=y_min+rnd()*(y_max-y_min);
		z=z_min+rnd()*(z_max-z_min);
		con.put(i,x,y,z);
	}

	// Open the file for the particle positions
	FILE *fp=safe_fopen("lammps_input","w");

	// Create a loop class to iterate over all of the generators
	// in the box
	c_loop_all cl(con);
	voronoicell c;
	if(cl.start()) do if(con.compute_cell(c,cl)) {

		// Generate the first vector of an orthonormal basis 
		x1=2*rnd()-1;y1=2*rnd()-1;z1=2*rnd()-1;r=1/sqrt(x1*x1+y1*y1+z1*z1);
		x1*=r;y1*=r;z1*=r;

		// Construct a second perpendicular vector
		if(abs(x1)>0.5||abs(y1)>0.5) {r=1/sqrt(x1*x1+y1*y1);x2=-y1*r;y2=x1*r;z2=0;}
		else {r=1/sqrt(x1*x1+z1*z1);x2=-z1*r;y2=0;z2=x1*r;}

		// Construct a third perpendicular vector using the vector product
		x=y2*z1-z2*y1;y=z2*x1-x2*z1;z=x2*y1-y2*x1;

		// Take a random rotation of the second and third vectors
		theta=2*pi*rnd();cth=cos(theta);sth=sin(theta);
		x3=x*cth+x2*sth;x2=-x*sth+x2*cth;
		y3=y*cth+y2*sth;y2=-y*sth+y2*cth;
		z3=z*cth+z2*sth;z2=-z*sth+z2*cth;

		// Get a bound on how far to search
		r=sqrt(0.25*c.max_radius_squared());
		v=(int(r/h)+2)*h;

		// Add small random displacement to lattice positioning,
		// so that it's not always perfectly aligned with the generator
		vx=-v+h*rnd();vy=-v+h*rnd();vz=-v+h*rnd();

		// Print diagnostic information about this generator 
		c_id=cl.pid();cl.pos(xc,yc,zc);
		printf("Generator %d at (%g,%g,%g), random basis:\n",c_id,xc,yc,zc);
		printf("%g %g %g\n",x1,y1,z1);
		printf("%g %g %g\n",x2,y2,z2);
		printf("%g %g %g\n\n",x3,y3,z3);

		// Loop over a local region of points
		for(z=vx;z<=v;z+=h) for(y=vy;y<=v;y+=h) for(x=vz;x<=v;x+=h) {

			// Construct points rotated into the random basis
			xt=xc+x*x1+y*x2+z*x3;
			yt=yc+x*y1+y*y2+z*y3;
			zt=zc+x*z1+y*z2+z*z3;

			// Skip if this lies outside the container
			if(xt<x_min||xt>x_max||yt<y_min||yt>y_max||zt<z_min||zt>z_max) continue;

			// Find the nearest generator
			con.find_voronoi_cell(xt,yt,zt,rx,ry,rz,i);

			// If the nearest generator matches, then save this point
			if(i==c_id) {fprintf(fp,"%d %g %g %g\n",j,xt,yt,zt);j++;}
		}
	} while(cl.inc());

	// Close the output file
	fclose(fp);

	// Output files for diagnosic purposes
	con.draw_particles("lammps_generators");
	con.draw_cells_gnuplot("lammps_cells");
}
