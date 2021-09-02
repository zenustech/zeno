// Cell cutting region example code
//
// Author   : Chris H. Rycroft (LBL / UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

#include "voro++_2d.hh"
using namespace voro;

const double pi=3.1415926535897932384626433832795;

// This constant sets the tolerance in the bisection search algorithm
const double tolwidth=1e-7;

// This constant determines the density of points to test
const double phi_step=pi/400;

int main() {
	double x,y,r,rmin,rmax;
	double phi;
	voronoicell_2d v;
	FILE *fp=safe_fopen("intersect_region.gnu","w");

	// Initialize the Voronoi cell to be an octahedron and make a single
	// plane cut to add some variation
	v.init(-1,1,-1,1);
	v.plane(1,1,1);

	// Output the cell in gnuplot format
	v.draw_gnuplot(0,0,"intersect_cell.gnu");

	// Now test over direction vectors from the center of the sphere. For
	// each vector, carry out a search to find the maximum distance along
	// that vector that a plane will intersect with cell, and save it to
	// the output file.
	for(phi=phi_step*0.5;phi<2*pi;phi+=phi_step) {

		// Calculate a direction to look along
		x=cos(phi);
		y=sin(phi);

		// Now carry out a bisection search. Here, we initialize a
		// minimum and a maximum guess for the distance along this
		// vector. Keep multiplying rmax by two until the plane no
		// longer makes a cut.
		rmin=0;rmax=1;
		while (v.plane_intersects(x,y,rmax)) rmax*=2;

		// We now know that the distance is somewhere between rmin and
		// rmax. Test the point halfway in-between these two. If it
		// intersects, then move rmin to this point; otherwise, move
		// rmax there. At each stage the bounding interval is divided
		// by two. Exit when the width of the interval is smaller than
		// the tolerance.
		while (rmax-rmin>tolwidth) {
			r=(rmax+rmin)*0.5;
			if (v.plane_intersects(x,y,r)) rmin=r;
			else rmax=r;
		}

		// Output this point to file
		r=(rmax+rmin)*0.5;
		x*=r;y*=r;
		fprintf(fp,"%g %g\n",x,y);
	}

	fclose(fp);
}
