// Ellipsoidal wall example
//
// Author   : Simon Konstandin (UMM Mannheim) 

#include "voro++.hh"
using namespace voro;

#include "quartic/quartic.hpp"
using namespace magnet::math;

// Set up constants for the container geometry
const double x_min=-6,x_max=6;
const double y_min=-6,y_max=6;
const double z_min=-6,z_max=6;

// Golden ratio constants
const double Phi=0.5*(1+sqrt(5.0));
const double phi=0.5*(1-sqrt(5.0));

// Set up the number of blocks that the container is divided
// into.
const int n_x=5,n_y=5,n_z=5;

// Set the number of particles that are going to be randomly introduced
const int particles=500;

// This function returns a random double between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

// Ellipsoidal wall object
class wall_ellipsoid : public wall {
	public:
		const double a;
		const double aa;
		const double aa_inv;
		const double a_cal;
		const double rr;
		const double rr_inv;
		const double xc;
		const double yc;
		const double zc;
		const int w_id;
		wall_ellipsoid(double a_,double r_,double xc_,double yc_,double zc_)
			: a(a_), aa(a_*a_), aa_inv(1/aa), a_cal(2.0+2.0/aa), rr(r_*r_),
			rr_inv(1/rr), xc(xc_), yc(yc_), zc(zc_), w_id(99) {}
		bool point_inside(double x,double y,double z) {
    			return x*x+y*y+z*z*a*a<rr;
		}
		virtual bool cut_cell(voronoicell &c,double x,double y,double z) {
			return cut_cell_base(c,x,y,z);
		}
		virtual bool cut_cell(voronoicell_neighbor &c,double x,double y,double z) {
			return cut_cell_base(c,x,y,z);
		}
		template<class v_cell>
		bool cut_cell_base(v_cell &c,double x,double y,double z) {
			x-=xc;y-=yc;z-=zc;
			double xx=x*x,yy=y*y,zz=z*z,s[4],
				b_cal=1.0-(xx+yy)*rr_inv+(4.0-zz*rr_inv)*aa_inv+aa_inv*aa_inv,
				c_cal=2.0*(1.0-(xx+yy+zz)*rr_inv)*aa_inv+2.0*aa_inv*aa_inv,
				d_cal=(1.0-(xx+yy+zz*aa)*rr_inv)*aa_inv*aa_inv,
				xell[4],yell[4],zell[4],d2[4];
			int i,j=0,q=quarticSolve(a_cal,b_cal,c_cal,d_cal,s[0],s[1],s[2],s[3]);

			for (i=0;i<q;i++) {
				xell[i]=x/(1+s[i]);
				yell[i]=y/(1+s[i]);	
				zell[i]=z/(1+s[i]*aa);
				d2[i]=(xell[i]-x)*(xell[i]-x)+(yell[i]-y)*(yell[i]-y)+(zell[i]-z)*(zell[i]-z);
			}

			for(i=1;i<4;i++) if ((d2[i]<=d2[j])&&(abs(xell[i]*xell[i]+yell[i]*yell[i]+aa*zell[i]*zell[i]-rr)<1e-4)) j=i;

			double xn=xell[j]-x,yn=yell[j]-y,zn=zell[j]-z,dn2=xn*xn+yn*yn+zn*zn;

			return c.nplane(xn,yn,zn,xx+yy+aa*zz<rr?2*dn2:0,w_id);
		}
};

int main() {
	int i;
	double x,y,z;

	// Create a container with the geometry given above. This is bigger
	// than the particle packing itself.
	container con(x_min,x_max,y_min,y_max,z_min,z_max,n_x,n_y,n_z,
			false,false,false,8);

	// Create the "initial shape" wall class and add it to the container
	wall_ellipsoid we(1.5,5,0,0,0);
	con.add_wall(we);
	
	// Randomly add particles into the container
	for(i=0;i<particles;) {
		x=x_min+rnd()*(x_max-x_min);
		y=y_min+rnd()*(y_max-y_min);
		z=z_min+rnd()*(z_max-z_min);
		if(con.point_inside(x,y,z)) con.put(i++,x,y,z);
	}

	// Save the particles and Voronoi cells in POV-Ray format
	con.draw_particles("ellipsoid_p.gnu");
	con.draw_cells_gnuplot("ellipsoid_v.gnu");
}
