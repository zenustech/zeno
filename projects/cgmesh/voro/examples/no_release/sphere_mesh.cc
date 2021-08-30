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
const double boxl=1.2;

// Set up the number of blocks that the container is divided into
const int bl=14;

// Set the number of particles that are going to be randomly introduced
const int particles=2000;

const int nface=11;

// This function returns a random double between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

struct wall_shell : public wall {
	public:
		wall_shell(double xc_,double yc_,double zc_,double rc,double sc,int w_id_=-99)
			: w_id(w_id_), xc(xc_), yc(yc_), zc(zc_), lc(rc-sc), uc(rc+sc) {}
		bool point_inside(double x,double y,double z) {
			double rsq=(x-xc)*(x-xc)+(y-yc)*(y-yc)+(z-zc)*(z-zc);
			return rsq>lc*lc&&rsq<uc*uc;
		}
		template<class v_cell>
		bool cut_cell_base(v_cell &c,double x,double y,double z) {
			double xd=x-xc,yd=y-yc,zd=z-zc,dq=xd*xd+yd*yd+zd*zd,dq2;
			if (dq>1e-5) {
				dq2=2*(sqrt(dq)*lc-dq);
				dq=2*(sqrt(dq)*uc-dq);
				return c.nplane(xd,yd,zd,dq,w_id)&&c.nplane(-xd,-yd,-zd,-dq2,w_id);
			}
			return true;
		}
		bool cut_cell(voronoicell &c,double x,double y,double z) {return cut_cell_base(c,x,y,z);}
		bool cut_cell(voronoicell_neighbor &c,double x,double y,double z) {return cut_cell_base(c,x,y,z);}
	private:
		const int w_id;
		const double xc,yc,zc,lc,uc;
};


int main() {
	int i=0,j,k,l,ll,o;
	double x,y,z,r,dx,dy,dz;
	int faces[nface],*fp;
	double p[3*particles];

	// Create a container with the geometry given above, and make it
	// non-periodic in each of the three coordinates. Allocate space for
	// eight particles within each computational block
	container con(-boxl,boxl,-boxl,boxl,-boxl,boxl,bl,bl,bl,false,false,false,8);

	wall_shell ws(0,0,0,1,0.00001);
	con.add_wall(ws);

	// Randomly add particles into the container
	while(i<particles) {
		x=boxl*(2*rnd()-1);
		y=boxl*(2*rnd()-1);
		z=boxl*(2*rnd()-1);
		r=x*x+y*y+z*z;
		if(r>1e-5) {
			r=1/sqrt(r);x*=r;y*=r;z*=r;
			con.put(i,x,y,z);
			i++;
		}
	}

	for(l=0;l<100;l++) {
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
		double fac=0;//l<9000?0.1/sqrt(double(l)):0;
		for(i=0;i<particles;i++) con.put(i,p[3*i]+fac*(2*rnd()-1),p[3*i+1]+fac*(2*rnd()-1),p[3*i+2]+fac*(2*rnd()-1));
		printf("%d",l);
		for(fp=faces;fp<faces+nface;fp++) printf(" %d",*fp);
		puts("");
	}

	// Output the particle positions in gnuplot format
	con.draw_particles("sphere_mesh_p.gnu");

	// Output the Voronoi cells in gnuplot format
	con.draw_cells_gnuplot("sphere_mesh_v.gnu");

	// Allocate memory for neighbor relations
	int *q=new int[particles*nface],*qn=new int[particles],*qp;
	for(l=0;l<particles;l++) qn[l]=0;

	// Create a table of all neighbor relations
	vector<int> vi;
	voronoicell_neighbor c;
	c_loop_all vl(con);
	if(vl.start()) do if(con.compute_cell(c,vl)) {
		i=vl.pid();qp=q+i*nface;
		c.neighbors(vi);
		if(vi.size()>nface+2) voro_fatal_error("Too many faces; boost nface",5);

		for(l=0;l<(signed int) vi.size();l++) if(vi[l]>=0) qp[qn[i]++]=vi[l];
	} while (vl.inc());

	// Sort the connections in anti-clockwise order
	bool connect;
	int tote=0;
	for(l=0;l<particles;l++) {
		tote+=qn[l];
		for(i=0;i<qn[l]-2;i++) {
			o=q[l*nface+i];
			//printf("---> %d,%d\n",i,o);
			j=i+1;
			while(j<qn[l]-1) {
				ll=q[l*nface+j];
			//	printf("-> %d %d\n",j,ll);
				connect=false;
				for(k=0;k<qn[ll];k++) {
			//		printf("%d %d %d\n",ll,k,q[ll*nface+k]);
					if(q[ll*nface+k]==o) {connect=true;break;}
				}
				if(connect) break;
				j++;
			}

			// Swap the connected vertex into this location
			//printf("%d %d\n",i+1,j);
			o=q[l*nface+i+1];
			q[l*nface+i+1]=q[l*nface+j];
			q[l*nface+j]=o;
		}
	
		// Reverse all connections if the have the wrong handedness
		j=3*l;k=3*q[l*nface];o=3*q[l*nface+1];
		x=p[j]-p[k];dx=p[j]-p[o];
		y=p[j+1]-p[k+1];dy=p[j+1]-p[o+1];
		z=p[j+2]-p[k+2];dz=p[j+2]-p[o+2];
		if(p[j]*(y*dz-z*dy)+p[j+1]*(z*dx-x*dz)+p[j+2]*(x*dy-y*dx)<0) {
			for(i=0;i<qn[l]/2;i++) {
				o=q[l*nface+i];
				q[l*nface+i]=q[l*nface+qn[l]-1-i];
				q[l*nface+qn[l]-1-i]=o;
			}
		}
	}

	FILE *ff=safe_fopen("sphere_mesh.net","w");
	int *mp=new int[particles],*mpi=new int[particles];
	for(i=0;i<particles;i++) mp[i]=-1;
	*mpi=0;*mp=0;l=1;o=0;
	while(o<l) {
		i=mpi[o];
		for(j=0;j<qn[i];j++) {
			k=q[i*nface+j];
			if(mp[k]==-1) {
				mpi[l]=k;
				mp[k]=l++;
			}
			if(mp[i]<mp[k]) 
				fprintf(ff,"%g %g %g\n%g %g %g\n\n\n",p[3*i],p[3*i+1],p[3*i+2],p[3*k],p[3*k+1],p[3*k+2]);
		}
		o++;
	}
	fclose(ff);

	// Save binary representation of the mesh
	FILE *fb=safe_fopen("sphere_mesh.bin","wb");
	
	// Write header
	int kk[3],sz=tote+particles+2,*red(new int[sz]),*rp=red;
	*kk=1;kk[1]=sz;kk[2]=3*particles;
	fwrite(kk,sizeof(int),3,fb);

	// Assemble the connections and write them
	*(rp++)=particles;*(rp++)=tote;
	for(l=0;l<particles;l++) *(rp++)=qn[mpi[l]];
	for(l=0;l<particles;l++) {
		i=mpi[l];printf("%d",l);
		for(j=0;j<qn[i];j++) {*(rp++)=mp[q[i*nface+j]];printf(" %d",*(rp-1));}
		puts("");
	}
	fwrite(red,sizeof(int),sz,fb);


	double *pm=new double[3*particles],*a=pm,*b;
	for(i=0;i<particles;i++) {
		b=p+3*mpi[i];
		*(a++)=*(b++);*(a++)=*(b++);*(a++)=*b;
	}
	fwrite(pm,sizeof(double),3*particles,fb);
	delete [] pm;

	// Free dynamically allocated arrays
	delete [] red;
	delete [] mpi;
	delete [] mp;
	delete [] qn;
	delete [] q;
}
