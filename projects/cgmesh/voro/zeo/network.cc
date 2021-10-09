// Voronoi calculation example code
//
// Author   : Chris H. Rycroft (LBL / UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

#include <cstring>

#include "voro++.hh"
using namespace voro;

#include "v_network.hh"
#include "r_table.cc"

// A guess for the memory allocation per region
const int memory=16;

// A maximum allowed number of regions, to prevent enormous amounts of memory
// being allocated
const int max_regions=16777216;

// A buffer size
const int bsize=2048;

// Output routine
template<class c_class>
void compute(c_class &con,char *buffer,int bp,double vol);

// Commonly used error message
void file_import_error() {
	voro_fatal_error("File import error",VOROPP_FILE_ERROR);
}

int main(int argc,char **argv) {
	char *farg,buffer[bsize];
	bool radial;int i,n,bp;
	double bx,bxy,by,bxz,byz,bz,x,y,z,vol;

	// Check the command line syntax
	if(argc==2) {
		radial=false;farg=argv[1];
	} else if(argc==3&&strcmp(argv[1],"-r")==0) {
		radial=true;farg=argv[2];
	} else {
		fputs("Syntax: ./network [-r] <filename.v1>\n",stderr);
		return VOROPP_CMD_LINE_ERROR;
	}

	// Check that the file has a ".v1" extension
	bp=strlen(farg);
	if(bp+2>bsize) {
		fputs("Filename too long\n",stderr);
		return VOROPP_CMD_LINE_ERROR;
	}
	if(bp<3||farg[bp-3]!='.'||farg[bp-2]!='v'||farg[bp-1]!='1') {
		fputs("Filename must end in '.v1'\n",stderr);
		return VOROPP_CMD_LINE_ERROR;
	}

	// Try opening the file
	FILE *fp(fopen(farg,"r"));
	if(fp==NULL) voro_fatal_error("Unable to open file for import",VOROPP_FILE_ERROR);

	// Read header line
	if(fgets(buffer,bsize,fp)!=buffer) file_import_error();
	if(strcmp(buffer,"Unit cell vectors:\n")!=0)
		voro_fatal_error("Invalid header line",VOROPP_FILE_ERROR);

	// Read in the box dimensions and the number of particles
	if(fscanf(fp,"%s %lg %lg %lg",buffer,&bx,&x,&x)!=4) file_import_error();
	if(strcmp(buffer,"va=")!=0) voro_fatal_error("Invalid first vector",VOROPP_FILE_ERROR);
	if(fscanf(fp,"%s %lg %lg %lg",buffer,&bxy,&by,&x)!=4) file_import_error();
	if(strcmp(buffer,"vb=")!=0) voro_fatal_error("Invalid second vector",VOROPP_FILE_ERROR);
	if(fscanf(fp,"%s %lg %lg %lg",buffer,&bxz,&byz,&bz)!=4) file_import_error();
	if(strcmp(buffer,"vc=")!=0) voro_fatal_error("Invalid third vector",VOROPP_FILE_ERROR);
	if(fscanf(fp,"%d",&n)!=1) file_import_error();

	// Print the box dimensions
	printf("Box dimensions:\n"
	       "  va=(%f 0 0)\n"
	       "  vb=(%f %f 0)\n"
	       "  vc=(%f %f %f)\n\n",bx,bxy,by,bxz,byz,bz);

	// Check that the input parameters make sense
	if(n<1) voro_fatal_error("Invalid number of particles",VOROPP_FILE_ERROR);
	if(bx<tolerance||by<tolerance||bz<tolerance)
		voro_fatal_error("Invalid box dimensions",VOROPP_FILE_ERROR);

	// Compute the internal grid size, aiming to make
	// the grid blocks square with around 6 particles
	// in each
	double ls=1.8*pow(bx*by*bz,-1.0/3.0);
	double nxf=bx*ls+1.5;
	double nyf=by*ls+1.5;
	double nzf=bz*ls+1.5;

	// Check the grid is not too huge, using floating point numbers to avoid
	// integer wrap-arounds
	if (nxf*nyf*nzf>max_regions) {
		fprintf(stderr,"voro++: Number of computational blocks exceeds the maximum allowed of %d\n"
			"Either increase the particle length scale, or recompile with an increased\nmaximum.\n",
			max_regions);
		return VOROPP_MEMORY_ERROR;
	}

	// Now that we are confident that the number of regions is reasonable,
	// create integer versions of them
	int nx=int(nxf);
	int ny=int(nyf);
	int nz=int(nzf);
	printf("Total particles = %d\n\nInternal grid size = (%d %d %d)\n\n",n,nx,ny,nz);

	vol=bx*by*bz;
	if(radial) {

		// Create a container with the geometry given above
		container_periodic_poly con(bx,bxy,by,bxz,byz,bz,nx,ny,nz,memory);

		// Read in the particles from the file
		for(i=0;i<n;i++) {
			if(fscanf(fp,"%s %lg %lg %lg",buffer,&x,&y,&z)!=4) file_import_error();
			con.put(i,x,y,z,radial_lookup(buffer));
		}
		fclose(fp);

		// Copy the output filename
		for(i=0;i<bp-2;i++) buffer[i]=farg[i];
		compute(con,buffer,bp,vol);
	} else {

		// Create a container with the geometry given above
		container_periodic con(bx,bxy,by,bxz,byz,bz,nx,ny,nz,memory);

		// Read in the particles from the file
		for(i=0;i<n;i++) {
			if(fscanf(fp,"%s %lg %lg %lg",buffer,&x,&y,&z)!=4)
				voro_fatal_error("File import error",VOROPP_FILE_ERROR);
			con.put(i,x,y,z);
		}
		fclose(fp);

		// Copy the output filename
		for(i=0;i<bp-2;i++) buffer[i]=farg[i];
		compute(con,buffer,bp,vol);
	}
}

inline void extension(const char *ext,char *bu) {
	char *ep((char*) ext);
	while(*ep!=0) *(bu++)=*(ep++);*bu=*ep;
}

template<class c_class>
void compute(c_class &con,char *buffer,int bp,double vol) {
	char *bu(buffer+bp-2);
	int id;
	double vvol(0),x,y,z,r;
	voronoicell c(con);
	voronoi_network vn(con,1e-5),vn2(con,1e-5);

	// Compute Voronoi cells and
	c_loop_all_periodic vl(con);
	if(vl.start()) do if(con.compute_cell(c,vl)) {
		vvol+=c.volume();
		vl.pos(id,x,y,z,r);
		vn.add_to_network(c,id,x,y,z,r);
		vn2.add_to_network_rectangular(c,id,x,y,z,r);
	} while(vl.inc());

	// Carry out the volume check
	printf("Volume check:\n  Total domain volume  = %f\n"
	       "  Total Voronoi volume = %f\n",vol,vvol);

	// Print non-rectangular cell network
	extension("nd2",bu);vn.draw_network(buffer);
	extension("nt2",bu);vn.print_network(buffer);

	// Print rectangular cell network
	extension("ntd",bu);vn2.draw_network(buffer);
	extension("net",bu);vn2.print_network(buffer);

	// Output the particles and any constructed periodic images
	extension("par",bu);con.draw_particles(buffer);

	// Output the Voronoi cells in gnuplot format
	extension("out",bu);con.draw_cells_gnuplot(buffer);

	// Output the unit cell in gnuplot format
	extension("dom",bu);con.draw_domain_gnuplot(buffer);
}
