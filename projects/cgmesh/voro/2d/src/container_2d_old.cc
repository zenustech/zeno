/** \file container_2d.cc
 * \brief Function implementations for the container_2d class. */

#include "container_2d.hh"

/** The class constructor sets up the geometry of container, initializing the
 * minimum and maximum coordinates in each direction, and setting whether each
 * direction is periodic or not. It divides the container into a rectangular
 * grid of blocks, and allocates memory for each of these for storing particle
 * positions and IDs.
 * \param[in] (ax_,bx_) the minimum and maximum x coordinates.
 * \param[in] (ay_,by_) the minimum and maximum y coordinates.
 * \param[in] (nx_,ny_) the number of grid blocks in each of the three
 *                       coordinate directions.
 * \param[in] (xperiodic_,yperiodic_) flags setting whether the container is periodic
 *                        in each coordinate direction.
 * \param[in] init_mem the initial memory allocation for each block. */
container_2d::container_2d(double ax_,double bx_,double ay_,
		double by_,int nx_,int ny_,bool xperiodic_,bool yperiodic_,int init_mem)
	: ax(ax_), bx(bx_), ay(ay_), by(by_), boxx((bx_-ax_)/nx_), boxy((by_-ay_)/ny_),
	xsp(1/boxx), ysp(1/boxy), nx(nx_), ny(ny_), nxy(nx*ny),
	xperiodic(xperiodic_), yperiodic(yperiodic_),
	co(new int[nxy]), mem(new int[nxy]), id(new int*[nxy]), p(new double*[nxy]) {
	int l;
	for(l=0;l<nxy;l++) co[l]=0;
	for(l=0;l<nxy;l++) mem[l]=init_mem;
	for(l=0;l<nxy;l++) id[l]=new int[init_mem];
	for(l=0;l<nxy;l++) p[l]=new double[2*init_mem];
}

/** The container destructor frees the dynamically allocated memory. */
container_2d::~container_2d() {
	int l;
	for(l=nxy-1;l>=0;l--) delete [] p[l];
	for(l=nxy-1;l>=0;l--) delete [] id[l];
	delete [] p;
	delete [] id;
	delete [] mem;
	delete [] co;
}

/** Put a particle into the correct region of the container.
 * \param[in] n the numerical ID of the inserted particle.
 * \param[in] (x,y) the position vector of the inserted particle. */
void container_2d::put(int n,double x,double y) {
	int ij;
	if(put_locate_block(ij,x,y)) {
		id[ij][co[ij]]=n;
		double *pp(p[ij]+2*co[ij]++);
		*(pp++)=x;*pp=y;
	}
}

/** This routine takes a particle position vector, tries to remap it into the
 * primary domain. If successful, it computes the region into which it can be
 * stored and checks that there is enough memory within this region to store
 * it.
 * \param[out] ij the region index.
 * \param[in,out] (x,y) the particle position, remapped into the primary
 *                      domain if necessary.
 * \return True if the particle can be successfully placed into the container,
 * false otherwise. */
inline bool container_2d::put_locate_block(int &ij,double &x,double &y) {
	if(put_remap(ij,x,y)) {
		if(co[ij]==mem[ij]) add_particle_memory(ij);
		return true;
	}
#if VOROPP_REPORT_OUT_OF_BOUNDS ==1
	fprintf(stderr,"Out of bounds: (x,y)=(%g,%g)\n",x,y);
#endif
	return false;
}

/** Takes a particle position vector and computes the region index into which
 * it should be stored. If the container is periodic, then the routine also
 * maps the particle position to ensure it is in the primary domain. If the
 * container is not periodic, the routine bails out.
 * \param[out] ij the region index.
 * \param[in,out] (x,y) the particle position, remapped into the primary
 *                      domain if necessary.
 * \return True if the particle can be successfully placed into the container,
 * false otherwise. */
inline bool container_2d::put_remap(int &ij,double &x,double &y) {
	int l;

	ij=step_int((x-ax)*xsp);
	if(xperiodic) {l=step_mod(ij,nx);x+=boxx*(l-ij);ij=l;}
	else if(ij<0||ij>=nx) return false;

	int j(step_int((y-ay)*ysp));
	if(yperiodic) {l=step_mod(j,ny);y+=boxy*(l-j);j=l;}
	else if(j<0||j>=ny) return false;

	ij+=nx*j;
	return true;
}

/** Increase memory for a particular region.
 * \param[in] i the index of the region to reallocate. */
void container_2d::add_particle_memory(int i) {
	int l,*idp;double *pp;
	mem[i]<<=1;
	if(mem[i]>max_particle_memory)
		voropp_fatal_error("Absolute maximum particle memory allocation exceeded",VOROPP_MEMORY_ERROR);
#if VOROPP_VERBOSE >=3
	fprintf(stderr,"Particle memory in region %d scaled up to %d\n",i,mem[i]);
#endif
	idp=new int[mem[i]];
	for(l=0;l<co[i];l++) idp[l]=id[i][l];
	pp=new double[2*mem[i]];
	for(l=0;l<2*co[i];l++) pp[l]=p[i][l];
	delete [] id[i];id[i]=idp;
	delete [] p[i];p[i]=pp;
}

/** Imports a list of particles from an input stream.
 * \param[in] fp a file handle to read from. */
void container_2d::import(FILE *fp) {
	int i,j;
	double x,y;
	while((j=fscanf(fp,"%d %lg %lg",&i,&x,&y))==3) put(i,x,y);
	if(j!=EOF) voropp_fatal_error("File import error",VOROPP_FILE_ERROR);
}

/** Clears a container of particles. */
void container_2d::clear() {
	for(int* cop=co;cop<co+nxy;cop++) *cop=0;
}

/** Dumps all the particle positions and IDs to a file.
 * \param[in] fp the file handle to write to. */
void container_2d::draw_particles(FILE *fp) {
	int ij,q;
	for(ij=0;ij<nxy;ij++) for(q=0;q<co[ij];q++)
		fprintf(fp,"%d %g %g\n",id[ij][q],p[ij][2*q],p[ij][2*q+1]);
}

/** Dumps all the particle positions in POV-Ray format.
 * \param[in] fp the file handle to write to. */
void container_2d::draw_particles_pov(FILE *fp) {
	int ij,q;
	for(ij=0;ij<nxy;ij++) for(q=0;q<co[ij];q++)
		fprintf(fp,"// id %d\nsphere{<%g,%g,0>,s\n",id[ij][q],p[ij][2*q],p[ij][2*q+1]);
}

/** Computes the Voronoi cells for all particles and saves the output in
 * gnuplot format.
 * \param[in] fp a file handle to write to. */
void container_2d::draw_cells_gnuplot(FILE *fp) {
	int i,j,ij=0,q;
	double x,y;
	voronoicell_2d c;
	for(j=0;j<ny;j++) for(i=0;i<nx;i++,ij++) for(q=0;q<co[ij];q++) {
		x=p[ij][2*q];y=p[ij][2*q+1];
		if(compute_cell_sphere(c,i,j,ij,q,x,y)) c.draw_gnuplot(x,y,fp);
	}
}

/** Computes the Voronoi cells for all particles within a rectangular box, and
 * saves the output in POV-Ray format.
 * \param[in] fp a file handle to write to. */
void container_2d::draw_cells_pov(FILE *fp) {
	int i,j,ij=0,q;
	double x,y;
	voronoicell_2d c;
	for(j=0;j<ny;j++) for(i=0;i<nx;i++,ij++) for(q=0;q<co[ij];q++) {
		x=p[ij][2*q];y=p[ij][2*q+1];
		if(compute_cell_sphere(c,i,j,ij,q,x,y)) {
			fprintf(fp,"// cell %d\n",id[ij][q]);
			c.draw_pov(x,y,0,fp);
		}
	}
}

/** Computes the Voronoi cells for all particles in the container, and for each
 * cell, outputs a line containing custom information about the cell structure.
 * The output format is specified using an input string with control sequences
 * similar to the standard C printf() routine.
 * \param[in] format the format of the output lines, using control sequences to
 *                   denote the different cell statistics.
 * \param[in] fp a file handle to write to. */
void container_2d::print_custom(const char *format,FILE *fp) {
	int i,j,ij=0,q;
	double x,y;
	voronoicell_2d c;
	for(j=0;j<ny;j++) for(i=0;i<nx;i++,ij++) for(q=0;q<co[ij];q++) {
		x=p[ij][2*q];y=p[ij][2*q+1];
		if(compute_cell_sphere(c,i,j,ij,q,x,y)) c.output_custom(format,id[ij][q],x,y,default_radius,fp);
	}
}

/** Initializes a voronoicell_2d class to fill the entire container.
 * \param[in] c a reference to a voronoicell_2d class.
 * \param[in] (x,y) the position of the particle that . */
bool container_2d::initialize_voronoicell(voronoicell_2d &c,double x,double y) {
	double x1,x2,y1,y2;
	if(xperiodic) x1=-(x2=0.5*(bx-ax));else {x1=ax-x;x2=bx-x;}
	if(yperiodic) y1=-(y2=0.5*(by-ay));else {y1=ay-y;y2=by-y;}
	c.init(x1,x2,y1,y2);
	return true;
}

/** Computes all Voronoi cells and sums their areas.
 * \return The computed area. */
double container_2d::sum_cell_areas() {
	int i,j,ij=0,q;
	double x,y,sum=0;
	voronoicell_2d c;
	for(j=0;j<ny;j++) for(i=0;i<nx;i++,ij++) for(q=0;q<co[ij];q++) {
		x=p[ij][2*q];y=p[ij][2*q+1];
		if(compute_cell_sphere(c,i,j,ij,q,x,y)) sum+=c.area();
	}
	return sum;
}

/** Computes all of the Voronoi cells in the container, but does nothing
 * with the output. It is useful for measuring the pure computation time
 * of the Voronoi algorithm, without any additional calculations such as
 * volume evaluation or cell output. */
void container_2d::compute_all_cells() {
	int i,j,ij=0,q;
	voronoicell_2d c;
	for(j=0;j<ny;j++) for(i=0;i<nx;i++,ij++) for(q=0;q<co[ij];q++)
		compute_cell_sphere(c,i,j,ij,q);
}

/** This routine computes the Voronoi cell for a give particle, by successively
 * testing over particles within larger and larger concentric circles. This
 * routine is simple and fast, although it may not work well for anisotropic
 * arrangements of particles.
 * \param[in,out] c a reference to a voronoicell object.
 * \param[in] (i,j) the coordinates of the block that the test particle is
 *                  in.
 * \param[in] ij the index of the block that the test particle is in, set to
 *               i+nx*j.
 * \param[in] s the index of the particle within the test block.
 * \param[in] (x,y) the coordinates of the particle.
 * \return False if the Voronoi cell was completely removed during the
 * computation and has zero volume, true otherwise. */
bool container_2d::compute_cell_sphere(voronoicell_2d &c,int i,int j,int ij,int s,double x,double y) {

	// This length scale determines how large the spherical shells should
	// be, and it should be set to approximately the particle diameter
	const double length_scale=0.5*sqrt((bx-ax)*(by-ay)/(nx*ny));

	double x1,y1,qx,qy,lr=0,lrs=0,ur,urs,rs;
	int q,t;
	voropp_loop_2d l(*this);

	if(!initialize_voronoicell(c,x,y)) return false;

	// Now the cell is cut by testing neighboring particles in concentric
	// shells. Once the test shell becomes twice as large as the Voronoi
	// cell we can stop testing.
	while(lrs<c.max_radius_squared()) {
		ur=lr+0.5*length_scale;urs=ur*ur;
		t=l.init(x,y,ur,qx,qy);
		do {
			for(q=0;q<co[t];q++) {
				x1=p[t][2*q]+qx-x;y1=p[t][2*q+1]+qy-y;
				rs=x1*x1+y1*y1;
				if(lrs-tolerance<rs&&rs<urs&&(q!=s||ij!=t)) {
					if(!c.plane(x1,y1,rs)) return false;
				}
			}
		} while((t=l.inc(qx,qy))!=-1);
		lr=ur;lrs=urs;
	}
	return true;
}

/** Creates a voropp_loop_2d object, by setting the necessary constants about the
 * container geometry from a pointer to the current container class.
 * \param[in] con a reference to the associated container class. */
voropp_loop_2d::voropp_loop_2d(container_2d &con) : boxx(con.bx-con.ax), boxy(con.by-con.ay),
	xsp(con.xsp),ysp(con.ysp),
	ax(con.ax),ay(con.ay),nx(con.nx),ny(con.ny),nxy(con.nxy),
	xperiodic(con.xperiodic),yperiodic(con.yperiodic) {}

/** Initializes a voropp_loop_2d object, by finding all blocks which are within a
 * given sphere. It calculates the index of the first block that needs to be
 * tested and sets the periodic displacement vector accordingly.
 * \param[in] (vx,vy) the position vector of the center of the sphere.
 * \param[in] r the radius of the sphere.
 * \param[out] (px,py) the periodic displacement vector for the first block to
 *                     be tested.
 * \return The index of the first block to be tested. */
int voropp_loop_2d::init(double vx,double vy,double r,double &px,double &py) {
	ai=step_int((vx-ax-r)*xsp);
	bi=step_int((vx-ax+r)*xsp);
	if(!xperiodic) {
		if(ai<0) {ai=0;if(bi<0) bi=0;}
		if(bi>=nx) {bi=nx-1;if(ai>=nx) ai=nx-1;}
	}
	aj=step_int((vy-ay-r)*ysp);
	bj=step_int((vy-ay+r)*ysp);
	if(!yperiodic) {
		if(aj<0) {aj=0;if(bj<0) bj=0;}
		if(bj>=ny) {bj=ny-1;if(aj>=ny) aj=ny-1;}
	}
	i=ai;j=aj;
	aip=ip=step_mod(i,nx);apx=px=step_div(i,nx)*boxx;
	ajp=jp=step_mod(j,ny);apy=py=step_div(j,ny)*boxy;
	inc1=aip-step_mod(bi,nx)+nx;
	s=aip+nx*ajp;
	return s;
}

/** Initializes a voropp_loop_2d object, by finding all blocks which overlap a given
 * rectangular box. It calculates the index of the first block that needs to be
 * tested and sets the periodic displacement vector (px,py,pz) accordingly.
 * \param[in] (xmin,xmax) the minimum and maximum x coordinates of the box.
 * \param[in] (ymin,ymax) the minimum and maximum y coordinates of the box.
 * \param[out] (px,py) the periodic displacement vector for the first block
 *                     to be tested.
 * \return The index of the first block to be tested. */
int voropp_loop_2d::init(double xmin,double xmax,double ymin,double ymax,double &px,double &py) {
	ai=step_int((xmin-ax)*xsp);
	bi=step_int((xmax-ax)*xsp);
	if(!xperiodic) {
		if(ai<0) {ai=0;if(bi<0) bi=0;}
		if(bi>=nx) {bi=nx-1;if(ai>=nx) ai=nx-1;}
	}
	aj=step_int((ymin-ay)*ysp);
	bj=step_int((ymax-ay)*ysp);
	if(!yperiodic) {
		if(aj<0) {aj=0;if(bj<0) bj=0;}
		if(bj>=ny) {bj=ny-1;if(aj>=ny) aj=ny-1;}
	}
	i=ai;j=aj;
	aip=ip=step_mod(i,nx);apx=px=step_div(i,nx)*boxx;
	ajp=jp=step_mod(j,ny);apy=py=step_div(j,ny)*boxy;
	inc1=aip-step_mod(bi,nx)+nx;
	s=aip+nx*ajp;
	return s;
}

/** Returns the next block to be tested in a loop, and updates the periodicity
 * vector if necessary.
 * \param[in,out] (px,py) the current block on entering the function, which is
 *                        updated to the next block on exiting the function. */
int voropp_loop_2d::inc(double &px,double &py) {
	if(i<bi) {
		i++;
		if(ip<nx-1) {ip++;s++;} else {ip=0;s+=1-nx;px+=boxx;}
		return s;
	} else if(j<bj) {
		i=ai;ip=aip;px=apx;j++;
		if(jp<ny-1) {jp++;s+=inc1;} else {jp=0;s+=inc1-nxy;py+=boxy;}
		return s;
	} else return -1;
}
