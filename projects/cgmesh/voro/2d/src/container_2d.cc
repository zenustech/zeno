// Voro++, a 3D cell-based Voronoi library
//
// Author   : Chris H. Rycroft (LBL / UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

/** \file container_2d.cc
 * \brief Function implementations for the container_2d and related classes. */

#include "container_2d.hh"

namespace voro {

/** The class constructor sets up the geometry of container, initializing the
 * minimum and maximum coordinates in each direction, and setting whether each
 * direction is periodic or not. It divides the container into a rectangular
 * grid of blocks, and allocates memory for each of these for storing particle
 * positions and IDs.
 * \param[in] (ax_,bx_) the minimum and maximum x coordinates.
 * \param[in] (ay_,by_) the minimum and maximum y coordinates.
 * \param[in] (nx_,ny_) the number of grid blocks in each of the three
 *                      coordinate directions.
 * \param[in] (xperiodic_,yperiodic_) flags setting whether the container is
 *				      periodic in each coordinate direction.
 * \param[in] init_mem the initial memory allocation for each block.
 * \param[in] ps_ the number of floating point entries to store for each
 *                particle. */
container_base_2d::container_base_2d(double ax_,double bx_,double ay_,double by_,
		int nx_,int ny_,bool xperiodic_,bool yperiodic_,int init_mem,int ps_)
	: voro_base_2d(nx_,ny_,(bx_-ax_)/nx_,(by_-ay_)/ny_),
	ax(ax_), bx(bx_), ay(ay_), by(by_), xperiodic(xperiodic_), yperiodic(yperiodic_),
	id(new int*[nxy]), p(new double*[nxy]), co(new int[nxy]), mem(new int[nxy]), ps(ps_) {
	int l;
	//totpar=0;
	for(l=0;l<nxy;l++) co[l]=0;
	for(l=0;l<nxy;l++) mem[l]=init_mem;
	for(l=0;l<nxy;l++) id[l]=new int[init_mem];
	for(l=0;l<nxy;l++) p[l]=new double[ps*init_mem];
}

/** The container destructor frees the dynamically allocated memory. */
container_base_2d::~container_base_2d() {
	int l;
	for(l=nxy-1;l>=0;l--) delete [] p[l];
	for(l=nxy-1;l>=0;l--) delete [] id[l];
	delete [] id;
	delete [] p;
	delete [] co;
	delete [] mem;
}

/** The class constructor sets up the geometry of container.
 * \param[in] (ax_,bx_) the minimum and maximum x coordinates.
 * \param[in] (ay_,by_) the minimum and maximum y coordinates.
 * \param[in] (nx_,ny_) the number of grid blocks in each of the three
 *                      coordinate directions.
 * \param[in] (xperiodic_,yperiodic_) flags setting whether the container is
 *				      periodic in each coordinate direction.
 * \param[in] init_mem the initial memory allocation for each block. */
container_2d::container_2d(double ax_,double bx_,double ay_,double by_,
	int nx_,int ny_,bool xperiodic_,bool yperiodic_,int init_mem)
	: container_base_2d(ax_,bx_,ay_,by_,nx_,ny_,xperiodic_,yperiodic_,init_mem,2),
	vc(*this,xperiodic_?2*nx_+1:nx_,yperiodic_?2*ny_+1:ny_) {}

/** The class constructor sets up the geometry of container.
 * \param[in] (ax_,bx_) the minimum and maximum x coordinates.
 * \param[in] (ay_,by_) the minimum and maximum y coordinates.
 * \param[in] (nx_,ny_) the number of grid blocks in each of the three
 *                      coordinate directions.
 * \param[in] (xperiodic_,yperiodic_) flags setting whether the container is
 *				      periodic in each coordinate direction.
 * \param[in] init_mem the initial memory allocation for each block. */
container_poly_2d::container_poly_2d(double ax_,double bx_,double ay_,double by_,
	int nx_,int ny_,bool xperiodic_,bool yperiodic_,int init_mem)
	: container_base_2d(ax_,bx_,ay_,by_,nx_,ny_,xperiodic_,yperiodic_,init_mem,3),
	vc(*this,xperiodic_?2*nx_+1:nx_,yperiodic_?2*ny_+1:ny_) {ppr=p;}

/** Put a particle into the correct region of the container.
 * \param[in] n the numerical ID of the inserted particle.
 * \param[in] (x,y) the position vector of the inserted particle. */
void container_2d::put(int n,double x,double y) {
	int ij;
	if(put_locate_block(ij,x,y)) {
		//totpar++;
		id[ij][co[ij]]=n;
		double *pp=p[ij]+2*co[ij]++;
		*(pp++)=x;*pp=y;
	}
}

/** Put a particle into the correct region of the container.
 * \param[in] n the numerical ID of the inserted particle.
 * \param[in] (x,y) the position vector of the inserted particle.
 * \param[in] r the radius of the particle. */
void container_poly_2d::put(int n,double x,double y,double r) {
	int ij;
	if(put_locate_block(ij,x,y)) {
		//totpar++;
		id[ij][co[ij]]=n;
		double *pp=p[ij]+3*co[ij]++;
		*(pp++)=x;*(pp++)=y;*pp=r;
		if(max_radius<r) max_radius=r;
	}
}

/** Put a particle into the correct region of the container, also recording
 * into which region it was stored.
 * \param[in] vo the ordering class in which to record the region.
 * \param[in] n the numerical ID of the inserted particle.
 * \param[in] (x,y) the position vector of the inserted particle. */
void container_2d::put(particle_order &vo,int n,double x,double y) {
	int ij;
	if(put_locate_block(ij,x,y)) {
		//totpar++;
		id[ij][co[ij]]=n;
		vo.add(ij,co[ij]);
		double *pp=p[ij]+2*co[ij]++;
		*(pp++)=x;*pp=y;
	}
}

/** Put a particle into the correct region of the container, also recording
 * into which region it was stored.
 * \param[in] vo the ordering class in which to record the region.
 * \param[in] n the numerical ID of the inserted particle.
 * \param[in] (x,y) the position vector of the inserted particle.
 * \param[in] r the radius of the particle. */
void container_poly_2d::put(particle_order &vo,int n,double x,double y,double r) {
	int ij;
	if(put_locate_block(ij,x,y)) {
		//totpar++;
		id[ij][co[ij]]=n;
		vo.add(ij,co[ij]);
		double *pp=p[ij]+3*co[ij]++;
		*(pp++)=x;*(pp++)=y;*pp=r;
		if(max_radius<r) max_radius=r;
	}
}

/** This routine takes a particle position vector, tries to remap it into the
 * primary domain. If successful, it computes the region into which it can be
 * stored and checks that there is enough memory within this region to store
 * it.
 * \param[out] ij the region index.
 * \param[in,out] (x,y) the particle position, remapped into the primary
 *                        domain if necessary.
 * \return True if the particle can be successfully placed into the container,
 * false otherwise. */
inline bool container_base_2d::put_locate_block(int &ij,double &x,double &y) {
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
 * \param[in,out] (x,y) the particle position, remapped into the primary domain
 *			if necessary.
 * \return True if the particle can be successfully placed into the container,
 * false otherwise. */
inline bool container_base_2d::put_remap(int &ij,double &x,double &y) {
	int l;

	ij=step_int((x-ax)*xsp);
	if(xperiodic) {l=step_mod(ij,nx);x+=boxx*(l-ij);ij=l;}
	else if(ij<0||ij>=nx) return false;

	int j=step_int((y-ay)*ysp);
	if(yperiodic) {l=step_mod(j,ny);y+=boxy*(l-j);j=l;}
	else if(j<0||j>=ny) return false;

	ij+=nx*j;
	return true;
}

/** Takes a position vector and attempts to remap it into the primary domain.
 * \param[out] (ai,aj) the periodic image displacement that the vector is in,
 *                     with (0,0,0) corresponding to the primary domain.
 * \param[out] (ci,cj) the index of the block that the position vector is
 *                     within, once it has been remapped.
 * \param[in,out] (x,y) the position vector to consider, which is remapped into
 *			the primary domain during the routine.
 * \param[out] ij the block index that the vector is within.
 * \return True if the particle is within the container or can be remapped into
 * it, false if it lies outside of the container bounds. */
inline bool container_base_2d::remap(int &ai,int &aj,int &ci,int &cj,double &x,double &y,int &ij) {
	ci=step_int((x-ax)*xsp);
	if(ci<0||ci>=nx) {
		if(xperiodic) {ai=step_div(ci,nx);x-=ai*(bx-ax);ci-=ai*nx;}
		else return false;
	} else ai=0;

	cj=step_int((y-ay)*ysp);
	if(cj<0||cj>=ny) {
		if(yperiodic) {aj=step_div(cj,ny);y-=aj*(by-ay);cj-=aj*ny;}
		else return false;
	} else aj=0;

	ij=ci+nx*cj;
	return true;
}

/** Takes a vector and finds the particle whose Voronoi cell contains that
 * vector. This is equivalent to finding the particle which is nearest to the
 * vector. Additional wall classes are not considered by this routine.
 * \param[in] (x,y) the vector to test.
 * \param[out] (rx,ry) the position of the particle whose Voronoi cell contains
 * 		       the vector. If the container is periodic, this may point
 * 		       to a particle in a periodic image of the primary domain.
 * \param[out] pid the ID of the particle.
 * \return True if a particle was found. If the container has no particles,
 * then the search will not find a Voronoi cell and false is returned. */
bool container_2d::find_voronoi_cell(double x,double y,double &rx,double &ry,int &pid) {
	int ai,aj,ci,cj,ij;
	particle_record_2d w;
	double mrs;

	// If the given vector lies outside the domain, but the container
	// is periodic, then remap it back into the domain
	if(!remap(ai,aj,ci,cj,x,y,ij)) return false;
	vc.find_voronoi_cell(x,y,ci,cj,ij,w,mrs);

	if(w.ij!=-1) {

		// Assemble the position vector of the particle to be returned,
		// applying a periodic remapping if necessary
		if(xperiodic) {ci+=w.di;if(ci<0||ci>=nx) ai+=step_div(ci,nx);}
		if(yperiodic) {cj+=w.dj;if(cj<0||cj>=ny) aj+=step_div(cj,ny);}
		rx=p[w.ij][2*w.l]+ai*(bx-ax);
		ry=p[w.ij][2*w.l+1]+aj*(by-ay);
		pid=id[w.ij][w.l];
		return true;
	}

	// If no particle if found then just return false
	return false;
}

/** Takes a vector and finds the particle whose Voronoi cell contains that
 * vector. Additional wall classes are not considered by this routine.
 * \param[in] (x,y) the vector to test.
 * \param[out] (rx,ry) the position of the particle whose Voronoi cell contains
 *		       the vector. If the container is periodic, this may point
 *		       to a particle in a periodic image of the primary domain.
 * \param[out] pid the ID of the particle.
 * \return True if a particle was found. If the container has no particles,
 * then the search will not find a Voronoi cell and false is returned. */
bool container_poly_2d::find_voronoi_cell(double x,double y,double &rx,double &ry,int &pid) {
	int ai,aj,ci,cj,ij;
	particle_record_2d w;
	double mrs;

	// If the given vector lies outside the domain, but the container
	// is periodic, then remap it back into the domain
	if(!remap(ai,aj,ci,cj,x,y,ij)) return false;
	vc.find_voronoi_cell(x,y,ci,cj,ij,w,mrs);

	if(w.ij!=-1) {

		// Assemble the position vector of the particle to be returned,
		// applying a periodic remapping if necessary
		if(xperiodic) {ci+=w.di;if(ci<0||ci>=nx) ai+=step_div(ci,nx);}
		if(yperiodic) {cj+=w.dj;if(cj<0||cj>=ny) aj+=step_div(cj,ny);}
		rx=p[w.ij][3*w.l]+ai*(bx-ax);
		ry=p[w.ij][3*w.l+1]+aj*(by-ay);
		pid=id[w.ij][w.l];
		return true;
	}

	// If no particle if found then just return false
	return false;
}

/** Increase memory for a particular region.
 * \param[in] i the index of the region to reallocate. */
void container_base_2d::add_particle_memory(int i) {
	int l,nmem=mem[i]<<1;

	// Carry out a check on the memory allocation size, and
	// print a status message if requested
	if(nmem>max_particle_memory_2d)
		voro_fatal_error("Absolute maximum memory allocation exceeded",VOROPP_MEMORY_ERROR);
#if VOROPP_VERBOSE >=3
	fprintf(stderr,"Particle memory in region %d scaled up to %d\n",i,nmem);
#endif

	// Allocate new memory and copy in the contents of the old arrays
	int *idp=new int[nmem];
	for(l=0;l<co[i];l++) idp[l]=id[i][l];
	double *pp=new double[ps*nmem];
	for(l=0;l<ps*co[i];l++) pp[l]=p[i][l];

	// Update pointers and delete old arrays
	mem[i]=nmem;
	delete [] id[i];id[i]=idp;
	delete [] p[i];p[i]=pp;
}

/** Import a list of particles from an open file stream into the container.
 * Entries of four numbers (Particle ID, x position, y position, z position)
 * are searched for. If the file cannot be successfully read, then the routine
 * causes a fatal error.
 * \param[in] fp the file handle to read from. */
void container_2d::import(FILE *fp) {
	int i,j;
	double x,y;
	while((j=fscanf(fp,"%d %lg %lg",&i,&x,&y))==3) put(i,x,y);
	if(j!=EOF) voro_fatal_error("File import error",VOROPP_FILE_ERROR);
}

/** Import a list of particles from an open file stream, also storing the order
 * of that the particles are read. Entries of four numbers (Particle ID, x
 * position, y position, z position) are searched for. If the file cannot be
 * successfully read, then the routine causes a fatal error.
 * \param[in,out] vo a reference to an ordering class to use.
 * \param[in] fp the file handle to read from. */
void container_2d::import(particle_order &vo,FILE *fp) {
	int i,j;
	double x,y;
	while((j=fscanf(fp,"%d %lg %lg",&i,&x,&y))==3) put(vo,i,x,y);
	if(j!=EOF) voro_fatal_error("File import error",VOROPP_FILE_ERROR);
}

/** Import a list of particles from an open file stream into the container.
 * Entries of five numbers (Particle ID, x position, y position, z position,
 * radius) are searched for. If the file cannot be successfully read, then the
 * routine causes a fatal error.
 * \param[in] fp the file handle to read from. */
void container_poly_2d::import(FILE *fp) {
	int i,j;
	double x,y,r;
	while((j=fscanf(fp,"%d %lg %lg %lg",&i,&x,&y,&r))==4) put(i,x,y,r);
	if(j!=EOF) voro_fatal_error("File import error",VOROPP_FILE_ERROR);
}

/** Import a list of particles from an open file stream, also storing the order
 * of that the particles are read. Entries of four numbers (Particle ID, x
 * position, y position, z position, radius) are searched for. If the file
 * cannot be successfully read, then the routine causes a fatal error.
 * \param[in,out] vo a reference to an ordering class to use.
 * \param[in] fp the file handle to read from. */
void container_poly_2d::import(particle_order &vo,FILE *fp) {
	int i,j;
	double x,y,r;
	while((j=fscanf(fp,"%d %lg %lg %lg",&i,&x,&y,&r))==4) put(vo,i,x,y,r);
	if(j!=EOF) voro_fatal_error("File import error",VOROPP_FILE_ERROR);
}

/** Outputs the a list of all the container regions along with the number of
 * particles stored within each. */
void container_base_2d::region_count() {
	int i,j,*cop=co;
	for(j=0;j<ny;j++) for(i=0;i<nx;i++)
		printf("Region (%d,%d): %d particles\n",i,j,*(cop++));
}

/** Clears a container of particles. */
void container_2d::clear() {
	for(int *cop=co;cop<co+nxy;cop++) *cop=0;
}

/** Clears a container of particles, also clearing resetting the maximum radius
 * to zero. */
void container_poly_2d::clear() {
	for(int *cop=co;cop<co+nxy;cop++) *cop=0;
	max_radius=0;
}

/** Computes all the Voronoi cells and saves customized information about them.
 * \param[in] format the custom output string to use.
 * \param[in] fp a file handle to write to. */
void container_2d::print_custom(const char *format,FILE *fp) {
	c_loop_all_2d vl(*this);
	print_custom(vl,format,fp);
}

/** Computes all the Voronoi cells and saves customized
 * information about them.
 * \param[in] format the custom output string to use.
 * \param[in] fp a file handle to write to. */
void container_poly_2d::print_custom(const char *format,FILE *fp) {
	c_loop_all_2d vl(*this);
	print_custom(vl,format,fp);
}

/** Computes all the Voronoi cells and saves customized information about them.
 * \param[in] format the custom output string to use.
 * \param[in] filename the name of the file to write to. */
void container_2d::print_custom(const char *format,const char *filename) {
	FILE *fp=safe_fopen(filename,"w");
	print_custom(format,fp);
	fclose(fp);
}

/** Computes all the Voronoi cells and saves customized
 * information about them
 * \param[in] format the custom output string to use.
 * \param[in] filename the name of the file to write to. */
void container_poly_2d::print_custom(const char *format,const char *filename) {
	FILE *fp=safe_fopen(filename,"w");
	print_custom(format,fp);
	fclose(fp);
}

/** Computes all of the Voronoi cells in the container, but does nothing
 * with the output. It is useful for measuring the pure computation time
 * of the Voronoi algorithm, without any additional calculations such as
 * volume evaluation or cell output. */
void container_2d::compute_all_cells() {
	voronoicell_2d c;
	c_loop_all_2d vl(*this);
	if(vl.start()) do compute_cell(c,vl);
	while(vl.inc());
}

/** Computes all of the Voronoi cells in the container, but does nothing
 * with the output. It is useful for measuring the pure computation time
 * of the Voronoi algorithm, without any additional calculations such as
 * volume evaluation or cell output. */
void container_poly_2d::compute_all_cells() {
	voronoicell_2d c;
	c_loop_all_2d vl(*this);
	if(vl.start()) do compute_cell(c,vl);while(vl.inc());
}

/** Calculates all of the Voronoi cells and sums their volumes. In most cases
 * without walls, the sum of the Voronoi cell volumes should equal the volume
 * of the container to numerical precision.
 * \return The sum of all of the computed Voronoi volumes. */
double container_2d::sum_cell_areas() {
	voronoicell_2d c;
	double area=0;
	c_loop_all_2d vl(*this);
	if(vl.start()) do if(compute_cell(c,vl)) area+=c.area();while(vl.inc());
	return area;
}

/** Calculates all of the Voronoi cells and sums their volumes. In most cases
 * without walls, the sum of the Voronoi cell volumes should equal the volume
 * of the container to numerical precision.
 * \return The sum of all of the computed Voronoi volumes. */
double container_poly_2d::sum_cell_areas() {
	voronoicell_2d c;
	double area=0;
	c_loop_all_2d vl(*this);
	if(vl.start()) do if(compute_cell(c,vl)) area+=c.area();while(vl.inc());
	return area;
}

/** This function tests to see if a given vector lies within the container
 * bounds and any walls.
 * \param[in] (x,y) the position vector to be tested.
 * \return True if the point is inside the container, false if the point is
 *         outside. */
bool container_base_2d::point_inside(double x,double y) {
	if(x<ax||x>bx||y<ay||y>by) return false;
	return point_inside_walls(x,y);
}

/** Draws an outline of the domain in gnuplot format.
 * \param[in] fp the file handle to write to. */
void container_base_2d::draw_domain_gnuplot(FILE *fp) {
	fprintf(fp,"%g %g\n%g %g\n%g %g\n%g %g\n%g %g\n",ax,ay,bx,ay,bx,by,ax,by,ax,ay);
}

/** Draws an outline of the domain in POV-Ray format.
 * \param[in] fp the file handle to write to. */
void container_base_2d::draw_domain_pov(FILE *fp) {
	fprintf(fp,"cylinder{<%g,%g,0>,<%g,%g,0>,rr}\n"
		   "cylinder{<%g,%g,0>,<%g,%g,0>,rr}\n",ax,ay,bx,ay,ax,by,bx,by);
	fprintf(fp,"cylinder{<%g,%g,0>,<%g,%g,0>,rr}\n"
		   "cylinder{<%g,%g,0>,<%g,%g,0>,rr}\n",ax,ay,ax,by,bx,ay,bx,by);
	fprintf(fp,"sphere{<%g,%g,0>,rr}\nsphere{<%g,%g,0>,rr}\n"
		   "sphere{<%g,%g,0>,rr}\nsphere{<%g,%g,0>,rr}\n",ax,ay,bx,ay,ax,by,bx,by);
}

/** The wall_list constructor sets up an array of pointers to wall classes. */
wall_list_2d::wall_list_2d() : walls(new wall_2d*[init_wall_size]), wep(walls), wel(walls+init_wall_size),
	current_wall_size(init_wall_size) {}

/** The wall_list destructor frees the array of pointers to the wall classes.
 */
wall_list_2d::~wall_list_2d() {
	delete [] walls;
}

/** Adds all of the walls on another wall_list to this class.
 * \param[in] wl a reference to the wall class. */
void wall_list_2d::add_wall(wall_list_2d &wl) {
	for(wall_2d **wp=wl.walls;wp<wl.wep;wp++) add_wall(*wp);
}

/** Deallocates all of the wall classes pointed to by the wall_list. */
void wall_list_2d::deallocate() {
	for(wall_2d **wp=walls;wp<wep;wp++) delete *wp;
}

/** Increases the memory allocation for the walls array. */
void wall_list_2d::increase_wall_memory() {
	current_wall_size<<=1;
	if(current_wall_size>max_wall_size)
		voro_fatal_error("Wall memory allocation exceeded absolute maximum",VOROPP_MEMORY_ERROR);
	wall_2d **nwalls=new wall_2d*[current_wall_size],**nwp=nwalls,**wp=walls;
	while(wp<wep) *(nwp++)=*(wp++);
	delete [] walls;
	walls=nwalls;wel=walls+current_wall_size;wep=nwp;
}

}
