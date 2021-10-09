// Voro++, a cell-based Voronoi library
//
// Authors  : Chris H. Rycroft (LBL / UC Berkeley)
//            Cody Robert Dance (UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

/** \file ctr_boundary_2d.cc
 * \brief Function implementations for the ctr_boundary_2d and related classes. */

#include <cstring>

#include "ctr_boundary_2d.hh"

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
container_boundary_2d::container_boundary_2d(double ax_,double bx_,double ay_,double by_,
		int nx_,int ny_,bool xperiodic_,bool yperiodic_,int init_mem)
	: voro_base_2d(nx_,ny_,(bx_-ax_)/nx_,(by_-ay_)/ny_),
	ax(ax_), bx(bx_), ay(ay_), by(by_), xperiodic(xperiodic_), yperiodic(yperiodic_),
	id(new int*[nxy]), p(new double*[nxy]), co(new int[nxy]), mem(new int[nxy]),
	wid(new int*[nxy]), nlab(new int*[nxy]), plab(new int**[nxy]), bndpts(new int*[nxy]),
	boundary_track(-1), edbc(0), edbm(init_boundary_size),
	edb(new int[2*edbm]), bnds(new double[2*edbm]), ps(2), soi(NULL),
	vc(*this,xperiodic_?2*nx_+1:nx_,yperiodic_?2*ny_+1:ny_)	{
	int l;
//        totpar=0;
	for(l=0;l<nxy;l++) co[l]=0;
	for(l=0;l<nxy;l++) mem[l]=init_mem;
	for(l=0;l<nxy;l++) id[l]=new int[init_mem];
	for(l=0;l<nxy;l++) p[l]=new double[ps*init_mem];
	for(l=0;l<nxy;l++) nlab[l]=new int[init_mem];
	for(l=0;l<nxy;l++) plab[l]=new int*[init_mem];
	for(l=0;l<nxy;l++) bndpts[l]=new int[init_mem];

	for(l=0;l<nxy;l++) {wid[l]=new int[init_wall_tag_size+2];*(wid[l])=0;wid[l][1]=init_wall_tag_size;}
}

/** The container destructor frees the dynamically allocated memory. */
container_boundary_2d::~container_boundary_2d() {
	int l;

	// Clear "sphere of influence" array if it has been allocated
	if(soi!=NULL) delete [] soi;

	// Deallocate the block-level arrays
	for(l=nxy-1;l>=0;l--) delete [] wid[l];
	for(l=nxy-1;l>=0;l--) delete [] bndpts[l];
	for(l=nxy-1;l>=0;l--) delete [] plab[l];
	for(l=nxy-1;l>=0;l--) delete [] nlab[l];
	for(l=nxy-1;l>=0;l--) delete [] p[l];
	for(l=nxy-1;l>=0;l--) delete [] id[l];

	// Delete the two-dimensional arrays
	delete [] id;
	delete [] p;
	delete [] co;
	delete [] mem;
}

/** Put a particle into the correct region of the container.
 * \param[in] n the numerical ID of the inserted particle.
 * \param[in] (x,y) the position vector of the inserted particle. */
void container_boundary_2d::put(int n,double x,double y) {
	int ij;
	if(put_locate_block(ij,x,y)) {
		//totpar++;
		id[ij][co[ij]]=n;
		if(boundary_track!=-1) {
			bndpts[ij][co[ij]]=edbc;
			register_boundary(x,y);
		} else bndpts[ij][co[ij]]=-1;
		double *pp=p[ij]+2*co[ij]++;
		*(pp++)=x;*pp=y;
	}
}

/** Put a particle into the correct region of the container, also recording
 * into which region it was stored.
 * \param[in] vo the ordering class in which to record the region.
 * \param[in] n the numerical ID of the inserted particle.
 * \param[in] (x,y) the position vector of the inserted particle. */
void container_boundary_2d::put(particle_order &vo,int n,double x,double y) {
	int ij;
	if(put_locate_block(ij,x,y)) {
		//totpar++;
		id[ij][co[ij]]=n;
		if(boundary_track!=-1) {
			bndpts[ij][co[ij]]=edbc;
			register_boundary(x,y);
		} else bndpts[ij][co[ij]]=-1;
		vo.add(ij,co[ij]);
		double *pp=p[ij]+2*co[ij]++;
		*(pp++)=x;*pp=y;
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
inline bool container_boundary_2d::put_locate_block(int &ij,double &x,double &y) {
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
inline bool container_boundary_2d::put_remap(int &ij,double &x,double &y) {
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

/** Increase memory for a particular region.
 * \param[in] i the index of the region to reallocate. */
void container_boundary_2d::add_particle_memory(int i) {
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
	int *nlabp=new int[nmem];
	for(l=0;l<co[i];l++) nlabp[l]=nlab[i][l];
	int **plabp=new int*[nmem];
	for(l=0;l<co[i];l++) plabp[l]=plab[i][l];
	int *bndptsp=new int[nmem];
	for(l=0;l<co[i];l++) bndptsp[l]=bndpts[i][l];

	// Update pointers and delete old arrays
	mem[i]=nmem;
	delete [] id[i];id[i]=idp;
	delete [] p[i];p[i]=pp;
	delete [] nlab[i];nlab[i]=nlabp;
	delete [] plab[i];plab[i]=plabp;
	delete [] bndpts[i];bndpts[i]=bndptsp;
}

/** Outputs the a list of all the container regions along with the number of
 * particles stored within each. */
void container_boundary_2d::region_count() {
	int i,j,*cop=co;
	for(j=0;j<ny;j++) for(i=0;i<nx;i++)
		printf("Region (%d,%d): %d particles\n",i,j,*(cop++));
}

/** Clears a container of particles. */
void container_boundary_2d::clear() {
	for(int *cop=co;cop<co+nxy;cop++) *cop=0;
}

/** Computes all the Voronoi cells and saves customized information about them.
 * \param[in] format the custom output string to use.
 * \param[in] fp a file handle to write to. */
void container_boundary_2d::print_custom(const char *format,FILE *fp) {
	c_loop_all_2d vl(*this);
	print_custom(vl,format,fp);
}

/** Computes all the Voronoi cells and saves customized information about them.
 * \param[in] format the custom output string to use.
 * \param[in] filename the name of the file to write to. */
void container_boundary_2d::print_custom(const char *format,const char *filename) {
	FILE *fp=safe_fopen(filename,"w");
	print_custom(format,fp);
	fclose(fp);
}

/** Computes all of the Voronoi cells in the container, but does nothing
 * with the output. It is useful for measuring the pure computation time
 * of the Voronoi algorithm, without any additional calculations such as
 * volume evaluation or cell output. */
void container_boundary_2d::compute_all_cells() {
	voronoicell_nonconvex_2d c;
	c_loop_all_2d vl(*this);
	if(vl.start()) do compute_cell(c,vl);
	while(vl.inc());
}

/** Calculates all of the Voronoi cells and sums their volumes. In most cases
 * without walls, the sum of the Voronoi cell volumes should equal the volume
 * of the container to numerical precision.
 * \return The sum of all of the computed Voronoi volumes. */
double container_boundary_2d::sum_cell_areas() {
	voronoicell_nonconvex_2d c;
	double area=0;
	c_loop_all_2d vl(*this);
	if(vl.start()) do if(compute_cell(c,vl)) area+=c.area();while(vl.inc());
	return area;
}

/** Draws an outline of the domain in gnuplot format.
 * \param[in] fp the file handle to write to. */
void container_boundary_2d::draw_domain_gnuplot(FILE *fp) {
	fprintf(fp,"%g %g\n%g %g\n%g %g\n%g %g\n%g %g\n",ax,ay,bx,ay,bx,by,ax,by,ax,ay);
}

/** Draws an outline of the domain in POV-Ray format.
 * \param[in] fp the file handle to write to. */
void container_boundary_2d::draw_domain_pov(FILE *fp) {
	fprintf(fp,"cylinder{<%g,%g,0>,<%g,%g,0>,rr}\n"
		   "cylinder{<%g,%g,0>,<%g,%g,0>,rr}\n",ax,ay,bx,ay,ax,by,bx,by);
	fprintf(fp,"cylinder{<%g,%g,0>,<%g,%g,0>,rr}\n"
		   "cylinder{<%g,%g,0>,<%g,%g,0>,rr}\n",ax,ay,ax,by,bx,ay,bx,by);
	fprintf(fp,"sphere{<%g,%g,0>,rr}\nsphere{<%g,%g,0>,rr}\n"
		   "sphere{<%g,%g,0>,rr}\nsphere{<%g,%g,0>,rr}\n",ax,ay,bx,ay,ax,by,bx,by);
}

/** This does the additional set-up for non-convex containers. We assume that
 * **p, **id, *co, *mem, *bnds, and edbc have already been setup. We then
 * proceed to setup **wid, *soi, and THE PROBLEM POINTS BOOLEAN ARRAY.
 * This algorithm keeps the importing seperate from the set-up */
void container_boundary_2d::setup(){
//	double lx,ly;//last (x,y)
	double cx,cy,nx,ny;//current (x,y),next (x,y)
	int widl=1,maxwid=1,fwid=1,nwid;//lwid;
//	bool first=true;

	tmp=tmpp=new int[3*init_temp_label_size];
	tmpe=tmp+3*init_temp_label_size;

	while(widl!=edbc){
		cx=bnds[2*widl];cy=bnds[2*widl+1];
		nwid=edb[2*widl];//lwid=edb[2*widl+1];
		//lx=bnds[lwid*2];ly=bnds[lwid*2+1];
		nx=bnds[2*nwid];ny=bnds[2*nwid+1];

		tag_walls(cx,cy,nx,ny,widl);
		semi_circle_labeling(cx,cy,nx,ny,widl);

		//make sure that the cos(angle)>1 and the angle points inward
		//probpts=(lx-cx)*(nx-cx)+(ly-cy)*(ny-cy)>tolerance &&
		//	cross_product(lx-cx,ly-cy,nx-cx,ny-cy);

		widl=edb[2*widl];
		if(widl>maxwid) maxwid=widl;
		if(widl==fwid){
			widl=maxwid+1;
			fwid=widl;
			maxwid++;
		//	first=false;
		}
	}

	// The temporary array can now be used to set up the label table
	create_label_table();

	// Remove temporary array
	delete [] tmp;
}

/** Given two points, tags all the computational boxes that the line segment
 * specified by the two points
 * goes through. param[in] (x1,y1) this is one point
 * \param[in] (x2,y2) this is the other point.
 * \param[in] wid this is the wall id bnds[2*wid] is the x index of the first
 *                vertex in the c-c direction. */
void container_boundary_2d::tag_walls(double x1,double y1,double x2,double y2,int wid_) {

	// Find which boxes these points are within
	int i1=int((x1-ax)*xsp),j1=int((y1-ay)*ysp);
	int i2=int((x2-ax)*xsp),j2=int((y2-ay)*ysp),k,ij;

	// Swap to ensure that i1 is smaller than i2
	double q,yfac;
	if(i2<i1) {
		q=x1;x1=x2;x2=q;q=y1;y1=y2;y2=q;
		k=i1;i1=i2;i2=k;k=j1;j1=j2;j2=k;
	}

	ij=i1+j1*nx;
	if(j1<j2) {
		yfac=1/(y2-y1);
		do {
			j1++;
			q=ay+j1*boxy;
			k=int((((q-y1)*x2+x1*(y2-q))*yfac-ax)*xsp);
			if(k>=nx) k=nx-1;
			tag_line(ij,(j1-1)*nx+k,wid_);
			ij+=nx;
		} while(j1<j2);
	} else if(j1>j2) {
		yfac=1/(y2-y1);
		do {
			q=ay+j1*boxy;
			k=int((((q-y1)*x2+x1*(y2-q))*yfac-ax)*xsp);
			if(k>=nx) k=nx-1;
			tag_line(ij,j1*nx+k,wid_);
			ij-=nx;
			j1--;
		} while(j1>j2);
	}
	tag_line(ij,i2+j2*nx,wid_);
}

void container_boundary_2d::tag_line(int &ij,int ije,int wid_) {
	tag(ij,wid_);
	while(ij<ije) {
		ij++;
		tag(ij,wid_);
	}
}

inline void container_boundary_2d::tag(int ij,int wid_) {
	int *&wp(wid[ij]);
	if(*wp==wp[1]) {
		int nws=wp[1]<<1;
		if(nws>max_wall_tag_size) voro_fatal_error("Maximum wall tag memory exceeded",VOROPP_MEMORY_ERROR);
		int *np=new int[nws+2];
		*np=*wp;np[1]=nws;
		for(int i=2;i<*wp+2;i++) np[i]=wp[i];
		delete [] wp;
		wp=np;
	}
	wp[2+(*wp)++]=wid_;
}

/* Tags particles that are within a semicircle (on the appropriate side) of a
 * boundary.
 * \param[in] (x1,y1) the start point of the wall segment, arranged so that it
 *                    is the first point reached in the counter-clockwise
 *                    direction.
 * \param[in] (x2,y2) the end points of the wall segment. */
void container_boundary_2d::semi_circle_labeling(double x1,double y1,double x2,double y2,int bid) {

	double radius=sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))*0.5,
	       midx=(x1+x2)*0.5,midy=(y1+y2)*0.5,cpx,cpy;
	int ai=int((midx-radius-ax)*xsp),
	    bi=int((midx+radius-ax)*xsp),
	    aj=int((midy-radius-ay)*ysp),
	    bj=int((midy+radius-ay)*ysp),i,j,ij,k;
	if(ai<0) ai=0;
	if(ai>=nx) ai=nx-1;
	if(bi<0) bi=0;
	if(bi>=nx) bi=nx-1;
	if(aj<0) aj=0;
	if(aj>=ny) aj=ny-1;
	if(bj<0) bj=0;
	if(bj>=ny) bj=ny-1;

	// Now loop through all the particles in the boxes we found, tagging
	// the ones that are within radius of (midx,midy) and are on the
	// appropriate side of the wall
	for(j=aj;j<=bj;j++) for(i=ai;i<=bi;i++) {
		ij=i+nx*j;
		for(k=0;k<co[ij];k++) {
			cpx=p[ij][2*k];
			cpy=p[ij][2*k+1];
			if((midx-cpx)*(midx-cpx)+(midy-cpy)*(midy-cpy)<=radius*radius&&
			cross_product((x1-x2),(y1-y2),(cpx-x2),(cpy-y2))&&
			(cpx!=x1||cpy==y1)&&(cpx!=x2||cpy!=y2)) {

				if(tmpp==tmpe) add_temporary_label_memory();
				*(tmpp++)=ij;
				*(tmpp++)=k;
				*(tmpp++)=bid;
			}
		}
	}
}

void container_boundary_2d::create_label_table() {
	int ij,q,*pp,tlab=0;

	// Clear label counters
	for(ij=0;ij<nxy;ij++) for(q=0;q<co[ij];q++) nlab[ij][q]=0;

	// Increment label counters
	for(pp=tmp;pp<tmpp;pp+=3) {nlab[*pp][pp[1]]++;tlab++;}

	// Check for case of no labels at all (which may be common)
	if(tlab==0) {
#if VOROPP_VERBOSE >=2
		fputs("No labels needed\n",stderr);
#endif
		return;
	}

	// If there was already a table from a previous call, remove it
	if(soi!=NULL) delete [] soi;

	// Allocate the label array, and set up pointers from each particle
	// to the corresponding location
	pp=soi=new int[tlab];
	for(ij=0;ij<nxy;ij++) for(q=0;q<co[ij];pp+=nlab[ij][q++]) plab[ij][q]=pp;

	// Fill in the label entries
	for(pp=tmp;pp<tmpp;pp+=3) *(plab[*pp][pp[1]]++)=pp[2];

	// Reset the label pointers
	pp=soi;
	for(ij=0;ij<nxy;ij++) for(q=0;q<co[ij];pp+=nlab[ij][q++]) plab[ij][q]=pp;
}

/** Draws the boundaries. (Note: this currently assumes that each boundary loop
 * is a continuous block in the bnds array, which will be true for the import
 * function. However, it may not be true in other cases, in which case this
 * routine would have to be extended.) */
void container_boundary_2d::draw_boundary_gnuplot(FILE *fp) {
	int i;

	for(i=0;i<edbc;i++) {
		fprintf(fp,"%g %g\n",bnds[2*i],bnds[2*i+1]);

		// If a loop is detected, than complete the loop in the output file
		// and insert a newline
		if(edb[2*i]<i) fprintf(fp,"%g %g\n\n",bnds[2*edb[2*i]],bnds[2*edb[2*i]+1]);
	}
}

bool container_boundary_2d::point_inside(double x,double y) {
	int i=0,j=0,k=0;
	bool sleft,left,nleft;

	while(i<edbc) {
		sleft=left=bnds[2*i]<x;
		do {
			j=edb[2*i];
			nleft=j<i?sleft:bnds[2*j]<x;
			if(nleft!=left) {
				if(left) {
					if(bnds[2*j+1]*(x-bnds[2*i])+bnds[2*i+1]*(bnds[2*j]-x)<y*(bnds[2*j]-bnds[2*i])) k++;
				} else {
					if(bnds[2*j+1]*(x-bnds[2*i])+bnds[2*i+1]*(bnds[2*j]-x)>y*(bnds[2*j]-bnds[2*i])) k--;
				}
			}
			left=nleft;
			i++;
		} while(j==i);
	}

#if VOROPP_VERBOSE >=2
	if(k<0) fprintf(stderr,"Negative winding number of %d for (%g,%g)\n",k,x,y);
	else if(k>1) fprintf(stderr,"Winding number of %d for (%g,%g)\n",k,x,y);
#endif
	return k>0;
}

template<class v_cell_2d>
bool container_boundary_2d::boundary_cuts(v_cell_2d &c,int ij,double x,double y) {
	int i,j,k;
	double lx,ly,dx,dy,dr;
	for(i=2;i<*(wid[ij])+2;i++) {
		j=2*wid[ij][i];k=2*edb[j];
		dx=bnds[k]-bnds[j];dy=bnds[k+1]-bnds[j+1];
		dr=dy*(bnds[j]-x)-dx*(bnds[j+1]-y);
		if(dr<tolerance) continue;
		lx=bnds[j]+bnds[k]-2*x;
		ly=bnds[j+1]+bnds[k+1]-2*y;
		if(lx*lx+ly*ly>dx*dx+dy*dy) continue;
		if(!c.plane(dy,-dx,2*dr)) return false;
	}
	return true;
}

bool container_boundary_2d::skip(int ij,int q,double x,double y) {
	int j;
	double wx1,wy1,wx2,wy2,dx,dy,lx,ly;
	double cx=p[ij][ps*q],cy=p[ij][ps*q+1];

	for(int i=0;i<nlab[ij][q];i++) {
		j=2*plab[ij][q][i];
		wx1=bnds[j];
		wy1=bnds[j+1];
		j=2*edb[j];
		wx2=bnds[j];
		wy2=bnds[j+1];
		dx=wx1-wx2;
		dy=wy1-wy2;
		wx1-=x;wy1-=y;
		if(dx*wy1-dy*wx1>tolerance) {
			lx=cx-x;ly=cy-y;
			if(wx1*ly-wy1*lx>tolerance&&lx*(wy2-y)-ly*(wx2-x)>tolerance) return true;
		}
	}
	return false;

}

/** Imports a list of particles from an input stream.
 * \param[in] fp a file handle to read from. */
void container_boundary_2d::import(FILE *fp) {
	int i;
	double x,y;
	char *buf(new char[512]);

	while(fgets(buf,512,fp)!=NULL) {
		if(strcmp(buf,"#Start\n")==0||strcmp(buf,"# Start\n")==0) {

			// Check that two consecutive start tokens haven't been
			// encountered
			if(boundary_track!=-1) voro_fatal_error("File import error - two consecutive start tokens found",VOROPP_FILE_ERROR);
			start_boundary();

		} else if(strcmp(buf,"#End\n")==0||strcmp(buf,"# End\n")==0||
			  strcmp(buf,"#End")==0||strcmp(buf,"# End")==0) {

			// Check that two consecutive end tokens haven't been
			// encountered
			if(boundary_track==-1) voro_fatal_error("File import error - found end token without start token",VOROPP_FILE_ERROR);
			end_boundary();
		} else {

			// Try and read three entries from the line
			if(sscanf(buf,"%d %lg %lg",&i,&x,&y)!=3) voro_fatal_error("File import error - can't parse particle information",VOROPP_FILE_ERROR);
			put(i,x,y);
		}
	}
	if(boundary_track!=-1) voro_fatal_error("File import error - end of file reached without finding end token",VOROPP_FILE_ERROR);

	if(!feof(fp)) voro_fatal_error("File import error - error reading string from file",VOROPP_FILE_ERROR);
	delete [] buf;
}

void container_boundary_2d::end_boundary() {
	if(boundary_track!=edbc) {
		edb[2*boundary_track+1]=edbc-1;
		edb[2*(edbc-1)]=boundary_track;
	}
	boundary_track=-1;
}

void container_boundary_2d::register_boundary(double x,double y) {
	if(edbc==edbm) add_boundary_memory();
	if(edbc!=boundary_track) {
		edb[2*edbc-2]=edbc;
		edb[2*edbc+1]=edbc-1;
	}
	bnds[2*edbc]=x;
	bnds[2*(edbc++)+1]=y;
}

/** Increases the size of the temporary label memory. */
void container_boundary_2d::add_temporary_label_memory() {
	int size(tmpe-tmp);
	size<<=1;
	if(size>3*max_temp_label_size)
		voro_fatal_error("Absolute temporary label memory allocation exceeded",VOROPP_MEMORY_ERROR);
#if VOROPP_VERBOSE >=3
	fprintf(stderr,"Temporary label memory in region scaled up to %d\n",size);
#endif
	int *ntmp(new int[size]),*tp(tmp);tmpp=ntmp;
	while(tp<tmpe) *(tmpp++)=*(tp++);
	delete [] tmp;
	tmp=ntmp;tmpe=tmp+size;
}

/** Increases the memory allocation for the boundary points. */
void container_boundary_2d::add_boundary_memory() {
	int i;
	edbm<<=1;
	if(edbm>max_boundary_size)
		voro_fatal_error("Absolute boundary memory allocation exceeded",VOROPP_MEMORY_ERROR);
#if VOROPP_VERBOSE >=3
	fprintf(stderr,"Boundary memory scaled up to %d\n",size);
#endif

	// Reallocate the boundary vertex information
	double *nbnds(new double[2*edbm]);
	for(i=0;i<2*edbc;i++) nbnds[i]=bnds[i];
	delete [] nbnds;bnds=nbnds;

	// Reallocate the edge information
	int *nedb(new int[2*edbm]);
	for(i=0;i<2*edbc;i++) nedb[i]=edb[i];
	delete [] edb;edb=nedb;
}

// Explicit instantiation
template bool container_boundary_2d::boundary_cuts(voronoicell_nonconvex_2d&,int,double,double);
template bool container_boundary_2d::boundary_cuts(voronoicell_nonconvex_neighbor_2d&,int,double,double);

}
