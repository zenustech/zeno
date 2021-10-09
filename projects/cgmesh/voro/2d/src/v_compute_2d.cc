// Voro++, a 3D cell-based Voronoi library
//
// Author   : Chris H. Rycroft (LBL / UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

/** \file v_compute_2d.cc
 * \brief Function implementantions for the 2D voro_compute class. */

#include "worklist_2d.hh"
#include "v_compute_2d.hh"
#include "rad_option.hh"
#include "container_2d.hh"
#include "ctr_boundary_2d.hh"

namespace voro {

/** The class constructor initializes constants from the container class, and
 * sets up the mask and queue used for Voronoi computations.
 * \param[in] con_ a reference to the container class to use.
 * \param[in] (hx_,hy_) the size of the mask to use. */
template<class c_class_2d>
voro_compute_2d<c_class_2d>::voro_compute_2d(c_class_2d &con_,int hx_,int hy_) :
	con(con_), boxx(con_.boxx), boxy(con_.boxy), xsp(con_.xsp),
	ysp(con_.ysp), hx(hx_), hy(hy_), hxy(hx_*hy_), ps(con_.ps),
	id(con_.id), p(con_.p), co(con_.co), bxsq(boxx*boxx+boxy*boxy),
	mv(0), qu_size(2*(2+hx+hy)), wl(con_.wl), mrad(con_.mrad),
	mask(new unsigned int[hxy]), qu(new int[qu_size]), qu_l(qu+qu_size) {
	reset_mask();
}

/** Scans all of the particles within a block to see if any of them have a
 * smaller distance to the given test vector. If one is found, the routine
 * updates the minimum distance and store information about this particle.
 * \param[in] ij the index of the block.
 * \param[in] (x,y) the test vector to consider (which may have already had a
 *                    periodic displacement applied to it).
 * \param[in] (di,dj) the coordinates of the current block, to store if the
 *		      particle record is updated.
 * \param[in,out] w a reference to a particle record in which to store
 *		    information about the particle whose Voronoi cell the
 *		    vector is within.
 * \param[in,out] mrs the current minimum distance, that may be updated if a
 * 		      closer particle is found. */
template<class c_class_2d>
inline void voro_compute_2d<c_class_2d>::scan_all(int ij,double x,double y,int di,int dj,particle_record_2d &w,double &mrs) {
	double x1,y1,rs;bool in_block=false;
	for(int l=0;l<co[ij];l++) {
		x1=p[ij][ps*l]-x;
		y1=p[ij][ps*l+1]-y;
		rs=con.r_current_sub(x1*x1+y1*y1,ij,l);
		if(rs<mrs) {mrs=rs;w.l=l;in_block=true;}
	}
	if(in_block) {w.ij=ij;w.di=di;w.dj=dj;}
}

/** Finds the Voronoi cell that given vector is within. For containers that are
 * not radially dependent, this corresponds to findig the particle that is
 * closest to the vector; for the radical tessellation containers, this
 * corresponds to a finding the minimum weighted distance.
 * \param[in] (x,y) the vector to consider.
 * \param[in] (ci,cj) the coordinates of the block that the test particle is
 *                       in relative to the container data structure.
 * \param[in] ij the index of the block that the test particle is in.
 * \param[out] w a reference to a particle record in which to store information
 * 		 about the particle whose Voronoi cell the vector is within.
 * \param[out] mrs the minimum computed distance. */
template<class c_class_2d>
void voro_compute_2d<c_class_2d>::find_voronoi_cell(double x,double y,int ci,int cj,int ij,particle_record_2d &w,double &mrs) {
	double qx=0,qy=0,rs;
	int i,j,di,dj,ei,ej,f,g,disp;
	double fx,fy,mxs,mys,*radp;
	unsigned int q,*e,*mij;

	// Init setup for parameters to return
	w.ij=-1;mrs=large_number;

	con.initialize_search(ci,cj,ij,i,j,disp);

	// Test all particles in the particle's local region first
	scan_all(ij,x,y,0,0,w,mrs);

	// Now compute the fractional position of the particle within its
	// region and store it in (fx,fy). We use this to compute an index
	// (di,dj) of which subregion the particle is within.
	unsigned int m1,m2;
	con.frac_pos(x,y,ci,cj,fx,fy);
	di=int(fx*xsp*wl_fgrid_2d);dj=int(fy*ysp*wl_fgrid_2d);

	// The indices (di,dj) tell us which worklist to use, to test the
	// blocks in the optimal order. But we only store worklists for the
	// eighth of the region where di, dj, and dk are all less than half the
	// full grid. The rest of the cases are handled by symmetry. In this
	// section, we detect for these cases, by reflecting high values of di,
	// dj. For these cases, a mask is constructed in m1 and m2
	// which is used to flip the worklist information when it is loaded.
	if(di>=wl_hgrid_2d) {
		mxs=boxx-fx;
		m1=127+(3<<21);m2=1+(1<<21);di=wl_fgrid_2d-1-di;if(di<0) di=0;
	} else {m1=m2=0;mxs=fx;}
	if(dj>=wl_hgrid_2d) {
		mys=boxy-fy;
		m1|=(127<<7)+(3<<24);m2|=(1<<7)+(1<<24);dj=wl_fgrid_2d-1-dj;if(dj<0) dj=0;
	} else mys=fy;

	// Do a quick test to account for the case when the minimum radius is
	// small enought that no other blocks need to be considered
	rs=con.r_max_add(mrs);
	if(mxs*mxs>rs&&mys*mys>rs) return;

	// Now compute which worklist we are going to use, and set radp and e to
	// point at the right offsets
	ij=di+wl_hgrid_2d*dj;
	radp=mrad+ij*wl_seq_length_2d;
	e=(const_cast<unsigned int*> (wl))+ij*wl_seq_length_2d;

	// Read in how many items in the worklist can be tested without having to
	// worry about writing to the mask
	f=e[0];g=0;
	do {

		// If mrs is less than the minimum distance to any untested
		// block, then we are done
		if(con.r_max_add(mrs)<radp[g]) return;
		g++;

		// Load in a block off the worklist, permute it with the
		// symmetry mask, and decode its position. These are all
		// integer bit operations so they should run very fast.
		q=e[g];q^=m1;q+=m2;
		di=q&127;di-=64;
		dj=(q>>7)&127;dj-=64;

		// Check that the worklist position is in range
		ei=di+i;if(ei<0||ei>=hx) continue;
		ej=dj+j;if(ej<0||ej>=hy) continue;

		// Call the compute_min_max_radius() function. This returns
		// true if the minimum distance to the block is bigger than the
		// current mrs, in which case we skip this block and move on.
		// Otherwise, it computes the maximum distance to the block and
		// returns it in crs.
		if(compute_min_radius(di,dj,fx,fy,mrs)) continue;

		// Now compute which region we are going to loop over, adding a
		// displacement for the periodic cases
		ij=con.region_index(ci,cj,ei,ej,qx,qy,disp);

		// If mrs is bigger than the maximum distance to the block,
		// then we have to test all particles in the block for
		// intersections. Otherwise, we do additional checks and skip
		// those particles which can't possibly intersect the block.
		scan_all(ij,x-qx,y-qy,di,dj,w,mrs);
	} while(g<f);

	// Update mask value and initialize queue
	mv++;
	if(mv==0) {reset_mask();mv=1;}
	int *qu_s=qu,*qu_e=qu;

	while(g<wl_seq_length_2d-1) {

		// If mrs is less than the minimum distance to any untested
		// block, then we are done
		if(con.r_max_add(mrs)<radp[g]) return;
		g++;

		// Load in a block off the worklist, permute it with the
		// symmetry mask, and decode its position. These are all
		// integer bit operations so they should run very fast.
		q=e[g];q^=m1;q+=m2;
		di=q&127;di-=64;
		dj=(q>>7)&127;dj-=64;

		// Compute the position in the mask of the current block. If
		// this lies outside the mask, then skip it. Otherwise, mark
		// it.
		ei=di+i;if(ei<0||ei>=hx) continue;
		ej=dj+j;if(ej<0||ej>=hy) continue;
		mij=mask+ei+hx*ej;
		*mij=mv;

		// Skip this block if it is further away than the current
		// minimum radius
		if(compute_min_radius(di,dj,fx,fy,mrs)) continue;

		// Now compute which region we are going to loop over, adding a
		// displacement for the periodic cases
		ij=con.region_index(ci,cj,ei,ej,qx,qy,disp);
		scan_all(ij,x-qx,y-qy,di,dj,w,mrs);

		if(qu_e>qu_l-8) add_list_memory(qu_s,qu_e);
		scan_bits_mask_add(q,mij,ei,ej,qu_e);
	}

	// Do a check to see if we've reached the radius cutoff
	if(con.r_max_add(mrs)<radp[g]) return;

	// We were unable to completely compute the cell based on the blocks in
	// the worklist, so now we have to go block by block, reading in items
	// off the list
	while(qu_s!=qu_e) {

		// Read the next entry of the queue
		if(qu_s==qu_l) qu_s=qu;
		ei=*(qu_s++);ej=*(qu_s++);
		di=ei-i;dj=ej-j;
		if(compute_min_radius(di,dj,fx,fy,mrs)) continue;

		ij=con.region_index(ci,cj,ei,ej,qx,qy,disp);

		scan_all(ij,x-qx,y-qy,di,dj,w,mrs);

		// Test the neighbors of the current block, and add them to the
		// block list if they haven't already been tested
		if((qu_s<=qu_e?(qu_l-qu_e)+(qu_s-qu):qu_s-qu_e)<18) add_list_memory(qu_s,qu_e);
		add_to_mask(ei,ej,qu_e);
	}
}

/** Scans the six orthogonal neighbors of a given block and adds them to the
 * queue if they haven't been considered already. It assumes that the queue
 * will definitely have enough memory to add six entries at the end.
 * \param[in] (ei,ej,ek) the block to consider.
 * \param[in,out] qu_e a pointer to the end of the queue. */
template<class c_class_2d>
inline void voro_compute_2d<c_class_2d>::add_to_mask(int ei,int ej,int *&qu_e) {
	unsigned int *mij=mask+ei+hx*ej;
	if(ej>0) if(*(mij-hx)!=mv) {if(qu_e==qu_l) qu_e=qu;*(mij-hx)=mv;*(qu_e++)=ei;*(qu_e++)=ej-1;}
	if(ei>0) if(*(mij-1)!=mv) {if(qu_e==qu_l) qu_e=qu;*(mij-1)=mv;*(qu_e++)=ei-1;*(qu_e++)=ej;}
	if(ei<hx-1) if(*(mij+1)!=mv) {if(qu_e==qu_l) qu_e=qu;*(mij+1)=mv;*(qu_e++)=ei+1;*(qu_e++)=ej;}
	if(ej<hy-1) if(*(mij+hx)!=mv) {if(qu_e==qu_l) qu_e=qu;*(mij+hx)=mv;*(qu_e++)=ei;*(qu_e++)=ej+1;}
}

/** Scans a worklist entry and adds any blocks to the queue
 * \param[in] (ei,ej,ek) the block to consider.
 * \param[in,out] qu_e a pointer to the end of the queue. */
template<class c_class_2d>
inline void voro_compute_2d<c_class_2d>::scan_bits_mask_add(unsigned int q,unsigned int *mij,int ei,int ej,int *&qu_e) {
	const unsigned int b1=1<<21,b2=1<<22,b3=1<<24,b4=1<<25;
	if((q&b2)==b2) {
		if(ei>0) {*(mij-1)=mv;*(qu_e++)=ei-1;*(qu_e++)=ej;}
		if((q&b1)==0&&ei<hx-1) {*(mij+1)=mv;*(qu_e++)=ei+1;*(qu_e++)=ej;}
	} else if((q&b1)==b1&&ei<hx-1) {*(mij+1)=mv;*(qu_e++)=ei+1;*(qu_e++)=ej;}
	if((q&b4)==b4) {
		if(ej>0) {*(mij-hx)=mv;*(qu_e++)=ei;*(qu_e++)=ej-1;}
		if((q&b3)==0&&ej<hy-1) {*(mij+hx)=mv;*(qu_e++)=ei;*(qu_e++)=ej+1;}
	} else if((q&b3)==b3&&ej<hy-1) {*(mij+hx)=mv;*(qu_e++)=ei;*(qu_e++)=ej+1;}
}

/** This routine computes a Voronoi cell for a single particle in the
 * container. It can be called by the user, but is also forms the core part of
 * several of the main functions, such as store_cell_volumes(), print_all(),
 * and the drawing routines. The algorithm constructs the cell by testing over
 * the neighbors of the particle, working outwards until it reaches those
 * particles which could not possibly intersect the cell. For maximum
 * efficiency, this algorithm is divided into three parts. In the first
 * section, the algorithm tests over the blocks which are in the immediate
 * vicinity of the particle, by making use of one of the precomputed worklists.
 * The code then continues to test blocks on the worklist, but also begins to
 * construct a list of neighboring blocks outside the worklist which may need
 * to be test. In the third section, the routine starts testing these
 * neighboring blocks, evaluating whether or not a particle in them could
 * possibly intersect the cell. For blocks that intersect the cell, it tests
 * the particles in that block, and then adds the block neighbors to the list
 * of potential places to consider.
 * \param[in,out] c a reference to a voronoicell object.
 * \param[in] ij the index of the block that the test particle is in.
 * \param[in] s the index of the particle within the test block.
 * \param[in] (ci,cj) the coordinates of the block that the test particle is
 *                       in relative to the container data structure.
 * \return False if the Voronoi cell was completely removed during the
 *         computation and has zero volume, true otherwise. */
template<class c_class_2d>
template<class v_cell_2d>
bool voro_compute_2d<c_class_2d>::compute_cell(v_cell_2d &c,int ij,int s,int ci,int cj) {
	static const int count_list[8]={7,11,15,19,26,35,45,59},*count_e=count_list+8;
	double x,y,x1,y1,qx=0,qy=0;
	double xlo,ylo,xhi,yhi,x2,y2,rs;
	int i,j,di,dj,ei,ej,f,g,l,disp;
	double fx,fy,gxs,gys,*radp;
	unsigned int q,*e,*mij;

	// Initialize the Voronoi cell to fill the entire container
	if(!con.initialize_voronoicell(c,ij,s,ci,cj,i,j,x,y,disp)) return false;
	con.r_init(ij,s);
	if(!con.boundary_cuts(c,ij,x,y)) return false;

	double crs,mrs;
	int next_count=3,*count_p=(const_cast<int*> (count_list));

	// Test all particles in the particle's local region first
	for(l=0;l<s;l++) {
		if(con.skip(ij,l,x,y)) continue;
		x1=p[ij][ps*l]-x;
		y1=p[ij][ps*l+1]-y;
		rs=con.r_scale(x1*x1+y1*y1,ij,l);
		if(!c.nplane(x1,y1,rs,id[ij][l])) return false;
	}
	l++;
	while(l<co[ij]) {
		if(con.skip(ij,l,x,y)) {l++;continue;}
		x1=p[ij][ps*l]-x;
		y1=p[ij][ps*l+1]-y;
		rs=con.r_scale(x1*x1+y1*y1,ij,l);
		if(!c.nplane(x1,y1,rs,id[ij][l])) return false;
		l++;
	}

	// Now compute the maximum distance squared from the cell center to a
	// vertex. This is used to cut off the calculation since we only need
	// to test out to twice this range.
	mrs=c.max_radius_squared();

	// Now compute the fractional position of the particle within its
	// region and store it in (fx,fy,fz). We use this to compute an index
	// (di,dj,dk) of which subregion the particle is within.
	unsigned int m1,m2;
	con.frac_pos(x,y,ci,cj,fx,fy);
	di=int(fx*xsp*wl_fgrid_2d);dj=int(fy*ysp*wl_fgrid_2d);

	// The indices (di,dj,dk) tell us which worklist to use, to test the
	// blocks in the optimal order. But we only store worklists for the
	// eighth of the region where di, dj, and dk are all less than half the
	// full grid. The rest of the cases are handled by symmetry. In this
	// section, we detect for these cases, by reflecting high values of di,
	// dj, and dk. For these cases, a mask is constructed in m1 and m2
	// which is used to flip the worklist information when it is loaded.
	if(di>=wl_hgrid_2d) {
		gxs=fx;
		m1=127+(3<<21);m2=1+(1<<21);di=wl_fgrid_2d-1-di;if(di<0) di=0;
	} else {m1=m2=0;gxs=boxx-fx;}
	if(dj>=wl_hgrid_2d) {
		gys=fy;
		m1|=(127<<7)+(3<<24);m2|=(1<<7)+(1<<24);dj=wl_fgrid_2d-1-dj;if(dj<0) dj=0;
	} else gys=boxy-fy;
	gxs*=gxs;gys*=gys;

	// Now compute which worklist we are going to use, and set radp and e to
	// point at the right offsets
	ij=di+wl_hgrid_2d*dj;
	radp=mrad+ij*wl_seq_length_2d;
	e=(const_cast<unsigned int*> (wl))+ij*wl_seq_length_2d;

	// Read in how many items in the worklist can be tested without having to
	// worry about writing to the mask
	f=e[0];g=0;
	do {

		// At the intervals specified by count_list, we recompute the
		// maximum radius squared
		if(g==next_count) {
			mrs=c.max_radius_squared();
			if(count_p!=count_e) next_count=*(count_p++);
		}

		// If mrs is less than the minimum distance to any untested
		// block, then we are done
		if(con.r_ctest(radp[g],mrs)) return true;
		g++;

		// Load in a block off the worklist, permute it with the
		// symmetry mask, and decode its position. These are all
		// integer bit operations so they should run very fast.
		q=e[g];q^=m1;q+=m2;
		di=q&127;di-=64;
		dj=(q>>7)&127;dj-=64;

		// Check that the worklist position is in range
		ei=di+i;if(ei<0||ei>=hx) continue;
		ej=dj+j;if(ej<0||ej>=hy) continue;

		// Call the compute_min_max_radius() function. This returns
		// true if the minimum distance to the block is bigger than the
		// current mrs, in which case we skip this block and move on.
		// Otherwise, it computes the maximum distance to the block and
		// returns it in crs.
		if(compute_min_max_radius(di,dj,fx,fy,gxs,gys,crs,mrs)) continue;

		// Now compute which region we are going to loop over, adding a
		// displacement for the periodic cases
		ij=con.region_index(ci,cj,ei,ej,qx,qy,disp);
		if(!con.boundary_cuts(c,ij,x,y)) return false;

		// If mrs is bigger than the maximum distance to the block,
		// then we have to test all particles in the block for
		// intersections. Otherwise, we do additional checks and skip
		// those particles which can't possibly intersect the block.
		if(co[ij]>0) {
			l=0;x2=x-qx;y2=y-qy;
			if(!con.r_ctest(crs,mrs)) {
				do {
					if(con.skip(ij,l,x,y)) {l++;continue;}
					x1=p[ij][ps*l]-x2;
					y1=p[ij][ps*l+1]-y2;
					rs=con.r_scale(x1*x1+y1*y1,ij,l);
					if(!c.nplane(x1,y1,rs,id[ij][l])) return false;
					l++;
				} while (l<co[ij]);
			} else {
				do {
					if(con.skip(ij,l,x,y)) {l++;continue;}
					x1=p[ij][ps*l]-x2;
					y1=p[ij][ps*l+1]-y2;
					rs=x1*x1+y1*y1;
					if(con.r_scale_check(rs,mrs,ij,l)&&!c.nplane(x1,y1,rs,id[ij][l])) return false;
					l++;
				} while (l<co[ij]);
			}
		}
	} while(g<f);

	// If we reach here, we were unable to compute the entire cell using
	// the first part of the worklist. This section of the algorithm
	// continues the worklist, but it now starts preparing the mask that we
	// need if we end up going block by block. We do the same as before,
	// but we put a mark down on the mask for every block that's tested.
	// The worklist also contains information about which neighbors of each
	// block are not also on the worklist, and we start storing those
	// points in a list in case we have to go block by block. Update the
	// mask counter, and if it wraps around then reset the whole mask; that
	// will only happen once every 2^32 tries.
	mv++;
	if(mv==0) {reset_mask();mv=1;}

	// Set the queue pointers
	int *qu_s=qu,*qu_e=qu;

	while(g<wl_seq_length_2d-1) {

		// At the intervals specified by count_list, we recompute the
		// maximum radius squared
		if(g==next_count) {
			mrs=c.max_radius_squared();
			if(count_p!=count_e) next_count=*(count_p++);
		}

		// If mrs is less than the minimum distance to any untested
		// block, then we are done
		if(con.r_ctest(radp[g],mrs)) return true;
		g++;

		// Load in a block off the worklist, permute it with the
		// symmetry mask, and decode its position. These are all
		// integer bit operations so they should run very fast.
		q=e[g];q^=m1;q+=m2;
		di=q&127;di-=64;
		dj=(q>>7)&127;dj-=64;

		// Compute the position in the mask of the current block. If
		// this lies outside the mask, then skip it. Otherwise, mark
		// it.
		ei=di+i;if(ei<0||ei>=hx) continue;
		ej=dj+j;if(ej<0||ej>=hy) continue;
		mij=mask+ei+hx*ej;
		*mij=mv;

		// Call the compute_min_max_radius() function. This returns
		// true if the minimum distance to the block is bigger than the
		// current mrs, in which case we skip this block and move on.
		// Otherwise, it computes the maximum distance to the block and
		// returns it in crs.
		if(compute_min_max_radius(di,dj,fx,fy,gxs,gys,crs,mrs)) continue;

		// Now compute which region we are going to loop over, adding a
		// displacement for the periodic cases
		ij=con.region_index(ci,cj,ei,ej,qx,qy,disp);
		if(!con.boundary_cuts(c,ij,x,y)) return false;

		// If mrs is bigger than the maximum distance to the block,
		// then we have to test all particles in the block for
		// intersections. Otherwise, we do additional checks and skip
		// those particles which can't possibly intersect the block.
		if(co[ij]>0) {
			l=0;x2=x-qx;y2=y-qy;
			if(!con.r_ctest(crs,mrs)) {
				do {
					if(con.skip(ij,l,x,y)) {l++;continue;}
					x1=p[ij][ps*l]-x2;
					y1=p[ij][ps*l+1]-y2;
					rs=con.r_scale(x1*x1+y1*y1,ij,l);
					if(!c.nplane(x1,y1,rs,id[ij][l])) return false;
					l++;
				} while (l<co[ij]);
			} else {
				do {
					if(con.skip(ij,l,x,y)) {l++;continue;}
					x1=p[ij][ps*l]-x2;
					y1=p[ij][ps*l+1]-y2;
					rs=x1*x1+y1*y1;
					if(con.r_scale_check(rs,mrs,ij,l)&&!c.nplane(x1,y1,rs,id[ij][l])) return false;
					l++;
				} while (l<co[ij]);
			}
		}

		// If there might not be enough memory on the list for these
		// additions, then add more
		if(qu_e>qu_l-8) add_list_memory(qu_s,qu_e);

		// Test the parts of the worklist element which tell us what
		// neighbors of this block are not on the worklist. Store them
		// on the block list, and mark the mask.
		scan_bits_mask_add(q,mij,ei,ej,qu_e);
	}

	// Do a check to see if we've reached the radius cutoff
	if(con.r_ctest(radp[g],mrs)) return true;

	// We were unable to completely compute the cell based on the blocks in
	// the worklist, so now we have to go block by block, reading in items
	// off the list
	while(qu_s!=qu_e) {

		// If we reached the end of the list memory loop back to the
		// start
		if(qu_s==qu_l) qu_s=qu;

		// Read in a block off the list, and compute the upper and lower
		// coordinates in each of the three dimensions
		ei=*(qu_s++);ej=*(qu_s++);
		xlo=(ei-i)*boxx-fx;xhi=xlo+boxx;
		ylo=(ej-j)*boxy-fy;yhi=ylo+boxy;

		// Carry out plane tests to see if any particle in this block
		// could possibly intersect the cell
		if(ei>i) {
			if(ej>j) {
				if(corner_test(c,xlo,ylo,xhi,yhi)) continue;
			} else if(ej<j) {
				if(corner_test(c,xlo,yhi,xhi,ylo)) continue;
			} else {
				if(edge_x_test(c,xlo,ylo,yhi)) continue;
			}
		} else if(ei<i) {
			if(ej>j) {
				if(corner_test(c,xhi,ylo,xlo,yhi)) continue;
			} else if(ej<j) {
				if(corner_test(c,xhi,yhi,xlo,ylo)) continue;
			} else {
				if(edge_x_test(c,xhi,ylo,yhi)) continue;
			}
		} else {
			if(ej>j) {
				if(edge_y_test(c,xlo,ylo,xhi)) continue;
			} else if(ej<j) {
				if(edge_y_test(c,xlo,yhi,xhi)) continue;
			} else voro_fatal_error("Compute cell routine revisiting central block, which should never\nhappen.",VOROPP_INTERNAL_ERROR);
		}

		// Now compute the region that we are going to test over, and
		// set a displacement vector for the periodic cases
		ij=con.region_index(ci,cj,ei,ej,qx,qy,disp);
		if(!con.boundary_cuts(c,ij,x,y)) return false;

		// Loop over all the elements in the block to test for cuts. It
		// would be possible to exclude some of these cases by testing
		// against mrs, but this will probably not save time.
		if(co[ij]>0) {
			l=0;x2=x-qx;y2=y-qy;
			do {
				if(con.skip(ij,l,x,y)) {l++;continue;}
				x1=p[ij][ps*l]-x2;
				y1=p[ij][ps*l+1]-y2;
				rs=con.r_scale(x1*x1+y1*y1,ij,l);
				if(!c.nplane(x1,y1,rs,id[ij][l])) return false;
				l++;
			} while (l<co[ij]);
		}

		// If there's not much memory on the block list then add more
		if((qu_s<=qu_e?(qu_l-qu_e)+(qu_s-qu):qu_s-qu_e)<8) add_list_memory(qu_s,qu_e);

		// Test the neighbors of the current block, and add them to the
		// block list if they haven't already been tested
		add_to_mask(ei,ej,qu_e);
	}

	return true;
}

/** This function checks to see whether a particular block can possibly have
 * any intersection with a Voronoi cell, for the case when the closest point
 * from the cell center to the block is on an edge which points along the z
 * direction.
 * \param[in,out] c a reference to a Voronoi cell.
 * \param[in] (xl,yl) the relative x and y coordinates of the corner of the
 *                    block closest to the cell center.
 * \param[in] (xh,yh) the relative x and y coordinates of the corner of the
 *                    block furthest away from the cell center.
 * \return False if the block may intersect, true if does not. */
template<class c_class_2d>
template<class v_cell_2d>
inline bool voro_compute_2d<c_class_2d>::corner_test(v_cell_2d &c,double xl,double yl,double xh,double yh) {
	con.r_prime(xl*xl+yl*yl);
	if(c.plane_intersects_guess(xl,yh,con.r_cutoff(xl*xl+yl*yh))) return false;
//	if(c.plane_intersects(xl,yl,con.r_cutoff(xl*xl+yl*yl))) return false;  XXX not needed?
	if(c.plane_intersects(xh,yl,con.r_cutoff(xl*xh+yl*yl))) return false;
	return true;
}

/** This function checks to see whether a particular block can possibly have
 * any intersection with a Voronoi cell, for the case when the closest point
 * from the cell center to the block is on a face aligned with the x direction.
 * \param[in,out] c a reference to a Voronoi cell.
 * \param[in] xl the minimum distance from the cell center to the face.
 * \param[in] (y0,y1) the minimum and maximum relative y coordinates of the
 *                    block.
 * \param[in] (z0,z1) the minimum and maximum relative z coordinates of the
 *                    block.
 * \return False if the block may intersect, true if does not. */
template<class c_class_2d>
template<class v_cell_2d>
inline bool voro_compute_2d<c_class_2d>::edge_x_test(v_cell_2d &c,double xl,double y0,double y1) {
	con.r_prime(xl*xl);
	if(c.plane_intersects_guess(xl,y0,con.r_cutoff(xl*xl))) return false;
	if(c.plane_intersects(xl,y1,con.r_cutoff(xl*xl))) return false;
	return true;
}

/** This function checks to see whether a particular block can possibly have
 * any intersection with a Voronoi cell, for the case when the closest point
 * from the cell center to the block is on a face aligned with the y direction.
 * \param[in,out] c a reference to a Voronoi cell.
 * \param[in] yl the minimum distance from the cell center to the face.
 * \param[in] (x0,x1) the minimum and maximum relative x coordinates of the
 *                    block.
 * \param[in] (z0,z1) the minimum and maximum relative z coordinates of the
 *                    block.
 * \return False if the block may intersect, true if does not. */
template<class c_class_2d>
template<class v_cell_2d>
inline bool voro_compute_2d<c_class_2d>::edge_y_test(v_cell_2d &c,double x0,double yl,double x1) {
	con.r_prime(yl*yl);
	if(c.plane_intersects_guess(x0,yl,con.r_cutoff(yl*yl))) return false;
	if(c.plane_intersects(x1,yl,con.r_cutoff(yl*yl))) return false;
	return true;
}

/** This routine checks to see whether a point is within a particular distance
 * of a nearby region. If the point is within the distance of the region, then
 * the routine returns true, and computes the maximum distance from the point
 * to the region. Otherwise, the routine returns false.
 * \param[in] (di,dj) the position of the nearby region to be tested,
 *                       relative to the region that the point is in.
 * \param[in] (fx,fy) the displacement of the point within its region.
 * \param[in] (gxs,gys) the maximum squared distances from the point to the
 *                          sides of its region.
 * \param[out] crs a reference in which to return the maximum distance to the
 *                 region (only computed if the routine returns positive).
 * \param[in] mrs the distance to be tested.
 * \return False if the region is further away than mrs, true if the region in
 *         within mrs.*/
template<class c_class_2d>
bool voro_compute_2d<c_class_2d>::compute_min_max_radius(int di,int dj,double fx,double fy,double gxs,double gys,double &crs,double mrs) {
	double xlo,ylo;
	if(di>0) {
		xlo=di*boxx-fx;
		crs=xlo*xlo;
		if(dj>0) {
			ylo=dj*boxy-fy;
			crs+=ylo*ylo;
			if(con.r_ctest(crs,mrs)) return true;
			crs+=bxsq+2*xlo*boxx+2*ylo*boxy;
		} else if(dj<0) {
			ylo=(dj+1)*boxy-fy;
			crs+=ylo*ylo;
			if(con.r_ctest(crs,mrs)) return true;
			crs+=bxsq+2*xlo*boxx-2*ylo*boxy;
		} else {
			if(con.r_ctest(crs,mrs)) return true;
			crs+=gys+boxx*(2*xlo+boxx);
		}
	} else if(di<0) {
		xlo=(di+1)*boxx-fx;
		crs=xlo*xlo;
		if(dj>0) {
			ylo=dj*boxy-fy;
			crs+=ylo*ylo;
			if(con.r_ctest(crs,mrs)) return true;
			crs+=bxsq-2*xlo*boxx+2*ylo*boxy;
		} else if(dj<0) {
			ylo=(dj+1)*boxy-fy;
			crs+=ylo*ylo;
			if(con.r_ctest(crs,mrs)) return true;
			crs+=bxsq-2*xlo*boxx-2*ylo*boxy;
		} else {
			if(con.r_ctest(crs,mrs)) return true;
			crs+=gys+boxx*(-2*xlo+boxx);
		}
	} else {
		if(dj>0) {
			ylo=dj*boxy-fy;
			crs=ylo*ylo;
			if(con.r_ctest(crs,mrs)) return true;
			crs+=boxy*(2*ylo+boxy);
		} else if(dj<0) {
			ylo=(dj+1)*boxy-fy;
			crs=ylo*ylo;
			if(con.r_ctest(crs,mrs)) return true;
			crs+=boxy*(-2*ylo+boxy);
		} else voro_fatal_error("Min/max radius function called for central block, which should never\nhappen.",VOROPP_INTERNAL_ERROR);
		crs+=gxs;
	}
	return false;
}

template<class c_class_2d>
bool voro_compute_2d<c_class_2d>::compute_min_radius(int di,int dj,double fx,double fy,double mrs) {
	double t,crs;

	if(di>0) {t=di*boxx-fx;crs=t*t;}
	else if(di<0) {t=(di+1)*boxx-fx;crs=t*t;}
	else crs=0;

	if(dj>0) {t=dj*boxy-fy;crs+=t*t;}
	else if(dj<0) {t=(dj+1)*boxy-fy;crs+=t*t;}

	return crs>con.r_max_add(mrs);
}

/** Adds memory to the queue.
 * \param[in,out] qu_s a reference to the queue start pointer.
 * \param[in,out] qu_e a reference to the queue end pointer. */
template<class c_class_2d>
inline void voro_compute_2d<c_class_2d>::add_list_memory(int*& qu_s,int*& qu_e) {
	qu_size<<=1;
	int *qu_n=new int[qu_size],*qu_c=qu_n;
#if VOROPP_VERBOSE >=2
	fprintf(stderr,"List memory scaled up to %d\n",qu_size);
#endif
	if(qu_s<=qu_e) {
		while(qu_s<qu_e) *(qu_c++)=*(qu_s++);
	} else {
		while(qu_s<qu_l) *(qu_c++)=*(qu_s++);
		qu_s=qu;
		while(qu_s<qu_e) *(qu_c++)=*(qu_s++);
	}
	delete [] qu;
	qu_s=qu=qu_n;
	qu_l=qu+qu_size;
	qu_e=qu_c;
}

// Explicit template instantiation
template voro_compute_2d<container_2d>::voro_compute_2d(container_2d&,int,int);
template voro_compute_2d<container_poly_2d>::voro_compute_2d(container_poly_2d&,int,int);
template voro_compute_2d<container_boundary_2d>::voro_compute_2d(container_boundary_2d&,int,int);
template bool voro_compute_2d<container_2d>::compute_cell(voronoicell_2d&,int,int,int,int);
template bool voro_compute_2d<container_2d>::compute_cell(voronoicell_neighbor_2d&,int,int,int,int);
template void voro_compute_2d<container_2d>::find_voronoi_cell(double,double,int,int,int,particle_record_2d&,double&);
template bool voro_compute_2d<container_poly_2d>::compute_cell(voronoicell_2d&,int,int,int,int);
template bool voro_compute_2d<container_poly_2d>::compute_cell(voronoicell_neighbor_2d&,int,int,int,int);
template void voro_compute_2d<container_poly_2d>::find_voronoi_cell(double,double,int,int,int,particle_record_2d&,double&);
template bool voro_compute_2d<container_boundary_2d>::compute_cell(voronoicell_nonconvex_2d&,int,int,int,int);
template bool voro_compute_2d<container_boundary_2d>::compute_cell(voronoicell_nonconvex_neighbor_2d&,int,int,int,int);
}
