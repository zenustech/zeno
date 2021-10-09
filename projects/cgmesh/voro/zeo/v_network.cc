#include "v_network.hh"

/** Initializes the Voronoi network object. The geometry is set up to match a
 * corresponding container class, and memory is allocated for the network.
 * \param[in] c a reference to a container or container_poly class. */
template<class c_class>
voronoi_network::voronoi_network(c_class &c,double net_tol_) :
	bx(c.bx), bxy(c.bxy), by(c.by), bxz(c.bxz), byz(c.byz), bz(c.bz),
	nx(c.nx), ny(c.ny), nz(c.nz), nxyz(nx*ny*nz),
	xsp(nx/bx), ysp(ny/by), zsp(nz/bz), net_tol(net_tol_) {
	int l;

	// Allocate memory for vertex structure
	pts=new double*[nxyz];
	idmem=new int*[nxyz];
	ptsc=new int[nxyz];
	ptsmem=new int[nxyz];
	for(l=0;l<nxyz;l++) {
		pts[l]=new double[4*init_network_vertex_memory];
		idmem[l]=new int[init_network_vertex_memory];
		ptsc[l]=0;ptsmem[l]=init_network_vertex_memory;
	}

	// Allocate memory for network edges and related statistics
	edc=0;edmem=init_network_vertex_memory*nxyz;
	ed=new int*[edmem];
	ne=new int*[edmem];
	pered=new unsigned int*[edmem];
	raded=new block*[edmem];
	nu=new int[edmem];
	nec=new int[edmem];
	numem=new int[edmem];

	// Allocate memory for back pointers
	reg=new int[edmem];
	regp=new int[edmem];

	// Allocate edge memory
	for(l=0;l<edmem;l++) {
		ed[l]=new int[2*init_network_edge_memory];
		ne[l]=ed[l]+init_network_edge_memory;
	}
	for(l=0;l<edmem;l++) raded[l]=new block[init_network_edge_memory];
	for(l=0;l<edmem;l++) pered[l]=new unsigned int[init_network_edge_memory];
	for(l=0;l<edmem;l++) {nu[l]=nec[l]=0;numem[l]=init_network_edge_memory;}

	// vertices
	vmap=new int[4*init_vertices];
	map_mem=init_vertices;
}

/** The voronoi_network destructor removes the dynamically allocated memory. */
voronoi_network::~voronoi_network() {
	int l;

	// Remove Voronoi mapping array
	delete [] vmap;

	// Remove individual edge arrays
	for(l=0;l<edmem;l++) delete [] pered[l];
	for(l=0;l<edmem;l++) delete [] raded[l];
	for(l=0;l<edmem;l++) delete [] ed[l];

	// Remove back pointers
	delete [] regp;delete [] reg;

	// Remove memory for edges and related statistics
	delete [] numem;delete [] nec;delete [] nu;
	delete [] raded;delete [] pered;
	delete [] ne;delete [] ed;

	// Remove vertex structure arrays
	for(l=0;l<nxyz;l++) {
		delete [] idmem[l];
		delete [] pts[l];
	}
	delete [] ptsmem;delete [] ptsc;
	delete [] idmem;delete [] pts;
}

/** Increase network memory for a particular region. */
void voronoi_network::add_network_memory(int l) {
	ptsmem[l]<<=1;

	// Check to see that an absolute maximum in memory allocation
	// has not been reached, to prevent runaway allocation
	if(ptsmem[l]>max_network_vertex_memory)
		voro_fatal_error("Container vertex maximum memory allocation exceeded",VOROPP_MEMORY_ERROR);

	// Allocate new arrays
	double *npts(new double[4*ptsmem[l]]);
	int *nidmem(new int[ptsmem[l]]);

	// Copy the contents of the old arrays into the new ones
	for(int i=0;i<4*ptsc[l];i++) npts[i]=pts[l][i];
	for(int i=0;i<ptsc[l];i++) nidmem[i]=idmem[l][i];

	// Delete old arrays and update pointers to the new ones
	delete [] pts[l];delete [] idmem[l];
	pts[l]=npts;idmem[l]=nidmem;
}

/** Increase edge network memory. */
void voronoi_network::add_edge_network_memory() {
	int i;
	edmem<<=1;

	// Allocate new arrays
	int **ned(new int*[edmem]);
	int **nne(new int*[edmem]);
	block **nraded(new block*[edmem]);
	unsigned int **npered(new unsigned int*[edmem]);
	int *nnu(new int[edmem]);
	int *nnec(new int[edmem]);
	int *nnumem(new int[edmem]);
	int *nreg(new int[edmem]);
	int *nregp(new int[edmem]);

	// Copy the contents of the old arrays into the new ones
	for(i=0;i<edc;i++) {
		ned[i]=ed[i];
		nne[i]=ne[i];
		nraded[i]=raded[i];
		npered[i]=pered[i];
		nnu[i]=nu[i];
		nnec[i]=nec[i];
		nnumem[i]=numem[i];
		nreg[i]=reg[i];
		nregp[i]=regp[i];
	}

	// Carry out new allocation
	while(i<edmem) {
		ned[i]=new int[2*init_network_edge_memory];
		nne[i]=ned[i]+init_network_edge_memory;
		nnu[i]=nnec[i]=0;nnumem[i]=init_network_edge_memory;
		nraded[i]=new block[init_network_edge_memory];
		npered[i++]=new unsigned int[init_network_edge_memory];
	}

	// Delete old arrays and update pointers to the new ones
	delete [] ed;ed=ned;
	delete [] ne;ne=nne;
	delete [] raded;raded=nraded;
	delete [] pered;pered=npered;
	delete [] nu;nu=nnu;
	delete [] nec;nec=nnec;
	delete [] numem;numem=nnumem;
	delete [] reg;reg=nreg;
	delete [] regp;regp=nregp;
}

/** Increase a particular vertex memory. */
void voronoi_network::add_particular_vertex_memory(int l) {
	numem[l]<<=1;

	// Check that the vertex allocation does not exceed a maximum safe
	// limit
	if(numem[l]>max_vertex_order)
		voro_fatal_error("Particular vertex maximum memory allocation exceeded",VOROPP_MEMORY_ERROR);

	// Allocate new arrays
	int *ned(new int[2*numem[l]]);
	int *nne(ned+numem[l]);
	block *nraded(new block[numem[l]]);
	unsigned int *npered(new unsigned int[numem[l]]);

	// Copy the contents of the old arrays into the new ones
	for(int i=0;i<nu[l];i++) {
		ned[i]=ed[l][i];
		nraded[i]=raded[l][i];
		npered[i]=pered[l][i];
	}
	for(int i=0;i<nec[l];i++) nne[i]=ne[l][i];

	// Delete old arrays and update pointers to the new ones
	delete [] ed[l];ed[l]=ned;ne[l]=nne;
	delete [] raded[l];raded[l]=nraded;
	delete [] pered[l];pered[l]=npered;
}


/** Increases the memory for the vertex mapping.
 * \param[in] pmem the amount of memory needed. */
void voronoi_network::add_mapping_memory(int pmem) {
	do {map_mem<<=1;} while(map_mem<pmem);
	delete [] vmap;
	vmap=new int[4*map_mem];
}

/** Clears the class of all vertices and edges. */
void voronoi_network::clear_network() {
	int l;
	edc=0;
	for(l=0;l<nxyz;l++) ptsc[l]=0;
	for(l=0;l<edmem;l++) nu[l]=0;
}

/** Outputs the network in a format that can be read by gnuplot.
 * \param[in] fp a file handle to write to. */
void voronoi_network::draw_network(FILE *fp) {
	int l,q,ai,aj,ak;
	double x,y,z,*ptsp;
	for(l=0;l<edc;l++) {
		ptsp=pts[reg[l]]+4*regp[l];
		x=*(ptsp++);y=*(ptsp++);z=*ptsp;
		for(q=0;q<nu[l];q++) {
			unpack_periodicity(pered[l][q],ai,aj,ak);
			if(ed[l][q]<l&&ai==0&&aj==0&&ak==0) continue;
			ptsp=pts[reg[ed[l][q]]]+4*regp[ed[l][q]];
			fprintf(fp,"%g %g %g\n%g %g %g\n\n\n",x,y,z,
				*ptsp+bx*ai+bxy*aj+bxz*ak,
				ptsp[1]+by*aj+byz*ak,ptsp[2]+bz*ak);
		}
	}
}

/** Prints out the network.
 * \param[in] fp a file handle to write to.
 * \param[in] reverse_remove a boolean value, setting whether or not to remove
 *                           reverse edges. */
void voronoi_network::print_network(FILE *fp,bool reverse_remove) {
	int ai,aj,ak,j,l,ll,q;
	double x,y,z,x2,y2,z2,*ptsp;

	// Print the vertex table
	fprintf(fp,"Vertex table:\n%d\n",edc);
	//os << edc << "\n";
	for(l=0;l<edc;l++) {
		ptsp=pts[reg[l]];j=4*regp[l];
		fprintf(fp,"%d %g %g %g %g",l,ptsp[j],ptsp[j+1],ptsp[j+2],ptsp[j+3]);
		for(ll=0;ll<nec[l];ll++) fprintf(fp," %d",ne[l][ll]);
		fputs("\n",fp);
	}

	// Print out the edge table, loop over vertices
	fputs("\nEdge table:\n",fp);
	for(l=0;l<edc;l++) {

		// Store the position of this vertex
		ptsp=pts[reg[l]];j=4*regp[l];
		x=ptsp[j];y=ptsp[j+1];z=ptsp[j+2];

		// Loop over edges of this vertex
		for(q=0;q<nu[l];q++) {

			unpack_periodicity(pered[l][q],ai,aj,ak);

			// If this option is enabled, then the code will not
			// print edges from i to j for j<i.
			if(reverse_remove) if(ed[l][q]<l&&ai==0&&aj==0&&ak==0) continue;

			fprintf(fp,"%d -> %d",l,ed[l][q]);
			raded[l][q].print(fp);

			// Compute and print the length of the edge
			ptsp=pts[reg[ed[l][q]]];j=4*regp[ed[l][q]];
			x2=ptsp[j]+ai*bx+aj*bxy+ak*bxz-x;
			y2=ptsp[j+1]+aj*by+ak*byz-y;
			z2=ptsp[j+2]+ak*bz-z;
			fprintf(fp," %d %d %d %g\n",ai,aj,ak,sqrt(x2*x2+y2*y2+z2*z2));
		}
	}
}

// Converts three periodic image displacements into a single unsigned integer.
// \param[in] i the periodic image in the x direction.
// \param[in] j the periodic image in the y direction.
// \param[in] k the periodic image in the z direction.
// \return The packed integer. */
inline unsigned int voronoi_network::pack_periodicity(int i,int j,int k) {
	unsigned int pa=((unsigned int) (127+i));
	pa<<=8;pa+=((unsigned int) (127+j));
	pa<<=8;pa+=((unsigned int) (127+k));
	return pa;
}

/** Unpacks an unsigned integer into three periodic image displacements.
 * \param[in] pa the packed integer.
// \param[out] i the periodic image in the x direction.
// \param[out] j the periodic image in the y direction.
// \param[out] k the periodic image in the z direction. */
inline void voronoi_network::unpack_periodicity(unsigned int pa,int &i,int &j,int &k) {
	i=((signed int) (pa>>16))-127;
	j=((signed int) ((pa>>8)&255))-127;
	k=((signed int) (pa&255))-127;
}

/** Adds a Voronoi cell to the network structure. The routine first checks all
 * of the Voronoi cell vertices and merges them with existing ones where
 * possible. Edges are then added to the structure.
 * \param[in] c a reference to a Voronoi cell.
 * \param[in] (x,y,z) the position of the Voronoi cell.
 * \param[in] idn the ID number of the particle associated with the cell. */
template<class v_cell>
void voronoi_network::add_to_network_internal(v_cell &c,int idn,double x,double y,double z,double rad,int *cmap) {
	int i,j,k,ijk,l,q,ai,aj,ak,*vmp(cmap);
	double gx,gy,vx,vy,vz,crad,*cp(c.pts);

	// Loop over the vertices of the Voronoi cell
	for(l=0;l<c.p;l++,vmp+=4) {

		// Compute the real position of this vertex, and evaluate its
		// position along the non-rectangular axes
		vx=x+cp[4*l]*0.5;vy=y+cp[4*l+1]*0.5;vz=z+cp[4*l+2]*0.5;
		gx=vx-vy*(bxy/by)+vz*(bxy*byz-by*bxz)/(by*bz);
		gy=vy-vz*(byz/bz);

		// Compute the adjusted radius, which will be needed either way
		crad=0.5*sqrt(cp[4*l]*cp[4*l]+cp[4*l+1]*cp[4*l+1]+cp[4*l+2]*cp[4*l+2])-rad;

		// Check to see if a vertex very close to this one already
		// exists in the network
		if(search_previous(gx,gy,vx,vy,vz,ijk,q,vmp[1],vmp[2],vmp[3])) {

			// If it does, then just map the Voronoi cell
			// vertex to it
			*vmp=idmem[ijk][q];

			// Store this radius if it smaller than the current
			// value
			if(pts[ijk][4*q+3]>crad) pts[ijk][4*q+3]=crad;
		} else {
			k=step_int(vz*zsp);if(k<0||k>=nz) {ak=step_div(k,nz);vx-=bxz*ak;vy-=byz*ak;vz-=bz*ak;k-=ak*nz;} else ak=0;
			j=step_int(gy*ysp);if(j<0||j>=ny) {aj=step_div(j,ny);vx-=bxy*aj;vy-=by*aj;j-=aj*ny;} else aj=0;
			i=step_int(gx*xsp);if(i<0||i>=nx) {ai=step_div(i,nx);vx-=bx*ai;i-=ai*nx;} else ai=0;

			vmp[1]=ai;vmp[2]=aj;vmp[3]=ak;
			ijk=i+nx*(j+ny*k);

			if(edc==edmem) add_edge_network_memory();
			if(ptsc[ijk]==ptsmem[ijk]) add_network_memory(ijk);

			reg[edc]=ijk;regp[edc]=ptsc[ijk];
			pts[ijk][4*ptsc[ijk]]=vx;
			pts[ijk][4*ptsc[ijk]+1]=vy;
			pts[ijk][4*ptsc[ijk]+2]=vz;
			pts[ijk][4*ptsc[ijk]+3]=crad;
			idmem[ijk][ptsc[ijk]++]=edc;
			*vmp=edc++;
		}

		// Add the neighbor information to this vertex
		add_neighbor(*vmp,idn);
	}

	add_edges_to_network(c,x,y,z,rad,cmap);
}

/** Adds a neighboring particle ID to a vertex in the Voronoi network, first
 * checking that the ID is not already recorded.
 * \param[in] k the Voronoi vertex.
 * \param[in] idn the particle ID number. */
inline void voronoi_network::add_neighbor(int k,int idn) {
	for(int i=0;i<nec[k];i++) if(ne[k][i]==idn) return;
	if(nec[k]==numem[k]) add_particular_vertex_memory(k);
	ne[k][nec[k]++]=idn;
}

/** Adds edges to the network structure, after the vertices have been
 * considered. This routine assumes that the cmap array has a mapping between
 * Voronoi cell vertices and Voronoi network vertices. */
template<class v_cell>
void voronoi_network::add_edges_to_network(v_cell &c,double x,double y,double z,double rad,int *cmap) {
	int i,j,ai,aj,ak,bi,bj,bk,k,l,q,*vmp;unsigned int cper;
	double vx,vy,vz,wx,wy,wz,dx,dy,dz,dis;double *pp;
	for(l=0;l<c.p;l++) {
		vmp=cmap+4*l;k=*(vmp++);ai=*(vmp++);aj=*(vmp++);ak=*vmp;
		pp=pts[reg[k]]+4*regp[k];
		vx=pp[0]+ai*bx+aj*bxy+ak*bxz;
		vy=pp[1]+aj*by+ak*byz;
		vz=pp[2]+ak*bz;
		for(q=0;q<c.nu[l];q++) {
			i=c.ed[l][q];
			vmp=cmap+4*i;
			j=*(vmp++);bi=*(vmp++);bj=*(vmp++);bk=*vmp;

			// Skip if this is a self-connecting edge
			if(j==k&&bi==ai&&bj==aj&&bk==ak) continue;
			cper=pack_periodicity(bi-ai,bj-aj,bk-ak);
			pp=pts[reg[j]]+(4*regp[j]);
			wx=pp[0]+bi*bx+bj*bxy+bk*bxz;
			wy=pp[1]+bj*by+bk*byz;
			wz=pp[2]+bk*bz;
			dx=wx-vx;dy=wy-vy;dz=wz-vz;
			dis=(x-vx)*dx+(y-vy)*dy+(z-vz)*dz;
			dis/=dx*dx+dy*dy+dz*dz;
			if(dis<0) dis=0;if(dis>1) dis=1;
			wx=vx-x+dis*dx;wy=vy-y+dis*dy;wz=vz-z+dis*dz;
			int nat=not_already_there(k,j,cper);
			if(nat==nu[k]) {
				if(nu[k]==numem[k]) add_particular_vertex_memory(k);
				ed[k][nu[k]]=j;
				raded[k][nu[k]].first(sqrt(wx*wx+wy*wy+wz*wz)-rad,dis);
				pered[k][nu[k]++]=cper;
			} else {
				raded[k][nat].add(sqrt(wx*wx+wy*wy+wz*wz)-rad,dis);
			}
		}
	}
}

template<class v_cell>
void voronoi_network::add_to_network_rectangular_internal(v_cell &c,int idn,double x,double y,double z,double rad,int *cmap) {
	int i,j,k,ijk,l,q,ai,aj,ak,*vmp(cmap);
	double vx,vy,vz,crad,*cp(c.pts);

	for(l=0;l<c.p;l++,vmp+=4) {
		vx=x+cp[4*l]*0.5;vy=y+cp[4*l+1]*0.5;vz=z+cp[4*l+2]*0.5;
		crad=0.5*sqrt(cp[4*l]*cp[4*l]+cp[4*l+1]*cp[4*l+1]+cp[4*l+2]*cp[4*l+2])-rad;
		if(safe_search_previous_rect(vx,vy,vz,ijk,q,vmp[1],vmp[2],vmp[3])) {
			*vmp=idmem[ijk][q];

			// Store this radius if it smaller than the current
			// value
			if(pts[ijk][4*q+3]>crad) pts[ijk][4*q+3]=crad;
		} else {
			k=step_int(vz*zsp);
			if(k<0||k>=nz) {
				ak=step_div(k,nz);
				vz-=ak*bz;vy-=ak*byz;vx-=ak*bxz;k-=ak*nz;
			} else ak=0;
			j=step_int(vy*ysp);
			if(j<0||j>=ny) {
				aj=step_div(j,ny);
				vy-=aj*by;vx-=aj*bxy;j-=aj*ny;
			} else aj=0;
			i=step_int(vx*xsp);
			if(i<0||i>=nx) {
				ai=step_div(i,nx);
				vx-=ai*bx;i-=ai*nx;
			} else ai=0;
			vmp[1]=ai;
			vmp[2]=aj;
			vmp[3]=ak;
			ijk=i+nx*(j+ny*k);
			if(edc==edmem) add_edge_network_memory();
			if(ptsc[ijk]==ptsmem[ijk]) add_network_memory(ijk);
			reg[edc]=ijk;regp[edc]=ptsc[ijk];
			pts[ijk][4*ptsc[ijk]]=vx;
			pts[ijk][4*ptsc[ijk]+1]=vy;
			pts[ijk][4*ptsc[ijk]+2]=vz;
			pts[ijk][4*ptsc[ijk]+3]=crad;
			idmem[ijk][ptsc[ijk]++]=edc;
			*vmp=edc++;
		}

		add_neighbor(*vmp,idn);
	}

	add_edges_to_network(c,x,y,z,rad,cmap);
}

int voronoi_network::not_already_there(int k,int j,unsigned int cper) {
	for(int i=0;i<nu[k];i++) if(ed[k][i]==j&&pered[k][i]==cper) return i;
	return nu[k];
}

bool voronoi_network::search_previous(double gx,double gy,double x,double y,double z,int &ijk,int &q,int &pi,int &pj,int &pk) {
	int ai=step_int((gx-net_tol)*xsp),bi=step_int((gx+net_tol)*xsp);
	int aj=step_int((gy-net_tol)*ysp),bj=step_int((gy+net_tol)*ysp);
	int ak=step_int((z-net_tol)*zsp),bk=step_int((z+net_tol)*zsp);
	int i,j,k,mi,mj,mk;
	double px,py,pz,px2,py2,px3,*pp;

	for(k=ak;k<=bk;k++) {
		pk=step_div(k,nz);px=pk*bxz;py=pk*byz;pz=pk*bz;mk=k-nz*pk;
		for(j=aj;j<=bj;j++) {
			pj=step_div(j,ny);px2=px+pj*bxy;py2=py+pj*by;mj=j-ny*pj;
			for(i=ai;i<=bi;i++) {
				pi=step_div(i,nx);px3=px2+pi*bx;mi=i-nx*pi;
				ijk=mi+nx*(mj+ny*mk);
				pp=pts[ijk];
				for(q=0;q<ptsc[ijk];q++,pp+=4) if(fabs(*pp+px3-x)<net_tol&&fabs(pp[1]+py2-y)<net_tol&&fabs(pp[2]+pz-z)<net_tol) return true;
			}
		}
	}
	return false;
}

bool voronoi_network::safe_search_previous_rect(double x,double y,double z,int &ijk,int &q,int &ci,int &cj,int &ck) {
	const double tol(0.5*net_tol);
	if(search_previous_rect(x+tol,y+tol,z+tol,ijk,q,ci,cj,ck)) return true;
	if(search_previous_rect(x-tol,y+tol,z+tol,ijk,q,ci,cj,ck)) return true;
	if(search_previous_rect(x+tol,y-tol,z+tol,ijk,q,ci,cj,ck)) return true;
	if(search_previous_rect(x-tol,y-tol,z+tol,ijk,q,ci,cj,ck)) return true;
	if(search_previous_rect(x+tol,y+tol,z-tol,ijk,q,ci,cj,ck)) return true;
	if(search_previous_rect(x-tol,y+tol,z-tol,ijk,q,ci,cj,ck)) return true;
	if(search_previous_rect(x+tol,y-tol,z-tol,ijk,q,ci,cj,ck)) return true;
	return search_previous_rect(x-tol,y-tol,z-tol,ijk,q,ci,cj,ck);
}

bool voronoi_network::search_previous_rect(double x,double y,double z,int &ijk,int &q,int &ci,int &cj,int &ck) {
	int k=step_int(z*zsp);
	if(k<0||k>=nz) {
		ck=step_div(k,nz);
		z-=ck*bz;y-=ck*byz;x-=ck*bxz;k-=ck*nz;
	} else ck=0;

	int j=step_int(y*ysp);
	if(j<0||j>=ny) {
		cj=step_div(j,ny);
		y-=cj*by;x-=cj*bxy;j-=cj*ny;
	} else cj=0;

	ijk=step_int(x*xsp);
	if(ijk<0||ijk>=nx) {
		ci=step_div(ijk,nx);
		x-=ci*bx;ijk-=ci*nx;
	} else ci=0;

	ijk+=nx*(j+ny*k);double *pp(pts[ijk]);
	for(q=0;q<ptsc[ijk];q++,pp+=4)
		if(fabs(*pp-x)<net_tol&&fabs(pp[1]-y)<net_tol&&fabs(pp[2]-z)<net_tol) return true;
	return false;
}

/** Custom int function, that gives consistent stepping for negative numbers.
 * With normal int, we have (-1.5,-0.5,0.5,1.5) -> (-1,0,0,1).
 * With this routine, we have (-1.5,-0.5,0.5,1.5) -> (-2,-1,0,1). */
inline int voronoi_network::step_int(double a) {
	return a<0?int(a)-1:int(a);
}

/** Custom integer division function, that gives consistent stepping for
 * negative numbers. */
inline int voronoi_network::step_div(int a,int b) {
	return a>=0?a/b:-1+(a+1)/b;
}

// Explicit instantiation
template voronoi_network::voronoi_network(container_periodic&, double);
template voronoi_network::voronoi_network(container_periodic_poly&, double);
template void voronoi_network::add_to_network<voronoicell>(voronoicell&, int, double, double, double, double);
template void voronoi_network::add_to_network<voronoicell_neighbor>(voronoicell_neighbor&, int, double, double, double, double);
template void voronoi_network::add_to_network_rectangular<voronoicell>(voronoicell&, int, double, double, double, double);
template void voronoi_network::add_to_network_rectangular<voronoicell_neighbor>(voronoicell_neighbor&, int, double, double, double, double);
