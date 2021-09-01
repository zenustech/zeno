#include "ctr_quad_2d.hh"
#include "quad_march.hh"

#include <vector>
#include <limits>
#include <deque>

namespace voro {

container_quad_2d::container_quad_2d(double ax_,double bx_,double ay_,double by_) :
	quadtree((ax_+bx_)*0.5,(ay_+by_)*0.5,(bx_-ax_)*0.5,(by_-ay_)*0.5,*this),
	ax(ax_), bx(bx_), ay(ay_), by(by_), bmask(0) {

}

quadtree::quadtree(double cx_,double cy_,double lx_,double ly_,container_quad_2d &parent_) :
	parent(parent_), cx(cx_), cy(cy_), lx(lx_), ly(ly_), ps(2),
	id(new int[qt_max]), p(new double[ps*qt_max]), co(0), mask(0), nco(0), nmax(0) {

}

quadtree::~quadtree() {
	if(id==NULL) {
		delete qne;delete qnw;
		delete qse;delete qsw;
	} else {
		delete [] p;
		delete [] id;
	}
	if(nmax>0) delete [] nei;
}

void quadtree::split() {
	double hx=0.5*lx,hy=0.5*ly;
	qsw=new quadtree(cx-hx,cy-hy,hx,hy,parent);
	qse=new quadtree(cx+hx,cy-hy,hx,hy,parent);
	qnw=new quadtree(cx-hx,cy+hy,hx,hy,parent);
	qne=new quadtree(cx+hx,cy+hy,hx,hy,parent);
	for(int i=0;i<co;i++)
		(p[ps*i]<cx?(p[ps*i+1]<cy?qsw:qnw)
			   :(p[ps*i+1]<cy?qse:qne))->quick_put(id[i],p[ps*i],p[ps*i+1]);
	delete [] id;id=NULL;
	delete [] p;
}

void quadtree::put(int i,double x,double y) {
	if(id!=NULL) {
		if(co==qt_max) split();
		else {
			quick_put(i,x,y);
			return;
		}
	}
	(x<cx?(y<cy?qsw:qnw):(y<cy?qse:qne))->put(i,x,y);
}

void quadtree::draw_cross(FILE *fp) {
	if(id==NULL) {
		fprintf(fp,"%g %g\n%g %g\n\n\n%g %g\n%g %g\n\n\n",
			cx-lx,cy,cx+ly,cy,cx,cy-ly,cx,cy+ly);
		qsw->draw_cross(fp);
		qse->draw_cross(fp);
		qnw->draw_cross(fp);
		qne->draw_cross(fp);
	}
}

void quadtree::reset_mask() {
	mask=0;
	if(id==NULL) {
		qsw->reset_mask();
		qse->reset_mask();
		qnw->reset_mask();
		qne->reset_mask();
	}
}

void container_quad_2d::draw_quadtree(FILE *fp) {
	fprintf(fp,"%g %g\n%g %g\n%g %g\n%g %g\n%g %g\n",ax,ay,bx,ay,bx,by,ax,by,ax,ay);
	draw_cross(fp);
}

void quadtree::draw_neighbors(FILE *fp) {
	for(int i=0;i<nco;i++)
		fprintf(fp,"%g %g %g %g\n",cx,cy,nei[i]->cx-cx,nei[i]->cy-cy);
	if(id==NULL) {
		qsw->draw_neighbors(fp);
		qse->draw_neighbors(fp);
		qnw->draw_neighbors(fp);
		qne->draw_neighbors(fp);
	}
}

void quadtree::draw_particles(FILE *fp) {
	if(id==NULL) {
		qsw->draw_particles(fp);
		qse->draw_particles(fp);
		qnw->draw_particles(fp);
		qne->draw_particles(fp);
	} else for(int i=0;i<co;i++)
		fprintf(fp,"%d %g %g\n",id[i],p[ps*i],p[ps*i+1]);
}

void quadtree::draw_cells_gnuplot(FILE *fp) {
	if(id==NULL) {
		qsw->draw_cells_gnuplot(fp);
		qse->draw_cells_gnuplot(fp);
		qnw->draw_cells_gnuplot(fp);
		qne->draw_cells_gnuplot(fp);
	} else {
		voronoicell_2d c;
		for(int j=0;j<co;j++) if(compute_cell(c,j))
			c.draw_gnuplot(p[ps*j],p[ps*j+1],fp);
	}
}

double quadtree::sum_cell_areas() {
	if(id==NULL)
		return qsw->sum_cell_areas()+qse->sum_cell_areas()
		      +qnw->sum_cell_areas()+qne->sum_cell_areas();
	double area=0;
	voronoicell_2d c;
	for(int j=0;j<co;j++) if(compute_cell(c,j))
		area+=c.area();
	return area;
}

void quadtree::compute_all_cells() {
	if(id==NULL) {
		qsw->compute_all_cells();
		qse->compute_all_cells();
		qnw->compute_all_cells();
		qne->compute_all_cells();
	} else{
		voronoicell_2d c;
		for(int j=0;j<co;j++) compute_cell(c,j);
	}
}

void quadtree::setup_neighbors() {
	if(id==NULL) {
		qsw->setup_neighbors();
		qse->setup_neighbors();
		qnw->setup_neighbors();
		qne->setup_neighbors();
		we_neighbors(qsw,qse);
		we_neighbors(qnw,qne);
		ns_neighbors(qsw,qnw);
		ns_neighbors(qse,qne);
	}
}

void quadtree::we_neighbors(quadtree *qw,quadtree *qe) {
	const int ma=1<<30;
	quad_march<0> mw(qw);
	quad_march<1> me(qe);
	while(mw.s<ma||me.s<ma) {
		mw.cu()->add_neighbor(me.cu());
		me.cu()->add_neighbor(mw.cu());
		if(mw.ns>me.ns) me.step();
		else {
			if(mw.ns==me.ns) me.step();
			mw.step();
		}
	}
}

void quadtree::ns_neighbors(quadtree *qs,quadtree *qn) {
	const int ma=1<<30;
	quad_march<2> ms(qs);
	quad_march<3> mn(qn);
	while(ms.s<ma||mn.s<ma) {
		ms.cu()->add_neighbor(mn.cu());
		mn.cu()->add_neighbor(ms.cu());
		if(ms.ns>mn.ns) mn.step();
		else {
			if(ms.ns==mn.ns) mn.step();
			ms.step();
		}
	}
}

void quadtree::add_neighbor_memory() {
	if(nmax==0) {
		nmax=4;
		nei=new quadtree*[nmax];
	}
	if(nmax>16777216) {
		fputs("Maximum quadtree neighbor memory exceeded\n",stderr);
		exit(1);
	}
	nmax<<=1;
	quadtree** pp=new quadtree*[nmax];
	for(int i=0;i<nco;i++) pp[i]=nei[i];
	delete [] nei;
	nei=pp;
}

bool quadtree::compute_cell(voronoicell_2d &c,int j) {
	int i;
	double x=p[ps*j],y=p[ps*j+1],x1,y1,xlo,xhi,ylo,yhi;
	quadtree *q;

	parent.initialize_voronoicell(c,x,p[ps*j+1]);
	for(i=0;i<j;i++) {
		x1=p[ps*i]-x;
		y1=p[ps*i+1]-y;
		if(!c.nplane(x1,y1,x1*x1+y1*y1,id[i])) return false;
	}
	i++;
	while(i<co) {
		x1=p[ps*i]-x;
		y1=p[ps*i+1]-y;
		if(!c.nplane(x1,y1,x1*x1+y1*y1,id[i])) return false;
		i++;
	}

	unsigned int &bm=parent.bmask;
	bm++;
	if(bm==0) {
		reset_mask();
		bm=1;
	}

	mask=bm;
	deque<quadtree*> dq;
	for(i=0;i<nco;i++) {
		dq.push_back(nei[i]);
		nei[i]->mask=bm;
	}

	while(!dq.empty()) {

		q=dq.front();dq.pop_front();
		q->bound(xlo,xhi,ylo,yhi);
		xlo-=x;xhi-=x;
		ylo-=y;yhi-=y;

		if(xlo>0) {
			if(ylo>0) {
				if(corner_test(c,xlo,ylo,xhi,yhi)) continue;
			} else if(yhi<0) {
				if(corner_test(c,xlo,yhi,xhi,ylo)) continue;
			} else {
				if(edge_x_test(c,xlo,ylo,yhi)) continue;
			}
		} else if(xhi<0) {
			if(ylo>0) {
				if(corner_test(c,xhi,ylo,xlo,yhi)) continue;
			} else if(yhi<0) {
				if(corner_test(c,xhi,yhi,xlo,ylo)) continue;
			} else {
				if(edge_x_test(c,xhi,ylo,yhi)) continue;
			}
		} else {
			if(ylo>0) {
				if(edge_y_test(c,xlo,ylo,xhi)) continue;
			} else if(yhi<0) {
				if(edge_y_test(c,xlo,yhi,xhi)) continue;
			} else voro_fatal_error("Compute cell routine revisiting central block, which should never\nhappen.",VOROPP_INTERNAL_ERROR);
		}

		for(i=0;i<q->co;i++) {
			x1=q->p[ps*i]-x;
			y1=q->p[ps*i+1]-y;
			if(!c.nplane(x1,y1,x1*x1+y1*y1,id[i])) return false;
		}

		for(i=0;i<q->nco;i++) if(q->nei[i]->mask!=bm) {
			dq.push_back(q->nei[i]);
			q->nei[i]->mask=bm;
		}
	} 
	return true;
}

inline bool quadtree::corner_test(voronoicell_2d &c,double xl,double yl,double xh,double yh) {
	if(c.plane_intersects_guess(xl,yh,xl*xl+yl*yh)) return false;
	if(c.plane_intersects(xh,yl,xl*xh+yl*yl)) return false;
	return true;
}

inline bool quadtree::edge_x_test(voronoicell_2d &c,double xl,double y0,double y1) {
	if(c.plane_intersects_guess(xl,y0,xl*xl)) return false;
	if(c.plane_intersects(xl,y1,xl*xl)) return false;
	return true;
}

inline bool quadtree::edge_y_test(voronoicell_2d &c,double x0,double yl,double x1) {
	if(c.plane_intersects_guess(x0,yl,yl*yl)) return false;
	if(c.plane_intersects(x1,yl,yl*yl)) return false;
	return true;
}

}
