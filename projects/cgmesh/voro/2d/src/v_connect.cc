#include "v_connect.hh"
#include <math.h>
#include <stdio.h>

namespace voro{

void v_connect::import(FILE *fp){
	bool neg_label=false,boundary_track=false,start=false;
	char *buf(new char[512]);
	int i=0,id;
	double x, y,pad=.05;

	while(fgets(buf,512,fp)!=NULL) {

		if(strcmp(buf,"#Start\n")==0||strcmp(buf,"# Start\n")==0) {

			// Check that two consecutive start tokens haven't been
			// encountered
			if(boundary_track) voro_fatal_error("File import error - two consecutive start tokens found",VOROPP_FILE_ERROR);
			start=true;boundary_track=true;

		} else if(strcmp(buf,"#End\n")==0||strcmp(buf,"# End\n")==0||
			  strcmp(buf,"#End")==0||strcmp(buf,"# End")==0) {

			// Check that two consecutive end tokens haven't been
			// encountered
			if(start) voro_fatal_error("File import error - end token immediately after start token",VOROPP_FILE_ERROR);
			if(!boundary_track) voro_fatal_error("File import error - found end token without start token",VOROPP_FILE_ERROR);
			vbd[i-1]|=2;boundary_track=false;
		} else {
			if(!boundary_track && bd==-1) bd=i;
			// Try and read three entries from the line
			if(sscanf(buf,"%d %lg %lg",&id,&x,&y)!=3) voro_fatal_error("File import error #1",VOROPP_FILE_ERROR);
			vid.push_back(id);
			vpos.push_back(x);
			vpos.push_back(y);
			vbd.push_back(start?1:0);
			i++;

			// Determine bounds
			if(id<0) neg_label=true;
			if(id>mid) mid=id;
			if(x<minx) minx=x;
			if(x>maxx) maxx=x;
			if(y<miny) miny=y;
			if(y>maxy) maxy=y;

			start=false;
		}
	}

	if(boundary_track) voro_fatal_error("File import error - boundary not finished",VOROPP_FILE_ERROR);
	if(!feof(fp)) voro_fatal_error("File import error #2",VOROPP_FILE_ERROR);
	delete [] buf;

	// Add small amount of padding to container bounds
	double dx=maxx-minx,dy=maxy-miny;
	minx-=pad*dx;maxx+=pad*dx;dx+=2*pad*dx;
	miny-=pad*dy;maxy+=pad*dy;dy+=2*pad*dy;

	// Guess the optimal computationl grid, aiming at eight particles per
	// grid square
	double lscale=sqrt(8.0*dx*dy/i);
	nx=(int)(dx/lscale)+1,ny=(int)(dy/lscale)+1;
	ng=i;
	gen_to_vert= new vector<int>[i];
	gen_to_ed= new vector<int>[i];
	gen_to_gen_e= new vector<int>[i];
	gen_to_gen_v=new vector<int>[i];
	generator_is_vertex=new int[i];
	for(int j=0;j<ng;j++) generator_is_vertex[j]=-1;
	mp=new int[mid+1];
	for(int j=0;j<ng;j++) mp[vid[j]]=j;
}
// Assemble vert_to_gen,gen_to_vert,vert_to_ed,ed_to_vert
void v_connect::assemble_vertex(){
	bool arrange=false;
	int cv,lv,cvi,lvi,fvi,j,id,pcurrent_vertices=init_vertices,vert_size=0,g1,g2,g3,gl1,gl2,gl3,i,pne=0;
	int *pmap;
	int *ped_to_vert=new int[2*current_edges];
	bool seencv=false,seenlv=false;
	vector<int> gens;
	vector<int> *pvert_to_gen= new vector<int>[pcurrent_vertices];
	vector<int> *pgen_to_vert=new vector<int>[ng];
	vector<int> problem_verts;
	vector<int> problem_verts21;
	vector<int> problem_verts32;
	vector<int> problem_gen_to_vert;
	double gx1,gy1,gx2,gy2;
	double *pvertl=new double[2*pcurrent_vertices];
	unsigned int *globvertc=new unsigned int[2*(ng+1)*ng*ng];

	for(int i=0;i<2*((ng+1)*ng*ng);i++){
		globvertc[i]=0;
	}

	cout << "2.1" << endl;

	double x,y,vx,vy;
	// Create container
	container_boundary_2d con(minx,maxx,miny,maxy,nx,ny,false,false,16);

	// Import data
	for(j=0;j<vid.size();j++) {
		if(vbd[j]&1) con.start_boundary();
		con.put(vid[j],vpos[2*j],vpos[2*j+1]);
		if(vbd[j]&2) con.end_boundary();
	}

	// Carry out all of the setup prior to computing any Voronoi cells
	con.setup();
	con.full_connect_on();
	voronoicell_nonconvex_neighbor_2d c;
	c_loop_all_2d cl(con);

	//Compute Voronoi Cells, adding vertices to potential data structures (p*)
	if(cl.start()) do if(con.compute_cell(c,cl)){
		cv=0;
		lv=c.ed[2*cv+1];
		id=cl.pid();
		x=vpos[2*mp[id]];
		y=vpos[2*mp[id]+1];
		groom_vertexg(c);
		do{
			gens=c.vertexg[cv];
			vx=.5*c.pts[2*cv]+x;
			vy=.5*c.pts[2*cv+1]+y;
			if(gens.size()==1){
				seencv=false;
				if(pcurrent_vertices==vert_size){
					pcurrent_vertices<<=1;
					add_memory_vector(pvert_to_gen,pcurrent_vertices);
					add_memory_array(pvertl,pcurrent_vertices);

				}
				pgen_to_vert[gens[0]].push_back(vert_size);
				pvertl[2*vert_size]=vx;
				pvertl[2*vert_size+1]=vy;
				pvert_to_gen[vert_size]=gens;
				cvi=vert_size;
				if(cv==0) fvi=cvi;
				vert_size++;
			}else if(gens.size()==2){
				gx1=vpos[2*mp[gens[0]]]-(c.pts[2*cv]*.5+x);gy1=vpos[2*mp[gens[0]]+1]-(c.pts[2*cv+1]*.5+y);
				gx2=vpos[2*mp[gens[1]]]-(c.pts[2*cv]*.5+x);gy2=vpos[2*mp[gens[1]]+1]-(c.pts[2*cv+1]*.5+y);
				if((((gx1*gy2)-(gy1*gx2))>-tolerance) && ((gx1*gy2)-(gy1*gx2))<tolerance){
					if((vbd[mp[gens[0]]]|2)!=0 && vbd[mp[gens[1]]]==1){

						g1=gens[1]; g2=gens[0];
					}else if((vbd[mp[gens[1]]]|2)!=0 && vbd[mp[gens[0]]]==1){

						g1=gens[0]; g2=gens[1];
					}else{
						if(mp[gens[0]]<mp[gens[1]]){
							g1=gens[1];g2=gens[0];
						}else{
							g1=gens[0];g2=gens[1];
						}
					}
				}else if(((gx1*gy2)-(gy1*gx2))>0){
					g1=gens[0]; g2=gens[1];
				}else{
					g1=gens[1]; g2=gens[0];
				}
				if(globvertc[2*((ng*ng*g1)+(ng*g2))]!=0){
					seencv=true;
					globvertc[2*((ng*ng*g1)+(ng*g2))]+=1;
					cvi=globvertc[2*((ng*ng*g1)+(ng*g2))+1];
					pgen_to_vert[id].push_back(cvi);
					if(cv==0) fvi=cvi;
				}else{
					seencv=false;
					if(pcurrent_vertices==vert_size){
						pcurrent_vertices<<=1;
						add_memory_vector(pvert_to_gen,pcurrent_vertices);
						add_memory_array(pvertl,pcurrent_vertices);

					}
	                                pgen_to_vert[id].push_back(vert_size);
					pvertl[2*vert_size]=vx;
					pvertl[2*vert_size+1]=vy;
					pvert_to_gen[vert_size]=gens;
					globvertc[2*((ng*ng*g1)+(ng*g2))]=(unsigned int)1;
					globvertc[2*((ng*ng*g1)+(ng*g2))+1]=vert_size;
					cvi=vert_size;
					if(cv==0) fvi=cvi;
					vert_size++;
				}
			}else{
				arrange_cc_x_to_gen(gens,vx,vy);
				g1=gens[0];g2=gens[1];g3=gens[2];
				if(g1<g2 && g1<g3){
					gl1=g1;gl2=g2;gl3=g3;
				}else if(g2<g1 && g2<g3){
					gl1=g2;gl2=g3;gl3=g1;
				}else{
					gl1=g3;gl2=g1;gl3=g2;
				}
				if(globvertc[2*((ng*ng*gl1)+(ng*gl2)+gl3)]!=0){
					seencv=true;
					globvertc[2*((ng*ng*gl1)+(ng*gl2)+gl3)]+=1;
					cvi=globvertc[2*((ng*ng*gl1)+(ng*gl2)+gl3)+1];
					pgen_to_vert[id].push_back(cvi);
					if(cv==0)fvi=cvi;
				}else{
					seencv=false;
					if(pcurrent_vertices==vert_size){
						pcurrent_vertices<<=1;
						add_memory_vector(pvert_to_gen,pcurrent_vertices);
						add_memory_array(pvertl,pcurrent_vertices);

					}
	                                pgen_to_vert[id].push_back(vert_size);
					pvertl[2*vert_size]=vx;
					pvertl[2*vert_size+1]=vy;
					pvert_to_gen[vert_size]=gens;
					globvertc[2*((ng*ng*gl1)+(ng*gl2)+gl3)]=(unsigned int)1;
					globvertc[2*((ng*ng*gl1)+(ng*gl2)+gl3)+1]=vert_size;
					cvi=vert_size;
					if(cv==0)fvi=cvi;
					vert_size++;
				}
			}

				//add edges to potential edge structures
				if(cv!=0){
					if(pne==current_edges){
							current_edges<<=1;
							add_memory_array(ped_to_vert,current_edges);
					}
					if(cvi<lvi){
						ped_to_vert[2*pne]=cvi;
						ped_to_vert[2*pne+1]=lvi;
					}else{
						ped_to_vert[2*pne]=lvi;
						ped_to_vert[2*pne+1]=cvi;
					}
					pne++;
				}

				lv=cv;
				lvi=cvi;
				seenlv=seencv;
				cv=c.ed[2*lv];

		}while(cv!=0);
			//deal with final edge (last vertex to first vertex)
			if(pne==current_edges){
				current_edges<<=1;
				add_memory_array(ped_to_vert,current_edges);
			}
			if(fvi<lvi){
				ped_to_vert[2*pne]=fvi;
				ped_to_vert[2*pne+1]=lvi;
			}else{
				ped_to_vert[2*pne]=lvi;
				ped_to_vert[2*pne+1]=fvi;
			}
			pne++;

	}while (cl.inc());
	cout << "2.2" << endl;
	//Add non-problem vertices(connectivity<=3) to class variables, add problem vertices to problem vertice data structures
	pmap=new int[pcurrent_vertices];
	j=0;
	int v1=0;
	for(int i=0;i<vert_size;i++){
		gens=pvert_to_gen[i];
		if(gens.size()==1){
			if(j==current_vertices){
				add_memory_vertices();
			}
			generator_is_vertex[gens[0]]=j;
			vertex_is_generator[j]=gens[0];
			vertl[2*j]=pvertl[2*i];
			vertl[2*j+1]=pvertl[2*i+1];
			vert_to_gen[j]=pvert_to_gen[i];
			pmap[i]=j;
			j++;
			v1++;
		}
		else if(gens.size()==2){
			gx1=vpos[2*mp[gens[0]]]-(pvertl[2*i]);gy1=vpos[2*mp[gens[0]]+1]-(pvertl[2*i+1]);
			gx2=vpos[2*mp[gens[1]]]-(pvertl[2*i]);gy2=vpos[2*mp[gens[1]]+1]-(pvertl[2*i+1]);
			if((((gx1*gy2)-(gy1*gx2))>-tolerance) && ((gx1*gy2)-(gy1*gx2))<tolerance){
				if((vbd[mp[gens[0]]]|2)!=0 && vbd[mp[gens[1]]]==1){
					g1=gens[1]; g2=gens[0];
				}else if((vbd[mp[gens[1]]]|2)!=0 && vbd[mp[gens[0]]]==1){
					g1=gens[0]; g2=gens[1];
				}else{
					if(mp[gens[0]]<mp[gens[1]]){
						g1=gens[1];g2=gens[0];
					}else{
						g1=gens[0];g2=gens[1];
					}
				}
			}else if(((gx1*gy2)-(gy1*gx2))>0){
				g1=gens[0]; g2=gens[1];
			}else{
				g1=gens[1]; g2=gens[0];
			}
			if(globvertc[2*((ng*ng*g1)+(ng*g2))]!=2){
				problem_verts21.push_back(i);
			}
			else{
				if(j==current_vertices){
					add_memory_vertices();
				}
				vertl[2*j]=pvertl[2*i];
				vertl[2*j+1]=pvertl[2*i+1];
				vert_to_gen[j]=pvert_to_gen[i];
				pmap[i]=j;
				j++;
			}
		}else{
			g1=gens[0];g2=gens[1];g3=gens[2];
			if(g1<g2 && g1<g3){
				gl1=g1;gl2=g2;gl3=g3;
			}else if(g2<g1 && g2<g3){
				gl1=g2;gl2=g3;gl3=g1;
			}else{
				gl1=g3;gl2=g1;gl3=g2;
			}
			if(globvertc[2*((ng*ng*gl1)+(ng*gl2)+gl3)]!=3){
				if(globvertc[2*((ng*ng*gl1)+(ng*gl2)+gl3)]==1) problem_verts.push_back(i);
				else problem_verts32.push_back(i);
			}else{
				if(j==current_vertices){
					add_memory_vertices();
				}
				vertl[2*j]=pvertl[2*i];
				vertl[2*j+1]=pvertl[2*i+1];
				vert_to_gen[j]=pvert_to_gen[i];
				pmap[i]=j;
				j++;
			}
		}

	}
	cout << "2.3\n problemverts21.size()=" << problem_verts21.size() << "\nproblem_verts32.size()=" << problem_verts32.size() << "\nproblem_verts.size()=" << problem_verts.size() << endl;
	nv=j;
	degenerate_vertices=j;
	//deal with problem verts
	while(problem_verts21.size()>problem_verts32.size()){
		for(int i=0;i<problem_verts21.size();i++){
			for(int j=i+1;j<problem_verts21.size();j++){
				one_in_common(pvert_to_gen[problem_verts21[i]],pvert_to_gen[problem_verts21[j]],g1);
				if(g1==-1) continue;
				gens=pvert_to_gen[problem_verts21[i]];
				gens.push_back(not_this_one(pvert_to_gen[problem_verts21[j]],g1));
				for(int k=0;k<problem_verts.size();k++){
					if(contain_same_elements(gens,pvert_to_gen[problem_verts[k]])){
						if(nv==current_vertices) add_memory_vertices();
						vertl[2*nv]=pvertl[2*problem_verts[k]];
						vertl[2*nv+1]=pvertl[2*problem_verts[k]+1];
						vert_to_gen[nv]=pvert_to_gen[problem_verts[k]];
						pmap[problem_verts21[i]]=nv;
						pmap[problem_verts21[j]]=nv;
						pmap[problem_verts[k]]=nv;
						nv++;
						problem_gen_to_vert.push_back(problem_verts21[i]);
						problem_gen_to_vert.push_back(problem_verts21[j]);
						problem_verts21.erase(problem_verts21.begin()+i);
						problem_verts21.erase(problem_verts21.begin()+(j-1));
						problem_verts.erase(problem_verts.begin()+k);
						break;
					}
				}
			}
		}
	}
	cout << "part 2" << endl;
	while(problem_verts32.size()>0){
		for(int i=0;i<problem_verts32.size();i++){
			gens=pvert_to_gen[problem_verts32[i]];
			for(int j=0;j<problem_verts21.size();j++){
				if(subset(pvert_to_gen[problem_verts21[j]],gens)){
					if(nv==current_vertices) add_memory_vertices();
					vertl[2*nv]=pvertl[2*problem_verts32[i]];
					vertl[2*nv+1]=pvertl[2*problem_verts32[i]+1];
					vert_to_gen[nv]=gens;
					pmap[problem_verts32[i]]=nv;
					pmap[problem_verts21[j]]=nv;
					nv++;
					problem_gen_to_vert.push_back(problem_verts21[j]);
					problem_verts21.erase(problem_verts21.begin()+j);
					problem_verts32.erase(problem_verts32.begin()+i);
					break;
				}
			}
		}
	}

	double standard,distance;
	cout << "part3" << endl;
	while(problem_verts.size()>0){

		if(nv==current_vertices) add_memory_vertices();
		gens=pvert_to_gen[problem_verts[0]];
		vx=pvertl[2*problem_verts[0]];vy=pvertl[2*problem_verts[0]+1];
		standard=pow(vx-vpos[2*mp[gens[0]]],2)+pow(vy-vpos[2*mp[gens[0]]+1],2);
		g1=gens[0];g2=gens[1];
		pmap[problem_verts[0]]=nv;
		problem_verts.erase(problem_verts.begin());
		i=0;
		while(true){
			g3=not_these_two(pvert_to_gen[problem_verts[i]],g1,g2);
			if(g3!=-1){
				distance=pow(vx-vpos[2*mp[g3]],2)+pow(vy-vpos[2*mp[g3]+1],2);
			}
			if(contains_two(pvert_to_gen[problem_verts[i]],g1,g2) &&
				(distance<(standard+tolerance) && (distance>(standard-tolerance)))){
				if(g3==gens[0]){
					pmap[problem_verts[i]]=nv;
					problem_verts.erase(problem_verts.begin()+i);
					break;
				}
				if(g3!=gens[2]) gens.push_back(g3);
				pmap[problem_verts[i]]=nv;
				g2=g3;
				g1=pvert_to_gen[problem_verts[i]][0];
				if(problem_verts.size()>1){
					problem_verts.erase(problem_verts.begin()+i);
				}else{
					break;
				}
				i=0;
			}else{
				i++;
			}
		}

		vertl[2*nv]=vx;
		vertl[2*nv+1]=vy;
		arrange_cc_x_to_gen(gens,vx,vy);
		vert_to_gen[nv]=gens;
		nv++;

	}
	delete [] pvert_to_gen;
	delete [] pvertl;
	delete [] globvertc;
	cout << "2.4" << endl;
	//assemble edge data structures
	ed_to_vert=new int[2*pne];
	vert_to_ed=new vector<int>[nv];
	unsigned int* globedgec=new unsigned int[nv*nv];
	for(int i=0;i<(nv*nv);i++){
		globedgec[i]=0;
	}
	for(int i=0;i<pne;i++){
		g1=pmap[ped_to_vert[2*i]];g2=pmap[ped_to_vert[2*i+1]];
		if(g2<g1){
			g2^=g1;
			g1^=g2;
			g2^=g1;
		}
		if(globedgec[(nv*g1+g2)]!=0){
			continue;
		}else{
			globedgec[(nv*g1+g2)]=1;
			ed_to_vert[2*ne]=g1;
			ed_to_vert[2*ne+1]=g2;
			vert_to_ed[g1].push_back(ne);
			vert_to_ed[g2].push_back(ne);
			ne++;
		}
	}
	for(int i=0;i<ng;i++){
		for(int k=0;k<pgen_to_vert[i].size();k++){
			gen_to_vert[i].push_back(pmap[pgen_to_vert[i][k]]);
			if(contains(problem_gen_to_vert,pgen_to_vert[i][k])) arrange=true;
		}
		if(arrange) arrange_cc_gen_to_vert(gen_to_vert[i],vpos[2*mp[i]],vpos[2*mp[i]+1]);
		arrange=false;
	}

	delete [] pgen_to_vert;
	delete [] pmap;
	delete [] ped_to_vert;
	delete [] globedgec;
	cout << "out" << endl;
}
//assemble gen_to_gen , gen_to_ed , ed_to_gen
void v_connect::assemble_gen_ed(){
	cout << "gen_ed 1" << endl;
	ed_to_gen=new vector<int>[ne];
	//though neither ed_on_bd or vert_on_bd are modified during this method, they are initialized here in case the user
	//does not require boundary information and will not be calling assemble_boundary
	ed_on_bd=new vector<int>[ne];
	vert_on_bd=new vector<int>[nv];
	vector<int> gens;
	int g1,g2,v1,v2,j,vi;
	double gx1,gy1,gx2,gy2,vx1,vy1,vx2,vy2;
		cout << "gen_ed 2" << endl;
	for(int i=0;i<ne;i++){

		v1=ed_to_vert[2*i];v2=ed_to_vert[2*i+1];
		two_in_common(vert_to_gen[v1],vert_to_gen[v2],g1,g2);
		if(g1!=-1 && g2!=-1){
			gen_to_gen_e[g1].push_back(g2);
			gen_to_gen_e[g2].push_back(g1);
			for(int k=0;k<gen_to_vert[g1].size();k++){
				if(gen_to_vert[g1][k]==v1){
					vi=k;
					break;
				}
			}
			if((vi!=(gen_to_vert[g1].size()-1) && gen_to_vert[g1][vi+1]==v2) ||
				(vi==(gen_to_vert[g1].size()-1) && gen_to_vert[g1][0]==v2)){
				ed_to_gen[i].push_back(g1);
				ed_to_gen[i].push_back(g2);
				gen_to_ed[g1].push_back(i);
				gen_to_ed[g2].push_back(~i);
			}else{
				ed_to_gen[i].push_back(g2);
				ed_to_gen[i].push_back(g1);
				gen_to_ed[g2].push_back(i);
				gen_to_ed[g1].push_back(~i);
			}
		}else{
			if(g1==-1) continue;
			ed_to_gen[i].push_back(g1);
			vx1=vertl[2*v1];vy1=vertl[2*v1+1];
			vx2=vertl[2*v2];vy2=vertl[2*v2+1];
			gx1=vpos[2*mp[g1]];gy1=vpos[2*mp[g1]+1];
			for(int k=0;k<gen_to_vert[g1].size();k++){
				if(gen_to_vert[g1][k]==v1){
					vi=k;
					break;
				}
			}
			if((vi!=(gen_to_vert[g1].size()-1) && gen_to_vert[g1][vi+1]==v2) ||
				(vi==(gen_to_vert[g1].size()-1) && gen_to_vert[g1][0]==v2)) gen_to_ed[g1].push_back(i);
			else gen_to_ed[g1].push_back(~i);
		}
	}
	cout << "gen_ed 3" << endl;
	for(int i=0;i<nv;i++){
		if(vert_to_ed[i].size()<=3) continue;
		else{
			gens=vert_to_gen[i];
			for(int j=0;j<gens.size();j++){
				for(int k=0;k<gens.size();k++){
					if(!contains(gen_to_gen_e[gens[j]],gens[k])) gen_to_gen_v[gens[j]].push_back(gens[k]);

				}
			}
		}
	}
	cout << "gen_ed 4" << endl;
	//arrange gen_to_ed gen_to_gen_e gen_to_gen_v counterclockwise
	for(int i=0;i<ng;i++){
		gx1=vpos[2*mp[i]];gy1=vpos[2*mp[i]+1];
		arrange_cc_gen_to_ed(gen_to_ed[i]);
		arrange_cc_x_to_gen(gen_to_gen_e[i],gx1,gy1);
		arrange_cc_x_to_gen(gen_to_gen_v[i],gx1,gy1);
	}
	cout << "gen_ed 5" << endl;
	//arrange vert_to_ed cc
	for(int i=0;i<nv;i++){
		gx1=vertl[2*i];gy1=vertl[2*i+1];
		arrange_cc_vert_to_ed(vert_to_ed[i],gx1,gy1,i);
	}

}

//assemble vert_on_bd and ed_on_bd as well as side edge information if neccessary.
void v_connect::assemble_boundary(){
	bool begun=false;
	int i=0,cg,ng,fg,lv,cv,nv,ev,ei,j;
	while(true){
		if(vbd[i]==1){
			begun=true;
			fg=mp[i];
			cg=fg;
			ng=mp[i+1];
			ev=gen_to_vert[ng][0];
		}else if(vbd[i]==2){
			begun=false;
			cg=mp[i];
			ng=fg;
			ev=gen_to_vert[ng][0];
		}else{
			if(!begun) break;
			cg=mp[i];
			ng=mp[i+1];
			ev=gen_to_vert[ng][0];
		}
		cv=gen_to_vert[cg][0];
		nv=gen_to_vert[cg][1];
		one_in_common(vert_to_ed[cv],vert_to_ed[nv],ei);
		vert_on_bd[cv].push_back(cg);
		vert_on_bd[cv].push_back(ng);
		vert_on_bd[nv].push_back(cg);
		vert_on_bd[nv].push_back(ng);
		ed_on_bd[ei].push_back(cg);
		ed_on_bd[ei].push_back(ng);
		lv=cv;
		cv=nv;
		while(nv!=ev){
			j=0;
			while(true){
				if(ed_to_vert[2*vert_to_ed[cv][j]]==lv) break;
				else if(ed_to_vert[2*vert_to_ed[cv][j]+1]==lv) break;
				j++;
			}
			if(j==(vert_to_ed[cv].size()-1)) ei=vert_to_ed[cv][0];
			else ei=vert_to_ed[cv][j+1];
			if(ed_to_vert[2*ei]==cv) nv=ed_to_vert[2*ei+1];
			else nv=ed_to_vert[2*ei];
			vert_on_bd[nv].push_back(cg);
			vert_on_bd[nv].push_back(ng);
			ed_on_bd[ei].push_back(cg);
			ed_on_bd[ei].push_back(ng);
			lv=cv;
			cv=nv;
		}
		i++;
	}

}

//(x,y)=my coordinates
//(vx,vy)=vertex coordinates
vector<int> v_connect::groom_vertexg_help(double x,double y,double vx, double vy,vector<int> &g){
	if(g.size()<2) return g;
	bool rightside=false;
	int g0=g[0],g1,bestg,besti;
	double d1;
	double standard=pow((vpos[2*mp[g0]]-vx),2)+pow((vpos[2*mp[g0]+1]-vy),2);
	double gx0,gy0,gx1,gy1,best,current;
	vector<int> newg,temp;
	temp.push_back(g0);
	for(int i=1;i<g.size();i++){
		g1=g[i];
		if(contains(temp,g1)) continue;
		d1=pow((vpos[2*mp[g1]]-vx),2)+pow((vpos[2*mp[g1]+1]-vy),2);
		if(d1<(standard+tolerance)){
			temp.push_back(g1);
		}
	}
	if(temp.size()<3) return temp;
	gx0=vpos[2*mp[g0]]; gy0=vpos[2*mp[g0]+1];
	newg.push_back(g0);
	//find right hand
	g1=temp[1];
	gx1=vpos[2*mp[g1]]; gy1=vpos[2*mp[g1]+1];
	if(cross_product(gx0-vx,gy0-vy,gx1-vx,gy1-vy)>0){
		rightside=true;
	}
	best=dot_product(gx0-vx,gy0-vy,gx1-vx,gy1-vy);
	bestg=g1;
	besti=1;
	for(int i=2;i<temp.size();i++){
		g1=temp[i];
		if(contains(newg,g1)) continue;
		gx1=vpos[2*mp[g1]]; gy1=vpos[2*mp[g1]+1];
		if(cross_product(gx0-vx,gy0-vy,gx1-vx,gy1-vy)>0){
			if(!rightside){
				rightside=true;
				best=dot_product(gx0-vx,gy0-vy,gx1-vx,gy1-vy);
				bestg=g1;
				besti=i;
			}else{
				current=dot_product(gx0-vx,gy0-vy,gx1-vx,gy1-vy);
				if(current>best){
					best=current;
					bestg=g1;
					besti=i;
				}
			}
		}else{
			if(rightside) continue;
			else{
				current=dot_product(gx0-vx,gy0-vy,gx1-vx,gy1-vy);
				if(current<best){
					best=current;
					bestg=g1;
					besti=i;
				}
			}
		}
	}
	if(!contains(newg,bestg)) newg.push_back(bestg);

	//find left hand
	rightside=false;
	g1=temp[1];
	gx1=vpos[2*mp[g1]]; gy1=vpos[2*mp[g1]+1];
	if(cross_product(gx0-vx,gy0-vy,gx1-vx,gy1-vy)<0){
		rightside=true;
	}
	best=dot_product(gx0-vx,gy0-vy,gx1-vy,gy1-vy);
	bestg=g1;
	for(int i=2;i<temp.size();i++){
		g1=temp[i];
		if(contains(newg,g1)) continue;
		gx1=vpos[2*mp[g1]]; gy1=vpos[2*mp[g1]+1];
		if(cross_product(gx0-vx,gy0-vy,gx1-vx,gy1-vy)<0){
			if(!rightside){
				rightside=true;
				best=dot_product(gx0-vx,gy0-vy,gx1-vx,gy1-vy);
				bestg=g1;
			}else{
				current=dot_product(gx0-vx,gy0-vy,gx1-vx,gy1-vy);
				if(current>best){
					best=current;
					bestg=g1;
				}
			}
		}else{
			if(rightside) continue;
			else{
				current=dot_product(gx0-vx,gy0-vx,gx1-vx,gy1-vy);
				if(current<best){
					best=current;
					bestg=g1;
				}
			}
		}
	}
	if(!contains(newg,bestg)) newg.push_back(bestg);
	if(newg.size()<3) return groom_vertexg_help2(x,y,vx,vy,g);
	return newg;

}
vector<int> v_connect::groom_vertexg_help2(double x,double y,double vx,double vy,vector<int> &g){
	if(g.size()<2) return g;
	int m0=g[0],m1=g[1],m2,p,i=1;
	double d0=pow((vpos[2*mp[m0]]-x),2)+pow((vpos[2*mp[m0]+1]-y),2);
	double d1=pow((vpos[2*mp[m1]]-vx),2)+pow((vpos[2*mp[m1]+1]-vy),2);
	double d2;
	double standard=pow((vpos[2*mp[m0]]-vx),2)+pow((vpos[2*mp[m0]+1]-vy),2);
	double dp,dcompare, temp;
	vector<int> newg;
	while(d1>=standard+tolerance){
		if(i==g.size()-1){
			newg.push_back(m0);
			return newg;
		}
			i++;
			m1=g[i];
			d1=pow((vpos[2*mp[m1]]-vx),2)+pow((vpos[2*mp[m1]+1]-vy),2);
	}
	if(i==g.size()-1){
		newg.push_back(m0);
		newg.push_back(m1);
		return newg;
	}
	i++;
	m2=g[i];
	d2=pow((vpos[2*mp[m2]]-vx),2)+pow((vpos[2*mp[m2]+1]-vy),2);
	while(d2>=standard+tolerance){
		if(i==g.size()-1){
			newg.push_back(m0);
			newg.push_back(m1);
			return newg;
		}
			i++;
			m2=g[i];
			d2=pow((vpos[2*mp[m2]]-vx),2)+pow((vpos[2*mp[m2]+1]-vy),2);
	}
	if(i==g.size()-1){
		newg.push_back(m0);
		newg.push_back(m1);
		newg.push_back(m2);
		return newg;
	}
	i++;
	d1=pow((vpos[2*mp[m1]]-x),2)+pow((vpos[2*mp[m1]+1]-y),2);
	d2=pow((vpos[2*mp[m2]]-x),2)+pow((vpos[2*mp[m2]+1]-y),2);
	if(d0<d2 && d2<d1){
		temp=d1;
		d1=d2;
		d2=temp;
		m2^=m1;
		m1^=m2;
		m2^=m1;
	}else if(d1<d0 && d0<d2){
		temp=d1;
		d1=d0;
		d0=temp;
		m0^=m1;
		m1^=m0;
		m0^=m1;
	}else if(d1<d2 && d2<d0){
		temp=d1;
		d1=d0;
		d0=temp;
		m0^=m1;
		m1^=m0;
		m0^=m1;
		temp=d1;
		d1=d2;
		d2=temp;
		m2^=m1;
		m1^=m2;
		m2^=m1;
	}else if(d2<d1 && d1<d0){
		temp=d2;
		d2=d0;
		d0=temp;
		m0^=m2;
		m2^=m0;
		m0^=m2;
	}else if(d2<d0 && d0<d1){
		temp=d2;
		d2=d0;
		d0=temp;
		m0^=m2;
		m2^=m0;
		m0^=m2;
		temp=d1;
		d1=d2;
		d2=temp;
		m2^=m1;
		m1^=m2;
		m2^=m1;
	}
	for(int j=i;j<g.size();j++){
		p=g[j];
		dcompare=pow((vpos[2*mp[p]]-vx),2)+pow((vpos[2*mp[p]+1]-vy),2);
		if(dcompare<=(standard+tolerance)){
			dp=pow((vpos[2*mp[p]]-x),2)+pow((vpos[2*mp[p]+1]-y),2);
			if(dp<d2){
				temp=d2;
				d2=dp;
				dp=temp;
				p^=m2;
				m2^=p;
				p^=m2;
				if(d2<d1){
					temp=d1;
					d1=d2;
					d2=temp;
					m2^=m1;
					m1^=m2;
					m2^=m1;
					if(d1<d0){
						temp=d1;
						d1=d0;
						d0=temp;
						m0^=m1;
						m1^=m0;
						m0^=m1;
					}
				}
			}
		}
	}
	newg.push_back(m0);
	newg.push_back(m1);
	newg.push_back(m2);
	return newg;
}

	void v_connect::arrange_cc_x_to_gen(vector<int> &list,double cx,double cy){
		if(list.size()==0) return;
		bool wrongside;
		int g1,ng,ni;
		double x1,y1,x2,y2,best,current;
		vector<int> newlist;
		vector<int> potential;
		newlist.push_back(list[0]);
		x1=vpos[2*mp[list[0]]];y1=vpos[2*mp[list[0]]+1];
		list.erase(list.begin());
		while(list.size()>0){
			wrongside=true;
			for(int i=0;i<list.size();i++){
				g1=list[i];
				x2=vpos[2*mp[g1]];y2=vpos[2*mp[g1]+1];
				if(cross_product(cx-x1,cy-y1,cx-x2,cy-y2)>=0){
					current=dot_product(cx-x1,cy-y1,cx-x2,cy-y2);
					if(wrongside){
						ng=g1;
						ni=i;
						best=current;
						wrongside=false;
					}else if(current>best){
						best=current;
						ng=g1;
						ni=i;
					}
				}else{
					if(!wrongside) continue;
					current=dot_product(cx-x1,cy-y1,cx-x2,cy-y2);
					if(i==0){
						best=current;
						ng=g1;
						ni=i;
					}else if(current<best){
						best=current;
						ng=g1;
						ni=i;
					}
				}
			}
			newlist.push_back(ng);
			list.erase(list.begin()+ni);
		}
		list=newlist;
	}

void v_connect::arrange_cc_gen_to_vert(vector<int> &list,double cx,double cy){
		if(list.size()==0) return;
		bool wrongside;
		int g1,ng,ni;
		double x1,y1,x2,y2,best,current;
		vector<int> newlist;
		vector<int> potential;
		newlist.push_back(list[0]);
		x1=vertl[2*list[0]];y1=vertl[2*list[0]+1];
		list.erase(list.begin());
		while(list.size()>0){
			wrongside=true;
			for(int i=0;i<list.size();i++){
				g1=list[i];
				x2=vertl[2*g1];y2=vertl[2*g1+1];
				if(cross_product(cx-x1,cy-y1,cx-x2,cy-y2)>=0){
					current=dot_product(cx-x1,cy-y1,cx-x2,cy-y2);
					if(wrongside){
						ng=g1;
						ni=i;
						best=current;
						wrongside=false;
					}else if(current>best){
						best=current;
						ng=g1;
						ni=i;
					}
				}else{
					if(!wrongside) continue;
					current=dot_product(cx-x1,cy-y1,cx-x2,cy-y2);
					if(i==0){
						best=current;
						ng=g1;
						ni=i;
					}else if(current<best){
						best=current;
						ng=g1;
						ni=i;
					}
				}
			}
			newlist.push_back(ng);
			list.erase(list.begin()+ni);
		}
		list=newlist;
}

void v_connect::arrange_cc_gen_to_ed(vector<int> &list){
	vector<int> newlist;
	int v1,v2,ed,i=0;
	newlist.push_back(list[0]);
	list.erase(list.begin());
	if(newlist[0]<0){
		ed=~newlist[0];
		v1=ed_to_vert[2*ed];
	}else{
		ed=newlist[0];
		v1=ed_to_vert[2*ed+1];
	}
	while(list.size()>0){
			if(list[i]>=0){
				v2=ed_to_vert[2*list[i]];
				if(v2==v1){
					v1=ed_to_vert[2*list[i]+1];
					newlist.push_back(list[i]);
					list.erase(list.begin()+i);
					i=0;
				}else i++;
			}else{
				v2=ed_to_vert[2*(~list[i])+1];

				if(v2==v1){
					v1=ed_to_vert[2*(~list[i])];
					newlist.push_back(list[i]);
					list.erase(list.begin()+i);
					i=0;
				}else i++;
			}
	}
	list=newlist;
}

void v_connect::arrange_cc_vert_to_ed(vector<int> &list,double cx, double cy,int id){

	if(list.size()==0) return;
	bool wrongside;
	int g1,ng,ni,index;
	double x1,y1,x2,y2,best,current;
	vector<int> newlist;
	vector<int> potential;
	newlist.push_back(list[0]);
	if(ed_to_vert[2*list[0]]==id) index=ed_to_vert[2*list[0]+1];
	else index=ed_to_vert[2*list[0]];
	x1=vertl[2*index];y1=vertl[2*index+1];
	list.erase(list.begin());
	while(list.size()>0){
		wrongside=true;
		for(int i=0;i<list.size();i++){
			g1=list[i];
			if(ed_to_vert[2*g1]==id) index=ed_to_vert[2*g1+1];
			else index=ed_to_vert[2*g1];
			x2=vertl[2*index];y2=vertl[2*index+1];
			if(cross_product(cx-x1,cy-y1,cx-x2,cy-y2)>=0){
				current=dot_product(cx-x1,cy-y1,cx-x2,cy-y2);
				if(wrongside){
					ng=g1;
					ni=i;
					best=current;
					wrongside=false;
				}else if(current>best){
					best=current;
					ng=g1;
					ni=i;
				}
			}else{
				if(!wrongside) continue;
				current=dot_product(cx-x1,cy-y1,cx-x2,cy-y2);
				if(i==0){
					best=current;
					ng=g1;
					ni=i;
				}else if(current<best){
					best=current;
					ng=g1;
					ni=i;
				}
			}
		}
		newlist.push_back(ng);
		list.erase(list.begin()+ni);
	}
	list=newlist;
}

void v_connect::draw_gnu(FILE *fp){
	int vert;
	for(int i=0;i<ng;i++){
		fprintf(fp,"# cell number %i\n",i);
		for(int j=0;j<gen_to_vert[i].size();j++){
			vert=gen_to_vert[i][j];
			fprintf(fp,"%g %g\n",vertl[2*vert],vertl[2*vert+1]);
		}
		fprintf(fp,"%g %g\n",vertl[2*gen_to_vert[i][0]],vertl[2*gen_to_vert[i][0]+1]);
		fprintf(fp,"\n");
	}
}

void v_connect::draw_vtg_gnu(FILE *fp){
	double vx, vy, gx, gy;
	int l2;
	for(int i=0;i<ng;i++){
		gx=vpos[2*mp[i]];gy=vpos[2*mp[i]+1];
		for(int k=0;k<gen_to_vert[i].size();k++){
			vx=vertl[2*gen_to_vert[i][k]];vy=vertl[2*gen_to_vert[i][k]+1];
			fprintf(fp, "%g %g\n %g %g\n\n\n",vx,vy,gx,gy);
		}
	}
}

void v_connect::draw_gen_gen(FILE *fp){
	int g2;
	double gx1,gy1,gx2,gy2;
	for(int i=0;i<ng;i++){
		gx1=vpos[2*mp[i]];gy1=vpos[2*mp[i]+1];
		for(int k=0;k<gen_to_gen_e[i].size();k++){
			g2=gen_to_gen_e[i][k];
			gx2=vpos[2*mp[g2]];gy2=vpos[2*mp[g2]+1];
			fprintf(fp, "%g %g\n %g %g\n\n\n",gx1,gy1,gx2,gy2);
		}
		for(int k=0;k<gen_to_gen_v[i].size();k++){
			g2=gen_to_gen_v[i][k];
			gx2=vpos[2*mp[g2]];gy2=vpos[2*mp[g2]+1];
			fprintf(fp, "%g %g\n %g %g \n\n\n",gx1,gy1,gx2,gy2);
		}
	}
}

void v_connect::label_vertices(FILE *fp){
	double vx,vy;
	for(int i=0;i<nv;i++){
		vx=vertl[2*i];vy=vertl[2*i+1];
		fprintf(fp,"set label '%i' at %g,%g point lt 2 pt 3 ps 2 offset -3,3\n",i,vx,vy);
	}
}

void v_connect::label_generators(FILE *fp){
	double gx,gy;
	for(int i=0;i<vid.size();i++){
		gx=vpos[2*i];gy=vpos[2*i+1];
		fprintf(fp,"set label '%i' at %g,%g point lt 4 pt 4 ps 2 offset 3,-3\n",vid[i],gx,gy);
	}
}

void v_connect::label_edges(FILE *fp){
	double ex,ey,vx1,vy1,vx2,vy2;
	for(int i=0;i<ne;i++){
		vx1=vertl[2*ed_to_vert[2*i]];vy1=vertl[2*ed_to_vert[2*i]+1];
		vx2=vertl[2*ed_to_vert[2*i+1]];vy2=vertl[2*ed_to_vert[2*i+1]+1];
		ex=(vx1+vx2)/2; ey=(vy1+vy2)/2;
		fprintf(fp,"set label '%i' at %g,%g point lt 3 pt 1 ps 2 offset 0,-3\n",i,ex,ey);
	}
}

void v_connect::label_centroids(FILE *fp){
	double x,y;
	for(int i=0;i<ng;i++){
		centroid(i,x,y);
		fprintf(fp,"set label '%i' at %g,%g point lt 5 pt 5 ps 2 offset 0,3\n",i,x,y);
	}
}

void v_connect::print_gen_to_ed_table(FILE *fp){
	fprintf(fp,"generator to edge connectivity, arranged counterclockwise. Negative edge number means edge with reverse orientation\n\n");
	for(int i=0;i<ng;i++){
		fprintf(fp,"generator %i\n", i);
		for(int k=0;k<gen_to_ed[i].size();k++){
			if(gen_to_ed[i][k]>=0) fprintf(fp,"\t %i\n",gen_to_ed[i][k]);
			else fprintf(fp,"\t -%i\n",~gen_to_ed[i][k]);
		}
		fprintf(fp,"\n\n");
	}
}

void v_connect::print_gen_to_vert_table(FILE *fp){
	fprintf(fp,"generator to vertex connectivity, arranged conterclockwise\n\n");
	for(int i=0;i<ng;i++){
		fprintf(fp,"generator %i\n",i);
		for(int k=0;k<gen_to_vert[i].size();k++){
			fprintf(fp,"\t %i\n",gen_to_vert[i][k]);
		}
	}
}

void v_connect::print_vert_to_gen_table(FILE *fp){
	fprintf(fp,"vertex to generator connectivity, arranged counterclockwise\n\n");
	for(int i=0;i<nv;i++){
		fprintf(fp,"vertex %i\n",i);
		for(int k=0;k<vert_to_gen[i].size();k++){
			fprintf(fp,"\t %i\n",vert_to_gen[i][k]);
		}
	}
}

void v_connect::print_ed_to_gen_table(FILE *fp){
	fprintf(fp,"edge to generator connectivity, arranged left-side, right-side\n\n");
	for(int i=0;i<ne;i++){
		fprintf(fp,"ed %i\n",i);
		for(int k=0;k<ed_to_gen[i].size();k++){
			fprintf(fp,"\t %i\n",ed_to_gen[i][k]);
		}
	}
}

void v_connect::print_vert_to_ed_table(FILE *fp){
	fprintf(fp,"vert to edge connectivity, arranged cc\n\n");
	for(int i=0;i<nv;i++){
		fprintf(fp,"vert %i\n",i);
		for(int k=0;k<vert_to_ed[i].size();k++){
			fprintf(fp,"\t %i\n",vert_to_ed[i][k]);
		}
	}
}

void v_connect::print_vert_boundary(FILE *fp){
	fprintf(fp,"vertex on boundary\n\n");
	for(int i=0;i<nv;i++){
		fprintf(fp,"\nvert %i\n",i);
		if(vert_on_bd[i].size()==0) fprintf(fp,"\t vertex not on bound\n");
		else{
			fprintf(fp,"\tvertex on bound");
			for(int k=0;k<vert_on_bd[i].size();k++) fprintf(fp,"\t %i",vert_on_bd[i][k]);
		}
	}
}

void v_connect::print_ed_boundary(FILE *fp){
	fprintf(fp,"edge on boundar \n\n");
	for(int i=0;i<ne;i++){
		fprintf(fp,"\nedge %i\n",i);
		if(ed_on_bd[i].size()==0) fprintf(fp,"\t edge not on bound\n");
		else{
			fprintf(fp,"\t edge on bound");
			for(int k=0;k<ed_on_bd[i].size();k++) fprintf(fp,"\t %i",ed_on_bd[i][k]);
		}
	}
}

void v_connect::ascii_output(FILE *fp){
	fprintf(fp,"\n# General Mesh Data\nNnp\t%i\nNel\t%i\nNel_tri3\t0\nNel_poly2d\t%i\nNel_quad4\t0\nNel_hexh8\t0\nNel_poly3d\t0\nNel_pyr5\t0\nNel_tet4\t0\nNel_wedget\t0\nNdim\t2\nNnd_sets\t0\nNsd_sets\t0\nendi\n",nv,ng,ng);
	fprintf(fp,"# element data: global id, block id, number of nodes, nodes\n");
	for(int i=0;i<ng;i++){
		fprintf(fp,"%i\t1\t%i",i,gen_to_vert[i].size());
		for(int j=0;j<gen_to_vert[i].size();j++){
			fprintf(fp,"\t%i",gen_to_vert[i][j]);
		}
		fprintf(fp,"\n");
	}
	fprintf(fp,"#\n#nodal data:global id, xcoord, ycoord\n#\n");
	for(int i=0;i<nv;i++){
		fprintf(fp,"%i\t%g\t%g\n",i,vertl[2*i],vertl[2*i+1]);
	}
}

void v_connect::add_memory_vertices(){
	double *vertle(vertl+2*current_vertices);
	int *vertex_is_generatore(vertex_is_generator+current_vertices);
	current_vertices<<=1;
	cout << "2.2.1" << endl;
	//copy vertl
	double *nvertl(new double[2*current_vertices]),*nvp(nvertl),*vp(vertl);
	while(vp<vertle) *(nvp++)=*(vp++);
	delete [] vertl;vertl=nvertl;
	cout << "2.2.2" << endl;
	//copy vert_to_gen, vert_to_ed
	vector<int> *nvert_to_gen(new vector<int>[current_vertices]);
	for(int i=0;i<(current_vertices>>1);i++){
		nvert_to_gen[i]=vert_to_gen[i];
	}
	delete [] vert_to_gen;vert_to_gen=nvert_to_gen;
	cout << "2.2.3" << endl;
	//copy vertex_is_generator
	int *nvertex_is_generator(new int[current_vertices]),*nvig(nvertex_is_generator),*vig(vertex_is_generator),*nvige(nvertex_is_generator+current_vertices);
	while(vig<vertex_is_generatore) *(nvig++)=*(vig++);
	cout << "inbetween" << endl;
	while(nvig<nvige) *(nvig++)=-1;
	cout << "2.2.4" << endl;
	delete [] vertex_is_generator; vertex_is_generator=nvertex_is_generator;
}

//returns the signed area of the cell corresponding to generator g
double v_connect::signed_area(int g){
	double area=0,vx1,vy1,vx2,vy2;
	vx1=vertl[2*gen_to_vert[g][0]];
	vy1=vertl[2*gen_to_vert[g][0]+1];
	for(int i=0;i<gen_to_vert[g].size();i++){
		if(i==gen_to_vert[g].size()-1){
			vx2=vertl[2*gen_to_vert[g][0]];
			vy2=vertl[2*gen_to_vert[g][0]+1];
		}else{
			vx2=vertl[2*gen_to_vert[g][i+1]];
			vy2=vertl[2*gen_to_vert[g][i+1]+1];
		}
		area+=((vx1*vy2)-(vx2*vy1));
		vx1=vx2;
		vy1=vy2;
	}
	area*=.5;
	return area;
}

//returns the centroid of the cell corresponding to generator g in x,y
void v_connect::centroid(int g,double &x,double &y){
	double area,vx1,vy1,vx2,vy2,cp;
	x=0; y=0; area=signed_area(g);
	vx1=vertl[2*gen_to_vert[g][0]];
	vy1=vertl[2*gen_to_vert[g][0]+1];
	for(int i=0;i<gen_to_vert[g].size();i++){
		if(i==gen_to_vert[g].size()-1){
			vx2=vertl[2*gen_to_vert[g][0]];
			vy2=vertl[2*gen_to_vert[g][0]+1];
		}else{
			vx2=vertl[2*gen_to_vert[g][i+1]];
			vy2=vertl[2*gen_to_vert[g][i+1]+1];
		}
		cp=cross_product(vx1,vy1,vx2,vy2);
		x+=((vx1+vx2)*cp);
		y+=((vy1+vy2)*cp);
		vx1=vx2;
		vy1=vy2;
	}
	x=((1/(6*area))*x);
	y=((1/(6*area))*y);
}
//outputs lloyds allgorithms in files lloyds# until max distance from generator to centroid < epsilon
void v_connect::lloyds(double epsilon){
	double gx,gy,cx,cy,max,current;
	int j=0;
	char *outfn1(new char[1000]);
	sprintf(outfn1,"lloyds");
	FILE *fp=safe_fopen(outfn1,"w");
	do{
		cout << "iteration" << j << endl;
		max=0;
		char *outfn2(new char[1000]);
		sprintf(outfn2,"lloyds%i",j);
		cout << "blah" << endl;
		draw_gnu(outfn2);
		cout << "yeah" << endl;
		delete[] outfn2;
		if(bd!=-1){
			for(int i=bd;i<ng;i++){
				cout << "i=" << i << endl;
				gx=vpos[2*mp[i]]; gy=vpos[2*mp[i]+1];
				centroid(i,cx,cy);
				current=pow(cx-gx,2)+pow(cy-gy,2);
				if(current>max) max=current;
				vpos[2*mp[i]]=cx;
				vpos[2*mp[i]+1]=cy;
			}
		}
		fprintf(fp,"iteration %i, max=%f, epsilon=%f",j,max,epsilon);
		nv=0;
		ne=0;
		cout << "1" << endl;
		current_vertices=init_vertices;
		current_edges=init_vertices;
		delete [] gen_to_gen_e;
		gen_to_gen_e=new vector<int>[ng];
		cout << "2" << endl;
		delete [] gen_to_gen_v;
		gen_to_gen_v=new vector<int>[ng];
		cout << "3" << endl;
		delete [] gen_to_ed;
		gen_to_ed=new vector<int>[ng];
		cout << "4" << endl;
		delete [] gen_to_vert;
		gen_to_vert=new vector<int>[ng];
		cout << "5" << endl;
		delete [] vertl;
		vertl=new double[2*init_vertices];
		cout << "6" << endl;
		delete [] vert_to_gen;
		vert_to_gen=new vector<int>[init_vertices];
		cout << "7" << endl;
		delete [] vertex_is_generator;
		cout << "7.1" << endl;
		vertex_is_generator=new int[init_vertices];
		cout << "7.2" << endl;
		for(int k=0;k<init_vertices;k++) vertex_is_generator[k]=-1;
		cout << "8" << endl;
		delete [] generator_is_vertex;
		generator_is_vertex=new int[ng];
		for(int k=0;k<ng;k++) generator_is_vertex[k]=-1;
		cout << "9" << endl;
		delete [] ed_to_vert;
		delete [] vert_to_ed;
		delete [] ed_to_gen;
		delete [] ed_on_bd;
		delete [] vert_on_bd;
		cout << "1..." << endl;
		assemble_vertex();
		cout << "2..." << endl;
		assemble_gen_ed();
		cout << "3..." << endl;
		assemble_boundary();
		cout << "4..." << endl;
		j++;
	}while(max>epsilon);
	fclose(fp);
	delete [] outfn1;
}

void v_connect::draw_median_mesh(FILE *fp){
	double ex,ey,cx,cy;
	int e;
	for(int i=0;i<ng;i++){
		if(generator_is_vertex[i]!=-1) continue;
		centroid(i,cx,cy);
		for(int j=0;j<gen_to_ed[i].size();j++){
			e=gen_to_ed[i][j];
			if(e<0) e=~e;
			ex=(vertl[2*ed_to_vert[2*e]] + vertl[2*ed_to_vert[2*e+1]])/2;
			ey=(vertl[2*ed_to_vert[2*e]+1] + vertl[2*ed_to_vert[2*e+1]+1])/2;
			fprintf(fp,"\n %g %g\n %g %g \n\n\n",ex,ey,cx,cy);
		}
	}

}

void v_connect::draw_closest_generator(FILE *fp,double x,double y){
	int cg,pg,ng=0;
	double gx,gy,best,bestl,current;
	gx=vpos[2*mp[0]]; gy=vpos[2*mp[0]+1];
	best = pow(gx-x,2)+pow(gy-y,2);
	cout << "best=" << best << endl;
	fprintf(fp,"%g %g\n",gx,gy);
	do{
		cg=ng;
		for(int i=0;i<gen_to_gen_e[cg].size();i++){
			pg=gen_to_gen_e[cg][i];
			gx=vpos[2*mp[pg]]; gy=vpos[2*mp[pg]+1];
			current=pow(gx-x,2)+pow(gy-y,2);
			cout << "current=" << current << endl;
			if(current<best){
				cout << "changed" << endl;
				best=current;
				ng=pg;
			}
		}
		fprintf(fp,"%g %g\n",vpos[2*mp[ng]],vpos[2*mp[ng]+1]);
	}while(cg!=ng);
}

}
