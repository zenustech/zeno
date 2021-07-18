/*
 *	octree3.cpp
 *
 *	Created by Ryoichi Ando on 5/30/12
 *	Email: and@verygood.aid.design.kyushu-u.ac.jp
 *
 */

#include <unordered_map>
#include <algorithm>
#include "vec.h"
#include "util.h"
#include "octree3.h"


//#include "opengl.h"
using namespace std;



octree3::octree3() {
	root = NULL;
	dontEnfortceWB = false;
	clearData();
}

octree3::octree3( const octree3 &octree ) {
	root = NULL;
	*this = octree;
}

octree3::~octree3() {
	clearData();
}

void octree3::operator=( const octree3 &octree ) {
	clearData();
	if( octree.root ) {
		maxdepth = octree.maxdepth;
		resolution = octree.resolution;
		root = new leaf3;
		copy(octree.root,root);
		terminals.resize(octree.terminals.size());
		
		// Build terminal array
		uint index = 0;
		countNumTerminal(root,index);
		terminals.resize(index);
		index = 0;
		buildArray(root,index);
		
		// Copy nodes
		nodes = octree.nodes;
	}
}

void octree3::copy( leaf3 *src, leaf3 *dest ) {
	*dest = *src;
	for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++){
		if( src->children[i][j][k] ) {
			dest->children[i][j][k] = new leaf3;
			copy(src->children[i][j][k],dest->children[i][j][k]);
		}
	}
}

bool octree3::buildOctree(/* const levelset3 *hint,*/ const std::vector<sphere3> &spheres, uint maxdepth ) {
	// Make default levelset
	//defaultLevelset3 defaultLevelset(maxdepth-1);
	//if( ! hint && spheres.empty()) hint = &defaultLevelset;

	// Clear the octree first
	if( ! clearData()) return false;

	//tick(); dump( ">>> Building octree started...\n" );

	// Set the maximal depth
	this->maxdepth = maxdepth;
	resolution = powf(2,maxdepth+2);

	// Allocate root leaf
	root = allocLeaf(FLUID::Vec3i(resolution/2,resolution/2,resolution/2),0,FLUID::Vec3i(0,0,0));

	// Subdivide this root leaf...
	//tick(); dump( "Subdividing octree..." );
	subdivide(root,spheres,maxdepth);
	//dump( "Done. Took %s.\n", stock("octree_subdivision_hinted"));

	// Enforce weak balance
	enforceWeakBalance();

	// Build terminal array
	//tick(); dump( "Building terminal list..." );
	uint index = 0;
	countNumTerminal(root,index);
	terminals.resize(index);
	index = 0;
	buildArray(root,index);
	//dump( "Done. Found %d terminals. Took %s.\n", index, stock());
	//writeNumber("octree_terminal_num", index);

	// Build corner nodes and its references
	//tick(); dump( "Building nodes list..." );
	buildNodes();
	//dump( "Done. Took %s.\n", stock("octree_node_list_hinted"));
	//writeNumber("octree_node_num", nodes.size());
	BuildTerminalsAllLevels(terminals);
	//dump( "<<< Octree done. Took %s.\n", stock("octree"));
	return true;
}


bool octree3::buildOctree(/* const levelset3 *hint,*/ const Array3f & detail_map, uint maxdepth ) {
	// Make default levelset
	//defaultLevelset3 defaultLevelset(maxdepth-1);
	//if( ! hint && spheres.empty()) hint = &defaultLevelset;

	// Clear the octree first
	if( ! clearData()) return false;

	//tick(); dump( ">>> Building octree started...\n" );

	// Set the maximal depth
	this->maxdepth = maxdepth;
	resolution = powf(2,maxdepth+2);

	// Allocate root leaf
	root = allocLeaf(FLUID::Vec3i(resolution/2,resolution/2,resolution/2),0,FLUID::Vec3i(0,0,0));

	// Subdivide this root leaf...
	//tick(); dump( "Subdividing octree..." );
	std::vector<Array3f *> level_desc;
	subdivide(root,level_desc,maxdepth);
	//dump( "Done. Took %s.\n", stock("octree_subdivision_hinted"));

	// Enforce weak balance
	enforceWeakBalance();

	// Build terminal array
	//tick(); dump( "Building terminal list..." );
	uint index = 0;
	countNumTerminal(root,index);
	terminals.resize(index);
	index = 0;
	buildArray(root,index);
	//dump( "Done. Found %d terminals. Took %s.\n", index, stock());
	//writeNumber("octree_terminal_num", index);

	// Build corner nodes and its references
	//tick(); dump( "Building nodes list..." );
	buildNodes();
	//dump( "Done. Took %s.\n", stock("octree_node_list_hinted"));
	//writeNumber("octree_node_num", nodes.size());
	

	//dump( "<<< Octree done. Took %s.\n", stock("octree"));
	return true;
}


void octree3::dontEnforceWeakBalance() {
	dontEnfortceWB = true;
}

int octree3::hitTest( FLUID::Vec3f p ) const {
	for( uint dim=0; dim<3; dim++ ) p[dim] = min(1.0f,max(0.0f,p[dim]));
	p = p * (float)resolution;
	std::vector<uint> array;
	hitTest(p,0.0,array,root);
	if( array.empty() ) return -1;
	return *array.begin();
}

bool octree3::hitTest( FLUID::Vec3f p, float r, std::vector<uint> &array, leaf3 *leaf ) const {
	if( ! leaf ) {
		p = p * (float)resolution;
		leaf = root;
	}
	FLUID::Vec3f center = FLUID::Vec3f(leaf->center[0],leaf->center[1],leaf->center[2]);
	FLUID::Vec3f p0 = center - (float)(leaf->dx)*FLUID::Vec3f(0.5,0.5,0.5);
	FLUID::Vec3f p1 = center + (float)(leaf->dx)*FLUID::Vec3f(0.5,0.5,0.5);
	bool hit = box(p,p0,p1) <= resolution*r;
	if( hit ) {
		if( leaf->subdivided ) {
			for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++) {
				hitTest(p,r,array,leaf->children[i][j][k]);
			}
		} else {
			array.push_back(leaf->index);
		}
	}
	return array.size();
}

octree3::leaf3* octree3::allocLeaf( FLUID::Vec3i center, uint depth, FLUID::Vec3i position ) {
	leaf3 *leaf = new leaf3;
	leaf->center = center;
	leaf->position = position;
	leaf->depth = depth;
	leaf->dx = resolution/powf(2,depth);
	leaf->subdivided = false;
	for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++) {
		leaf->children[i][j][k] = NULL;
		leaf->corners[i][j][k] = 0;
	} 
	return leaf;
}

static FLUID::Vec3i computeCenterPos( FLUID::Vec3i center, uint dx, uint i, uint j, uint k ) {
	return center-((int)(dx)*FLUID::Vec3i(1,1,1))/4+((int)(dx)*FLUID::Vec3i(i,j,k))/2;
}

static FLUID::Vec3i computeCornerPos( FLUID::Vec3i center, uint dx, uint i, uint j, uint k ) {
	return center-((int)(dx)*FLUID::Vec3i(1,1,1))/2+(int)(dx)*FLUID::Vec3i(i,j,k);
}

//bool octree3::checkSubdivision( FLUID::Vec3i pos, uint dx, const levelset3 *hint, int threshold, int depth, uint max_nest ) const {
//	if( ! max_nest ) return false;
//	if( hint->evalLevelset(FLUID::Vec3f(pos[0],pos[1],pos[2])/(float)resolution) < powf(0.5,depth) ) {
//		return true;
//	}
//	if( dx > threshold ) {
//		// Compute the center position of this children
//		for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++) {
//			// Compute the levelset at this position
//			FLUID::Vec3i sub_pos = computeCenterPos(pos,dx,i,j,k);
//			if( checkSubdivision(sub_pos,dx/2,hint,threshold,depth,max_nest-1)) return true;
//		} 
//	}
//	return false;
//}

void octree3::subdivide( leaf3 *leaf, /*const levelset3 *hint,*/ const std::vector<sphere3> &spheres, uint maxdepth ) {
	bool doSubdivision = false;

	// Compute the center position of this children
	float dx = leaf->dx/(float)resolution;

	// See this octree contains large small particles
	for( uint n=0; n<spheres.size(); n++ ) {
		const sphere3 &sphere = spheres[n];
		if( sphere.r <= 0.5*dx ) {
			doSubdivision = true;
			break;
		}
	}

	// See hint indicates a subdivision
	//if( hint && ! doSubdivision ) {
	//	int threshold = 8;
	//	doSubdivision = checkSubdivision(leaf->center,leaf->dx,hint,threshold,leaf->depth,5);
	//}

	// If to subdivide, do it
	if( doSubdivision ) {
		uint depth = leaf->depth+1;
		if( depth <= maxdepth ) {
			leaf->subdivided = true;
			for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++) {
				// Compute the center position for this children
				FLUID::Vec3i center = computeCenterPos(leaf->center,leaf->dx,i,j,k);
				leaf3 *child = allocLeaf(center,depth,FLUID::Vec3i(i,j,k));
				// Make a new sphere array for this child
				std::vector<sphere3 > child_spheres;
				for( uint n=0; n<spheres.size(); n++ ) {
					const sphere3 sphere = spheres[n];
					FLUID::Vec3f child_pos = FLUID::Vec3f(center[0],center[1],center[2])/(float)resolution;
					float child_dx = (float)(child->dx)/(float)resolution;
					if( dist2(child_pos, sphere.p) < sqr(child_dx) ) {
						child_spheres.push_back(spheres[n]);
					}
				}				
				leaf->children[i][j][k] = child;
				child->father = leaf;
				subdivide(child,child_spheres,maxdepth);
			} 
		}
	}
}




void octree3::subdivide( leaf3 *leaf,const std::vector<Array3f *> &level_discptor, uint maxdepth ) {
	bool doSubdivision = false;

	// Compute the center position of this children
	float dx = leaf->dx/(float)resolution;

	// See this octree contains large small particles
	//for( uint n=0; n<spheres.size(); n++ ) {
	//	const sphere3 &sphere = *spheres[n];
	//	if( sphere.r <= 0.5*dx ) {
	//		doSubdivision = true;
	//		break;
	//	}
	//}
	

	//see if this octant needs refinement
	FLUID::Vec3i center_pos = leaf->center;
	FLUID::Vec3i level_pos = FLUID::Vec3i(center_pos[0], center_pos[1], center_pos[2])/(int)(leaf->dx);
	
	if ((*(level_discptor[leaf->depth]))(level_pos[0],level_pos[1],level_pos[2])<powf(0.5,leaf->depth))
	{
		doSubdivision = true;
	}
	
	// See hint indicates a subdivision
	//if( hint && ! doSubdivision ) {
	//	int threshold = 8;
	//	doSubdivision = checkSubdivision(leaf->center,leaf->dx,hint,threshold,leaf->depth,5);
	//}

	// If to subdivide, do it
	if( doSubdivision ) {
		uint depth = leaf->depth+1;
		if( depth <= maxdepth ) {
			leaf->subdivided = true;
			for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++) {
				// Compute the center position for this children
				FLUID::Vec3i center = computeCenterPos(leaf->center,leaf->dx,i,j,k);
				leaf3 *child = allocLeaf(center,depth,FLUID::Vec3i(i,j,k));
				// Make a new sphere array for this child
/*				std::vector<sphere3 *> child_spheres;
				for( uint n=0; n<spheres.size(); n++ ) {
					const sphere3 &sphere = *spheres[n];
					FLUID::Vec3f child_pos = FLUID::Vec3f(center[0],center[1],center[2])/(float)resolution;
					float child_dx = (double)(child->dx)/(float)resolution;
					if( dist2(child_pos, sphere.p) < sqr(child_dx) ) {
						child_spheres.push_back(spheres[n]);
					}
				}	*/			
				leaf->children[i][j][k] = child;
				subdivide(child,level_discptor,maxdepth);
			} 
		}
	}
}

void octree3::enforceWeakBalance() {
	if( dontEnfortceWB ) return;
	
	//tick(); dump( "Enforcing Weak Balance condition..." );
	uint itnum = 0;
	uint first_num = 0;
	uint last_num = 0;
	uint subdiv_num = 0;
	// Repeat while more than 2-level T-junction exists
	while(true) {
		// Collect terminals
		std::unordered_map<uint,leaf3 *> terminals_collapse;
		
		// Build terminal array
		uint index = 0;
		countNumTerminal(root,index);
		terminals.resize(index);
		if( ! first_num ) first_num = index;
		last_num = index;
		index = 0;
		buildArray(root,index);
		
		// For each terminal
		for( uint n=0; n<terminals.size(); n++ ) {
			// Look for neighbors
			for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++) {
				FLUID::Vec3f p = FLUID::Vec3f(FLUID::Vec3f(terminals[n]->center[0],terminals[n]->center[1],terminals[n]->center[2])+(float)(terminals[n]->dx)*FLUID::Vec3f(2*i-1,2*j-1,2*k-1))/(float)resolution;
				int neigh = hitTest(p);
				if( neigh >= 0 ) {
					if( terminals[neigh]->dx > 2*terminals[n]->dx && terminals_collapse.find(terminals[neigh]->index) == terminals_collapse.end()) {
						terminals_collapse[terminals[neigh]->index] = terminals[neigh];
						subdiv_num ++;
					}
				}
			} 
		}
		if( terminals_collapse.empty() ) break;
		
		// Collapse
		std::unordered_map<uint,leaf3 *>::iterator it;
		for( it=terminals_collapse.begin(); it!=terminals_collapse.end(); it++ ) {
			leaf3 *leaf = it->second;
			uint depth = leaf->depth+1;
			if( depth <= maxdepth ) {
				leaf->subdivided = true;
				for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++) {
					// Compute the center position for this children
					FLUID::Vec3i center = computeCenterPos(leaf->center,leaf->dx,i,j,k);
					leaf3 *child = allocLeaf(center,depth,FLUID::Vec3i(i,j,k));
					leaf->children[i][j][k] = child;
					child->father = leaf;
				} 
			}
		}
		itnum ++;
	}
	//dump( "Done. Looped %d times. %d terminals are subdivided and extra %d terminals are generated. Took %s.\n", itnum, subdiv_num, last_num-first_num, stock("octree_weakbalance"));
	//writeNumber("octree_weakbalace_generated", last_num-first_num);
}

void octree3::countNumTerminal( leaf3 *leaf, uint &count ) {
	if( leaf->subdivided ) {
		for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++) {
			countNumTerminal(leaf->children[i][j][k],count);
		} 
	} else {
		count++;
	}
}

void octree3::buildArray( leaf3 *leaf, uint &index ) {
	if( leaf->subdivided ) {
		for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++) {
			buildArray(leaf->children[i][j][k],index);
		} 
	} else {
		terminals[index] = leaf;
		leaf->index = index;
		index ++;
	}
}

void octree3::buildNodes() {
	std::unordered_map<uint64,uint64> nodeDictionary;
	uint64 index = 0;
	for( uint n=0; n<terminals.size(); n++ ) {
		leaf3 *leaf = terminals[n];
		for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++) {
			// Compute the center position for this children
			FLUID::Vec3i corner = computeCornerPos(leaf->center,leaf->dx,i,j,k);
			uint64 idx = computeCornerIndex(corner);
			if( nodeDictionary.find(idx) == nodeDictionary.end() ) {
				nodeDictionary[idx] = index;
				leaf->corners[i][j][k] = index;
				index ++;
			} else {
				leaf->corners[i][j][k] = nodeDictionary[idx];
			}
		} 
	}
	nodes.resize(index);
	for( uint n=0; n<terminals.size(); n++ ) {
		leaf3 *leaf = terminals[n];
		for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++) {
			uint64 index = leaf->corners[i][j][k];
			nodes[index] = FLUID::Vec3f(computeCornerPos(leaf->center,leaf->dx,i,j,k))/(float)resolution;
		} 
	}
	for( uint n=0; n<terminals.size(); n++ ) {
		leaf3 *leaf = terminals[n];
		FLUID::Vec3i center = leaf->center;
		nodes.push_back(FLUID::Vec3f(center)/(float)resolution);
	}
	
}
void octree3::buildNodes(vector<leaf3 *> &leafs, vector<double> &values, int level)
{
	
	
	for (uint n=0;n<leafs.size();n++)
	{
		leaf3 *leaf = leafs[n];
		//if the leaf is not subdived, just copy it's finer level's node index
		if(leaf->subdivided!=true)
		{
			for (int i=0; i<leaf->node_list[level-1].size();i++)
			{
				leaf->node_list[level].push_back(leaf->node_list[level-1][i]);
			}
		}
		else//if this leaf is subdived, it has 8 finer level cells
		{
			for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++) {
				leaf->node_list[level].push_back(leaf->corners[i][j][k]);
			}
		}
		
	}

	std::unordered_map<uint64, uint64> node_map;
	
	uint64 index = 0;
	for (uint n=0;n<leafs.size();n++)
	{
		leaf3 *leaf = leafs[n];
		
		for (int i=0; i<leaf->node_list[level].size();i++)
		{
			if (node_map.find(leaf->node_list[level][i])==node_map.end())//cannot locate one
			{
				leaf->value_list[level].push_back(index);
				node_map[leaf->node_list[level][i]] = index;
				index++;
			}
			else //located one already allocated value
			{
				leaf->value_list[level].push_back(node_map[leaf->node_list[level][i]]);
			}
			
		}
		

	}
	values.resize(index);
	values.assign(values.size(),0);
	
	
}
void octree3::BuildTerminalsAllLevels(std::vector<leaf3 *> &terminals_level0)
{
	for (uint i=0;i<terminal_level_k.size();i++)
	{
		terminal_level_k[i].resize(0);
	}
	terminal_level_k.resize(0);
	multigrid_levels = 0;
	terminal_level_k.push_back(terminals_level0);
	uint problem_size = terminal_level_k[0].size();
	uint dx = terminals_level0[0]->dx;
	for (uint i=1; i<terminals_level0.size();i++)
	{
		if (terminals_level0[i]->dx<dx)
		{
			dx = terminals_level0[i]->dx;
		}
		
	}
	
	//printf("%d\n",dx);
	while(problem_size>1024)
	{
		leaf_table.clear();
		uint index=0;
		//printf("%d\n",problem_size);
		dx = dx*2;
		terminal_level_k.push_back(vector<leaf3 *>());
		for (uint i=0; i<terminal_level_k[multigrid_levels].size();i++)
		{
			//printf("%d\n", i);
			//if(i%1000000==0) printf("%d\n",i/1000000);
			if (terminal_level_k[multigrid_levels][i]->dx>=dx)
			{
				terminal_level_k[multigrid_levels+1].push_back(terminal_level_k[multigrid_levels][i]);
			}
			else
			{
				leaf3 *coarse_leaf = terminal_level_k[multigrid_levels][i]->father;
				for(int ii=0;ii<2;ii++)for(int jj=0;jj<2;jj++)for(int kk=0;kk<2;kk++) {
					coarse_leaf->corners[ii][jj][kk]=coarse_leaf->children[ii][jj][kk]->corners[ii][jj][kk];
				} 
				uint leaf_index;
				if (!coarse_leaf->counted)
				{
					coarse_leaf->counted = true;
					terminal_level_k[multigrid_levels+1].push_back(coarse_leaf);
					leaf_table.add(coarse_leaf->center,index);
					index++;
				}
			}
			
		}
		multigrid_levels++;
		problem_size = terminal_level_k[multigrid_levels].size();
		
	}
	multigrid_levels++;
	for (int i=0;i<multigrid_levels;i++)
	{
		for (uint t=0;t<terminal_level_k[i].size();t++)
		{
			terminal_level_k[i][t]->node_list.resize(multigrid_levels);
			terminal_level_k[i][t]->value_list.resize(multigrid_levels);
			terminal_level_k[i][t]->counted = false;
		}

	}

	for (int i=0;i<multigrid_levels;i++)
	{
		for (uint t=0;t<terminal_level_k[i].size();t++)
		{
			terminal_level_k[i][t]->node_list[i].resize(0);
			terminal_level_k[i][t]->value_list[i].resize(0);
		}
	}
	
}
void octree3::finalizeMultigridLevels()
{
	if(Dofs_level_k.size()>0)
	{
		for(int i=0;i<Dofs_level_k.size();i++)
		{
			Dofs_level_k[i].resize(0);
			Dofs_level_k[i].shrink_to_fit();
		}
	}


}
bool octree3::isIn(const FLUID::Vec3f &p, leaf3 *leaf)
{
	//compute bounding box of leaf
	FLUID::Vec3f bmin = FLUID::Vec3f(computeCornerPos(leaf->center,leaf->dx,0,0,0))/(float)resolution;
	FLUID::Vec3f bmax = FLUID::Vec3f(computeCornerPos(leaf->center,leaf->dx,1,1,1))/(float)resolution;

	if (p[0]>=bmin[0] && p[0]<=bmax[0] 
	 && p[1]>=bmin[1] && p[1]<=bmax[1]
	 && p[2]>=bmin[2] && p[2]<=bmax[2])
	{
		return true;
	}
	return false;
}
octree3::leaf3* octree3::findLeaf(const FLUID::Vec3f & p, leaf3 *leaf)
{
	if (isIn(p,leaf))
	{
		if(leaf->subdivided == false)
		{
			return leaf;
		}
		else
		{
			leaf3* l;
			for (int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++)
			{
				l=findLeaf(p,leaf->children[i][j][k]);
				if(l!=NULL)
				{
					return l;
				}
			}
			return NULL;
		}
	}
	else
	{
		return NULL;
		
	}
	
}
void octree3::initMultigridLevels(const vector<FLUID::Vec3f> &vertices)
{
	Dofs_level_k.resize(multigrid_levels);
	finalizeMultigridLevels();

	//level 0 would be given from the outside:
	//a set of vertex position, push back their
	//index to corresponding leaf nodes.
	uint num_oct_nodes = nodes.size();
	for (uint i=0; i<vertices.size();i++)
	{
		uint64 index = nodes.size();
		leaf3* leaf = findLeaf(vertices[i], root);
		if (leaf!=NULL)//we find one
		{
			leaf->node_list[0].push_back(index);
			leaf->value_list[0].push_back(index - num_oct_nodes);
			nodes.push_back(vertices[i]);
		}
	}
	assert((nodes.size()-num_oct_nodes) == vertices.size());
	Dofs_level_k[0].resize(vertices.size());
	Dofs_level_k[0].assign(vertices.size(),0);
	

	//from level 1 up, first allocate unknowns
	//
	for (int i=1;i<multigrid_levels;i++)
	{
		buildNodes(terminal_level_k[i],Dofs_level_k[i],i);
	}
	int dx = terminal_level_k[0][0]->dx;
	for (int n=1;n<terminal_level_k[0].size();n++)
	{
		if (terminal_level_k[0][n]->dx<dx)
		{
			dx = terminal_level_k[0][n]->dx;
		}
		
	}
	
	P_L.resize(0);
	R_L.resize(0);
	//form prolongation and restriction
	for (int i=0; i<multigrid_levels-1;i++)
	{
		P_L.push_back(new FixedSparseMatrixd);
		R_L.push_back(new FixedSparseMatrixd);
		dx = dx * 2;
		formProlongationAndRestriction(*(P_L[i]),*(R_L[i]),i,dx);
		//ostringstream pout,rout;
		//pout<<"prolongation"<<i<<".m";
		//rout<<"restriction"<<i<<".m";
		cout<<i<<endl;

	}
	
	
}
double compute_weight(double r)
{
	if(fabs(r)<1.0)
	{
		return 1-fabs(r);
	}
	else
	{
		return 0;
	}
}
void octree3::formProlongationAndRestriction(FixedSparseMatrixd &P, 
											 FixedSparseMatrixd &R, 
											 uint level, int dx)
{
	//algorithm:
	//for all terminals at current level
	//	if terminal_i->dx < dx
	//		for all its Dof nodes
	//			index1 = index of this Dof node
	//			for all Dof' in this terminal's father level
	//				index2 = index of the Dof'
	//				w = compute interpolating weight
	//				p(index1, index2) = w;
	//				r(index2, index1) = w;
	//			endfor
	//		endfor
	//	else
	//		for all its Dof nodes
	//			index1 = index of this Dof node at this level
	//			index2 = index of this Dof node at higher level
	//			p(index1,index2) = 1;
	//			r(index2,index1) = 1;
	//		endfor
	//	endif
	//endfor
	SparseMatrixd p;
	SparseMatrixd r;
	p.resize(Dofs_level_k[level].size());
	r.resize(Dofs_level_k[level+1].size());
	p.zero();
	r.zero();

	for (uint n=0;n<terminal_level_k[level].size();n++)
	{
		leaf3* leaf = terminal_level_k[level][n];
		if(leaf->dx<dx)
		{
			for (uint nn=0;nn<leaf->value_list[level].size();nn++)
			{
				uint64 index1 = leaf->value_list[level][nn];
				leaf3* father = leaf->father;
				FLUID::Vec3f vertex1 = nodes[leaf->node_list[level][nn]];
				for (uint nnn=0;nnn<father->value_list[level+1].size();nnn++)
				{
					uint64 index2 = father->value_list[level+1][nnn];
					FLUID::Vec3f vertex2 = nodes[father->node_list[level+1][nnn]];
					float h = (float)dx/(float)resolution;
					FLUID::Vec3f d = vertex2-vertex1;
					double w = compute_weight(d[0]/h)*compute_weight(d[1]/h)*compute_weight(d[2]/h);
					//assign corresponding values
					p.set_element(index1,index2,w);
					r.set_element(index2,index1,w);


				}
				
			}
			
		}
		else
		{
			for (uint nn=0;nn<leaf->value_list[level].size();nn++)
			{
				uint64 index1 = leaf->value_list[level][nn];
				uint64 index2 = leaf->value_list[level+1][nn];
				p.set_element(index1,index2,1);
				r.set_element(index2,index1,1);
			}
			
		}
	}
	P.construct_from_matrix(p);
	R.construct_from_matrix(r);
	p.clear();
	r.clear();
	
}
uint64 octree3::computeCornerIndex( FLUID::Vec3i &p ) const {
	uint64 R = resolution;
	return p[0]+p[1]*R+p[2]*R*R;
}

bool octree3::clearData() {
	if( root ) {
		releaseChilren(root);
		root = NULL;
	}
	maxdepth = 1;
	terminals.clear();
	nodes.clear();
	return true;
}

bool octree3::releaseChilren( leaf3 *leaf ) {
	if( ! leaf ) return false;
	// Make sure we release all the chilren first
	for(int i=0;i<2;i++)for(int j=0;j<2;j++)for(int k=0;k<2;k++) {
		if( leaf->children[i][j][k] ) {
			releaseChilren(leaf->children[i][j][k]);
			leaf->children[i][j][k] = NULL;
		}
	} 
	
	// After that we deallocate this structure
	delete leaf;
	return true;
}

float octree3::box( FLUID::Vec3f p, FLUID::Vec3f p0, FLUID::Vec3f p1 ) const {
	float sd = -9999.0;
	sd = max(sd,p0[0]-p[0]);
	sd = max(sd,p0[1]-p[1]);
	sd = max(sd,p0[2]-p[2]);
	sd = max(sd,p[0]-p1[0]);
	sd = max(sd,p[1]-p1[1]);
	sd = max(sd,p[2]-p1[2]);
	return sd;
}


void octree3::writeObj( const char *path ) const {
	FILE *fp = fopen(path,"w");
	if( ! fp ) return;
	uint index=0;
	for( uint n=0; n<terminals.size(); n++ ) {
		leaf3* leaf = terminals[n];
		FLUID::Vec3f p = FLUID::Vec3f(terminals[n]->center)/(float)resolution;
		float dx = leaf->dx/(float)resolution;
		for( uint dim=0; dim<3; dim++ ) {
			FLUID::Vec3f vec0, vec1;
			if( dim == 0 ) {
				vec0 = FLUID::Vec3f(0,1,0);
				vec1 = FLUID::Vec3f(0,0,1);
			} else if( dim == 1 ) {
				vec0 = FLUID::Vec3f(1,0,0);
				vec1 = FLUID::Vec3f(0,0,1);
			} else if( dim == 2 ) {
				vec0 = FLUID::Vec3f(1,0,0);
				vec1 = FLUID::Vec3f(0,1,0);
			}
			for( int dir=-1; dir<=1; dir+=2 ) {
				FLUID::Vec3f q[4] = { -0.5*vec0-0.5*vec1, 0.5*vec0-0.5*vec1, 0.5*vec0+0.5*vec1, -0.5*vec0+0.5*vec1 };
				for( uint m=0; m<4; m++ ) {
					FLUID::Vec3f corner = p+dir*0.5f*dx*FLUID::Vec3f((float)(dim==0),(float)(dim==1),(float)(dim==2))+dx*q[m];
					fprintf(fp,"v %f %f %f\n", corner[0], corner[1], corner[2] );
					index ++;
				}
				//fprintf(fp,"f %d %d %d %d\n", index-2, index-1, index, index+1 );
			}
		}
	}
	fprintf(fp,"\n\n");
	index=0;
	for( uint n=0; n<terminals.size(); n++ ) {
		leaf3* leaf = terminals[n];
		FLUID::Vec3f p = FLUID::Vec3f(terminals[n]->center)/(float)resolution;
		float dx = leaf->dx/(float)resolution;
		for( uint dim=0; dim<3; dim++ ) {
			FLUID::Vec3f vec0, vec1;
			if( dim == 0 ) {
				vec0 = FLUID::Vec3f(0,1,0);
				vec1 = FLUID::Vec3f(0,0,1);
			} else if( dim == 1 ) {
				vec0 = FLUID::Vec3f(1,0,0);
				vec1 = FLUID::Vec3f(0,0,1);
			} else if( dim == 2 ) {
				vec0 = FLUID::Vec3f(1,0,0);
				vec1 = FLUID::Vec3f(0,1,0);
			}
			for( int dir=-1; dir<=1; dir+=2 ) {
				FLUID::Vec3f q[4] = { -0.5*vec0-0.5*vec1, 0.5*vec0-0.5*vec1, 0.5*vec0+0.5*vec1, -0.5*vec0+0.5*vec1 };
				for( uint m=0; m<4; m++ ) {
					FLUID::Vec3f corner = p+dir*0.5f*dx*FLUID::Vec3f((float)(dim==0),(float)(dim==1),(float)(dim==2))+dx*q[m];
					//fprintf(fp,"v %f %f %f\n", corner[0], corner[1], corner[2] );
					index ++;
				}
				fprintf(fp,"f %d %d %d %d\n", index-3, index-2, index-1, index );
			}
		}
	}

	fclose(fp);
}



void octree3::writeObjLevel( const char *path, int level) const {
	char filename[256];
	sprintf(filename, "%s%d.txt",path,level); 
	FILE *fp = fopen(filename,"w");
	if( ! fp ) return;
	//uint index=0;
	//for( uint n=0; n<terminal_level_k[level].size(); n++ ) {
	//	leaf3* leaf = terminal_level_k[level][n];
	//	FLUID::Vec3f p = FLUID::Vec3f(terminal_level_k[level][n]->center)/(float)resolution;
	//	float dx = leaf->dx/(float)resolution;
	//	for( uint dim=0; dim<3; dim++ ) {
	//		FLUID::Vec3f vec0, vec1;
	//		if( dim == 0 ) {
	//			vec0 = FLUID::Vec3f(0,1,0);
	//			vec1 = FLUID::Vec3f(0,0,1);
	//		} else if( dim == 1 ) {
	//			vec0 = FLUID::Vec3f(1,0,0);
	//			vec1 = FLUID::Vec3f(0,0,1);
	//		} else if( dim == 2 ) {
	//			vec0 = FLUID::Vec3f(1,0,0);
	//			vec1 = FLUID::Vec3f(0,1,0);
	//		}
	//		for( int dir=-1; dir<=1; dir+=2 ) {
	//			FLUID::Vec3f q[4] = { -0.5*vec0-0.5*vec1, 0.5*vec0-0.5*vec1, 0.5*vec0+0.5*vec1, -0.5*vec0+0.5*vec1 };
	//			for( uint m=0; m<4; m++ ) {
	//				FLUID::Vec3f corner = p+dir*0.5f*dx*FLUID::Vec3f((float)(dim==0),(float)(dim==1),(float)(dim==2))+dx*q[m];
	//				fprintf(fp,"v %f %f %f\n", corner[0], corner[1], corner[2] );
	//				index ++;
	//			}
	//			//fprintf(fp,"f %d %d %d %d\n", index-2, index-1, index, index+1 );
	//		}
	//	}
	//}
	vector<FLUID::Vec3f> vertex_list;
	vector<vector<uint64>> index_list;
	vertex_list.resize(Dofs_level_k[level].size());
	index_list.resize(terminal_level_k[level].size());

	uint64 index = 0;
	int dx = terminal_level_k[level][0]->dx;
	std::unordered_map<uint64, uint64> vertex_map;
	for (uint n=0;n<terminal_level_k[level].size();n++)
	{
		leaf3 *leaf = terminal_level_k[level][n];

		for (int i=0; i<leaf->node_list[level].size();i++)
		{
			if (vertex_map.find(leaf->node_list[level][i])==vertex_map.end())//cannot locate one
			{
				index_list[n].push_back(index);
				vertex_list[index] = nodes[leaf->node_list[level][i]];
				vertex_map[leaf->node_list[level][i]] = index;
				index++;
			}
			else //located one already allocated value
			{
				index_list[n].push_back(vertex_map[leaf->node_list[level][i]]);
			}

		}
		if (leaf->dx<dx)
		{
			dx = leaf->dx;
		}
		
	}
	fprintf(fp,"%d %f \n", vertex_list.size(),(float)dx/(float)resolution/4.0);
	for( uint n=0;n<vertex_list.size();n++ ) 
	{
		FLUID::Vec3f v = vertex_list[n];
		fprintf(fp,"%f %f %f\n", v[0], v[1], v[2] );
					
	}



	fprintf(fp,"\n\n");

	/*for( uint n=0; n<terminal_level_k[level].size(); n++ ) {
		
		
		fprintf(fp,"f %d %d %d %d\n", index_list[n][0]+1,
			index_list[n][1]+1,
			index_list[n][3]+1,
			index_list[n][2]+1);
		fprintf(fp,"f %d %d %d %d\n", index_list[n][4]+1,
			index_list[n][5]+1,
			index_list[n][7]+1,
			index_list[n][6]+1);
		fprintf(fp,"f %d %d %d %d\n", leaf->corners[0][0][0]+1,
			leaf->corners[1][0][0]+1,
			leaf->corners[1][1][0]+1,
			leaf->corners[0][1][0]+1);
		fprintf(fp,"f %d %d %d %d\n", leaf->corners[0][0][1]+1,
			leaf->corners[1][0][1]+1,
			leaf->corners[1][1][1]+1,
			leaf->corners[0][1][1]+1);
		

	}*/

	fclose(fp);
}

