/*
 *	octree3.h
 *
 *	Created by Ryoichi Ando on 5/30/12
 *	Email: and@verygood.aid.design.kyushu-u.ac.jp
 *
 */


#include "vec.h"
#include "util.h"
#include <vector>
#include <list>
#include "array3.h"
#ifndef _OCTREE3_H
#define _OCTREE3_H
#include "hashtable.h"
#include "sparse_matrix.h"
using namespace std;

class octree3 {
public:
	// Octree cell structure
	struct leaf3 {
		leaf3(){counted = false;}
		~leaf3(){}
		leaf3 *father;
		leaf3 *children[2][2][2];
		FLUID::Vec3i position;
		bool subdivided;
		FLUID::Vec3i center;
		uint depth;
		int dx;
		uint64 corners[2][2][2];	// index reference to the corner indices
		uint index;
		vector<vector<uint64>> node_list;// index reference to DoFs at level k, their corresponding 
		vector<vector<uint64>>   value_list;// position and value index. 
		bool counted;
	};
	
	typedef struct sphere3{
		sphere3(){ p = FLUID::Vec3f(0);r=0;}
		sphere3(const FLUID::Vec3f &_p, const float &_r)
		{
			p=_p;
			r=_r;
		}
		sphere3(const sphere3 &s)
		{
			 p = s.p;
			 r = s.r;
		}
		~sphere3(){}
		
		FLUID::Vec3f p;
		float r;
	};
	
	octree3();
	octree3( const octree3 &octree );
	void operator=( const octree3 &octree );
	~octree3();

	
	
	//bool buildOctree( const levelset3 *hint, uint maxdepth );
	bool buildOctree( const std::vector<sphere3> &spheres, uint maxdepth );
	bool buildOctree( const Array3f & detail_map, uint maxdepth);
	//bool buildOctree( const levelset3 *hint, const std::vector<sphere3 *> &spheres, uint maxdepth );
	void dontEnforceWeakBalance();
	
	int hitTest( FLUID::Vec3f p ) const;
	bool hitTest( FLUID::Vec3f p, float r, std::vector<uint> &array, leaf3 *leaf=NULL ) const;
	void finalizeMultigridLevels();
	void initMultigridLevels(const vector<FLUID::Vec3f> &vertices = vector<FLUID::Vec3f>());
	void writeObj( const char *path ) const;
	void writeObjLevel(const char *path,int level) const;
	
	leaf3 *root;
	HashTable<FLUID::Vec3i, unsigned int> leaf_table;
	std::vector<leaf3 *> terminals; // Octree terminal list
	std::vector<FLUID::Vec3f> nodes;	// Octree corner list
	std::vector<std::vector<double>> Dofs_level_k;    //unknowns values at level k
	std::vector<FLUID::Vec3f> nodes_level_k;   //all nodes positions for all levels, 
	std::vector<std::vector<leaf3 *>> terminal_level_k; //terminal list at level k
	std::vector<FixedSparseMatrixd *> P_L;
	std::vector<FixedSparseMatrixd *> R_L;

	
	uint maxdepth;
	uint multigrid_levels;
	uint resolution;
	bool dontEnfortceWB;
	void subdivide( leaf3 *leaf, /*const levelset3 *hint,*/ const std::vector<sphere3> &spheres, uint maxdepth );
	void subdivide( leaf3 *leaf,const std::vector<Array3f *> &level_discptor, uint maxdepth );
	void enforceWeakBalance();

private:
	void formProlongationAndRestriction(FixedSparseMatrixd &P, 
		                                FixedSparseMatrixd &R,
										uint level, int dx);
	void countNumTerminal( leaf3 *leaf, uint &index );
	void BuildTerminalsAllLevels(std::vector<leaf3 *> &terminals_level0);
	void getTerminalsLevel_K(leaf3 *leaf, uint k);
	void buildArray( leaf3 *leaf, uint &index );
	void buildNodes();
	void buildNodes(vector<leaf3 *> &leafs, vector<double> &values, int level);
	bool clearData();
	bool releaseChilren( leaf3 *leaf );
	bool isIn(const FLUID::Vec3f &p, leaf3 *leaf);
	leaf3* allocLeaf( FLUID::Vec3i center, uint depth, FLUID::Vec3i position );
	leaf3* findLeaf(const FLUID::Vec3f &p, leaf3 *leaf);
	float box( FLUID::Vec3f p, FLUID::Vec3f p0, FLUID::Vec3f p1 ) const;
	void copy( leaf3 *src, leaf3 *dest );
	uint64 computeCornerIndex( FLUID::Vec3i &p ) const;
	//bool checkSubdivision( FLUID::Vec3i pos, uint dx, const levelset3 *hint, int threshold, int depth, uint max_nest ) const;
};

#endif