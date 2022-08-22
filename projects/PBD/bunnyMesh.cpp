#include <string>
#include <iostream>
#include <vector>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>

#include "BunnyMeshData.h"

namespace zeno{

struct bunnyMesh : zeno::INode
{
	BunnyMeshData theBunnyMesh;
	virtual void apply() override 
    {
	    auto prim = std::make_shared<zeno::PrimitiveObject>();

        auto &pos = prim->verts;
        auto &tet = prim->quads;
        auto &edge = prim->lines;
        auto &surf = prim->tris;

		int numParticles = theBunnyMesh.pos.size()/3;
		int numTets = theBunnyMesh.tet.size()/4;
		int numEdges = theBunnyMesh.edge.size()/2;
		int numSurfs = theBunnyMesh.surf.size()/3;

		pos.resize(numParticles);
		tet.resize(numTets);
		edge.resize(numEdges);
		surf.resize(numSurfs);

		for(int i = 0; i < numParticles; i++)
			for (int j = 0; j < 3; j++)
				pos[i][j] = theBunnyMesh.pos[i * 3 + j];

		for(int i = 0; i < numTets; i++)
			for (int j = 0; j < 4; j++)
				tet[i][j] = theBunnyMesh.tet[i * 4 + j];

		for(int i = 0; i < numEdges; i++)
			for (int j = 0; j < 2; j++)
				edge[i][j] = theBunnyMesh.edge[i * 2 + j];

		for(int i = 0; i < numSurfs; i++)
			for (int j = 0; j < 3; j++)
				surf[i][j] = theBunnyMesh.surf[i * 3 + j];

		// std::cout << "created a bunny tetrahedron mesh!" << std::endl;
		// std::cout << "numParticles:" << numParticles<< std::endl;
		// std::cout << "numEdges:" << numEdges<< std::endl;
		// std::cout << "numTets:" << numTets<< std::endl;
		// std::cout << "numSurfs:" << numSurfs<< std::endl;

        set_output("prim", std::move(prim));
	};
};

ZENDEFNODE(bunnyMesh, {
    {},
    {"prim"},
    {},
    {"PBD"},
});

}
