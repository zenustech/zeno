#include <string>
#include <vector>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/zeno.h>
#include <zeno/utils/log.h>

#include "BunnyMeshData.h"
#include "../Utils/myPrint.h"

namespace zeno{

struct BunnyMesh : zeno::INode
{
	inline static const BunnyMeshData theBunnyMesh;
	virtual void apply() override 
    {
	    auto prim = std::make_shared<zeno::PrimitiveObject>();

        auto &pos = prim->verts;
        auto &tet = prim->quads;
        auto &edge = prim->lines;
        auto &surf = prim->tris;

		int numParticles = std::size(theBunnyMesh.pos)/3;
		int numTets = std::size(theBunnyMesh.tet)/4;
		int numEdges = std::size(theBunnyMesh.edge)/2;
		int numSurfs = std::size(theBunnyMesh.surf)/3;

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
		
		//move above to test the dropping scence
		for(int i = 0; i < numParticles; i++)
			pos[i][1] += 1.0;

		log_info("created a bunny tetrahedron mesh");
		log_info("numParticles: {}", numParticles);
		log_info("numEdges: {}", numEdges);
		log_info("numTets: {}", numTets);
		log_info("numSurfs: {}", numSurfs);

        set_output("prim", std::move(prim));
	};
};

ZENDEFNODE(BunnyMesh, {
    {},
    {gParamType_Primitive, "prim"},
    {},
    {"PBD"},
});

}
