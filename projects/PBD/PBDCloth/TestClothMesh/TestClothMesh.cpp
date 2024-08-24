#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>

#include "TestClothMeshData.h"


namespace zeno{

struct TestClothMesh : zeno::INode
{
	inline static const TestClothMeshData mesh;
	virtual void apply() override 
    {
	    auto prim = std::make_shared<zeno::PrimitiveObject>();

        auto &pos = prim->verts;
        auto &tris = prim->tris;

		pos.resize(std::size(mesh.pos)/3);
		tris.resize(std::size(mesh.tris)/3);

		for(int i = 0; i < pos.size(); i++)
			for (int j = 0; j < 3; j++)
				pos[i][j] = mesh.pos[i * 3 + j];

		for(int i = 0; i < tris.size(); i++)
			for (int j = 0; j < 3; j++)
				tris[i][j] = mesh.tris[i * 3 + j];

		// //move above 
		// for(int i = 0; i < pos.size(); i++)
		// 	pos[i][1] += 1.0;

        set_output("prim", std::move(prim));
	};
};

ZENDEFNODE(TestClothMesh, {
    {},
    {gParamType_Primitive, "prim"},
    {},
    {"PBD"},
});

}
