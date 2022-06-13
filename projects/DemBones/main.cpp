#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/ListObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/vec.h>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <fstream>


#include "DemBonesExt.h"
#include "MatBlocks.h"

using namespace std;
using namespace Eigen;
using namespace Dem;

class MyDemBones: public DemBonesExt<double, float> {
public:
	double tolerance;
	int patience;

	MyDemBones(): tolerance(1e-3), patience(3) { nIters=100; }

	void compute() {
		prevErr=-1;
		np=patience;
		DemBonesExt<double, float>::compute();
	}

	void cbIterBegin() {
        fmt::print("Iter {}: ", iter);
	}

	bool cbIterEnd() {
		double err=rmse();
        fmt::print("RMSE =  {}\n", err);
		if ((err<prevErr*(1+weightEps))&&((prevErr-err)<tolerance*prevErr)) {
			np--;
			if (np==0) {
				fmt::print("Convergence is reached!\n");
				return true;
			}
		} else np=patience;
		prevErr=err;
		return false;
	}

	void cbInitSplitBegin() {
	}

	void cbInitSplitEnd() {
	}

	void cbWeightsBegin() {
	}

	void cbWeightsEnd() {
	}

	void cbTranformationsBegin() {
	}

	void cbTransformationsEnd() {
	}

	bool cbTransformationsIterEnd() {
		return false;
	}

	bool cbWeightsIterEnd() {
		return false;
	}

private:
	double prevErr;
	int np;
};

namespace zeno {

struct BoneBindInfo: zeno::IObject {
    MatrixXd boneRestRotation;
    MatrixXd boneRestTranslate;
};

struct SkiningInfo: zeno::IObject {
    MatrixXd boneRestMatrix;
    SparseMatrix<double> skiningWeight;
};

struct FrameInfo: zeno::IObject {
    MatrixXd boneFrameRotation;
    MatrixXd boneFrameTranslate;
};

struct DemBones : zeno::INode {
    virtual void apply() override {
        auto list = get_input<ListObject>("BakedPrimList");
        auto vec_prims = list->get<shared_ptr<PrimitiveObject>>();
        auto prim = get_input<PrimitiveObject>("RefPrim");
        int bone_number = get_param<int>("bone_number");

        MyDemBones model;
        model.nS = 1;
        model.nV = prim->verts.size();
        model.nF = vec_prims.size();
        fmt::print("------------- 1 ----------------\n");
        fmt::print("------------- model.nF: {} ----------------\n", model.nF);
        fmt::print("------------- model.nV: {} ----------------\n", model.nV);

        size_t number_ref_model_vert = prim->verts.size();
        for (size_t f = 0; f < model.nF; f++) {
            size_t cur_frame_model_vert_number = vec_prims[f]->get()->verts.size();
            if (cur_frame_model_vert_number != number_ref_model_vert) {
                throw("dem bones vert num not match!");
            }
        }

        fmt::print("start read mesh sequence\n");
        // read mesh sequence
        {
            model.v.resize(3*model.nF, model.nV);
            model.fTime.resize(model.nF);
            model.fStart.resize(model.nS+1);
            model.fStart(0)=0;

            for(int k = 0; k < model.nF; k++) {
                for (int i = 0; i < model.nV; i++) {
                    auto pos = vec_prims[k]->get()->verts[i];
                    model.v.col(i).segment<3>(k * 3) << pos[0], pos[1], pos[2];
                }
            }
            model.fStart(1) = model.fStart(0) + model.nF;

            model.subjectID.resize(model.nF);
            for (int k=model.fStart(0); k<model.fStart(1); k++)  {
                model.subjectID(k)=0;
            }
        }

        fmt::print("start read reference mesh\n");

        // read reference mesh
        {
            MatrixXd v;
            v.resize(3, model.nV);
            for (int i = 0; i < model.nV; i++)  {
                auto &pos = prim->verts;
                v.col(i) << pos[i][0], pos[i][1], pos[i][2];
            }
            model.u.resize(3, model.nV);
            model.u.block(0, 0, 3, model.nV) = v;

            // model.fv=importer.fv;
            for (auto const& f: prim->tris) {
                model.fv.push_back({f[0], f[1], f[2]});
            }

            model.nB = bone_number;
            model.init();
        }
        fmt::print("Computing Skinning Decomposition:\n");

        model.compute();

        MatrixXd lr, lt, gb, lbr, lbt;
        model.computeRTB(0, lr, lt, gb, lbr, lbt);
        BoneBindInfo boneBindInfo;
        boneBindInfo.boneRestRotation = lbr;
        boneBindInfo.boneRestTranslate = lbt;

        SkiningInfo skiningInfo;
        skiningInfo.boneRestMatrix = gb;
        skiningInfo.skiningWeight = model.w;

        FrameInfo frameInfo;
        frameInfo.boneFrameRotation = lr;
        frameInfo.boneFrameTranslate = lt;


        set_output("BoneBindInfo", make_shared<BoneBindInfo>(boneBindInfo));
        set_output("SkiningInfo", make_shared<SkiningInfo>(skiningInfo));
        set_output("FrameInfo", make_shared<FrameInfo>(frameInfo));
    }
};

ZENDEFNODE(DemBones,{
    {"BakedPrimList", "RefPrim"},
    {
        "BoneBindInfo",
        "SkiningInfo",
        "FrameInfo",
    },
    {{"int", "bone_number", "20"}},
    {"DemBones"},
});

}
