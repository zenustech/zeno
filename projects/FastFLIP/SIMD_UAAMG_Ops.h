#pragma once

namespace simd_uaamg {
struct TouchCoarseLeafReducer {

	TouchCoarseLeafReducer(const std::vector<openvdb::Int32Tree::LeafNodeType*>& fineLeaves) :
		mFineLeaves(fineLeaves) {
		mCoarseDofGrid = openvdb::Int32Grid::create(-1);
	}

	TouchCoarseLeafReducer(const TouchCoarseLeafReducer& other, tbb::split) :
		mFineLeaves(other.mFineLeaves) {
		mCoarseDofGrid = openvdb::Int32Grid::create(-1);
	}

	void operator()(const tbb::blocked_range<openvdb::Index>& r) {
		for (openvdb::Index i = r.begin(); i != r.end(); ++i) {
			mCoarseDofGrid->tree().touchLeaf(openvdb::Coord(mFineLeaves[i]->origin().asVec3i() / 2));
		}
	}

	void join(TouchCoarseLeafReducer& other) {
		auto& grid = *other.mCoarseDofGrid;
		//merge the counter grid
		for (auto leaf = grid.tree().beginLeaf(); leaf; ++leaf) {
			auto* newLeaf = mCoarseDofGrid->tree().probeLeaf(leaf->origin());
			if (!newLeaf) {
				// if the leaf doesn't yet exist in the new tree, steal it
				auto& tree = const_cast<openvdb::Int32Grid&>(grid).tree();
				mCoarseDofGrid->tree().addLeaf(tree.template stealNode<openvdb::Int32Tree::LeafNodeType>(leaf->origin(),
					-1, false));
			}
		}
	}

	const std::vector<openvdb::Int32Tree::LeafNodeType* >& mFineLeaves;
	openvdb::Int32Grid::Ptr mCoarseDofGrid;
};
}