#include "usdnodealigner.h"

#include <zeno/utils/log.h>

#include <iostream>
#include <queue>

using MyTreeNode = USDNodeAligner::TreeNode;

const static float NODE_OFFSET_X = 1000.0f;
const static float NODE_OFFSET_Y = 300.0f;

USDNodeAligner* USDNodeAligner::_instance = nullptr;

USDNodeAligner& USDNodeAligner::instance() {
	if (_instance == nullptr) {
		_instance = new USDNodeAligner;
	}
	return *_instance;
}

USDNodeAligner::USDNodeAligner() {
	mLinkMap.clear();
	mDepthToSize.clear();
	mRootNode = 0;
	mMainGraph = 0;
}

void USDNodeAligner::setupGraph(ZENO_HANDLE mainGraph, ZENO_HANDLE rootNode) {
	mMainGraph = mainGraph;
	mRootNode = rootNode;
}

void USDNodeAligner::doAlign(const std::pair<float, float>& anchorPos) {
	if (mRootNode == 0) {
		zeno::log_warn("[doAlign] illegal root node for the graph, no aligning...");
		return;
	}
	if (mMainGraph == 0) {
		zeno::log_warn("[doAlign] illegal graph handle, no aligning...");
		return;
	}

	_beforeDFS();

	int maxDepth = mDepthToSize.size();
	if (maxDepth == 0) {
		zeno::log_warn("no nodes in the graph, no aligning...");
		return;
	}

	std::vector<int> visited(maxDepth + 1, 0); // recording how many nodes have been visited for each depth
	/*
	* in zeno node system, the graph starts from leaves
	* so we should put the root on the right side
	*/
	std::pair<float, float> rootPos = { anchorPos.first + maxDepth * NODE_OFFSET_X, anchorPos.second};

	_DFS(rootPos, visited, mRootNode);

	mLinkMap.clear();
	mDepthToSize.clear();
	mRootNode = 0;
	mMainGraph = 0;
}

void USDNodeAligner::_beforeDFS() {
	/*
	* traverse the graph in bfs order and set depth
	*/
	if (mRootNode == 0) {
		zeno::log_error("[_beforeDFS]: root node is illegal");
		return;
	}
	std::queue<ZENO_HANDLE> tobeVisited;

	tobeVisited.push(mRootNode);
	mLinkMap[mRootNode].mDepth = 0;

	while (!tobeVisited.empty()) {
		auto handle = tobeVisited.front();
		tobeVisited.pop();
		auto& node = mLinkMap[handle];

		for (auto childHandle : node.mChilds) {
			auto& childNode = mLinkMap[childHandle];
			childNode.mDepth = node.mDepth + 1;

			if (mDepthToSize.find(childNode.mDepth) == mDepthToSize.end()) {
				mDepthToSize[childNode.mDepth] = 1;
			}
			else {
				mDepthToSize[childNode.mDepth] += 1;
			}

			tobeVisited.push(childHandle);
		}
	}
}

void USDNodeAligner::_DFS(const std::pair<float, float>& rootPos, std::vector<int>& visitMap, ZENO_HANDLE curNodeHandle) {
	const MyTreeNode& node = mLinkMap[curNodeHandle];
	if (node.mHandle != curNodeHandle) {
		zeno::log_error("found illegal node handle info in the node graph: %d %d", curNodeHandle, node.mHandle);
		return;
	}

	std::pair<float, float> pos;
	if (curNodeHandle == mRootNode) {
		pos = rootPos;
	}
	else {
		int& visitOrder = visitMap[node.mDepth];

		float heightOfDepth = NODE_OFFSET_Y * (mDepthToSize[node.mDepth] - 1);

		pos.first = rootPos.first - node.mDepth * NODE_OFFSET_X;
		pos.second = rootPos.second - heightOfDepth * 0.5f + visitOrder * NODE_OFFSET_Y;

		++visitOrder;
	}
	Zeno_SetPos(mMainGraph, curNodeHandle, pos);

	for (auto handle : node.mChilds) {
		_DFS(rootPos, visitMap, handle);
	}
}

void USDNodeAligner::addChild(ZENO_HANDLE parent, ZENO_HANDLE child) {
	if (!parent || !child) {
		zeno::log_error("illegal link pair: %d %d", parent, child);
		return;
	}

	MyTreeNode& pNode = mLinkMap[parent];
	MyTreeNode& cNode = mLinkMap[child];

	pNode.mHandle = parent;
	pNode.mChilds.push_back(child);
	pNode.mHandle = parent;

	cNode.mHandle = child;
}
