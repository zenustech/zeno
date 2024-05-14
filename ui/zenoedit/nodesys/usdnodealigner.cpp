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
	mGraphToNodes.clear();
	mGraphToAnchor.clear();
	mTrees.clear();
	mSizeOfNodes = 0;
}

void USDNodeAligner::doAlign() {
	if (mGraphToNodes.empty()) {
		zeno::log_info("No nodes to be aligned.");
		return;
	}

	mSizeOfNodes = 0;

	// construct node trees from all graphs
	for (auto graphIt = mGraphToNodes.begin(); graphIt !=mGraphToNodes.end(); ++graphIt) {
		auto& graphNodes = graphIt->second;

		for (auto nodeIt = graphNodes.begin(); nodeIt != graphNodes.end(); ++nodeIt) {
			if (nodeIt->second.mDepth == 0) {
				_buildTree(graphNodes, graphIt->first, nodeIt->second.mHandle);
			}
		}
	}

	zeno::log_info("size of nodes to align: {}", mSizeOfNodes);

	for (auto& treeInfo : mTrees) {
		int maxDepth = treeInfo.mDepthToSize.size();
		if (maxDepth == 0) {
			continue;
		}

		std::vector<int> visited(maxDepth + 1, 0); // recording how many nodes have been visited for each depth

		/*
		* in zeno node system, the graph starts from leaves
		* so we should put the root on the right side
		*/
		std::pair<float, float> rootPos;
		if (mGraphToAnchor.find(treeInfo.mGraph) == mGraphToAnchor.end()) {
			rootPos = {50.0f + maxDepth * NODE_OFFSET_X, 500.0f};
		}
		else {
			auto& anchor = mGraphToAnchor[treeInfo.mGraph];
			rootPos = { anchor.first + maxDepth * NODE_OFFSET_X, anchor.second };
		}

		_DFS(treeInfo, mGraphToNodes[treeInfo.mGraph], rootPos, visited, treeInfo.mRootNode);
	}

	mGraphToNodes.clear();
	mGraphToAnchor.clear();
	mTrees.clear();
	mSizeOfNodes = 0;
}

void USDNodeAligner::_buildTree(std::map<ZENO_HANDLE, TreeNode>& graphNodes, ZENO_HANDLE graphHandle, ZENO_HANDLE rootNode) {
	/*
	* traverse the graph in bfs order and set depth
	*/
	if (rootNode == 0) {
		zeno::log_error("[_buildTree]: root node is illegal");
		return;
	}
	std::queue<ZENO_HANDLE> tobeVisited;

	tobeVisited.push(rootNode);
	graphNodes[rootNode].mDepth = 0;

	auto& treeInfo = mTrees.emplace_back();
	treeInfo.mGraph = graphHandle;
	treeInfo.mRootNode = rootNode;

	while (!tobeVisited.empty()) {
		++mSizeOfNodes;
		auto handle = tobeVisited.front();
		tobeVisited.pop();
		auto& node = graphNodes[handle];

		for (auto childHandle : node.mChilds) {
			auto& childNode = graphNodes[childHandle];
			childNode.mDepth = node.mDepth + 1;

			if (treeInfo.mDepthToSize.find(childNode.mDepth) == treeInfo.mDepthToSize.end()) {
				treeInfo.mDepthToSize[childNode.mDepth] = 1;
			}
			else {
				treeInfo.mDepthToSize[childNode.mDepth] += 1;
			}

			tobeVisited.push(childHandle);
		}
	}
}

void USDNodeAligner::_DFS(TreeInfo& tree, std::map<ZENO_HANDLE, TreeNode>& graphNodes, const std::pair<float, float>& rootPos, std::vector<int>& visitMap, ZENO_HANDLE curNodeHandle) {
	auto nodeIt = graphNodes.find(curNodeHandle);

	const MyTreeNode& node = graphNodes[curNodeHandle];

	std::pair<float, float> pos;
	int& visitOrder = visitMap[node.mDepth];

	float heightOfDepth = NODE_OFFSET_Y * (tree.mDepthToSize[node.mDepth] - 1);

	pos.first = rootPos.first - node.mDepth * NODE_OFFSET_X;
	pos.second = rootPos.second - heightOfDepth * 0.5f + visitOrder * NODE_OFFSET_Y;

	++visitOrder;
	Zeno_SetPos(tree.mGraph, curNodeHandle, pos);

	for (auto handle : node.mChilds) {
		_DFS(tree, graphNodes, rootPos, visitMap, handle);
	}
}

void USDNodeAligner::addChild(ZENO_HANDLE graph, ZENO_HANDLE parent, ZENO_HANDLE child) {
	if (!parent || !child) {
		zeno::log_error("illegal link pair: %d %d", parent, child);
		return;
	}

	auto& graphNodes = mGraphToNodes[graph];

	MyTreeNode& pNode = graphNodes[parent];
	MyTreeNode& cNode = graphNodes[child];

	pNode.mHandle = parent;
	pNode.mChilds.push_back(child);

	cNode.mHandle = child;
	// mark this node as child, then we know the node with depth == 0 is the root of the tree
	cNode.mDepth = 1;
}

void USDNodeAligner::setGraphAnchor(ZENO_HANDLE graph, const std::pair<float, float>& anthorPos) {
	mGraphToAnchor[graph] = anthorPos;
}
