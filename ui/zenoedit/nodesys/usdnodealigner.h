#pragma once
#include <zenomodel/include/api.h>

#include <vector>
#include <map>
#include <set>

class USDNodeAligner {
public:
	struct TreeNode {
		std::vector<ZENO_HANDLE> mChilds;
		ZENO_HANDLE mHandle = 0;
		int mDepth = 0;
	};

	struct TreeInfo {
		std::map<int, int> mDepthToSize;
		ZENO_HANDLE mRootNode;
		ZENO_HANDLE mGraph;
	};

	static USDNodeAligner& instance();

	void setGraphAnchor(ZENO_HANDLE graph, const std::pair<float, float>& anthorPos);

	void addChild(ZENO_HANDLE graph, ZENO_HANDLE parent, ZENO_HANDLE child);

	void doAlign();

private:
	static USDNodeAligner* _instance;

	USDNodeAligner();

	void _buildTree(std::map<ZENO_HANDLE, TreeNode>& graphNodes, ZENO_HANDLE graphHandle, ZENO_HANDLE rootNode);
	void _DFS(TreeInfo& tree, std::map<ZENO_HANDLE, TreeNode>& graphNodes, const std::pair<float, float>& rootPos, std::vector<int>& visitMap, ZENO_HANDLE curNodeHandle);

	std::map<ZENO_HANDLE, std::map<ZENO_HANDLE, TreeNode>> mGraphToNodes; // arr[graph handle][zeno handle] = TreeNode()
	std::map<ZENO_HANDLE, std::pair<float, float>> mGraphToAnchor;
	std::vector<TreeInfo> mTrees;
	int mSizeOfNodes;
};
