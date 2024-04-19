#pragma once
#include <zenomodel/include/api.h>

#include <map>
#include <set>

class USDNodeAligner {
public:
	struct TreeNode {
		std::vector<ZENO_HANDLE> mChilds;
		ZENO_HANDLE mHandle = 0;
		int mDepth = 0;
	};

	static USDNodeAligner& instance();

	void setupGraph(ZENO_HANDLE mainGraph, ZENO_HANDLE rootNode);

	void addChild(ZENO_HANDLE parent, ZENO_HANDLE child);

	void doAlign(const std::pair<float, float>& anchorPos);

private:
	static USDNodeAligner* _instance;

	USDNodeAligner();

	void _beforeDFS();
	void _DFS(const std::pair<float, float>& rootPos, std::vector<int>& visitMap, ZENO_HANDLE curNodeHandle);

	std::map<ZENO_HANDLE, TreeNode> mLinkMap;
	std::map<int, int> mDepthToSize;
	ZENO_HANDLE mRootNode;
	ZENO_HANDLE mMainGraph;
};
