#ifndef kd_search_H
#define kd_search_H

#include <vector>
#include <set>
#include <zeno/utils/vec.h>
#include <zeno/utils/log.h>

namespace zeno{
class KdNode;

class KdTree {
protected:
	int n_pts;
	int bkt_size;
	std::vector<vec3f> pts;
	std::vector<int> pidx;
	KdNode* root;
    void split(const std::pair<vec3f, vec3f> &bnds, int lid, int rid, int &cut_dim, float &cut_val, int &n_lo);
    KdNode* construct_tree(int lid, int rid, std::pair<vec3f, vec3f>& bnd_box);

public:
	KdTree(const std::vector<vec3f>& pa, int n, int bs = 10);
	~KdTree() {
        if (root != NULL) delete root;
    }
	std::set<int> fix_radius_search(const vec3f& point, float radius);

    friend class KdSplit;
    friend class KdLeaf;
};

class KdNode {
protected:
    KdTree* tree;
public:
	virtual ~KdNode() {}
	virtual void fix_radius_search(float box_dist, float radius, const vec3f& point, std::set<int>& closest) = 0;

	friend class KdTree;
};

class KdSplit : public KdNode {
protected:
	int				cut_dim;
	float			cut_val;
	float			cd_bnds[2];
	KdNode*		    child[2];
public:
	KdSplit(int cd, float cv, float lv, float hv, KdNode* lc, KdNode* hc) {
        cut_dim		= cd;
        cut_val		= cv;
        cd_bnds[0]  = lv;
        cd_bnds[1]  = hv;
        child[0]	= lc;
        child[1]	= hc;
    }

	~KdSplit() {
        if (child[0])
            delete child[0];
        if (child[1])
            delete child[1];
    }
	void fix_radius_search(float box_dist, float radius, const vec3f& point, std::set<int>& closest);
};		


class KdLeaf : public KdNode {
protected:
	int lid;
    int rid;
public:
	KdLeaf(int l, int r) : lid(l), rid(r) {}
	~KdLeaf() {}
	void fix_radius_search(float box_dist, float radius, const vec3f& point, std::set<int>& closest);
};

} // namespace zeno
#endif