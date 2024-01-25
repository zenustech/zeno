#include "./kdsearch.h"

namespace zeno{
void KdTree::split(const std::pair<vec3f, vec3f> &bnds, int lid, int rid, int &cut_dim, float &cut_val, int &n_lo) {
	float max_length = std::max(std::max(bnds.second[0]-bnds.first[0], bnds.second[1]-bnds.first[1]), bnds.second[2]-bnds.first[2]);
	float max_spread = -1;
	for (int d = 0; d < 3; d++)
		if ((bnds.second[d] - bnds.first[d]) >= (1-1e-3)*max_length) {
            float dmin = pts[pidx[lid]][d], dmax = pts[pidx[lid]][d];
            for (int i = lid + 1; i < rid; i++) {
                float c = pts[pidx[i]][d];
                if (c < dmin) dmin = c;
                else if (c > dmax) dmax = c;
            }
			float spr = dmax - dmin;
			if (spr > max_spread) {
				max_spread = spr;
				cut_dim = d;
			}
		}

	float ideal_cut_val = (bnds.first[cut_dim] + bnds.second[cut_dim]) / 2;
    float min = pts[pidx[lid]][cut_dim], max = pts[pidx[lid]][cut_dim];
    for (int i = lid + 1; i < rid; i++) {
        float c = pts[pidx[i]][cut_dim];
        if (c < min) min = c;
        else if (c > max) max = c;
    }

	if (ideal_cut_val < min)
		cut_val = min;
	else if (ideal_cut_val > max)
		cut_val = max;
	else
		cut_val = ideal_cut_val;


	int br1, br2;
	int l = lid;
	int r = rid-1;
	while(1) {
		while (l < rid && pts[pidx[l]][cut_dim] < cut_val) l++;
		while (r >= lid && pts[pidx[r]][cut_dim] >= cut_val) r--;
		if (l > r) break;
        pidx[l]^=pidx[r]; pidx[r]^=pidx[l]; pidx[l]^=pidx[r];
		l++; r--;
	}
	br1 = l;
	r = rid-1;
	while(1) {
		while (l < rid && pts[pidx[l]][cut_dim] <= cut_val) l++;
		while (r >= br1 && pts[pidx[r]][cut_dim] > cut_val) r--;
		if (l > r) break;
        pidx[l]^=pidx[r]; pidx[r]^=pidx[l]; pidx[l]^=pidx[r];
		l++; r--;
	}
	br2 = l;

	if (ideal_cut_val < min) n_lo = lid+1;
	else if (ideal_cut_val > max) n_lo = rid-1;
	else if (br1 > (lid+rid)/2) n_lo = br1;
	else if (br2 < (lid+rid)/2) n_lo = br2;
	else n_lo = (lid+rid)/2;
}

KdNode* KdTree::construct_tree(int lid, int rid, std::pair<vec3f, vec3f>& bnd_box) {
	if (rid - lid <= bkt_size) {
		if (rid - lid == 0)
			return nullptr;
		else {
            KdLeaf* leaf = new KdLeaf(lid, rid);
            leaf->tree = this;
			return leaf;
        }
	} else {
		int cd;
		float cv;
		int n_lo;

		split(bnd_box, lid, rid, cd, cv, n_lo);

		float lv = bnd_box.first[cd];
		float hv = bnd_box.second[cd];

		bnd_box.second[cd] = cv;
		KdNode* lo = construct_tree(lid, n_lo, bnd_box);
		bnd_box.second[cd] = hv;

		bnd_box.first[cd] = cv;
		KdNode* hi = construct_tree(n_lo, rid, bnd_box);
		bnd_box.first[cd] = lv;

        KdSplit* split = new KdSplit(cd, cv, lv, hv, lo, hi);
        split->tree = this;
		return split;
	}
} 

KdTree::KdTree(const std::vector<zeno::vec3f>& pa, int n, int bs) : n_pts(n), bkt_size(bs) {
	if (n == 0) return;
	pts = std::move(pa);
    pidx.resize(n);
    for (int i = 0; i < n; ++i)
        pidx[i] = i;

	std::pair<vec3f, vec3f> bnd_box = std::make_pair(vec3f(0), vec3f(0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < 3; ++j) {
            if(pts[i][j] < bnd_box.first[j]) bnd_box.first[j] = pts[i][j];
            if(pts[i][j] > bnd_box.second[j]) bnd_box.second[j] = pts[i][j];
        }

	root = construct_tree(0, n_pts, bnd_box);
}

std::set<int> KdTree::fix_radius_search(const zeno::vec3f& point, float radius) {
    std::set<int> closest{};
	closest.clear();
	root->fix_radius_search(0, radius*radius, point, closest);
	return std::move(closest);
}

void KdSplit::fix_radius_search(float box_dist, float radius, const vec3f& point, std::set<int>& closest) {
	float cut_diff = point[cut_dim] - cut_val;

	if (cut_diff < 0) {
        if (child[0])
		    child[0]->fix_radius_search(box_dist, radius, point, closest);

		float box_diff = cd_bnds[0] - point[cut_dim];
		if (box_diff < 0)
			box_diff = 0;
		box_dist = box_dist + cut_diff * cut_diff - box_diff * box_diff;

		if (child[1] && box_dist <= radius)
			child[1]->fix_radius_search(box_dist, radius, point, closest);
	} else {
        if (child[1])
		    child[1]->fix_radius_search(box_dist, radius, point, closest);

		float box_diff = point[cut_dim] - cd_bnds[1];
		if (box_diff < 0)
			box_diff = 0;
		box_dist = box_dist + cut_diff *cut_diff - box_diff * box_diff;

		if (child[0] && box_dist <= radius)
			child[0]->fix_radius_search(box_dist, radius, point, closest);
	}
}

void KdLeaf::fix_radius_search(float box_dist, float radius, const vec3f& point, std::set<int>& closest) {
	for (int i = lid; i < rid; i++) {
		float dist = lengthSquared(tree->pts[tree->pidx[i]] - point);
		if (dist < radius)
			closest.insert(tree->pidx[i]);
	}
}
} // namespace zeno
