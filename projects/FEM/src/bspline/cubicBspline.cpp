#include <cubicBspline.h>

UniformCubicBasisSpline::UniformCubicBasisSpline(){
    base_matrix << -1, 3, -3, 1,
		3, -6, 3, 0,
		-3, 3, 0, 0,
		1, 0, 0, 0;
	Clear();
}

void UniformCubicBasisSpline::Resize(size_t nm_ctrlps) {
	assert(nm_ctrlps > 1);
	ctrlps = MatXd::Zero(nm_ctrlps,2);
	poly_bezier_ctrlps = MatXd::Zero(nm_ctrlps * 3 - 2,2);
	integration_offset.resize(nm_ctrlps);

	std::vector<Eigen::Triplet<double>> triplets;
	double one_third = 1.0 / 3.0;
	double two_third = 2.0 / 3.0;
	double one_six = 1.0 / 6.0;

	cp2pp_map = SpMat(nm_ctrlps*3 - 2, nm_ctrlps);
	triplets.clear();
	triplets.emplace_back(0, 0, 1.0);
	for (int i = 0; i < nm_ctrlps - 2; ++i) {
		triplets.emplace_back(i * 3 + 1, i, two_third);
		triplets.emplace_back(i * 3 + 1, i + 1, one_third);
		triplets.emplace_back(i * 3 + 2, i, one_third);
		triplets.emplace_back(i * 3 + 2, i + 1, two_third);

		triplets.emplace_back(i * 3 + 3, i, one_six);
		triplets.emplace_back(i * 3 + 3, i + 1, two_third);
		triplets.emplace_back(i * 3 + 3, i + 2, one_six);
	}
	triplets.emplace_back(nm_ctrlps * 3 - 5, nm_ctrlps - 2, two_third);
	triplets.emplace_back(nm_ctrlps * 3 - 5, nm_ctrlps - 1, one_third);
	triplets.emplace_back(nm_ctrlps * 3 - 4, nm_ctrlps - 2, one_third);
	triplets.emplace_back(nm_ctrlps * 3 - 4, nm_ctrlps - 1, two_third);
	triplets.emplace_back(nm_ctrlps * 3 - 3, nm_ctrlps - 1, 1.0);
	cp2pp_map.setFromTriplets(triplets.begin(), triplets.end());
    cp2pp_map.makeCompressed();
}

void UniformCubicBasisSpline::Interpolate(const VecXd& interps,const Vec2d& _inter_range) {
    int n = interps.size();
    Resize(n);
    inner_range = _inter_range;

    int start_idx = 0;
	assert(n >= 2);// for cubic interpolation, we should have at least 3 interpolation points

    ctrlps = interps;

	if(n == 3){
		ctrlps[1] = (3./2.) * interps[1] - (1./4.) * interps[0] - (1./4.) * interps[2];
	}else if (n > 3){
		VecXd a = VecXd::Ones(n);a[n - 1] = 0;
		VecXd b = VecXd::Ones(n) * 4;b[0] = b[n - 1] = 1;
		VecXd c = VecXd::Ones(n);c[0] = 0.0;
		ctrlps.segment(1,n - 2) *= 6.0;
		thomas_algorithm(a,b,c, ctrlps);
	}

	poly_bezier_ctrlps = cp2pp_map * ctrlps;

    update_integration_offset();
}

void UniformCubicBasisSpline::SetCtrlPoints(const VecXd& _ctrlps) {
    assert(_ctrlsps.size() == ctrlps.size());
	ctrlps = _ctrlps;
	poly_bezier_ctrlps = cp2pp_map * ctrlps;
    update_integration_offset();
}

void UniformCubicBasisSpline::SetCtrlPointsAt(size_t idx,const FEM_Scaler value) {
    assert(idx >=0 && idx < GetNumCtrlPoints());
    ctrlps[idx] = value;
    poly_bezier_ctrlps = cp2pp_map * ctrlps;
	update_integration_offset();
}