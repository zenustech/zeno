#ifndef CUBIC_B_SPLINE_HPP
#define CUBIC_B_SPLINE_HPP

#include <matrix_helper.hpp>
#include <cmath>

class UniformCubicBasisSpline {
public:
	UniformCubicBasisSpline();
	void Resize(size_t nm_ctrlps);
	void Interpolate(const VecXd& interps,const Vec2d& inter_range);

	void SetCtrlPoints(const VecXd& interps);
    void SetCtrlPointsAt(size_t idx,FEM_Scaler value);

	inline size_t GetNumCtrlPoints() const { return ctrlps.rows(); }
	inline size_t GetNumBezierCtrlPoints() const { return poly_bezier_ctrlps.rows(); }
	inline size_t GetDegree() const { return 3; }
	inline size_t GetOrder() const { return GetDegree() + 1; }

	inline const VecXd& GetCtrlPoints() const { return ctrlps; }
	inline FEM_Scaler GetCtrlPointAt(int idx) const { return ctrlps[idx]; }
	inline const VecXd& GetPolyBezierCtrlps() const { return poly_bezier_ctrlps; }
	inline FEM_Scaler GetPolyBezierCtrlpAt(int idx) const { return poly_bezier_ctrlps[idx]; }

	FEM_Scaler EvalOnKnot(double knot) const {
		double u;
		int bezier_idx;
		query_knot(knot, u, bezier_idx);
		VecXd U = VecXd::Ones(GetOrder());
		for (int i = GetOrder() - 2; i >= 0; --i)
			U[i] = U[i + 1] * u;

		FEM_Scaler result = U.transpose() * base_matrix * poly_bezier_ctrlps.segment(bezier_idx * GetDegree(), GetOrder());
		// std::cout << "bezier_idx : " << knot << "\t" << u  << "\t" << bezier_idx << std::endl;

		return result;
	}

	FEM_Scaler EvalDerivativeOnKnot(double knot) const {
		double u;// the local position
		int bezier_idx;
		query_knot(knot, u, bezier_idx);

		FEM_Scaler width = inner_range[1] - inner_range[0];
		FEM_Scaler D = width / (GetNumCtrlPoints() - 1);


		VecXd DU = VecXd::Zero(GetOrder());
		DU[GetOrder() - 1] = 0;
		DU[GetOrder() - 2] = 1;
		for (int i = GetOrder() - 3; i >= 0; --i)
			DU[i] = DU[i + 1] * u;
		for (int i = GetOrder() - 3; i >= 0; --i)
			DU[i] *= (GetOrder() - i - 1);

		FEM_Scaler result = DU.transpose() * base_matrix * poly_bezier_ctrlps.segment(bezier_idx * GetDegree(), GetOrder());
		result /= D;
		return result;
	}

	FEM_Scaler EvalIntegrationOnKnot(double knot) const {
		double u;
		int bezier_idx;
		query_knot(knot, u, bezier_idx);

		FEM_Scaler width = inner_range[1] - inner_range[0];
		FEM_Scaler D = width / (GetNumCtrlPoints() - 1);

		VecXd IU = VecXd::Ones(GetOrder()) * u;
		for (int i = GetOrder() - 2; i >= 0; --i)
			IU[i] = IU[i + 1] * u;
		for (int i = 0;i < GetOrder();++i)
			IU[i] /= (GetOrder() - i);

		IU *= D;

		auto yc = poly_bezier_ctrlps.segment(bezier_idx * GetDegree(), GetOrder());

		FEM_Scaler result = IU.transpose() * base_matrix * yc;
		return integration_offset[bezier_idx] + result;
	}

	VecXd SampleCurve(int num_sample_points,double extent = 0.5) const {
		assert(num_sample_points > 1);
		VecXd samples = VecXd::Zero(num_sample_points);
		double min = inner_range[0] - extent;
		double max = inner_range[1] + extent;
		double step = (max - min) / (num_sample_points - 1);
		for (int i = 0; i < num_sample_points; ++i) {
			double u = min + i * step;
			// std::cout << "sample_u : " << u << std::endl;
			samples[i] = EvalOnKnot(u);
		}
		return samples;
	}

	bool IsFunction() const {return true;}

	void Clear() {
		ctrlps = MatXd::Zero(0, 0);
		poly_bezier_ctrlps = MatXd::Zero(0, 0);
		integration_offset.clear();
	}

	static void DebugCode() {
		UniformCubicBasisSpline spline;
		Vec2d range = Vec2d(-2,2);
		FEM_Scaler width = range[1] - range[0];
		size_t nm_segs = 50;

		VecXd interps = VecXd::Zero(nm_segs + 1);

		const FEM_Scaler DEBUG_PI = 3.141592653;

		FEM_Scaler period = width;
		for(size_t i = 0;i < nm_segs + 1;++i){
			FEM_Scaler phase = i * 2*DEBUG_PI / nm_segs;
			interps[i] = sin(phase) * 2;
		}
		spline.Interpolate(interps,range);

		size_t deriv_sample_segs = 100;
		VecXd ref_deriv = VecXd::Zero(deriv_sample_segs + 1);	
		for(size_t i = 0;i < deriv_sample_segs + 1;++i){
			FEM_Scaler phase = i * 2*DEBUG_PI / deriv_sample_segs;
			ref_deriv[i] = cos(phase) * 2 * 2*DEBUG_PI / period;
		}
		FEM_Scaler step = width / (deriv_sample_segs);
		VecXd deriv_cmp = VecXd::Zero(deriv_sample_segs + 1);
		for(size_t i = 0;i < deriv_sample_segs + 1;++i) {
			deriv_cmp[i] = spline.EvalDerivativeOnKnot(range[0] + step * i);
		}

		// for(size_t i = 0;i < deriv_cmp.size();++i)
		// 	std::cout << "D<" << i << "> : \t" << ref_deriv[i] << "\t" << deriv_cmp[i] << std::endl;


		// std::cout << "update_int_interval : " << std::endl;
		// for(size_t i = 0;i < spline.integration_offset.size();++i){
		// 	std::cout << "INT<" << i << "> : \t" << spline.integration_offset[i] << std::endl;
		// }

		size_t int_sample_segs = 200;
		VecXd ref_int = VecXd::Zero(int_sample_segs + 1);
		VecXd cmp_int = VecXd::Zero(int_sample_segs + 1);

		step = width / int_sample_segs;
		FEM_Scaler C = 0;
		for(size_t i = 0;i < ref_int.size();++i){
			FEM_Scaler phase = i * 2*DEBUG_PI / int_sample_segs;
			ref_int[i] = -cos(phase) * 2 * period / 2 / DEBUG_PI;
			if(i == 0)
				C = ref_int[i];
			ref_int[i] -= C;
			cmp_int[i] = spline.EvalIntegrationOnKnot(range[0] + step * i);
		}

		// for(size_t i = 0;i < ref_int.size();++i){
		// 	std::cout << "I<" << i << "> : " << ref_int[i] << "\t" << cmp_int[i] << std::endl;
		// }
		// std::cout << "interps : " << std::endl << interps << std::endl;
		// std::cout << "polybezier : " << std::endl << spline.GetPolyBezierCtrlps() << std::endl;
		// std::cout << "ctrlps : " << std::endl << spline.GetCtrlPoints() << std::endl;
		// auto samples = spline.SampleCurve(101,0);
		// std::cout << "sample : " << std::endl << samples << std::endl;

		
	}

protected:
    Vec2d inner_range;
	VecXd ctrlps;
	VecXd poly_bezier_ctrlps;
	Mat4x4d base_matrix;
	std::vector<double> integration_offset;

protected:
	void update_integration_offset() {
		integration_offset[0] = 0;
		FEM_Scaler width = inner_range[1] - inner_range[0];
		FEM_Scaler D = width / (GetNumCtrlPoints() - 1);

		Vec4d IU;
		IU << 1.0/4,1.0/3,1.0/2,1.0;
		IU *= D;


		for (int i = 0; i < GetNumCtrlPoints() - 1; ++i) {
			auto yc = poly_bezier_ctrlps.segment(i * GetDegree(), GetOrder());
			FEM_Scaler delta = yc.transpose() * base_matrix * IU;
			integration_offset[i + 1] = integration_offset[i] + delta;
		}
	}

	void query_knot(double knot, double& u, int& bezier_idx) const {
		FEM_Scaler width = inner_range[1] - inner_range[0];
		FEM_Scaler D = width / (GetNumCtrlPoints() - 1);
		bezier_idx = std::floor((knot - inner_range[0]) / D);
		bezier_idx = bezier_idx < 0 ? 0 : bezier_idx;
		bezier_idx = bezier_idx > GetNumCtrlPoints() - 2 ? GetNumCtrlPoints() - 2 : bezier_idx;

		u = (knot - bezier_idx * D - inner_range[0]) / D;
		// std::cout << "quergy : " << bezier_idx << "\t" << knot << "\t" << u << "\t" << inner_range.transpose() << std::endl;
		if (bezier_idx > 0 && bezier_idx < GetNumCtrlPoints() - 2)
			assert(u >= 0 && u <= 1);
	}

private:
	SpMat cp2pp_map;

	void thomas_algorithm(const VecXd& a, const VecXd& b, VecXd& gamma, VecXd& R) const {
		int n = R.size();
		assert(n > 2);
		assert(a.size() == n && b.size() == n && gamma.size() == n);
		gamma[0] = gamma[0] / b[0];
		R[0] = R[0] / b[0];
		for (int i = 1; i < n; ++i) {
			gamma[i] = gamma[i] / (b[i] - a[i]*gamma[i-1]);
			R[i] = (R[i] - a[i] * R[i-1]) / (b[i] - a[i] * gamma[i-1]);
		}
		// back substitution
		for (int i = n - 2; i >= 0; --i)
			R[i] = R[i] - gamma[i] * R[i+1];
	}

};

#endif //MR_BASIS_SPLINE_H