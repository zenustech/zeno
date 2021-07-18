#ifndef __LINEAR_WAVE_SOLVER__
#define __LINEAR_WAVE_SOLVER__
#include"array2.h"
#include"AlgebraicMultigrid.h"
#include"array3.h"
#include <tbb\tbb.h>
#include <fstream>
//this solver tries to solve the linearized wave equation, given by
//
// dphi/dt = -gh   at y = 0
// dh/dt = dphi/dn at y = 0
// with laplace phi = 0


class LinearWaveSolver {
public:
	LinearWaveSolver() {}
	~LinearWaveSolver() {}
	void outputObj(std::string path, int frame)
	{

		ostringstream strout;
		strout << path << "/result_" << setfill('0') << setw(5) << frame << ".obj";

		string filepath = strout.str();

		ofstream outfile(filepath.c_str());
		//write vertices
		for (unsigned int i = 0; i < verts.size(); ++i)
		{
			outfile << "v" << " " << verts[i][0] << " " << verts[i][1] << " " << verts[i][2] << std::endl;
		}
		//write triangle faces
		for (unsigned int i = 0; i < faces.size(); ++i)
			outfile << "f" << " " << faces[i][0] + 1 << " " << faces[i][1] + 1 << " " << faces[i][2] + 1 << " " << faces[i][3] + 1 << std::endl;
		outfile.close();
	}
	void solveDphiDt(float dt)
	{
		int compute_num = ni*nk;
		tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
		{
			int k = thread_idx / ni;
			int i = thread_idx%ni;
			boundary_phi(i, k) += -9.8*dt*(verts[k*ni + i][1] - m_sea_level);
		});
	}
	void solveDhDt(float dt)
	{
		solveLaplaceFDM(dt);
		int compute_num = ni*nk;
		tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
		{
			int k = thread_idx / ni;
			int i = thread_idx%ni;
			verts[thread_idx][1] += 2.0*dt*(boundary_phi(i, k) - phi[k*ni*nj + (nj - 1)*ni + i]) / m_hx;
		});
	}
	void timestep(float dt)
	{
		if (start == 0)
		{
			start = 1;
			solveDphiDt(0.5*dt);
		}
		else
		{
			solveDphiDt(dt);
		}
		solveDhDt(dt);
	}
	void init(int nx, int ny, float sl, float L)
	{
		start = 0;
		ni = nx; nk = ny;nj = 64;
		m_hx = L / (float)nx;
		m_sea_level = sl;
		buildSurfaceMesh();
		buildLaplaceSolver();
	}
	void buildSurfaceMesh()
	{
		verts.resize(0);
		faces.resize(0);
		for (int j = 0;j<nk;j++)
			for (int i = 0;i < ni;i++)
			{
				verts.push_back(Vec3f(i, 0, j)*m_hx + Vec3f(0.5f*m_hx, m_sea_level, 0.5f*m_hx));
			}
		for (int j = 0;j<nk - 1;j++)
			for (int i = 0;i < ni - 1;i++)
			{
				faces.push_back(
					Vec4i(j*ni + i,
						j*ni + i + 1,
						(j + 1)*ni + i + 1,
						(j + 1)*ni + i));
			}
	}
	void solveLaplaceFDM(float dt) {
		phi.assign(ni*nj*nk, 0);
		rhs.assign(ni*nj*nk, 0);
		int compute_num = ni*nk;
		tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
		{
			int k = thread_idx / ni;
			int i = thread_idx%ni;
			double boundary_value = boundary_phi(i, k);
			rhs[k*ni*nj + (nj - 1)*ni + i] = 2.0*boundary_value / sqr(m_hx);
		});
		double res; int iters;
		AMGPCGSolvePrebuilt(matrix_fix, rhs, phi, A_L, R_L, P_L, S_L, total_level, 1e-12, 100, res, iters, ni, nj, nk);
		std::cout << "solver converged to " << res << " within " << iters << " iterations" << std::endl;
	}
	//todo 
	void solveLaplaceBEM() {}
	void buildLaplaceSolver()
	{
		int n = ni*nj*nk;
		boundary_phi.resize(ni, nk);
		phi.resize(n);
		matrix.resize(n);
		rhs.resize(n);
		int compute_num = ni*nj*nk;
		int slice = ni*nj;
		tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
		{
			int k = thread_idx / slice;
			int j = (thread_idx%slice) / ni;
			int i = thread_idx%ni;
			int g_index = k*slice + j*ni + i;
			//left
			if (i - 1 >= 0)
			{
				matrix.add_to_element(g_index, g_index, 1 / sqr(m_hx));
				matrix.add_to_element(g_index, g_index - 1, -1 / sqr(m_hx));
			}
			//right
			if (i + 1 < ni)
			{
				matrix.add_to_element(g_index, g_index, 1 / sqr(m_hx));
				matrix.add_to_element(g_index, g_index + 1, -1 / sqr(m_hx));
			}
			//down
			if (j - 1 >= 0)
			{
				matrix.add_to_element(g_index, g_index, 1 / sqr(m_hx));
				matrix.add_to_element(g_index, g_index - ni, -1 / sqr(m_hx));
			}
			//up
			if (j + 1 < nj)
			{
				matrix.add_to_element(g_index, g_index, 1 / sqr(m_hx));
				matrix.add_to_element(g_index, g_index + ni, -1 / sqr(m_hx));
			}
			if (j + 1 == nj)
			{
				matrix.add_to_element(g_index, g_index, 2.0 / sqr(m_hx));
			}
			//front
			if (k - 1 >= 0)
			{
				matrix.add_to_element(g_index, g_index, 1 / sqr(m_hx));
				matrix.add_to_element(g_index, g_index - slice, -1 / sqr(m_hx));
			}
			//back
			if (k + 1 < nk)
			{
				matrix.add_to_element(g_index, g_index, 1 / sqr(m_hx));
				matrix.add_to_element(g_index, g_index + slice, -1 / sqr(m_hx));
			}
		});
		matrix_fix.construct_from_matrix(matrix);

		amgLevelGenerator.generateLevelsGalerkinCoarsening(A_L, R_L, P_L, S_L, total_level, matrix_fix, ni, nj, nk);
	}
	int start;
	int ni, nj, nk;
	float m_hx;
	float m_sea_level;
	SparseMatrixd matrix;
	FixedSparseMatrixd matrix_fix;
	levelGen<double> amgLevelGenerator;
	Array2d boundary_phi;
	std::vector<Vec3f> verts;
	std::vector<Vec4i> faces;
	std::vector<double> rhs;
	std::vector<double> phi;
	std::vector<FixedSparseMatrixd *> A_L;
	std::vector<FixedSparseMatrixd *> R_L;
	std::vector<FixedSparseMatrixd *> P_L;
	std::vector<Vec3i>                S_L;
	int total_level;
};

#endif
