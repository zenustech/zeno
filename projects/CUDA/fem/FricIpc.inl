namespace zeno {

void CodimStepping::IPCSystem::computeFrictionBarrierGradientAndHessian(
    zs::CudaExecutionPolicy &pol, const zs::SmallString &gTag,
    bool includeHessian) {
  using namespace zs;
  constexpr auto space = execspace_e::cuda;
  T activeGap2 = dHat * dHat + (T)2.0 * xi * dHat;
  using Vec12View = zs::vec_view<T, zs::integer_seq<int, 12>>;
  using Vec9View = zs::vec_view<T, zs::integer_seq<int, 9>>;
  using Vec6View = zs::vec_view<T, zs::integer_seq<int, 6>>;
  auto numFPP = nFPP.getVal();
  pol(range(numFPP),
      [vtemp = proxy<space>({}, vtemp), fricPP = proxy<space>({}, fricPP),
       FPP = proxy<space>(FPP), gTag, epsvh = epsv * dt, fricMu = fricMu,
       projectDBC = projectDBC, includeHessian] __device__(int fppi) mutable {
        auto fpp = FPP[fppi];
        auto p0 = vtemp.pack<3>("xn", fpp[0]) - vtemp.pack<3>("xhat", fpp[0]);
        auto p1 = vtemp.pack<3>("xn", fpp[1]) - vtemp.pack<3>("xhat", fpp[1]);
#if 1
        auto basis = fricPP.pack<3, 2>("basis", fppi);
        auto fn = fricPP("fn", fppi);
        auto relDX3D = point_point_rel_dx(p0, p1);
        auto relDX = basis.transpose() * relDX3D;
        auto relDXNorm2 = relDX.l2NormSqr();
        auto relDXNorm = zs::sqrt(relDXNorm2);
        auto f1_div_relDXNorm = zs::f1_SF_div_rel_dx_norm(relDXNorm2, epsvh);
        relDX *= f1_div_relDXNorm * fricMu * fn;
        auto TTTDX = -point_point_rel_dx_tan_to_mesh(relDX, basis);
        // gradient
        for (int d = 0; d != 3; ++d) {
          atomic_add(exec_cuda, &vtemp(gTag, d, fpp[0]), TTTDX(0, d));
          atomic_add(exec_cuda, &vtemp(gTag, d, fpp[1]), TTTDX(1, d));
        }
        // hessian
        if (!includeHessian)
          return;
        relDX = basis.transpose() * relDX3D;
        auto TT = point_point_TT(basis); // 2x6
        auto f2_term = f2_SF_term(relDXNorm2, epsvh);
        using HessT = zs::vec<T, 6, 6>;
        auto hess = HessT::zeros();
        if (relDXNorm2 >= epsvh * epsvh) {
          zs::vec<T, 2> ubar{-relDX[1], relDX[0]};
          hess = dyadic_prod(
              TT.transpose() *
                  ((fricMu * fn * f1_div_relDXNorm / relDXNorm2) * ubar),
              ubar * TT);
        } else {
          if (relDXNorm == 0) {
            if (fn > 0)
              hess = fricMu * fn * f1_div_relDXNorm * TT.transpose() * TT;
            // or ignored
          } else {
            auto innerMtr = dyadic_prod((f2_term / relDXNorm) * relDX, relDX);
            innerMtr(0, 0) += f1_div_relDXNorm;
            innerMtr(1, 1) += f1_div_relDXNorm;
            innerMtr *= fricMu * fn;
            //
            make_pd(innerMtr);
            hess = TT.transpose() * innerMtr * TT;
          }
        }
#else
#endif
        // rotate and project
        mat3 BCbasis[2];
        int BCorder[2];
        int BCfixed[2];
        for (int i = 0; i != 2; ++i) {
          BCbasis[i] = vtemp.pack<3, 3>("BCbasis", fpp[i]);
          BCorder[i] = vtemp("BCorder", fpp[i]);
          BCfixed[i] = vtemp("BCfixed", fpp[i]);
        }
        rotate_hessian(hess, BCbasis, BCorder, BCfixed, projectDBC);
        // pp[0], pp[1]
        fricPP.tuple<36>("H", fppi) = hess;
        /// construct P
        for (int vi = 0; vi != 2; ++vi) {
          for (int i = 0; i != 3; ++i)
            for (int j = 0; j != 3; ++j) {
              atomic_add(exec_cuda, &vtemp("P", i * 3 + j, fpp[vi]),
                         hess(vi * 3 + i, vi * 3 + j));
            }
        }
      });
  auto numFPE = nFPE.getVal();
  pol(range(numFPE),
      [vtemp = proxy<space>({}, vtemp), fricPE = proxy<space>({}, fricPE),
       FPE = proxy<space>(FPE), gTag, epsvh = epsv * dt, fricMu = fricMu,
       projectDBC = projectDBC, includeHessian] __device__(int fpei) mutable {
        auto fpe = FPE[fpei];
        auto p = vtemp.pack<3>("xn", fpe[0]) - vtemp.pack<3>("xhat", fpe[0]);
        auto e0 = vtemp.pack<3>("xn", fpe[1]) - vtemp.pack<3>("xhat", fpe[1]);
        auto e1 = vtemp.pack<3>("xn", fpe[2]) - vtemp.pack<3>("xhat", fpe[2]);
#if 1
        auto basis = fricPE.pack<3, 2>("basis", fpei);
        auto fn = fricPE("fn", fpei);
        auto yita = fricPE("yita", fpei);
        auto relDX3D = point_edge_rel_dx(p, e0, e1, yita);
        auto relDX = basis.transpose() * relDX3D;
        auto relDXNorm2 = relDX.l2NormSqr();
        auto relDXNorm = zs::sqrt(relDXNorm2);
        auto f1_div_relDXNorm = zs::f1_SF_div_rel_dx_norm(relDXNorm2, epsvh);
        relDX *= f1_div_relDXNorm * fricMu * fn;
        auto TTTDX = -point_edge_rel_dx_tan_to_mesh(relDX, basis, yita);
        // gradient
        for (int d = 0; d != 3; ++d) {
          atomic_add(exec_cuda, &vtemp(gTag, d, fpe[0]), TTTDX(0, d));
          atomic_add(exec_cuda, &vtemp(gTag, d, fpe[1]), TTTDX(1, d));
          atomic_add(exec_cuda, &vtemp(gTag, d, fpe[2]), TTTDX(2, d));
        }
        // hessian
        if (!includeHessian)
          return;
        relDX = basis.transpose() * relDX3D;
        auto TT = point_edge_TT(basis, yita); // 2x9
        auto f2_term = f2_SF_term(relDXNorm2, epsvh);
        using HessT = zs::vec<T, 9, 9>;
        auto hess = HessT::zeros();
        if (relDXNorm2 >= epsvh * epsvh) {
          zs::vec<T, 2> ubar{-relDX[1], relDX[0]};
          hess = dyadic_prod(
              TT.transpose() *
                  ((fricMu * fn * f1_div_relDXNorm / relDXNorm2) * ubar),
              ubar * TT);
        } else {
          if (relDXNorm == 0) {
            if (fn > 0)
              hess = fricMu * fn * f1_div_relDXNorm * TT.transpose() * TT;
            // or ignored
          } else {
            auto innerMtr = dyadic_prod((f2_term / relDXNorm) * relDX, relDX);
            innerMtr(0, 0) += f1_div_relDXNorm;
            innerMtr(1, 1) += f1_div_relDXNorm;
            innerMtr *= fricMu * fn;
            //
            make_pd(innerMtr);
            hess = TT.transpose() * innerMtr * TT;
          }
        }
#else
#endif
        // rotate and project
        mat3 BCbasis[3];
        int BCorder[3];
        int BCfixed[3];
        for (int i = 0; i != 3; ++i) {
          BCbasis[i] = vtemp.pack<3, 3>("BCbasis", fpe[i]);
          BCorder[i] = vtemp("BCorder", fpe[i]);
          BCfixed[i] = vtemp("BCfixed", fpe[i]);
        }
        rotate_hessian(hess, BCbasis, BCorder, BCfixed, projectDBC);
        // pe[0], pe[1], pe[2]
        fricPE.tuple<81>("H", fpei) = hess;
        /// construct P
        for (int vi = 0; vi != 3; ++vi) {
          for (int i = 0; i != 3; ++i)
            for (int j = 0; j != 3; ++j) {
              atomic_add(exec_cuda, &vtemp("P", i * 3 + j, fpe[vi]),
                         hess(vi * 3 + i, vi * 3 + j));
            }
        }
      });
  auto numFPT = nFPT.getVal();
  pol(range(numFPT),
      [vtemp = proxy<space>({}, vtemp), fricPT = proxy<space>({}, fricPT),
       FPT = proxy<space>(FPT), gTag, epsvh = epsv * dt, fricMu = fricMu,
       projectDBC = projectDBC, includeHessian] __device__(int fpti) mutable {
        auto fpt = FPT[fpti];
        auto p = vtemp.pack<3>("xn", fpt[0]) - vtemp.pack<3>("xhat", fpt[0]);
        auto v0 = vtemp.pack<3>("xn", fpt[1]) - vtemp.pack<3>("xhat", fpt[1]);
        auto v1 = vtemp.pack<3>("xn", fpt[2]) - vtemp.pack<3>("xhat", fpt[2]);
        auto v2 = vtemp.pack<3>("xn", fpt[3]) - vtemp.pack<3>("xhat", fpt[3]);
#if 1
        auto basis = fricPT.pack<3, 2>("basis", fpti);
        auto fn = fricPT("fn", fpti);
        auto betas = fricPT.pack<2>("beta", fpti);
        auto relDX3D = point_triangle_rel_dx(p, v0, v1, v2, betas[0], betas[1]);
        auto relDX = basis.transpose() * relDX3D;
        auto relDXNorm2 = relDX.l2NormSqr();
        auto relDXNorm = zs::sqrt(relDXNorm2);
        auto f1_div_relDXNorm = zs::f1_SF_div_rel_dx_norm(relDXNorm2, epsvh);
        relDX *= f1_div_relDXNorm * fricMu * fn;
        auto TTTDX = -point_triangle_rel_dx_tan_to_mesh(relDX, basis, betas[0],
                                                        betas[1]);
        // gradient
        for (int d = 0; d != 3; ++d) {
          atomic_add(exec_cuda, &vtemp(gTag, d, fpt[0]), TTTDX(0, d));
          atomic_add(exec_cuda, &vtemp(gTag, d, fpt[1]), TTTDX(1, d));
          atomic_add(exec_cuda, &vtemp(gTag, d, fpt[2]), TTTDX(2, d));
          atomic_add(exec_cuda, &vtemp(gTag, d, fpt[3]), TTTDX(3, d));
        }
        // hessian
        if (!includeHessian)
          return;
        relDX = basis.transpose() * relDX3D;
        auto TT = point_triangle_TT(basis, betas[0], betas[1]); // 2x12
        auto f2_term = f2_SF_term(relDXNorm2, epsvh);
        using HessT = zs::vec<T, 12, 12>;
        auto hess = HessT::zeros();
        if (relDXNorm2 >= epsvh * epsvh) {
          zs::vec<T, 2> ubar{-relDX[1], relDX[0]};
          hess = dyadic_prod(
              TT.transpose() *
                  ((fricMu * fn * f1_div_relDXNorm / relDXNorm2) * ubar),
              ubar * TT);
        } else {
          if (relDXNorm == 0) {
            if (fn > 0)
              hess = fricMu * fn * f1_div_relDXNorm * TT.transpose() * TT;
            // or ignored
          } else {
            auto innerMtr = dyadic_prod((f2_term / relDXNorm) * relDX, relDX);
            innerMtr(0, 0) += f1_div_relDXNorm;
            innerMtr(1, 1) += f1_div_relDXNorm;
            innerMtr *= fricMu * fn;
            //
            make_pd(innerMtr);
            hess = TT.transpose() * innerMtr * TT;
          }
        }
#else
#endif
        // rotate and project
        mat3 BCbasis[4];
        int BCorder[4];
        int BCfixed[4];
        for (int i = 0; i != 4; ++i) {
          BCbasis[i] = vtemp.pack<3, 3>("BCbasis", fpt[i]);
          BCorder[i] = vtemp("BCorder", fpt[i]);
          BCfixed[i] = vtemp("BCfixed", fpt[i]);
        }
        rotate_hessian(hess, BCbasis, BCorder, BCfixed, projectDBC);
        // pt[0], pt[1], pt[2], pt[3]
        fricPT.tuple<144>("H", fpti) = hess;
        /// construct P
        for (int vi = 0; vi != 4; ++vi) {
          for (int i = 0; i != 3; ++i)
            for (int j = 0; j != 3; ++j) {
              atomic_add(exec_cuda, &vtemp("P", i * 3 + j, fpt[vi]),
                         hess(vi * 3 + i, vi * 3 + j));
            }
        }
      });
  auto numFEE = nFEE.getVal();
  pol(range(numFEE),
      [vtemp = proxy<space>({}, vtemp), fricEE = proxy<space>({}, fricEE),
       FEE = proxy<space>(FEE), gTag, epsvh = epsv * dt, fricMu = fricMu,
       projectDBC = projectDBC, includeHessian] __device__(int feei) mutable {
        auto fee = FEE[feei];
        auto e0 = vtemp.pack<3>("xn", fee[0]) - vtemp.pack<3>("xhat", fee[0]);
        auto e1 = vtemp.pack<3>("xn", fee[1]) - vtemp.pack<3>("xhat", fee[1]);
        auto e2 = vtemp.pack<3>("xn", fee[2]) - vtemp.pack<3>("xhat", fee[2]);
        auto e3 = vtemp.pack<3>("xn", fee[3]) - vtemp.pack<3>("xhat", fee[3]);
#if 1
        auto basis = fricEE.pack<3, 2>("basis", feei);
        auto fn = fricEE("fn", feei);
        auto gammas = fricEE.pack<2>("gamma", feei);
        auto relDX3D = edge_edge_rel_dx(e0, e1, e2, e3, gammas[0], gammas[1]);
        auto relDX = basis.transpose() * relDX3D;
        auto relDXNorm2 = relDX.l2NormSqr();
        auto relDXNorm = zs::sqrt(relDXNorm2);
        auto f1_div_relDXNorm = zs::f1_SF_div_rel_dx_norm(relDXNorm2, epsvh);
        relDX *= f1_div_relDXNorm * fricMu * fn;
        auto TTTDX =
            -edge_edge_rel_dx_tan_to_mesh(relDX, basis, gammas[0], gammas[1]);
        // gradient
        for (int d = 0; d != 3; ++d) {
          atomic_add(exec_cuda, &vtemp(gTag, d, fee[0]), TTTDX(0, d));
          atomic_add(exec_cuda, &vtemp(gTag, d, fee[1]), TTTDX(1, d));
          atomic_add(exec_cuda, &vtemp(gTag, d, fee[2]), TTTDX(2, d));
          atomic_add(exec_cuda, &vtemp(gTag, d, fee[3]), TTTDX(3, d));
        }
        // hessian
        if (!includeHessian)
          return;
        relDX = basis.transpose() * relDX3D;
        auto TT = edge_edge_TT(basis, gammas[0], gammas[1]); // 2x12
        auto f2_term = f2_SF_term(relDXNorm2, epsvh);
        using HessT = zs::vec<T, 12, 12>;
        auto hess = HessT::zeros();
        if (relDXNorm2 >= epsvh * epsvh) {
          zs::vec<T, 2> ubar{-relDX[1], relDX[0]};
          hess = dyadic_prod(
              TT.transpose() *
                  ((fricMu * fn * f1_div_relDXNorm / relDXNorm2) * ubar),
              ubar * TT);
        } else {
          if (relDXNorm == 0) {
            if (fn > 0)
              hess = fricMu * fn * f1_div_relDXNorm * TT.transpose() * TT;
            // or ignored
          } else {
            auto innerMtr = dyadic_prod((f2_term / relDXNorm) * relDX, relDX);
            innerMtr(0, 0) += f1_div_relDXNorm;
            innerMtr(1, 1) += f1_div_relDXNorm;
            innerMtr *= fricMu * fn;
            //
            make_pd(innerMtr);
            hess = TT.transpose() * innerMtr * TT;
          }
        }
#else
#endif
        // rotate and project
        mat3 BCbasis[4];
        int BCorder[4];
        int BCfixed[4];
        for (int i = 0; i != 4; ++i) {
          BCbasis[i] = vtemp.pack<3, 3>("BCbasis", fee[i]);
          BCorder[i] = vtemp("BCorder", fee[i]);
          BCfixed[i] = vtemp("BCfixed", fee[i]);
        }
        rotate_hessian(hess, BCbasis, BCorder, BCfixed, projectDBC);
        // ee[0], ee[1], ee[2], ee[3]
        fricEE.tuple<144>("H", feei) = hess;
        /// construct P
        for (int vi = 0; vi != 4; ++vi) {
          for (int i = 0; i != 3; ++i)
            for (int j = 0; j != 3; ++j) {
              atomic_add(exec_cuda, &vtemp("P", i * 3 + j, fee[vi]),
                         hess(vi * 3 + i, vi * 3 + j));
            }
        }
      });
  return;
}

} // namespace zeno