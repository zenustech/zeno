#ifndef IDEAL_GAS_H
#define IDEAL_GAS_H

#include <Bow/Macros.h>
#include <Bow/Types.h>
#include <Bow/Utils/Logging.h>

namespace Bow {
namespace ConstitutiveModel {
namespace IdealGas {

template <class Scalar>
BOW_INLINE Scalar get_pressure_from_int_energy(const Scalar gamma, const Scalar internelEnergy)
{
    Scalar P = (gamma - 1) * internelEnergy;
    return P;
};

template <class Scalar>
BOW_INLINE Scalar get_int_energy_from_pressure(const Scalar gamma, const Scalar P)
{
    Scalar internelEnergy = P / (gamma - 1);
    return internelEnergy;
};

template <class Scalar>
BOW_INLINE Scalar get_pressure_from_T(const Scalar rho, const Scalar R, const Scalar T)
{
    Scalar P = rho * R * T;
    return P;
};

template <class Scalar>
BOW_INLINE Scalar get_T_from_int_energy(const Scalar gamma, const Scalar internelEnergy, const Scalar rho, const Scalar R)
{
    Scalar T = (gamma - 1) * internelEnergy / rho / R;
    return T;
};

// conversion from conserved variables and primitives
// q: rho tot_e mv etc
// p: rho P v etc
template <class Scalar, int dim>
Array<Scalar, dim + 2, 1> getPFromQ(const Array<Scalar, dim + 2, 1>& q, Scalar gamma)
{
    BOW_ASSERT_INFO(q(0) > 0, "inputted negative density");
    // BOW_ASSERT_INFO(q(1) > 0, "inputted negative total energy");
    Scalar int_e = q(1) - 0.5 * (q.template tail<dim>() * q.template tail<dim>()).sum() / q(0);
    // BOW_ASSERT_INFO(int_e > 0, "inputted negative internal energy");
    Array<Scalar, dim + 2, 1> p = q;
    p(1) = get_pressure_from_int_energy(gamma, int_e);
    p.template tail<dim>() /= p(0);
    return p;
};

template <class Scalar, int dim>
Array<Scalar, dim + 2, 1> getQFromP(const Array<Scalar, dim + 2, 1>& p, Scalar gamma)
{
    BOW_ASSERT_INFO(p(0) > 0, "inputted negative density");
    // BOW_ASSERT_INFO(p(1) > 0, "inputted negative pressure");
    Array<Scalar, dim + 2, 1> q = p;
    q(1) = get_int_energy_from_pressure(gamma, q(1)) + 0.5 * q(0) * (q.template tail<dim>() * q.template tail<dim>()).sum();
    q.template tail<dim>() *= q(0);
    return q;
};

}
}
} // namespace Bow::ConstitutiveModel::IdealGas

#endif