#include "UnifiedSolver.cuh"

namespace zeno {

template <typename T>
void UnifiedIPCSystem::SystemHessian<T>::initializePreconditioner(zs::CudaExecutionPolicy &pol,
                                                                  UnifiedIPCSystem &system) {
    // compute nLevels, nTotalEntries
    ;
    nTotalEntries = buildPreconditioner(pol, system);
    //
    Pm = zs::Vector<zs::vec<T, 96, 96>>{spmat.get_allocator(), (std::size_t)nTotalEntries};
    Rm = zs::Vector<zs::vec<T, 3>>{spmat.get_allocator(), (std::size_t)nTotalEntries};
    Zm = zs::Vector<zs::vec<T, 3>>{spmat.get_allocator(), (std::size_t)nTotalEntries};
    traversed = spmat._ptrs;
}

template <typename T>
int UnifiedIPCSystem::SystemHessian<T>::buildPreconditioner(zs::CudaExecutionPolicy &pol, UnifiedIPCSystem &system) {
    // Pm
    return 0;
}

template <typename T>
void UnifiedIPCSystem::SystemHessian<T>::precondition(zs::CudaExecutionPolicy &pol, dtiles_t &vtemp,
                                                      const zs::SmallString srcTag, const zs::SmallString dstTag) {
    // Pm, Rm, Zm
}

///
/// instantiations
///
template void
UnifiedIPCSystem::SystemHessian<typename UnifiedIPCSystem::T>::initializePreconditioner(zs::CudaExecutionPolicy &pol,
                                                                                        UnifiedIPCSystem &system);
template int
UnifiedIPCSystem::SystemHessian<typename UnifiedIPCSystem::T>::buildPreconditioner(zs::CudaExecutionPolicy &pol,
                                                                                   UnifiedIPCSystem &system);

template void UnifiedIPCSystem::SystemHessian<typename UnifiedIPCSystem::T>::precondition(
    zs::CudaExecutionPolicy &pol, typename UnifiedIPCSystem::dtiles_t &vtemp, const zs::SmallString srcTag,
    const zs::SmallString dstTag);

} // namespace zeno