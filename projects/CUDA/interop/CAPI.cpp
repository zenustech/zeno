#include <zeno/extra/CAPI.h>
#include "Vector.hpp"
#include "vec.hpp"
using namespace zeno;

#define DEFINE_CREATE_ZS_VEC_SCALAR_TYPE(type)                                                                                       \
    ZENO_CAPI Zeno_Error ZS_CreateObjectZsSmallVec_##type##_scalar(Zeno_Object *objectRet_) ZENO_CAPI_NOEXCEPT                       \
    {                                                                                                                                \
        return PyZeno::lastError.catched([=] { *objectRet_ = PyZeno::lutObject.create(std::make_shared<SmallVecObject>(type{})); }); \
    }
#define DEFINE_CREATE_ZS_VEC_1D_TYPE_DIM(type, dim)                                                                                                \
    ZENO_CAPI Zeno_Error ZS_CreateObjectZsSmallVec_##type##_##dim(Zeno_Object *objectRet_) ZENO_CAPI_NOEXCEPT                                      \
    {                                                                                                                                              \
        return PyZeno::lastError.catched([=] { *objectRet_ = PyZeno::lutObject.create(std::make_shared<SmallVecObject>(zs::vec<type, dim>{})); }); \
    }
#define DEFINE_CREATE_ZS_VEC_1D_TYPE(type) DEFINE_CREATE_ZS_VEC_1D_TYPE_DIM(type, 1) DEFINE_CREATE_ZS_VEC_1D_TYPE_DIM(type, 2) DEFINE_CREATE_ZS_VEC_1D_TYPE_DIM(type, 3) DEFINE_CREATE_ZS_VEC_1D_TYPE_DIM(type, 4)
#define DEFINE_CREATE_ZS_VEC_2D_TYPE_DIMX_DIMY(type, dim_x, dim_y)                                                                                          \
    ZENO_CAPI Zeno_Error ZS_CreateObjectZsSmallVec_##type##_##dim_x##x##dim_y(Zeno_Object *objectRet_) ZENO_CAPI_NOEXCEPT                                   \
    {                                                                                                                                                       \
        return PyZeno::lastError.catched([=] { *objectRet_ = PyZeno::lutObject.create(std::make_shared<SmallVecObject>(zs::vec<type, dim_x, dim_y>{})); }); \
    }
#define DEFINE_CREATE_ZS_VEC_2D_TYPE_DIMX(type, dim_x) DEFINE_CREATE_ZS_VEC_2D_TYPE_DIMX_DIMY(type, dim_x, 1) DEFINE_CREATE_ZS_VEC_2D_TYPE_DIMX_DIMY(type, dim_x, 2) DEFINE_CREATE_ZS_VEC_2D_TYPE_DIMX_DIMY(type, dim_x, 3) DEFINE_CREATE_ZS_VEC_2D_TYPE_DIMX_DIMY(type, dim_x, 4)
#define DEFINE_CREATE_ZS_VEC_2D_TYPE(type) DEFINE_CREATE_ZS_VEC_2D_TYPE_DIMX(type, 1) DEFINE_CREATE_ZS_VEC_2D_TYPE_DIMX(type, 2) DEFINE_CREATE_ZS_VEC_2D_TYPE_DIMX(type, 3) DEFINE_CREATE_ZS_VEC_2D_TYPE_DIMX(type, 4)

#define INSTANTIATE_ZPC_VECTOR_ZENO_APIS(T)                                                                                                                                             \
    ZENO_CAPI Zeno_Error container_obj##__##v##_##T(Zeno_Object *ptr,                                                                                                                   \
                                                    const zs::ZSPmrAllocator<false> *allocator, zs::size_t size) ZENO_CAPI_NOEXCEPT                                                     \
    {                                                                                                                                                                                   \
        return PyZeno::lastError.catched([=] { *ptr = PyZeno::lutObject.create(                                                                                                         \
                                                   std::make_shared<VectorViewLiteObject>(                                                                                              \
                                                       zs::Vector<T, zs::ZSPmrAllocator<false>>{*allocator, size})); }); \
    }                                                                                                                                                                                   \
    ZENO_CAPI Zeno_Error container_obj##__##v##_##T##_virtual(Zeno_Object *ptr,                                                                                                         \
                                                              const zs::ZSPmrAllocator<true> *allocator, zs::size_t size) ZENO_CAPI_NOEXCEPT                                            \
    {                                                                                                                                                                                   \
        return PyZeno::lastError.catched([=] { *ptr = PyZeno::lutObject.create(                                                                                                         \
                                                   std::make_shared<VectorViewLiteObject>(                                                                                              \
                                                       zs::Vector<T, zs::ZSPmrAllocator<true>>{*allocator, size})); });  \
    }

namespace PyZeno
{
    template <class T>
    ZS_DataType getZSdataType()
    {
        ZS_DataType ret;
        if constexpr (std::is_same_v<T, int>)
            ret = ZS_DataType::ZS_DataType_int;
        else if constexpr (std::is_same_v<T, float>)
            ret = ZS_DataType::ZS_DataType_float;
        else if constexpr (std::is_same_v<T, double>)
            ret = ZS_DataType::ZS_DataType_double;
        else
            throw zeno::makeError<zeno::Error>("unknown zs scalar type encountered");
        return ret;
    }
}

extern "C"
{
    DEFINE_CREATE_ZS_VEC_SCALAR_TYPE(int)
    DEFINE_CREATE_ZS_VEC_SCALAR_TYPE(float)
    DEFINE_CREATE_ZS_VEC_SCALAR_TYPE(double)
    DEFINE_CREATE_ZS_VEC_1D_TYPE(int)
    DEFINE_CREATE_ZS_VEC_1D_TYPE(float)
    DEFINE_CREATE_ZS_VEC_1D_TYPE(double)
    DEFINE_CREATE_ZS_VEC_2D_TYPE(int)
    DEFINE_CREATE_ZS_VEC_2D_TYPE(float)
    DEFINE_CREATE_ZS_VEC_2D_TYPE(double)

    ZENO_CAPI Zeno_Error ZS_GetObjectZsVecData(Zeno_Object object_, void **ptrRet_,
                                               size_t *dims_Ret_, size_t *dim_xRet_,
                                               size_t *dim_yRet_,
                                               ZS_DataType *typeRet_) ZENO_CAPI_NOEXCEPT
    {
        return PyZeno::lastError.catched([=]
                                         {
            auto optr = PyZeno::lutObject.access(object_).get();
            auto& vec = dynamic_cast<SmallVecObject *>(optr)->value;
            std::visit(
                [dims_Ret_ = dims_Ret_, dim_xRet_ = dim_xRet_, dim_yRet_ = dim_yRet_, typeRet_ = typeRet_, ptrRet_ = ptrRet_] (auto &vec) {
                    using vec_t = RM_CVREF_T(vec); 
                    if constexpr (zs::is_scalar_v<vec_t>) {
                        *typeRet_ = PyZeno::getZSdataType<vec_t>(); 
                        *dims_Ret_ = 0; 
                        *ptrRet_ = reinterpret_cast<void *>(&vec); 
                    } else {
                        *typeRet_ = PyZeno::getZSdataType<typename vec_t::value_type>(); 
                        *ptrRet_ = reinterpret_cast<void *>(vec.data()); 
                        constexpr auto dim = vec_t::dim; 
                        *dims_Ret_ = dim; 
                        *dim_xRet_ = vec_t::template range_t<0>::value;
                        if constexpr (dim == 2)
                        {
                            *dim_yRet_ = vec_t::template range_t<1>::value;
                        }
                    }
            }, vec); });
    }

    INSTANTIATE_ZPC_VECTOR_ZENO_APIS(int)
    INSTANTIATE_ZPC_VECTOR_ZENO_APIS(float)
    INSTANTIATE_ZPC_VECTOR_ZENO_APIS(double)

    ZENO_CAPI Zeno_Error Zs_GetVectorData(Zeno_Object object, void **ptrRet, int *memsrcRet,
                                          size_t *sizeRet, ZS_DataType *dtypeRet, zs::ProcID *devIdRet,
                                          bool *is_virtual) ZENO_CAPI_NOEXCEPT
    {
        return PyZeno::lastError.catched([=]
                                         {
            auto optr = PyZeno::lutObject.access(object).get();
            auto& vector = dynamic_cast<VectorViewLiteObject *>(optr)->value;
            std::visit([=](auto &vector){
                *ptrRet = reinterpret_cast<void *>(&vector);
                *memsrcRet = static_cast<int>(vector.memspace());
                *sizeRet = vector.size(); 
                *dtypeRet = PyZeno::getZSdataType<typename RM_CVREF_T(vector)::value_type>(); 
                *devIdRet = vector.devid(); 
                *is_virtual = RM_CVREF_T(vector.get_allocator())::is_virtual::value; 
            }, vector); });
    }
}