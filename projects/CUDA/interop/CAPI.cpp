#include <zeno/extra/CAPI.h>
#include "vec.hpp"
using namespace zeno; 

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

extern "C" {
#if 1
    // create_zs_vec_{type} 
    // for testing, use float first 

    // Zeno_CreateObjectInt
#if 0 
    ZENO_CAPI Zeno_Error ZS_CreateObjectZsSmallVecInt(Zeno_Object *objectRet_, const int *value_, size_t dim_x_, size_t dim_y_) ZENO_CAPI_NOEXCEPT {
        return PyZeno::lastError.catched([=] {
            if (dim_x_ == -1) {
                // scalar 
            } else if (dim_y_ == -1) {
                // (dim_x_, )
            } else {
                // (dim_x_, dim_y_)
            }
        }); 
    }
#endif 
    // int, float, double 

    ZENO_CAPI Zeno_Error ZS_GetObjectZsVecData(Zeno_Object object_, void **ptrRet_, size_t *dims_Ret_, size_t *dim_xRet_, size_t *dim_yRet_, ZS_DataType *typeRet_) ZENO_CAPI_NOEXCEPT {
        return PyZeno::lastError.catched([=] {
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
            }, vec); 
        }); 
    }
#endif 
}

