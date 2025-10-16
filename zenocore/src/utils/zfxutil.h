#ifndef __ZFX_UTIL_H__
#define __ZFX_UTIL_H__

#include <zeno/core/data.h>
#include <zeno/formula/syntax_tree.h>
#include <zeno/types/AttributeVector.h>

namespace zeno
{
    namespace zfx
    {
        zeno::reflect::Any zfxvarToAny(const zfxvariant& var);
        AttrVar zfxvarToAttrvar(const zfxvariant& var);
        std::vector<zfxvariant> extractAttrValue(zeno::reflect::Any anyval, int size);
        AttrVar convertToAttrVar(const ZfxVector& zfxvec);
        AttrVar getInitValueFromVariant(const ZfxVector& zfxvec);
        std::vector<glm::vec3> zvec3toglm(const std::vector<zeno::vec3f>& vec);
        void setAttrValue(std::string attrname, const std::string& channel, const ZfxVariable& var, operatorVals opVal, ZfxElemFilter& filter, ZfxContext* pContext);
        ZfxVariable getAttrValue(const std::string& attrname, ZfxContext* pContext, char channel = 0);
        ZfxVariable initVarFromZvar(const zfxvariant& var);
        zfxvariant getZfxVarElement(const ZfxVector& vec, int idx);
        NodeImpl* getNodeAndParamFromRefString(
            const std::string& ref, 
            ZfxContext* pContext, 
            std::string& paramName,
            std::string& paramPath
            );
    }
}



#endif