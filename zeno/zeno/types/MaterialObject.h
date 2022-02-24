#pragma once

#include <zeno/core/IObject.h>
#include <string>

namespace zeno
{

    struct MaterialObject
        : IObjectClone<MaterialObject>
    {
        std::string vert;
        std::string frag;

        const std::string &getVert() const
        {
            return vert;
        }

        std::string &getVert()
        {
            return vert;
        }

        void setVert(const std::string &vert_)
        {
            vert = vert_;
        }

        const std::string &getFrag() const
        {
            return frag;
        }

        std::string &getFrag()
        {
            return frag;
        }

        void setFrag(const std::string &frag_)
        {
            frag = frag_;
        }

    }; // struct Material

} // namespace zeno
