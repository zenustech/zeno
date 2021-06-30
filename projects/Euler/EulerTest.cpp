
#if 0
#include "Libs/AdvectionOp.h"
#include "Libs/Assembler.h"
#include "Libs/BasicOp.h"
#include "Libs/Builder.h"
#include "Libs/CoupledGasSolidSimulator.h"
#include "Libs/EulerGasDenseGrid.h"
#include "Libs/GasAssembler.h"
#include "Libs/IdealGas.h"
#include "Libs/ProjectionOp.h"
#include "Libs/SolidAssembler.h"
#include "Libs/StateDense.h"
#include "Libs/TVDRK.h"
#include "Libs/WENO.h"
#include <omp.h>
#include <zeno/MeshObject.h>
#include <zeno/NumericObject.h>
#include <zeno/VDBGrid.h>
#include <zeno/zeno.h>
namespace zeno {

struct ZenCompressAero : zeno::IObject {
    Bow::EulerGas::FieldHelperDenseDouble3Ptr gas;
};


struct DenseField : zeno::IObject {
    virtual std::string getType() {return std::string();}
};





template <class T>
struct DenseFieldWrapper : DenseField 
{
    DenseFieldWrapper(){ m_grid = new Bow::Field<T>; dx=0; bmin=zeno::vec3f(0); bmax = zeno::vec3f(0);}
    Bow::Field<T>* m_grid;
    float dx;
    zeno::vec3f bmin, bmax;
    int nx, ny, nz;
    std::string spatialType;
    

    virtual std::string getType()
    {
        if(std::is_same<T, double>::value)
        {
        return std::string("FloatGrid");
        }
        else if(std::is_same<T, int>::value)
        {
            return std::string("Int32Grid");
        }
        else if(std::is_same<T, Bow::Vector<double, 3>>::value)
        {
        return std::string("Vec3fGrid");
        }
        else {
        return std::string("");
        }
    }
};


using DenseFloatGrid = VDBGridWrapper<double>;
using DenseIntGrid = VDBGridWrapper<int>;
using DenseFloat3Grid = VDBGridWrapper<Bow::Vector<double, 3>>;


struct DenseFieldToVDB  : zeno::INode {
    virtual void apply() override {

        auto inField = get_input("inDenseField")->as<DenseField>();
        if(inField->getType() == std::string("FloatGrid"))
        {
            auto oField = zeno::IObject::make<VDBFloatGrid>();
            auto transform = openvdb::math::Transform::createLinearTransform(inField->dx);
            if(inField->spatialType == std::string("vertex"))
            {
                transform->postTranslate(openvdb::Vec3d{ -0.5,-0.5,-0.5 }*double(inField->dx));
            }
            
            oField->m_grid->setTransform(transform);

            openvdb::FloatGrid::Accessor writeTo = oField->m_grid->getAccessor();

#omp parallel for
            for(size_t index = 0; index<inField->m_gird->size(); index++)
            {
                int i = index%(ni+4);
                int j = index/(ni+4) % (nj+4);
                int k = index/((ni+4)*(nj+4));


            }
        }

    
    }
};

} // namespace zeno

#endif