
#include "event.h"
#include <zeno/extra/EventCallbacks.h>
#include <zeno/zeno.h>
#include <cassert>

::grpc::Status EventBusService::TriggerEvent(::grpc::ServerContext *context, const ::zeno::event::TriggerEventQuery *request, ::zeno::event::TriggerEventResponse *response) {
    std::string requestEventName = zeno::event::EventType_Name(request->eventtype());

    zeno::getSession().eventCallbacks->triggerEvent(requestEventName);

    response->set_status(zeno::common::SUCCESS);

    return grpc::Status::OK;
}

::grpc::Status EventBusService::PushPrimitiveNotify(::grpc::ServerContext *context, const ::zeno::event::PutPrimitiveObjectQuery *request, ::zeno::event::PutPrimitiveObjectResponse *response) {
    try {
        for (const auto& InPrimitive : request->primitives()) {
            std::shared_ptr<zeno::PrimitiveObject> Primitive = FromProtobuf(&InPrimitive.second);
            zeno::getSession().eventCallbacks->triggerEvent("rpcIncomingPrimitive", Primitive );
        }
    } catch (const std::runtime_error& Err) {}

    return grpc::Status::CANCELLED;
}

inline constexpr bool CheckIsSameType(int32_t &Lhs, const int32_t Rhs) {
    if (Lhs == -1) {
        Lhs = Rhs;
        return true;
    } else
        return Rhs == Lhs;
}

/**
 * @class InsertHelper
 * @brief A class that provides a helper for inserting values into a container.
 *
 * This class allows for inserting and appending values of any type into a specified container.
 * It also provides the ability to associate an attribute name and perform type checking on the value being inserted.
 * The type check result is stored in an integer.
 *
 * Usage:
 * 1. Create an instance of InsertHelper with the desired value, container, attribute name, and type check reference.
 * 2. Use the Insert() member function to insert the value into the container.
 *
 * Example:
 * ```
 * std::vector<int> vec;
 * std::string attr = "count";
 * int32_t typeCheckResult = 0;
 *
 * InsertHelper ih(42, vec, attr, typeCheckResult);
 * ih.Insert();
 * ```
 */
template <typename ValueType, typename ContainerType>
void InsertHelper(const ValueType& Value, ContainerType& Container, const std::string& AttrName, int32_t& InTypeCheck) {
    const size_t ValueTypeId = typeid(ValueType).hash_code();
    if (!CheckIsSameType(InTypeCheck, ValueTypeId)) {
        throw std::runtime_error("It must be same types in an array");
    }
    if (Container.has_attr(AttrName)) Container.add_attr<ValueType>(AttrName);
    Container.attr<ValueType>(AttrName).push_back(Value);
}

/**
 * @class UnpackHelper
 * @brief This class provides a helper function to unpack attributes from a list and store them in a container.
 *
 * The UnpackHelper class is used to extract attributes from an attribute list and store them in a container.
 * This helper class can be used when attributes need to be unpacked and processed individually.
 */
template <typename AttrListType, typename ContainerType>
void UnpackHelper(AttrListType& AttrList, ContainerType& Conatiner) {
    for (const auto &Attr: AttrList) {
        const std::string &AttrName = Attr.first;
        if (Attr.second.numericpack_size() > Conatiner.size()) Conatiner.reserve(Attr.second.numericpack_size());
        int32_t Type = 0;
        for (const auto &Data: Attr.second.numericpack()) {
            if (Data.has_floatvalue()) {
                InsertHelper(Data.floatvalue(), Conatiner, AttrName, Type);
            } else if (Data.has_int32value()) {
                InsertHelper(Data.int32value(), Conatiner, AttrName, Type);
            } else if (Data.has_vector2fvalue()) {
                InsertHelper(zeno::vec2f { Data.vector2fvalue().x(), Data.vector2fvalue().y() }, Conatiner, AttrName, Type);
            } else if (Data.has_vector3fvalue()) {
                InsertHelper(zeno::vec3f { Data.vector3fvalue().x(), Data.vector3fvalue().y(), Data.vector3fvalue().z() }, Conatiner, AttrName, Type);
            } else if (Data.has_point2ivalue()) {
                InsertHelper(zeno::vec2i { Data.point2ivalue().x(), Data.point2ivalue().y() }, Conatiner, AttrName, Type);
            } else if (Data.has_point3ivalue()) {
                InsertHelper(zeno::vec3i { Data.point3ivalue().x(), Data.point3ivalue().y(), Data.point3ivalue().z() }, Conatiner, AttrName, Type);
            } else if (Data.has_point4ivalue()) {
                InsertHelper(zeno::vec4i { Data.point4ivalue().x(), Data.point4ivalue().y(), Data.point4ivalue().z(), Data.point4ivalue().w() }, Conatiner, AttrName, Type);
            }
        }
    }
}

/**
 * @brief Convert a Protobuf PrimitiveObject to native format.
 *
 * This function takes a Pointer to a Protobuf PrimitiveObject and converts it to a native format.
 *
 * @param Object Pointer to a Protobuf PrimitiveObject to convert.
 *
 * @return Pointer to a converted native format object.
 */
std::shared_ptr<zeno::PrimitiveObject> FromProtobuf(const zeno::event::PrimitiveObject *Object) {
    assert(nullptr != Object);

    std::shared_ptr<zeno::PrimitiveObject> NewPrimitive = std::make_shared<zeno::PrimitiveObject>();

    
    auto &Vertices = NewPrimitive->verts;
    auto &Points = NewPrimitive->points;
    auto &Lines = NewPrimitive->lines;
    auto &Triangles = NewPrimitive->tris;
    auto &Quadratics = NewPrimitive->quads;
    auto &Loops = NewPrimitive->loops;
    auto &Polys = NewPrimitive->polys;
    auto &Edges = NewPrimitive->edges;
    auto &TextureCoords = NewPrimitive->uvs;

    using namespace zeno::event;

    UnpackHelper(Object->vertices().attributes(), Vertices);
    UnpackHelper(Object->points().attributes(), Points);
    UnpackHelper(Object->lines().attributes(), Lines);
    UnpackHelper(Object->triangles().attributes(), Triangles);
    UnpackHelper(Object->quadratics().attributes(), Quadratics);
    UnpackHelper(Object->loops().attributes(), Loops);
    UnpackHelper(Object->polys().attributes(), Polys);
    UnpackHelper(Object->edges().attributes(), Edges);
    UnpackHelper(Object->texturecoords().attributes(), TextureCoords);

    return NewPrimitive;
}

namespace {
    [[maybe_unused]] StaticServiceRegister<EventBusService> AutoRegisterForEventBusService{};
}
