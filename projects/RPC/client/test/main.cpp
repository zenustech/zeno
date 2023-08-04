
#include <common/ping.pb.h>
#include <common/ping.grpc.pb.h>
#include <event/event_bus.pb.h>
#include <event/event_bus.grpc.pb.h>

#include <grpcpp/grpcpp.h>
#include <grpcpp/create_channel.h>

int main(int Argc, char* Argv[]) {
    zeno::event::PutPrimitiveObjectQuery Query;
    zeno::event::PutPrimitiveObjectResponse Response;

    zeno::event::PrimitiveObject Obj;
    zeno::event::AttributeList PosAttr;

    zeno::common::GenericNumeric Numeric;
    zeno::common::Vector3f Data;
    Data.set_x(1.0f);
    Data.set_y(1.5f);
    Data.set_z(2.0f);

    (*Numeric.mutable_vector3fvalue()) = Data;

    zeno::common::GenericNumeric Numeric2;
    Numeric2.CopyFrom(Numeric);

    zeno::common::GenericNumeric Numeric3;
    Numeric3.CopyFrom(Numeric);

    PosAttr.mutable_numericpack()->Add( std::move(Numeric) );
    PosAttr.mutable_numericpack()->Add( std::move(Numeric3) );
    PosAttr.mutable_numericpack()->Add( std::move(Numeric2) );

    (*Obj.mutable_vertices()->mutable_attributes())["pos"] = PosAttr;
    (*Query.mutable_primitives())["123"] = Obj;

    auto Channel = grpc::CreateChannel("localhost:25561", grpc::InsecureChannelCredentials());
    std::unique_ptr<zeno::event::EventBus::Stub> Stub = zeno::event::EventBus::NewStub(Channel);
    grpc::ClientContext Context;
    grpc::Status Status = Stub->PushPrimitiveNotify(&Context, Query, &Response);

    if (!Status.ok()) {
        std::cout << "Failed to make rpc call." << std::endl;
        std::cout << Status.error_message() << std::endl;
//        return 1;
    }

    std::cout << "Status: " << Response.status() << std::endl;

    return 0;
}
