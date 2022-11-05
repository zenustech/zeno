#include <zenovis/StageManager.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/UserData.h>

#include <pxr/usd/usdGeom/mesh.h>

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <filesystem>

PXR_NAMESPACE_USING_DIRECTIVE

namespace zenovis {

class con_handler : public boost::enable_shared_from_this<con_handler>
{
  private:
    boost::asio::ip::tcp::socket sock;
    std::string message="Hello From Server!";
    enum { max_length = 1024 };
    char data[max_length];

  public:
    typedef boost::shared_ptr<con_handler> pointer;
    con_handler(boost::asio::io_context& io_context): sock(io_context){}
    // creating the pointer
    static pointer create(boost::asio::io_context& io_context)
    {
        return pointer(new con_handler(io_context));
    }
    //socket creation
    boost::asio::ip::tcp::socket& socket()
    {
        return sock;
    }

    void start()
    {
        sock.async_read_some(
            boost::asio::buffer(data, max_length),
            boost::bind(&con_handler::handle_read,
                        shared_from_this(),
                        boost::asio::placeholders::error,
                        boost::asio::placeholders::bytes_transferred));

        sock.async_write_some(
            boost::asio::buffer(message, max_length),
            boost::bind(&con_handler::handle_write,
                        shared_from_this(),
                        boost::asio::placeholders::error,
                        boost::asio::placeholders::bytes_transferred));
    }

    void handle_read(const boost::system::error_code& err, size_t bytes_transferred)
    {
        if (!err) {
            std::cout << data << std::endl;
        } else {
            std::cerr << "error: " << err.message() << std::endl;
            sock.close();
        }
    }
    void handle_write(const boost::system::error_code& err, size_t bytes_transferred)
    {
        if (!err) {
            std::cout << "Server sent Hello message!"<< std::endl;
        } else {
            std::cerr << "error: " << err.message() << std::endl;
            sock.close();
        }
    }
};

class Server
{
  private:
    boost::asio::ip::tcp::acceptor acceptor_;
    void start_accept()
    {
        // socket  // acceptor_.get_io_service()
        auto& a = acceptor_.get_executor().context();
        con_handler::pointer connection = con_handler::create(static_cast<boost::asio::io_context &>(a));

        // asynchronous accept operation and wait for a new connection.
        acceptor_.async_accept(connection->socket(),
                               boost::bind(&Server::handle_accept, this, connection,
                                           boost::asio::placeholders::error));
    }
  public:
    //constructor for accepting connection from client
    Server(boost::asio::io_context& io_context): acceptor_(io_context, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 1234))
    {
        start_accept();
    }
    void handle_accept(con_handler::pointer connection, const boost::system::error_code& err)
    {
        if (!err) {
            connection->start();
        }
        start_accept();
    }
};

void client(){
    boost::asio::io_context io_context;
    //socket creation
    boost::asio::ip::tcp::socket socket(io_context);
    //connection
    socket.connect( boost::asio::ip::tcp::endpoint( boost::asio::ip::address::from_string("127.0.0.1"), 1234 ));
    // request/message from client
    const std::string msg = "Hello from Client!\n";
    boost::system::error_code error;
    boost::asio::write( socket, boost::asio::buffer(msg), error );
    if( !error ) {
        std::cout << "Client sent hello message!" << std::endl;
    }
    else {
        std::cout << "send failed: " << error.message() << std::endl;
    }
    // getting response from server
    boost::asio::streambuf receive_buffer;
    boost::asio::read(socket, receive_buffer, boost::asio::transfer_all(), error);
    if( error && error != boost::asio::error::eof ) {
        std::cout << "receive failed: " << error.message() << std::endl;
    }
    else {
        const char* data = boost::asio::buffer_cast<const char*>(receive_buffer.data());
        std::cout << data << std::endl;
    }
}

void UpdateTimer(std::function<void(void)> func, unsigned int interval)
{
    std::thread([func, interval]() {
        while (true)
        {
            func();
            std::this_thread::sleep_for(std::chrono::milliseconds(interval));
        }
    }).detach();
}

std::string Execute( std::string cmd )
{
    std::string file_name = "result.txt" ;
    std::system( ( cmd + " > " + file_name ).c_str() ) ;
    std::ifstream file(file_name) ;
    return { std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() } ;
}

TF_DEFINE_PRIVATE_TOKENS(
    _tokens,
    // light
    (UsdLuxDiskLight)
    (UsdLuxCylinderLight)
    (UsdLuxDistantLight)
    (UsdLuxDomeLight)
    (UsdLuxRectLight)
    (UsdLuxSphereLight)
    // prim
    (UsdGeomMesh)
    (UsdGeomCurves)
    (UsdGeomPoints)
    (UsdVolume)
    (UsdGeomCamera)
);

StageManager::StageManager(){
    zeno::log_info("USD: StageManager Constructed");

    UpdateTimer(std::bind(&StageManager::update, this), 500);

    //cStagePtr = UsdStage::CreateInMemory();
    cStagePtr = UsdStage::CreateNew("projects/USD/usd.usda");
    sStagePtr = UsdStage::Open(confInfo.cPath + "/" + confInfo.cRoot);

    std::thread([&]() {
        try
        {
            std::cout << "USD: StageManager Server Running.\n";
            boost::asio::io_service io_service;
            Server server(io_service);
            io_service.run();
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }).detach();

    zeno::log_info("USD: StageManager Initialized");
};
StageManager::~StageManager(){
    zeno::log_info("USD: StageManager Destroyed");
};

int zenovis::StageManager::_UsdGeomMesh(const PrimInfo& primInfo){
    auto zenoPrim = dynamic_cast<zeno::PrimitiveObject *>(primInfo.iObject.get());
    std::filesystem::path p(primInfo.pPath); std::string nodeName = p.filename().string();
    zeno::log_info("USD: GeomMesh {}", nodeName);
    SdfPath objPath(primInfo.pPath);
    _CreateUSDHierarchy(objPath);

    UsdGeomMesh mesh = UsdGeomMesh::Define(cStagePtr, objPath);
    UsdPrim usdPrim = mesh.GetPrim();

    pxr::VtArray<pxr::GfVec3f> Points;
    pxr::VtArray<pxr::GfVec3f> DisplayColor;
    pxr::VtArray<int> FaceVertexCounts;
    pxr::VtArray<int> FaceVertexIndices;

    // Points
    for(auto const& vert:zenoPrim->verts)
        Points.emplace_back(vert[0], vert[1], vert[2]);
    // Face
    if(zenoPrim->loops.size() && zenoPrim->polys.size()){
        // TODO Generate UsdGeomMesh based on these attributes
    }else{
        for(auto const& ind:zenoPrim->tris){
            FaceVertexIndices.emplace_back(ind[0]);
            FaceVertexIndices.emplace_back(ind[1]);
            FaceVertexIndices.emplace_back(ind[2]);
            FaceVertexCounts.emplace_back(3);
        }
    }
    // DisplayColor
    if(zenoPrim->verts.has_attr("clr0")){
        for(auto const& clr0:zenoPrim->verts.attr<zeno::vec3f>("clr0")){
            DisplayColor.emplace_back(clr0[0], clr0[1], clr0[2]);
        }
    }

    mesh.CreatePointsAttr(pxr::VtValue{Points});
    mesh.CreateFaceVertexCountsAttr(pxr::VtValue{FaceVertexCounts});
    mesh.CreateFaceVertexIndicesAttr(pxr::VtValue{FaceVertexIndices});
    mesh.CreateDisplayColorAttr(pxr::VtValue{DisplayColor});

    mesh.GetDisplayColorPrimvar().SetInterpolation(UsdGeomTokens->vertex);
}

bool zenovis::StageManager::load_objects(const std::map<std::string, std::shared_ptr<zeno::IObject>> &objs) {
    auto ins = zenoObjects.insertPass();
    bool inserted = false;
    for (auto const &[key, obj] : objs) {
        if (ins.may_emplace(key)) {

            std::string p_path, p_type;
            PrimInfo primInfo;
            obj->userData().has("P_Path") ? p_path = obj->userData().get2<std::string>("P_Path") : p_path = "";
            obj->userData().has("P_Type") ? p_type = obj->userData().get2<std::string>("P_Type") : p_type = "";
            primInfo.pPath = p_path; primInfo.iObject = obj;
            zeno::log_info("USD: StageManager Emplace {}, P_Type {}, P_Path {}", key, p_type, p_path);

            if(p_type == _tokens->UsdGeomMesh.GetString()){
                _UsdGeomMesh(primInfo);
            }else if(p_type == _tokens->UsdLuxDiskLight.GetString()){

            }else{
                zeno::log_info("USD: Unsupported type {}", p_type);
            }

            ins.try_emplace(key, std::move(obj));
            inserted = true;
        }
    }

    // Debug
    //std::string stageString;
    //cStagePtr->ExportToString(&stageString);
    //std::cout << "USD: Stage " << std::endl << stageString << std::endl;
    cStagePtr->Save();
    sStagePtr->Save();

    return inserted;
}
void StageManager::update() {

    //client();

    // TODO Use webhook instead of timed update
    std::string cmd = "git -C " + confInfo.cPath + " pull";
    std::string path = ";C:\\Windows\\System32;"+confInfo.cGit;
    std::string path2 = "PATH=C:/Windows/System32;"+confInfo.cGit+";%PATH%";

    // XXX git and cmd (on Windows) environment variables need to be set up in PATH
    std::string res = Execute(cmd);
    std::cout << "USD: Update Res " << res << std::endl;

    // SS(R) -> SS(L)
    if(res.find("Already up to date") == std::string::npos) {
        std::cout << "USD: Update Dirty Stage" << std::endl;

        sStagePtr->Reload();  // The effect is same as layers.reload

        // Debug
        //std::string str;
        //sStagePtr->ExportToString(&str);
        //std::cout << "USD Stage " << std::endl << str << std::endl;

        // SS(L) -> CS
    }
}

}