#include "zenovis/Stage.h"
#include "zeno/types/PrimitiveObject.h"
#include "zeno/types/UserData.h"
#include "zeno/utils/logger.h"

#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdUtils/flattenLayerStack.h>
#include <pxr/usd/usdUtils/stitch.h>

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

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

/// ######################################################

ZenoStage::ZenoStage() {
    UpdateTimer(std::bind(&ZenoStage::update, this), 500);

    //cStagePtr = UsdStage::CreateInMemory();
    cStagePtr = UsdStage::CreateNew(confInfo.cPath + "/_inter_stage.usda");
    fStagePtr = UsdStage::CreateNew(confInfo.cPath + "/_inter_final.usda");
    sStagePtr = UsdStage::Open(confInfo.cPath + "/test.usda");

    UsdStage::CreateNew(confInfo.cPath + "/_inter_compLayer.usda");  // making sure the file exists
    fStagePtr->GetRootLayer()->InsertSubLayerPath("_inter_compLayer.usda");
    fStagePtr->Save();

    pxr::UsdGeomSetStageUpAxis(cStagePtr, pxr::TfToken("Y"));
    pxr::UsdGeomSetStageUpAxis(fStagePtr, pxr::TfToken("Y"));

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
}

void ZenoStage::CreateUSDHierarchy(const SdfPath &path)
{
    if (path == SdfPath::AbsoluteRootPath())
        return;
    CreateUSDHierarchy(path.GetParentPath());
    UsdGeomXform::Define(cStagePtr, path);
}

int ZenoStage::RemoveStagePrims(){
    std::vector<std::string> primPaths;
    for (UsdPrim prim: cStagePtr->TraverseAll()) {
        primPaths.emplace_back(prim.GetPrimPath().GetAsString());
    }
    for(auto const &p: primPaths){
        cStagePtr->RemovePrim(SdfPath(p));
    }
    if(! cStagePtr->TraverseAll().empty()){
        std::cout << "USD: ERROR Didn't clean out\n";
    }
}

int ZenoStage::CheckPathConflict(){
    std::vector<SdfPath> primPaths;
    for (UsdPrim prim: cStagePtr->TraverseAll()) {
        auto at = _typeTokens->allTokens;
        auto it = std::find(at.begin(), at.end(), prim.GetTypeName());
        if(it != at.end())
            primPaths.emplace_back(prim.GetPrimPath());
    }
    for (UsdPrim prim: sStagePtr->TraverseAll()) {
        auto it = std::find(primPaths.begin(), primPaths.end(), prim.GetPrimPath());
        if(it != primPaths.end()){
            zeno::log_warn("USD: Path Conflict {}", prim.GetPrimPath());
        }
    }
}

int ZenoStage::CheckAttrVisibility(const UsdPrim& prim){
    UsdAttribute attr_visibility = prim.GetAttribute(TfToken("visibility"));
    TfToken visibility;
    attr_visibility.Get(&visibility, 0.0);
    if (visibility == UsdGeomTokens->invisible)
        return 1;
    return 0;
}

int ZenoStage::TraverseStageObjects(UsdStageRefPtr stage, std::map<std::string, UPrimInfo>& consis) {
    for (UsdPrim prim: stage->TraverseAll()){
        // Visibility
        if(CheckAttrVisibility(prim))
            continue;

        UPrimInfo primInfo{prim, std::make_shared<zeno::PrimitiveObject>()};
        std::string primPath = prim.GetPrimPath().GetAsString();

        // Mesh
        if(prim.GetTypeName() == _typeTokens->Mesh){
            consis[primPath] = primInfo;
            Convert2ZenoPrimitive(primInfo);
        }
    }
}

int ZenoStage::CompositionArcsStage(){
    auto roots = std::vector<pxr::SdfLayerHandle> {};
    roots.reserve(1);
    roots.push_back(sStagePtr->GetRootLayer());
    std::vector<std::string> identifiers;
    identifiers.reserve(roots.size());
    for (auto const &root : roots) {
        identifiers.push_back(root->GetIdentifier());
    }

    pxr::UsdUtilsResolveAssetPathFn anonymous_path_remover =
        [&identifiers](pxr::SdfLayerHandle const &sourceLayer, std::string const &path) {
            if (std::find(std::begin(identifiers), std::end(identifiers), path) != std::end(identifiers)) {
                return std::string {};
            }

            return std::string {path.c_str()};
        };

    composLayer = pxr::UsdUtilsFlattenLayerStack(cStagePtr, anonymous_path_remover);
    for (auto const &root : roots) {
        pxr::UsdUtilsStitchLayers(composLayer, root);
    }

    // Output
    composLayer->Export(confInfo.cPath + "/_inter_compLayer.usda");
    return 0;
}

void ZenoStage::update() {
    // TODO Use webhook instead of timed update
    std::string cmd = "git -C " + confInfo.cPath + " pull";

    // XXX git and cmd (on Windows) environment variables need to be set up in PATH
    std::string res = Execute(cmd);
    res.erase(std::remove(res.begin(), res.end(), '\n'), res.cend());
    //std::cout << "USD: Update Res " << res << std::endl;

    // SS(R) -> SS(L)
    if(res.find("Already up to date") == std::string::npos) {
        std::cout << "USD: Update Dirty Stage" << std::endl;

        sStagePtr->Reload();  // The effect is same as layers.reload
        CompositionArcsStage();
        fStagePtr->Reload();

        std::string stageString;
        fStagePtr->ExportToString(&stageString);
        std::cout << "USD: Stage " << std::endl << stageString << std::endl;
    }
}

int ZenoStage::Convert2UsdGeomMesh(const ZPrimInfo& primInfo){
    auto zenoPrim = dynamic_cast<zeno::PrimitiveObject *>(primInfo.iObject.get());
    std::filesystem::path p(primInfo.pPath); std::string nodeName = p.filename().string();
    //std::cout << "USD: Convert2UsdGeomMesh " << nodeName << std::endl;
    SdfPath objPath(primInfo.pPath);
    CreateUSDHierarchy(objPath);

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
    // DisplayColor  (clr0 - FBX, clr - ZENO)
    // TODO Order of reading
    if(zenoPrim->verts.has_attr("clr0")){
        for(auto const& clr0:zenoPrim->verts.attr<zeno::vec3f>("clr0")){
            DisplayColor.emplace_back(clr0[0], clr0[1], clr0[2]);
        }
    }else if(zenoPrim->verts.has_attr("clr")){
        for(auto const& clr:zenoPrim->verts.attr<zeno::vec3f>("clr")){
            DisplayColor.emplace_back(clr[0], clr[1], clr[2]);
        }
    }

    mesh.CreatePointsAttr(pxr::VtValue{Points});
    mesh.CreateFaceVertexCountsAttr(pxr::VtValue{FaceVertexCounts});
    mesh.CreateFaceVertexIndicesAttr(pxr::VtValue{FaceVertexIndices});
    mesh.CreateDisplayColorAttr(pxr::VtValue{DisplayColor});

    mesh.GetDisplayColorPrimvar().SetInterpolation(UsdGeomTokens->vertex);
}

int ZenoStage::Convert2ZenoPrimitive(const UPrimInfo &primInfo) {
    auto obj = primInfo.iObject;
    auto prim = primInfo.usdPrim;
    auto timeCode = 0.0;

    TfToken tf_displayColor("primvars:displayColor");

    UsdGeomMesh processGeom(prim);
    UsdAttribute attr_geoHole = processGeom.GetHoleIndicesAttr();
    UsdAttribute attr_faceVertexIndices = prim.GetAttribute(UsdGeomTokens->faceVertexIndices);
    UsdAttribute attr_points = prim.GetAttribute(UsdGeomTokens->points);
    UsdAttribute attr_displayColor = prim.GetAttribute(tf_displayColor);
    UsdAttribute attr_faceVertexCounts = prim.GetAttribute(TfToken("faceVertexCounts"));

    VtArray<int> vt_faceVertexCounts, vt_faceVertexIndices;
    VtArray<GfVec3f> vt_points, vt_displayColor;
    VtArray<int> vt_holeIndices;
    UsdGeomXformable xformable(prim);
    GfMatrix4d ModelTransform = xformable.ComputeLocalToWorldTransform(timeCode);

    attr_faceVertexIndices.Get(&vt_faceVertexIndices, timeCode);
    attr_points.Get(&vt_points, timeCode);
    attr_displayColor.Get(&vt_displayColor, timeCode);
    attr_geoHole.Get(&vt_holeIndices, timeCode);
    attr_faceVertexCounts.Get(&vt_faceVertexCounts, timeCode);

    int index_start = 0;

    auto & verts = obj->verts;
    auto & loops = obj->loops;
    auto & polys  = obj->polys;
    for(int i=0; i<vt_points.size(); i++){
        auto g_point = ModelTransform.Transform(vt_points[i]);
        verts.emplace_back(g_point[0],g_point[1],g_point[2]);
    }
    for(int i=0;i<vt_faceVertexCounts.size();i++){
        int count = vt_faceVertexCounts[i];
        for(int j=index_start;j<index_start+count;j++){
            loops.emplace_back(vt_faceVertexIndices[j]);
        }
        polys.push_back({i * count, count});
        index_start+=count;
    }
    if(! vt_displayColor.empty()){
        auto & clr = obj->verts.add_attr<zeno::vec3f>("clr");
        for(int i=0;i<vt_displayColor.size();i++){
            auto color = vt_displayColor[i];
            clr[i] = {color[0],color[1],color[2]};
        }
    }

    return 0;
}

