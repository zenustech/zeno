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

// https://www.codeproject.com/Articles/1264257/Socket-Programming-in-Cplusplus-using-boost-asio-T
struct StageSyncServer;

struct StageConHandler : boost::enable_shared_from_this<StageConHandler>
{
    StageSyncServer* syncServer;
    boost::asio::ip::tcp::socket sock;
    std::string message="Message From Server!";
    enum { max_length = 1024 };
    char data[max_length];

    typedef boost::shared_ptr<StageConHandler> pointer;
    StageConHandler(boost::asio::io_context& io_context): sock(io_context){}

    // creating the pointer
    static pointer Create(boost::asio::io_context& io_context)
    {
        std::cout << "USD: StageConHandler Create\n";
        return pointer(new StageConHandler(io_context));
    }

    //socket creation
    boost::asio::ip::tcp::socket& Socket()
    {
        return sock;
    }

    void Start()
    {
        std::cout << "USD: StageConHandler Start\n";
        sock.async_read_some(
            boost::asio::buffer(data, max_length),
            boost::bind(&StageConHandler::HandleRead,
                        shared_from_this(),
                        boost::asio::placeholders::error,
                        boost::asio::placeholders::bytes_transferred));

        sock.async_write_some(
            boost::asio::buffer(message, max_length),
            boost::bind(&StageConHandler::HandleWrite,
                        shared_from_this(),
                        boost::asio::placeholders::error,
                        boost::asio::placeholders::bytes_transferred));
    }

    void HandleRead(const boost::system::error_code& err, size_t bytes_transferred);
    void HandleWrite(const boost::system::error_code& err, size_t bytes_transferred)
    {
        if (!err) {
            std::cout << "USD: ConHandler Server Sent Message"<< std::endl;
        } else {
            std::cerr << "USD: ConHandler Error Write: " << err.message() << std::endl;
            sock.close();
        }
    }
};

struct StageSyncServer{
    boost::asio::ip::tcp::acceptor acceptor_;
    std::function<void(SyncInfo)> cSyncCallback;
    void StartAccept()
    {
        std::cout << "USD: StageSyncServer StartAccept\n";
        // socket  // acceptor_.get_io_service()
        auto& a = acceptor_.get_executor().context();
        StageConHandler::pointer connection = StageConHandler::Create(static_cast<boost::asio::io_context &>(a));
        connection->syncServer = this;

        // asynchronous accept operation and wait for a new connection.
        acceptor_.async_accept(connection->Socket(),
                               boost::bind(&StageSyncServer::HandleAccept, this, connection,
                                           boost::asio::placeholders::error));
    }

    //constructor for accepting connection from client
    StageSyncServer(boost::asio::io_context& io_context): acceptor_(io_context, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 1234))
    {
        std::cout << "USD: StageSyncServer Constructed\n";
        StartAccept();
    }

    void HandleAccept(StageConHandler::pointer connection, const boost::system::error_code& err)
    {
        std::cout << "USD: StageSyncServer HandleAccept\n";
        if (!err) {
            connection->Start();
        }
        StartAccept();
    }
};

void StageConHandler::HandleRead(const boost::system::error_code& err, size_t bytes_transferred)
{
    if (!err) {
        std::cout << "USD: ConHandler Receive A Message: " << data << std::endl;
        SyncInfo syncInfo{data};
        syncServer->cSyncCallback(syncInfo);
    } else {
        std::cerr << "USD: ConHandler Error Read: " << err.message() << std::endl;
        sock.close();
    }
}

// https://www.cppstories.com/2018/07/string-view-perf-followup/
std::vector<std::string> StrSplit(const std::string& str, const std::string& delims = " ")
{
    std::vector<std::string> output;
    auto first = std::cbegin(str);

    while (first != std::cend(str))
    {
        const auto second = std::find_first_of(first, std::cend(str),
                                               std::cbegin(delims), std::cend(delims));

        if (first != second)
            output.emplace_back(first, second);

        if (second == std::cend(str))
            break;

        first = std::next(second);
    }

    return output;
}

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


void ZenoStage::init(){
    UpdateTimer(std::bind(&ZenoStage::update, this), 500);
    composLayer = SdfLayer::CreateAnonymous();

    cStagePtr = UsdStage::CreateInMemory();
    //cStagePtr = UsdStage::CreateNew(stateInfo->cPath + "/_inter_stage.usda");
    fStagePtr = UsdStage::CreateNew(stateInfo->cPath + "/_inter_final.usda");
    sStagePtr = UsdStage::Open(stateInfo->cPath + "/test.usda");

    // Create a placeholder usd file
    UsdStage::CreateNew(stateInfo->cPath + "/_inter_compLayer.usda");  // making sure the file exists
    fStagePtr->GetRootLayer()->InsertSubLayerPath("_inter_compLayer.usda");
    fStagePtr->Save();

    pxr::UsdGeomSetStageUpAxis(cStagePtr, pxr::TfToken("Y"));
    pxr::UsdGeomSetStageUpAxis(fStagePtr, pxr::TfToken("Y"));

    stateInfo->cSyncCallback = std::bind(&ZenoStage::sync, this, std::placeholders::_1);

    std::thread([&]() {
        try
        {
            std::cout << "USD: StageManager Server Running.\n";
            boost::asio::io_service io_service;
            //StageSyncServer server(io_service);
            auto server = std::make_shared<StageSyncServer>(io_service);
            server->cSyncCallback = stateInfo->cSyncCallback;
            io_service.run();
        }
        catch(std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }).detach();
}

void ZenoStage::update() {
    // TODO Use webhook instead of timed update
    std::string cmd = "git -C " + stateInfo->cPath + " pull";

    // XXX git and cmd (on Windows) environment variables need to be set up in PATH
    std::string res = Execute(cmd);
    res.erase(std::remove(res.begin(), res.end(), '\n'), res.cend());
    //std::cout << "USD: Update Res " << res << std::endl;

    // SS(R) -> SS(L)
    if(res.find("Already up to date") == std::string::npos) {
        std::cout << "USD: Update Dirty Stage" << std::endl;

        TIMER_START(StageUpdate)
        sStagePtr->Reload();  // The effect is same as layers.reload
        CompositionArcsStage();
        fStagePtr->Reload();
        TIMER_END(StageUpdate)

        // FIXME The child thread calls the main thread update function
        //  Need better synchronization mechanisms and consider moving
        //  this logic into the UI module
        stateInfo->cUpdateFunction();
    }
}

void ZenoStage::sync(SyncInfo info) {
    std::cout << "USD: ZenoStage Sync\n";

    auto splittedStr = StrSplit(info.sMsg, ";");
    auto primPath = splittedStr[0];
    auto transformType = splittedStr[1];
    auto valueType = splittedStr[2];
    // Let's assume that the type is vec3
    double x = atof(splittedStr[3].c_str());
    double y = atof(splittedStr[4].c_str());
    double z = atof(splittedStr[5].c_str());
    auto value = zeno::vec3f(float(x),float(y),float(z));
    auto end = splittedStr[6];

    std::cout << "USD: Sync " << primPath << " Type " << transformType <<" "<< valueType
              << " Value " << value[0]<<","<<value[1]<<","<<value[2] << " END " << end << "\n";

    auto sdfPrimPath = SdfPath(primPath);
    auto obj = convertedObject[sdfPrimPath];
    if(objectsTransform.find(sdfPrimPath) == objectsTransform.end())
        objectsTransform[sdfPrimPath] = ZTransInfo();

    auto& objTrans = objectsTransform[sdfPrimPath];
    objTrans.zTrans = glm::vec3(0,0,0);
    objTrans.zRotate = glm::vec3(0,0,0);
    objTrans.zScale = glm::vec3(1,1,1);
    if(transformType == "Translate"){
        objTrans.zTrans = glm::vec3(value[0],value[1],value[2]);
    }else if(transformType == "Rotate"){
        objTrans.zRotate = glm::vec3(value[0],value[1],value[2]);
    }else if(transformType == "Scale"){
        objTrans.zScale = glm::vec3(value[0],value[1],value[2]);
    }

    if(! std::count(syncedObject.begin(), syncedObject.end(), primPath))
        syncedObject.emplace_back(primPath);

    stateInfo->cUpdateFunction();
}

ZenoStage::ZenoStage() {

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

int ZenoStage::PrintStageString(UsdStageRefPtr stage){
    std::string result;
    stage->ExportToString(&result);
    std::cout << "USD: Stage\n" << result << "\n";
}

int ZenoStage::PrintLayerString(SdfLayerRefPtr layer){
    std::string result;
    layer->ExportToString(&result);
    std::cout << "USD: Layer\n" << result << "\n";
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

int ZenoStage::CheckConvertConsistency(UsdPrim& prim){
    if(convertedObject.find(prim.GetPrimPath()) != convertedObject.end()){
        return 1;
    }
    return 0;
}

int ZenoStage::TraverseStageObjects(UsdStageRefPtr stage, std::map<std::string, UPrimInfo>& consis) {
    for (UsdPrim prim: stage->TraverseAll()){
        UPrimInfo primInfo;
        pxr::SdfPath primPath = prim.GetPrimPath();
        std::string primStrPath = primPath.GetAsString();

        // Visibility
        if(CheckAttrVisibility(prim))
            continue;
        if(CheckConvertConsistency(prim)){
            primInfo = {prim, convertedObject[primPath]};
            consis[primStrPath] = primInfo;
            continue;
        }

        // Mesh
        if(prim.GetTypeName() == _typeTokens->Mesh){
            primInfo = {prim, std::make_shared<zeno::PrimitiveObject>()};
            consis[primStrPath] = primInfo;
            std::cout << "USD: Converting " << primStrPath << "\n";
            Convert2ZenoPrimitive(primInfo);
            convertedObject[primPath] = primInfo.iObject;
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
    bool exportResult = composLayer->Export(stateInfo->cPath + "/_inter_compLayer.usda");
    // Debug
    //PrintLayerString(composLayer);

    if(! exportResult){
        zeno::log_error("USD: ComposLayer Export Error");
    }

    return 0;
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

    // TODO Interpolation for primvars read.

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
    auto path = prim.GetPrimPath().GetAsString();
    auto timeCode = 0.0;

    TfToken tf_displayColor("primvars:displayColor");

    UsdGeomMesh processGeom(prim);
    UsdAttribute attr_geoHole = processGeom.GetHoleIndicesAttr();
    UsdAttribute attr_faceVertexIndices = prim.GetAttribute(UsdGeomTokens->faceVertexIndices);
    UsdAttribute attr_points = prim.GetAttribute(UsdGeomTokens->points);
    UsdAttribute attr_normals = prim.GetAttribute(UsdGeomTokens->normals);
    UsdAttribute attr_displayColor = prim.GetAttribute(tf_displayColor);
    UsdAttribute attr_faceVertexCounts = prim.GetAttribute(TfToken("faceVertexCounts"));

    VtArray<int> vt_faceVertexCounts, vt_faceVertexIndices;
    VtArray<GfVec3f> vt_points, vt_normals, vt_displayColor;
    VtArray<int> vt_holeIndices;
    UsdGeomXformable xformable(prim);
    GfMatrix4d ModelTransform = xformable.ComputeLocalToWorldTransform(timeCode);

    attr_faceVertexIndices.Get(&vt_faceVertexIndices, timeCode);
    attr_points.Get(&vt_points, timeCode);
    attr_normals.Get(&vt_normals, timeCode);
    attr_displayColor.Get(&vt_displayColor, timeCode);
    attr_geoHole.Get(&vt_holeIndices, timeCode);
    attr_faceVertexCounts.Get(&vt_faceVertexCounts, timeCode);

    // TODO Interpolation for primvars write.

    int index_start = 0;

    auto & verts = obj->verts;
    auto & loops = obj->loops;
    auto & polys  = obj->polys;
    auto & tris  = obj->tris;
    for(int i=0; i<vt_points.size(); i++){
        // TODO Use transform data like userData instead of actually calculating vertex positions
        auto g_point = ModelTransform.Transform(vt_points[i]);
        verts.emplace_back(g_point[0],g_point[1],g_point[2]);
    }
    // FIXME If the mesh does not have a normal attribute, zeno will crash
    if(! vt_normals.empty()){
        auto & nrm = obj->verts.add_attr<zeno::vec3f>("nrm");
        for(int i=0;i<vt_normals.size();i++){
            auto normal = vt_normals[i];
            nrm[i] = {normal[0],normal[1],normal[2]};
        }
    }else{
        std::cout << "USD: No normal property " << path << "\n";
    }

    // TODO zeno support for quadrilateral and changeable faces
    for(int i=0;i<vt_faceVertexCounts.size();i++){
        // FIXME With tris, it is no longer black in Optix mode,
        //  but the face is wrong because it is quad from ServerStage
        //  Use loops-polys mode, normal under raster, wrong under Optix
        int count = vt_faceVertexCounts[i];

        //for(int j=index_start;j<index_start+count;j++){
        //    loops.emplace_back(vt_faceVertexIndices[j]);
        //}
        //polys.push_back({i * count, count});

        // TODO Support any number of face vertex counts
        if(count==3){
            tris.emplace_back(vt_faceVertexIndices[index_start],
                              vt_faceVertexIndices[index_start+1],
                              vt_faceVertexIndices[index_start+2]);
        }
        if(count==4){
            tris.emplace_back(vt_faceVertexIndices[index_start],
                              vt_faceVertexIndices[index_start+2],
                              vt_faceVertexIndices[index_start+1]);
            tris.emplace_back(vt_faceVertexIndices[index_start],
                            vt_faceVertexIndices[index_start+3],
                            vt_faceVertexIndices[index_start+2]);
        }
        index_start+=count;
    }

    if(! vt_displayColor.empty()){
        auto & clr = obj->verts.add_attr<zeno::vec3f>("clr");
        for(int i=0;i<vt_displayColor.size();i++){
            auto color = vt_displayColor[i];
            clr[i] = {color[0],color[1],color[2]};
        }
    }else{
        std::cout << "USD: No displayColor property " << path << "\n";
    }

    return 0;
}
