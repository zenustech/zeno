#include "zenosolver.h"
#include <zenoio/reader/zsgreader.h>
#include <zenomodel/include/zenomodel.h>
#include <zenomodel/include/nodesmgr.h>
#include "serialize.h"
#include <iostream>

int runner_start(std::string const& progJson, int sessionid, HANDLE hPipeWrite, const LAUNCH_PARAM& param);
//#define DEBUG_SOME_NODE

namespace zensolver
{
    void flip_solve(
        HANDLE hPipeWrite,
        const std::string& init_fluid,                   /*初始流体*/    //目前只传共享内存的名字，不存具体的字节
        int size_fluid,
        const std::string& static_collider,              /*静态碰撞体*/
        int size_collider,
        const std::string& emission_source,              /*发射源*/
        int size_emission,
        float accuracy,                     /*精度*/
        float timestep,                     /*时间步长*/
        float max_substep,                  /*最大子步数*/
        /*重力*/
        float gravity_x,
        float gravity_y,
        float gravity_z,
        /*发射源速度*/
        float emit_vx,
        float emit_vy,
        float emit_vz,
        bool is_emission,                               /*是否发射*/
        float dynamic_collide_strength,                 /*动态碰撞强度*/
        float density,               /*密度*/
        float surface_tension,       /*表面张力*/
        float viscosity,             /*粘性*/
        float wall_viscosity,        /*壁面粘性*/
        float wall_viscosityRange,   /*壁面粘性作用范围*/
        char* curve_force,           /*曲线力*/
        int n_curve_force,
        int curve_endframe,          /*曲线终止帧*/
        float curve_range,           /*曲线作用范围*/
        float preview_size,          /*预览大小*/
        float preview_minVelocity,   /*预览最小速度*/
        float preview_maxVelocity,   /*预览最大速度*/
        char* FSD,                   /*流固耦合*/
        int n_size_FSD
    ) {
        //步骤一：打开流体的标准zsg文件（只有流体求解器自身）
        QString filename = ":/FlipGraph_V05.zsg";
        QFile file(filename);
        bool ret = file.open(QIODevice::ReadOnly | QIODevice::Text);
        if (!ret) {
            return;
        }

        QByteArray byteArray = file.readAll();
        IGraphsModel* pModel = zeno_model::createModel(nullptr);    //todo: leak
        std::shared_ptr<IAcceptor> acceptor(zeno_model::createIOAcceptor(pModel, false));
        ret = ZsgReader::getInstance().parseZsg(filename, byteArray, acceptor.get());
        if (!ret) {
            return;
        }

        QModelIndex subgIdx = pModel->index("main");
        QString name = QString::fromUtf8("流体求解器");
        QModelIndexList nodes = pModel->findSubgraphNode(name);
        ZASSERT_EXIT(nodes.size() == 1);
        QModelIndex solverIdx = nodes[0];
        QString identSolver = solverIdx.data(ROLE_OBJID).toString();

#if 1
        //步骤二：新创建几个zenobjcache节点，把init_fluid等encoding的对象塞进去
        NODE_DATA node = NodesMgr::newNodeData(pModel, "ZenCacheNode", QPointF());
        pModel->addNode(node, subgIdx);
        QString identInitFluid = node[ROLE_OBJID].toString();
        QModelIndex initFluidIdx = pModel->index(identInitFluid, subgIdx);
        PARAM_UPDATE_INFO info;
        info.name = "sharedmemory";
        info.newValue = QString::fromStdString(init_fluid); //共享内存的地址
        pModel->updateSocketDefl(identInitFluid, info, subgIdx);
        info.name = "size";
        info.newValue = size_fluid;
        pModel->updateSocketDefl(identInitFluid, info, subgIdx);
#endif

#if 1
        node = NodesMgr::newNodeData(pModel, "ZenCacheNode", QPointF());
        pModel->addNode(node, subgIdx);
        QString identStaticColl = node[ROLE_OBJID].toString();
        QModelIndex staticColliderIdx = pModel->index(identStaticColl, subgIdx);
        info.name = "sharedmemory";
        info.newValue = QString::fromStdString(static_collider);
        pModel->updateSocketDefl(identStaticColl, info, subgIdx);
        info.name = "size";
        info.newValue = size_collider;
        pModel->updateSocketDefl(identStaticColl, info, subgIdx);
#endif

#if 1
        //创建PrimitiveToSDF，将静态碰撞体传进去
        node = NodesMgr::newNodeData(pModel, "PrimitiveToSDF", QPointF());
        pModel->addNode(node, subgIdx);
        QString identPrim2SDF = node[ROLE_OBJID].toString();
        QModelIndex prim2sdfIdx = pModel->index(identPrim2SDF, subgIdx);

        STATUS_UPDATE_INFO status;
        status.role = ROLE_OPTIONS;
        status.newValue = OPT_ONCE;
        status.oldValue = 0;
        pModel->updateNodeStatus(identPrim2SDF, status, subgIdx);
#endif

        //步骤三：连接这几个zenobjcache节点至流体求解器
#if 1
        EdgeInfo edge;
        edge.outSockPath = QString("main:%1:[node]/outputs/output").arg(identInitFluid);// "main:5420a971-ZenCacheNode:[node]/outputs/output";
        edge.inSockPath = QString::fromUtf8("main:%1:[node]/inputs/初始流体").arg(identSolver);
        pModel->addLink(subgIdx, edge);
#endif

#if 1
        edge.outSockPath = QString("main:%1:[node]/outputs/output").arg(identStaticColl);// "main:5420a971-ZenCacheNode:[node]/outputs/output";
        edge.inSockPath = QString::fromUtf8("main:%1:[node]/inputs/PrimitiveMesh").arg(identPrim2SDF);
        pModel->addLink(subgIdx, edge);

        edge.outSockPath = QString("main:%1:[node]/outputs/sdf").arg(identPrim2SDF);
        edge.inSockPath = QString::fromUtf8("main:%1:[node]/inputs/静态碰撞体").arg(identSolver);
        pModel->addLink(subgIdx, edge);
#endif


        //步骤四：序列化这个图
        LAUNCH_PARAM launch;
        launch.beginFrame = 0;
        launch.endFrame = 100;
        launch.enableCache = true;
        launch.cacheDir = "C:/tmp";

        rapidjson::StringBuffer s;
        RAPIDJSON_WRITER writer(s);
        {
            JsonArrayBatch batch(writer);
            JsonHelper::AddVariantList({ "setBeginFrameNumber", launch.beginFrame }, "int", writer);
            JsonHelper::AddVariantList({ "setEndFrameNumber", launch.endFrame }, "int", writer);
            serializeScene(pModel, writer, launch);
        }

        //步骤五：直接调内核处理
        std::string progJson(s.GetString());
        runner_start(progJson, 0, hPipeWrite, launch);
        
        //TODO: 内核要和调用方(zeno)进行通信，约定好zencache的位置
    }
}