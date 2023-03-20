#include "ubipcclient.h"
#include "msgpack.h"
#include "unrealregistry.h"
#include <QTcpSocket>
#include <QThread>

UnrealBridge::IPCClient::IPCClient(QObject *parent) : QObject(parent) {
}

void UnrealBridge::IPCClient::setup() {
    UnrealBridge::bIsWorkProcess = true;
}

UnrealBridge::IPCServer::IPCServer(QObject *parent) : QObject(parent) {
    m_server_thread = new QThread(this);
    moveToThread(m_server_thread);
    connect(m_server_thread, &QThread::started, this, &IPCServer::serverLoop, Qt::QueuedConnection);
}

void UnrealBridge::IPCServer::setup() {
    m_server.Get("/texture/heightfield", UnrealBridge::IPCHandler::heightField);
    m_server.Post("/livelink/heightfield", UnrealBridge::IPCHandler::addHeightFieldSubject);
    m_server_thread->start();
}

void UnrealBridge::IPCServer::serverLoop() {
    m_server.listen("127.0.0.1", 23344);
}

void UnrealBridge::IPCHandler::heightField(const httplib::Request &req, httplib::Response &res) {
    auto subjectName = req.params.find("subject");
    if (subjectName != req.params.end()) {
        auto sub = ZenoSubjectRegistry::find<zeno::UnrealZenoHeightFieldSubject>(subjectName->second);
        if (sub.has_value()) {
            std::vector<uint8_t> data = msgpack::pack(*sub.value());
            res.set_content(reinterpret_cast<const char *>(data.data()), data.size(), "application/x-binary");
        }
    }
    else {
        res.status = 404;
    }
}

void UnrealBridge::IPCHandler::addHeightFieldSubject(const httplib::Request &req, httplib::Response &res) {
    auto subjectName = req.headers.find("content-type");
    if (subjectName != req.headers.end()) {
        std::error_code err;
        auto subject = msgpack::unpack<UnrealHeightFieldSubject>(reinterpret_cast<const uint8_t *>(req.body.c_str()), req.body.size(), err);
        if (!err) {
            UnrealSubjectRegistry::getStatic().put(subjectName->second, subject);
            UnrealSubjectRegistry::getStatic().markDirty(true);
        }
    }
    else {
        res.status = 400;
    }
}
