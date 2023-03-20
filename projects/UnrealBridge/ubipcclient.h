#ifndef ZENO_UBIPCCLIENT_H
#define ZENO_UBIPCCLIENT_H

#include "3rd/httplib.h"
#include "unrealregistry.h"
#include "msgpack.h"
#include <QObject>
#include <QThread>
#include <optional>

class QTcpSocket;

namespace UnrealBridge{

static bool bIsWorkProcess = false;

class IPCClient : public QObject {
    Q_OBJECT

public:
    void setup();

private:
    httplib::Client m_client { "localhost", 23344 };

    explicit IPCClient(QObject* parent);

public:
    inline static IPCClient& getStatic() {
        static IPCClient sIPCClient(nullptr);

        return sIPCClient;
    }

    template <typename T, typename U = std::remove_reference<T>::type>
    static std::optional<U> fetchSubject(const std::string& subjectName) { return std::nullopt; }

    template <>
    static std::optional<zeno::UnrealZenoHeightFieldSubject> fetchSubject<zeno::UnrealZenoHeightFieldSubject>(const std::string& subjectName) {
        IPCClient& client = getStatic();
        httplib::Params params;
        params.insert(std::make_pair("subject", subjectName));
        auto res = client.m_client.Get("/texture/heightfield", params, httplib::Headers {}, httplib::Progress {});
        if (res && res->status != 404) {
            std::error_code err;
            auto subject = msgpack::unpack<zeno::UnrealZenoHeightFieldSubject>(reinterpret_cast<const uint8_t *>(res->body.c_str()), res->body.size(), err);
            if (!err) {
                return subject;
            }
        }
        return std::nullopt;
    }

    template <typename T, typename U = std::remove_reference<T>::type>
    static void sendSubject(const std::string& subjectName, U& subject) {}

    template <>
    static void sendSubject<UnrealHeightFieldSubject>(const std::string& subjectName, UnrealHeightFieldSubject& subject) {
        IPCClient& client = getStatic();
        auto data = msgpack::pack(subject);
        auto res = client.m_client.Post("/livelink/heightfield", reinterpret_cast<const char *>(data.data()), data.size(), subjectName);
    }
};

class IPCServer : public QObject{
  Q_OBJECT

private:
    QThread* m_server_thread;
    httplib::Server m_server;

    explicit IPCServer(QObject* parent);

public:
    void setup();

private slots:
    void serverLoop();

public:
    inline static IPCServer& getStatic() {
        static IPCServer sIPCServer(nullptr);

        return sIPCServer;
    }
};

struct IPCHandler {
    static void heightField(const httplib::Request& req, httplib::Response& res);
    static void addHeightFieldSubject(const httplib::Request& req, httplib::Response& res);
};

}

#endif //ZENO_UBIPCCLIENT_H
