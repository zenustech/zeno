#include <zeno/extra/GlobalStatus.h>
#include <zeno/core/INode.h>
#include <zeno/utils/log.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

namespace zeno {

ZENO_API void GlobalStatus::clearState() {
    nodeName = {};
    error = nullptr;
}

ZENO_API std::string GlobalStatus::toJson() const {
    if (!failed()) return {};

    rapidjson::Document doc(rapidjson::kObjectType);
    rapidjson::Value nodeNameJson(rapidjson::kStringType);
    nodeNameJson.SetString(nodeName.data(), nodeName.size());
    doc.AddMember("nodeName", nodeNameJson, doc.GetAllocator());

    auto const &errorMessage = error->message;
    rapidjson::Value errorMessageJson(rapidjson::kStringType);
    errorMessageJson.SetString(errorMessage.data(), errorMessage.size());
    doc.AddMember("errorMessage", errorMessageJson, doc.GetAllocator());

    rapidjson::StringBuffer buf;
    rapidjson::Writer writer(buf);
    doc.Accept(writer);
    return {buf.GetString(), buf.GetLength()};
}

ZENO_API void GlobalStatus::fromJson(std::string_view json) {
    if (json.empty()) { *this = {}; return; }

    rapidjson::Document doc;
    doc.Parse(json.data(), json.size());
    log_debug("got error from json: {}", json);

    auto obj = doc.GetObject();

    auto it = obj.FindMember("nodeName");
    if (it == obj.MemberEnd()) {
        log_warn("document has no nodeName!");
        return;
    }
    nodeName.assign(it->value.GetString(), it->value.GetStringLength());

    it = obj.FindMember("errorMessage");
    if (it == obj.MemberEnd()) {
        log_warn("document has no errorMessage!");
        return;
    }
    std::string errorMessage{it->value.GetString(), it->value.GetStringLength()};
    error = std::make_shared<Error>(errorMessage);
}

}
