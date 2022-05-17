#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/document.h>
#include <stack>

class pybjsonparser {
    rapidjson::Document root;
    std::stack<rapidjson::Value *> stk;

public:
    pybjsonparser &str(std::string_view s) {
        root.Parse(s.data(), s.size());
        return *this;
    }

    pybjsonparser &obj() {
        stk.push(&root);
        return *this;
    }

    pybjsonparser &eobj() {
        stk.pop();
        return *this;
    }

    pybjsonparser &key(const char *s) {
        rapidjson::Value const *val = stk.top();
        auto it = val->FindMember(s);
        if (it == val->MemberEnd())
            throw;
        stk.top() = const_cast<rapidjson::Value *>(&it->value);
        return *this;
    }

    pybjsonparser &val(std::string &s) {
        s.assign(stk.top()->GetString(), stk.top()->GetStringLength());
        return *this;
    }

    pybjsonparser &val(int32_t &i) {
        i = stk.top()->GetInt();
        return *this;
    }

    pybjsonparser &val(uint32_t &i) {
        i = stk.top()->GetUint();
        return *this;
    }

    pybjsonparser &val(int64_t &i) {
        i = stk.top()->GetInt64();
        return *this;
    }

    pybjsonparser &val(uint64_t &i) {
        i = stk.top()->GetUint64();
        return *this;
    }

    pybjsonparser &val(float &i) {
        i = stk.top()->GetFloat();
        return *this;
    }

    pybjsonparser &val(double &i) {
        i = stk.top()->GetDouble();
        return *this;
    }

    pybjsonparser &val_bool(bool &i) {
        i = stk.top()->GetBool();
        return *this;
    }
    
    pybjsonparser &val_null() {
        RAPIDJSON_ASSERT(stk.top()->IsNull());
        return *this;
    }

    template <class FGet, class FSet>
    pybjsonparser &val_f(FGet &&fget, FSet &&fset) {
        std::decay_t<std::invoke_result_t<FGet>> t;
        val(t);
        std::forward<FSet>(fset)(std::move(t));
        return *this;
    }
};

class pybjsonwriter {
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer{buffer};

public:
    std::string str() const {
        return {buffer.GetString(), buffer.GetSize()};
    }

    std::string_view str_view() const {
        return {buffer.GetString(), buffer.GetSize()};
    }

    pybjsonwriter &obj() {
        writer.StartObject();
        return *this;
    }

    pybjsonwriter &eobj() {
        writer.EndObject();
        return *this;
    }

    pybjsonwriter &key(std::string_view s) {
        writer.Key(s.data(), s.size());
        return *this;
    }

    pybjsonwriter &val(std::string_view s) {
        writer.String(s.data(), s.size());
        return *this;
    }

    pybjsonwriter &val(int32_t i) {
        writer.Int(i);
        return *this;
    }

    pybjsonwriter &val(uint32_t i) {
        writer.Uint(i);
        return *this;
    }

    pybjsonwriter &val(int64_t i) {
        writer.Int64(i);
        return *this;
    }

    pybjsonwriter &val(uint64_t i) {
        writer.Uint64(i);
        return *this;
    }

    pybjsonwriter &val(float i) {
        writer.Double(i);
        return *this;
    }

    pybjsonwriter &val(double i) {
        writer.Double(i);
        return *this;
    }

    pybjsonwriter &val_bool(bool i) {
        writer.Bool(i);
        return *this;
    }
    
    pybjsonwriter &val_null() {
        writer.Null();
        return *this;
    }

    template <class FGet, class FSet>
    pybjsonwriter &val_f(FGet &&fget, FSet &&fset) {
        val(std::forward<FGet>(fget)());
        return *this;
    }
};

int main() {
    pybjsonwriter w;
    w.obj()
     .key("a")
     .val("b")
     .eobj();
    auto r = w.str();
    std::cout << r << std::endl;
    pybjsonparser p;
    std::string b;
    p.str(r)
     .obj()
     .key("a")
     .val(b)
     .eobj();
    std::cout << b << std::endl;
    return 0;
}
