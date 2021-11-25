#include <zeno/dop/Functor.h>


ZENO_NAMESPACE_BEGIN
namespace dop {


ztd::map<std::string, FuncOverloads> &overloading_table() {
    static ztd::map<std::string, FuncOverloads> impl;
    return impl;
}


static int match_signature(FuncSignature const &lhs, FuncSignature const &rhs) {
    int score = 0;
    for (int i = 0; i < std::min(lhs.size(), rhs.size()); i++) {
        if (rhs[i] == std::type_index(typeid(void))) {
            continue;
        }
        if (lhs[i] == rhs[i]) {
            score += 1;
        }
    }
    return score;
}


void FuncOverloads::invoke(FuncContext *ctx) const {
    FuncSignature sig;
    for (auto const &val: ctx->inputs) {
        sig.push_back(val.type());
    }

    std::map<int, Functor const *> matches;
    for (auto const &[key, func]: functors) {
        if (int prio = match_signature(sig, key); prio != -1) {
            matches.emplace(prio, &func);
        }
    }
    [[unlikely]] if (matches.empty())
        throw ztd::error("no suitable overloading found");
    auto const &func = *matches.begin()->second;
    func(ctx);
}


void add_overloading(const char *kind, Functor func, FuncSignature const &sig) {
    bool success = overloading_table()[kind].functors.emplace(sig, func).second;
    [[unlikely]] if (!success)
        printf("[zeno-init] dop::define: redefined overload: kind=[%s]\n", kind);
}


}
ZENO_NAMESPACE_END
