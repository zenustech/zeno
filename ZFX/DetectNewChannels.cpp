#include "IRVisitor.h"
#include "Stmts.h"

namespace zfx {

struct DetectNewChannels : Visitor<DetectNewChannels> {
    using visit_stmt_types = std::tuple
        < TempSymbolStmt
        >;

    std::map<int, std::string> const &temps;
    DetectNewChannels(std::map<int, std::string> const &temps)
        : temps(temps) {}

    std::map<std::string, int> tempdims;

    void visit(TempSymbolStmt *stmt) {
        auto name = temps.at(stmt->tmpid);
        if (name.size() && name[0] == '@') {
            if (stmt->dim == 0) {
                error("undefined type for new channel: %s\n", name.c_str());
            }
#ifdef ZFX_PRINT_IR
            printf("detected new channel: %s (dim %d)\n",
                name.c_str(), stmt->dim);
#endif
            tempdims[name] = stmt->dim;
        }
    }
};

std::map<std::string, int> apply_detect_new_channels(IR *ir,
        std::map<int, std::string> const &temps) {
    DetectNewChannels visitor(temps);
    visitor.apply(ir);
    return visitor.tempdims;
}

}
