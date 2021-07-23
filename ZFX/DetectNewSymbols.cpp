#include "IRVisitor.h"
#include "Stmts.h"

namespace zfx {

struct DetectNewSymbols : Visitor<DetectNewSymbols> {
    using visit_stmt_types = std::tuple
        < TempSymbolStmt
        >;

    std::map<int, std::string> const &temps;
    DetectNewSymbols(std::map<int, std::string> const &temps)
        : temps(temps) {}

    std::map<std::string, int> tempdims;

    void visit(TempSymbolStmt *stmt) {
        auto name = temps.at(stmt->tmpid);
        if (name.size() && name[0] == '@') {
            if (stmt->dim == 0) {
                error("undefined type for new channel: %s\n", name.c_str());
            }
            //printf("detected new channel: %s (dim %d)\n",
            //name.c_str(), stmt->dim);
            tempdims[name] = stmt->dim;
        }
    }
};

struct AppendNewSymbols : Visitor<AppendNewSymbols> {
    using visit_stmt_types = std::tuple
        < TempSymbolStmt
        , Statement
        >;

    std::map<int, std::string> const &temps;
    std::map<std::string, int> const &tempdims;
    std::vector<std::pair<std::string, int>> &symbols;

    AppendNewSymbols(
        std::map<int, std::string> const &temps,
        std::map<std::string, int> const &tempdims,
        std::vector<std::pair<std::string, int>> &symbols)
        : temps(temps), tempdims(tempdims), symbols(symbols)
    {}

    std::unique_ptr<IR> ir = std::make_unique<IR>();

    void visit(TempSymbolStmt *stmt) {
        auto name = temps.at(stmt->tmpid);
        if (auto it = tempdims.find(name); it != tempdims.end()) {
            auto dim = it->second;
            std::vector<int> symids;
            for (int i = 0; i < dim; i++) {
                int symid = symbols.size();
                symids.push_back(symid);
                symbols.emplace_back(name, i);
                //printf("%s.%d\n", name.c_str(), i);
            }
            auto new_stmt = ir->emplace_back<SymbolStmt>(symids);
            new_stmt->dim = stmt->dim;
            ir->mark_replacement(stmt, new_stmt);

        } else {
            ir->push_clone_back(stmt);
        }
    }

    void visit(Statement *stmt) {
        ir->push_clone_back(stmt);
    }
};

std::map<std::string, int> apply_detect_new_symbols(IR *ir,
        std::map<int, std::string> const &temps,
        std::vector<std::pair<std::string, int>> &symbols) {
    DetectNewSymbols detector(temps);
    detector.apply(ir);
    AppendNewSymbols visitor(temps, detector.tempdims, symbols);
    visitor.apply(ir);
    *ir = *visitor.ir;
    return detector.tempdims;
}

}
