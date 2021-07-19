#include "LowerAST.h"
#include "Visitors.h"
#include <zfx/zfx.h>

namespace zfx {

std::tuple
    < std::string
    , std::vector<std::pair<std::string, int>>
    , std::vector<std::pair<std::string, int>>
    > compile_to_assembly
    ( std::string const &code
    , Options const &options
    ) {
#ifdef ZFX_PRINT_IR
    cout << "=== ZFX" << endl;
    cout << code << endl;
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== Parse" << endl;
#endif
    auto asts = parse(code);
#ifdef ZFX_PRINT_IR
    for (auto const &a: asts) {
        a->print();
        cout << endl;
    }
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== LowerAST" << endl;
#endif
    auto
        [ ir
        , symbols
        , params
        ] = lower_ast
        ( std::move(asts)
        , options.symdims
        , options.pardims
        );
#ifdef ZFX_PRINT_IR
    ir->print();
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== SymbolCheck" << endl;
#endif
    apply_symbol_check(ir.get());
#ifdef ZFX_PRINT_IR
    ir->print();
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== TypeCheck" << endl;
#endif
    apply_type_check(ir.get());
#ifdef ZFX_PRINT_IR
    ir->print();
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== ExpandFunctions" << endl;
#endif
    ir = apply_expand_functions(ir.get());
#ifdef ZFX_PRINT_IR
    ir->print();
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== TypeCheck" << endl;
#endif
    apply_type_check(ir.get());
#ifdef ZFX_PRINT_IR
    ir->print();
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== LowerMath" << endl;
#endif
    ir = apply_lower_math(ir.get());
#ifdef ZFX_PRINT_IR
    ir->print();
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== LowerAccess" << endl;
#endif
    ir = apply_lower_access(ir.get());
#ifdef ZFX_PRINT_IR
    ir->print();
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== ReassignParameters" << endl;
#endif
    auto uniforms = apply_reassign_parameters(ir.get());
#ifdef ZFX_PRINT_IR
    ir->print();
#endif
    std::vector<std::pair<std::string, int>> new_params;
    for (int i = 0; i < params.size(); i++) {
        auto it = uniforms.find(i);
        if (it == uniforms.end())
            continue;
        auto dst = it->second;
        if (new_params.size() < dst + 1)
            new_params.resize(dst + 1);
        new_params[dst] = params[i];
    }

    std::ostringstream oss_end;
    if (options.const_parametrize) {
#ifdef ZFX_PRINT_IR
        cout << "=== ConstParametrize" << endl;
#endif
        auto constants = apply_const_parametrize(ir.get(), uniforms.size());
#ifdef ZFX_PRINT_IR
        ir->print();
#endif
        for (auto const &[idx, expr]: constants) {
            oss_end << "const " << idx << " " << expr << "\n";
        }
    }

    if (options.arch_maxregs != 0) {
#ifdef ZFX_PRINT_IR
        cout << "=== RegisterAllocation" << endl;
#endif
        apply_register_allocation(ir.get(), options.arch_maxregs);
#ifdef ZFX_PRINT_IR
        ir->print();
#endif
    }

    // TODO: if (options.reassign_globals)
#ifdef ZFX_PRINT_IR
    cout << "=== ReassignGlobals" << endl;
#endif
    auto globals = apply_reassign_globals(ir.get());
#ifdef ZFX_PRINT_IR
    ir->print();
#endif
    std::vector<std::pair<std::string, int>> new_symbols;
    for (int i = 0; i < symbols.size(); i++) {
        auto it = globals.find(i);
        if (it == globals.end())
            continue;
        auto dst = it->second;
        if (new_symbols.size() < dst + 1)
            new_symbols.resize(dst + 1);
        new_symbols[dst] = symbols[i];
    }

    if (options.global_localize) {
#ifdef ZFX_PRINT_IR
        cout << "=== GlobalLocalize" << endl;
#endif
        apply_global_localize(ir.get(), globals.size());
#ifdef ZFX_PRINT_IR
        ir->print();
#endif
    }

#ifdef ZFX_PRINT_IR
    cout << "=== EmitAssembly" << endl;
#endif
    auto assem = apply_emit_assembly(ir.get());
    assem = oss_end.str() + assem;
#ifdef ZFX_PRINT_IR
    cout << assem;
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== Assemble" << endl;
#endif
    return
        { assem
        , new_symbols
        , new_params
        };
}

}
