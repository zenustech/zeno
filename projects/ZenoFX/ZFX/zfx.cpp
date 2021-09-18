#include "LowerAST.h"
#include "Visitors.h"
#include <zfx/zfx.h>

namespace zfx {

std::tuple
    < std::string
    , std::vector<std::pair<std::string, int>>
    , std::vector<std::pair<std::string, int>>
    , std::map<std::string, int>
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
        , temporaries
        ] = lower_ast
        ( std::move(asts)
        , options.symdims
        , options.pardims
        );
#ifdef ZFX_PRINT_IR
    ir->print();
#endif

#ifdef ZFX_PRINT_IR
    cout << "=== ControlCheck" << endl;
#endif
    apply_control_check(ir.get());
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

    std::map<std::string, int> newsyms;
    if (options.detect_new_symbols) {
#ifdef ZFX_PRINT_IR
        cout << "=== DetectNewSymbols" << endl;
#endif
        newsyms = apply_detect_new_symbols(
                ir.get(), temporaries, symbols);
#ifdef ZFX_PRINT_IR
        ir->print();
#endif
    }

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

    if (options.demote_math_funcs) {
#ifdef ZFX_PRINT_IR
        cout << "=== DemoteMathFuncs" << endl;
#endif
        ir = apply_demote_math_funcs(ir.get());
#ifdef ZFX_PRINT_IR
        ir->print();
#endif
    }

#ifdef ZFX_PRINT_IR
    cout << "=== LowerAccess" << endl;
#endif
    ir = apply_lower_access(ir.get());
#ifdef ZFX_PRINT_IR
    ir->print();
#endif

    if (options.constant_fold) {
#ifdef ZFX_PRINT_IR
        cout << "=== ConstantFold" << endl;
#endif
        ir = apply_constant_fold(ir.get());
#ifdef ZFX_PRINT_IR
        ir->print();
#endif
    }

    if (options.kill_unreachable) {
#ifdef ZFX_PRINT_IR
        cout << "=== KillUnreachable" << endl;
#endif
        ir = apply_kill_unreachable(ir.get());
#ifdef ZFX_PRINT_IR
        ir->print();
#endif
    }

    if (options.reassign_parameters) {
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
        params = new_params;
    }

    std::stringstream oss_end;
    if (options.const_parametrize) {
#ifdef ZFX_PRINT_IR
        cout << "=== ConstParametrize" << endl;
#endif
        auto constants = apply_const_parametrize(ir.get());
#ifdef ZFX_PRINT_IR
        ir->print();
#endif
        for (auto const &[idx, value]: constants) {
            oss_end << "const " << idx << " " << value << "\n";
        }
    }

    if (options.arch_maxregs != 0) {
#ifdef ZFX_PRINT_IR
        cout << "=== RegisterAllocation" << endl;
#endif
        int memsize = apply_register_allocation(ir.get(),
                options.arch_maxregs);
#ifdef ZFX_PRINT_IR
        ir->print();
#endif

        if (options.save_math_registers) {
#ifdef ZFX_PRINT_IR
            cout << "=== SaveMathRegisters" << endl;
#endif
            ir = apply_save_math_registers(ir.get(),
                    options.arch_maxregs, memsize);
#ifdef ZFX_PRINT_IR
            ir->print();
#endif
        }
    }

    if (options.kill_unreachable) {
#ifdef ZFX_PRINT_IR
        cout << "=== KillUnreachable" << endl;
#endif
        ir = apply_kill_unreachable(ir.get());
#ifdef ZFX_PRINT_IR
        ir->print();
#endif
    }

    if (options.merge_identical) {
#ifdef ZFX_PRINT_IR
        cout << "=== MergeIdentical" << endl;
#endif
        ir = apply_merge_identical(ir.get());
#ifdef ZFX_PRINT_IR
        ir->print();
#endif
    }

    if (options.reassign_channels) {
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
        symbols = new_symbols;
    }

    if (options.global_localize) {
#ifdef ZFX_PRINT_IR
        cout << "=== GlobalLocalize" << endl;
#endif
        apply_global_localize(ir.get());
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
        , symbols
        , params
        , newsyms
        };
}

}
