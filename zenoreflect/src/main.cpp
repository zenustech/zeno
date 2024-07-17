#include "args.hpp"
#include "utils.hpp"
#include "codegen.hpp"
#include "parser.hpp"

int main(int argc, char* argv[]) {
    ControlFlags flags = parse_args(argc, argv);
    GLOBAL_CONTROL_FLAGS = &flags;

    ReflectionModel model{};
    pre_generate_reflection_model();

    int32_t result = 0;
    zeno::reflect::CodeCompilerState compiler_state {nullptr};
    for (const std::string& filepath : GLOBAL_CONTROL_FLAGS->input_sources) {
        std::optional<std::string> source_str = zeno::reflect::read_file(filepath);
        if (!source_str.has_value()) {
            std::cerr << std::format("Can't read source file {}", filepath) << std::endl;
            return 2;
        }
        std::string source = source_str.value();

        result += static_cast<int32_t>(generate_reflection_model({
            .identity_name = filepath,
            .source = source,
            .type = TranslationUnitType::Header,
        }, model, compiler_state));
    }

    post_generate_reflection_model(model, compiler_state);

    return result;
}
