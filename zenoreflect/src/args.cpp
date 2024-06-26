#include "args.hpp"

ControlFlags parse_args(int argc, char* argv[]) {
    return argparse::parse<ControlFlags>(argc, argv);
}

ControlFlags* GLOBAL_CONTROL_FLAGS = nullptr;
