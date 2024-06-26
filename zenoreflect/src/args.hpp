#pragma once

#include "argparse.hpp"

struct ControlFlags : public argparse::Args {
    std::vector<std::string>& input_sources = kwarg("input_source,S", "Source file path passing to generator");
    std::string& output_dir = kwarg("header_output,o", "Output directory");
    std::string& target_type_register_source_path = kwarg("generated_source_path", "Path to target source contains generate type static register");
    std::string& cpp_version = kwarg("stdc++", "Set cpp standard (default: 17)").set_default("17");
    bool& verbose = flag("v,verbose", "Print extra information");
    std::vector<std::string>& include_dirs = kwarg("I,include_dirs", "Include directories").multi_argument().set_default(std::vector<std::string>{});
    std::vector<std::string>& pre_include_headers = kwarg("H,pre_include_header", "Automatic place those headers in all sources").set_default(std::vector<std::string>{});
};

ControlFlags parse_args(int argc, char** argv);

extern ControlFlags* GLOBAL_CONTROL_FLAGS;
