#include "template_literal"

using namespace zeno::reflect;

const char* text::RTTI =
    #include "RTTI.inja"
;

const char* text::GENERATED_TEMPLATE_HEADER_TEMPLATE =
    #include "generated_template_header.inja"
;

const char* text::REFLECTED_TYPE_REGISTER =
    #include "reflected_type_register.inja"
;

const char* text::REFLECTED_METADATA = 
    #include "metadata.inja"
;
