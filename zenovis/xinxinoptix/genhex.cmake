if (NOT DEFINED FILE_PATH)
    message(FATAL_ERROR "FILE_PATH not defined")
endif()
if (NOT DEFINED FILE_NAME)
    message(FATAL_ERROR "FILE_NAME not defined")
endif()
if (NOT DEFINED DUMMY_SOURCE_FILE)
    message(FATAL_ERROR "DUMMY_SOURCE_FILE not defined")
endif()

message(STATUS "Processing NVRTC header: ${FILE_PATH}")
file(READ ${FILE_PATH} HEX_CONTENTS HEX)

string(REGEX MATCHALL "([A-Za-z0-9][A-Za-z0-9])" SEPARATED_HEX "${HEX_CONTENTS}")
list(JOIN SEPARATED_HEX ",\n0x" FORMATTED_HEX)
string(PREPEND FORMATTED_HEX "0x")

file(WRITE ${DUMMY_SOURCE_FILE} "/* generated from: ${FILE_PATH} */\n#include <vector>\nnamespace sutil {\nstd::vector<const char *> &getIncFileTab();\nstd::vector<const char *> &getIncPathTab();\nstatic const unsigned char mydata[] = {\n${FORMATTED_HEX},\n0};\nstatic int helper = (getIncPathTab().push_back(\"${FILE_NAME}\"), getIncFileTab().push_back((const char *)mydata), 0);\n}")
