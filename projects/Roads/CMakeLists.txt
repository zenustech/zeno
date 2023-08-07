CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

if (NOT ZENO_WITH_RPC)
    message(SEND_ERROR "Add flag -DZENO_WITH_RPC=ON ")
    message(FATAL_ERROR "You must enable RPC to use road system.")
endif ()
