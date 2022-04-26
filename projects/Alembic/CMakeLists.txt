file(GLOB ABC_SOURCE *.cpp *.h)

target_sources(zeno PRIVATE ${ABC_SOURCE})

target_include_directories(zeno PRIVATE .)

option(ZENO_SYSTEM_ALEMBIC "Use Alembic found in system instead of submodule" OFF)
if (ZENO_SYSTEM_ALEMBIC)
    find_package(Alembic REQUIRED)
    target_link_libraries(zeno PRIVATE Alembic::Alembic)
else()
    if (NOT EXISTS Alembic/CMakeLists.txt)
        message(FATAL_ERROR "Alembic submodule not found! Please run: git submodule update --init --recursive")
    endif()
    add_subdirectory(Alembic)
    target_link_libraries(zeno PRIVATE Alembic)
endif()
