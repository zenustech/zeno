file(GLOB ABC_SOURCE *.cpp *.h)

target_sources(zeno PRIVATE ${ABC_SOURCE})

target_include_directories(zeno PRIVATE .)

if (ZENO_SYSTEM_ALEMBIC)
    find_package(Alembic REQUIRED)
    target_link_libraries(zeno PRIVATE Alembic::Alembic)
else()
    add_subdirectory(Alembic)
    target_link_libraries(zeno PRIVATE Alembic)
endif()
