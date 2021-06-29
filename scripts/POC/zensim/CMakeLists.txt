add_executable(zensim a.cpp)

install(TARGETS zensim DESTINATION bin)

set(CPACK_PACKAGE_NAME "zensim")
set(CPACK_SET_DESTDIR ON)
set(CPACK_INSTALL_PREFIX "/opt/zensim")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_RPM_PACKAGE_GROUP "zensim-dev")
set(CPACK_PACKAGE_VENDOR "zensim-dev")
set(CPACK_PACKAGE_LICENSE "Mozilla Public License 2.0")
include(CPack)
