find_package(Qt5 REQUIRED COMPONENTS Widgets)

file(GLOB_RECURSE source CONFIGURE_DEPENDS *.h *.cpp *.ui *.qrc)



add_library(zenoio OBJECT ${source})

set_property(TARGET zenoio PROPERTY AUTOUIC ON)
set_property(TARGET zenoio PROPERTY AUTOMOC ON)
set_property(TARGET zenoio PROPERTY AUTORCC ON)

target_link_libraries(zenoio PUBLIC Qt5::Widgets glad zeno)
target_include_directories(zenoio
	PUBLIC .
	PRIVATE ../3rdparty
	PRIVATE ../zenoui
	PRIVATE ${Qt5Widgets_PRIVATE_INCLUDE_DIRS}
	PRIVATE ../zeno/zeno
)
