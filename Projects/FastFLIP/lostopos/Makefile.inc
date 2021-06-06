
# Source files
LIB_SRC = accelerationgrid.cpp broadphasegrid.cpp collisionpipeline.cpp \
          dynamicsurface.cpp edgecollapser.cpp edgeflipper.cpp edgesplitter.cpp \
          eltopo.cpp impactzonesolver.cpp meshmerger.cpp meshpincher.cpp meshsmoother.cpp \
          meshrenderer.cpp nondestructivetrimesh.cpp subdivisionscheme.cpp surftrack.cpp \
          trianglequality.cpp \

# Common
LIB_SRC += ../common/collisionqueries.cpp ../common/wallclocktime.cpp \
           ../common/newsparse/sparse_matrix.cpp \
           ../common/newsparse/krylov_solvers.cpp \

LIB_SRC += ../common/tunicate/expansion.cpp ../common/tunicate/intersection.cpp ../common/tunicate/neg.cpp \
           ../common/tunicate/orientation.cpp 

LIB_SRC += ../common/ccd_wrapper.cpp ../common/cubic_ccd_wrapper.cpp ../common/collisionqueries.cpp

LIB_SRC += ../common/lexer.cpp ../common/newparser.cpp 

# object files
LIB_RELEASE_OBJ = $(patsubst %.cpp,obj/%.o,$(notdir $(LIB_SRC)))
LIB_DEBUG_OBJ = $(patsubst %.cpp,obj_debug/%.o,$(notdir $(LIB_SRC)))

# Build the library
$(LIBRARY)_release.a: $(LIB_RELEASE_OBJ)
	ar r $@ $(LIB_RELEASE_OBJ)
	ranlib $@
		  
$(LIBRARY)_debug.a: $(LIB_DEBUG_OBJ)
	ar r $@ $(LIB_DEBUG_OBJ)
	ranlib $@

