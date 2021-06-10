# --------------------
#
# Makefile for El Topo.
#
# Create a file called Makefile.local_defs before running make.  See Makefile.example_defs for suggestions.
#
# --------------------

# This is for GNU make; other versions of make may not run correctly.

# Local machine settings
include Makefile.local_defs

# --------------------
# El Topo library
# --------------------

LIBRARY = libeltopo
include Makefile.inc

# how to compile each file
.SUFFIXES:
obj/%.o:
	$(CC) -c $(RELEASE_FLAGS) $(INCLUDE_PATH) -o $@ $<
obj_debug/%.o:
	$(CC) -c $(DEBUG_FLAGS) $(INCLUDE_PATH) -o $@ $<

.PHONY: release
release: $(LIBRARY)_release.a 

.PHONY: debug
debug: $(LIBRARY)_debug.a 

# cleaning up
.PHONY: clean
clean:
	-rm -f obj/*.o obj/depend $(LIBRARY)_debug.a obj_debug/*.o obj_debug/depend $(LIBRARY)_release.a *core

# dependencies are automatically generated
.PHONY: depend
depend:
	-mkdir obj
	-rm -f obj/depend
	$(foreach srcfile,$(LIB_SRC),$(DEPEND) -MM $(srcfile) -MT $(patsubst %.cpp,obj/%.o,$(notdir $(srcfile))) $(INCLUDE_PATH) >> obj/depend;)
	-mkdir obj_debug
	-rm -f obj_debug/depend
	$(foreach srcfile,$(LIB_SRC),$(DEPEND) -MM $(srcfile) -MT $(patsubst %.cpp,obj_debug/%.o,$(notdir $(srcfile))) $(INCLUDE_PATH) >> obj_debug/depend;)

-include obj/depend
-include obj_debug/depend
