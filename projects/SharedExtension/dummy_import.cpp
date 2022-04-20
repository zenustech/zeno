namespace zeno {

// prevent the linker from optimizing out linkage against zenosharedext.so
#ifdef _MSC_VER
__declspec(dllimport)
#endif
int _zenosharedext_link_what_you_use();

static int _zenosharedext_dummy_var = _zenosharedext_link_what_you_use();

}
