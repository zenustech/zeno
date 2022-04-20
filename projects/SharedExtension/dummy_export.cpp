namespace zeno {

// prevent the linker from optimizing out linkage against zenosharedext.so
#ifdef _MSC_VER
__declspec(dllexport)
#endif
int _zenosharedext_link_what_you_use();
int _zenosharedext_link_what_you_use() { return 1; }

}
