set(sources "")
set(SCATGEN
    slatm1.c
    slaran.c
    slarnd.c
    slaruv.c
    slabad.c
    slarnv.c
)
set(DZATGEN
    dlatm1.c
    dlaran.c
    dlarnd.c
    dlaruv.c
    dlabad.c
    dlarnv.c
)

if(enable_single)
  list(APPEND sources ${SCATGEN})
  list(APPEND sources
    slatb4.c
    slaset.c
    slartg.c
    slatms.c
    slagge.c
    slagsy.c
    slarge.c
    slaror.c
    slarot.c
    slatm2.c
    slatm3.c
    r_lg10.c
    r_sign.c
    pow_dd.c
    pow_ri.c
  )
endif()

if(enable_double)
  list(APPEND sources ${DZATGEN})
  list(APPEND sources
    dlatb4.c
    dlaset.c
    dlartg.c
    dlatms.c
    dlagge.c
    dlagsy.c
    dlarge.c
    dlaror.c
    dlarot.c
    dlatm2.c
    dlatm3.c
    d_lg10.c
    d_sign.c
    pow_dd.c
    pow_di.c
  )
endif()

if(enable_complex)
  list(APPEND sources ${SCATGEN})
  list(APPEND sources
    clatb4.c
    claset.c
    clartg.c
    clarnv.c
    clacgv.c
    csymv.c
    clatms.c
    clagge.c
    clagsy.c
    clarge.c
    claror.c
    clarot.c
    clatm2.c
    clatm3.c
    claghe.c
    clarnd.c
#    cdotc.c
    r_lg10.c
    r_sign.c
    pow_dd.c
    pow_ri.c
)
endif()

if(enable_complex16)
  list(APPEND sources ${DZATGEN})
  list(APPEND sources
    zlatb4.c
    zlaset.c
    zlartg.c
    zlarnv.c
    zlacgv.c
    zsymv.c
    zlatms.c
    zlagge.c
    zlagsy.c
    zlarge.c
    zlaror.c
    zlarot.c
    zlatm2.c
    zlatm3.c
    zlaghe.c
    zlarnd.c
#    zdotc.c
    d_lg10.c
    d_sign.c
    pow_dd.c
    pow_di.c
  )
endif()

add_library(matgen ${sources})
target_link_libraries(matgen superlu)
