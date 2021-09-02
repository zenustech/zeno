include_directories(${SuperLU_SOURCE_DIR}/SRC)

# Libs linked to all of the tests
set(test_link_libs matgen)

add_subdirectory(MATGEN)

set(ALINTST sp_ienv.c)
set(NVAL   9 19)
set(NRHS   2)
set(LWORK  0 10000000)

function(cat IN_FILE OUT_FILE)
  file(READ ${IN_FILE} CONTENTS)
  file(APPEND ${OUT_FILE} "${CONTENTS}")
endfunction()


function(add_superlu_test output input target)
  set(TEST_INPUT "${SuperLU_SOURCE_DIR}/EXAMPLE/${input}")
  set(TEST_OUTPUT "${SuperLU_BINARY_DIR}/TESTING/${output}")

  # Prepare a temporary file to "cat" to:
  # "== LAPACK test matrices"
  file(WRITE ${TEST_OUTPUT} "")

##  get_target_property(TEST_LOC ${target} LOCATION)
  set(TEST_LOC ${CMAKE_CURRENT_BINARY_DIR})

  foreach (n ${NVAL})
    foreach (s ${NRHS})
      foreach(l ${LWORK})
          set(testName "${target}_${n}_${s}_${l}")
	  set(SINGLE_OUTPUT ${SuperLU_BINARY_DIR}/TESTING/${testName}.out)
          add_test( ${testName}_LA  "${CMAKE_COMMAND}"
	    -DTEST=${TEST_LOC} -t "LA" -n ${n} -s ${s} -l ${l}
	    -DOUTPUT=${SINGLE_OUTPUT}
	    -DALL_OUTPUT=${TEST_OUTPUT}
	    -DHEADING=Dense\ matrix\ --\ n=${n},\ s=${s},\ lwork=${l}
	    -P "${SuperLU_SOURCE_DIR}/TESTING/runtest.cmake"
	  )
      endforeach()
    endforeach()
  endforeach()

#  file(APPEND ${TEST_OUTPUT} "== sparse matrix\n")

  foreach (s ${NRHS})
      foreach(l ${LWORK})
          set(testName "${target}_${s}_${l}")
	  set(SINGLE_OUTPUT ${SuperLU_BINARY_DIR}/TESTING/${testName}.out)
          add_test( ${testName}_SP  "${CMAKE_COMMAND}"
	    -DTEST=${TEST_LOC} -t "SP" -s ${s} -l ${l} -f ${TEST_INPUT}
	    -DOUTPUT=${SINGLE_OUTPUT}
	    -DALL_OUTPUT=${TEST_OUTPUT}
	    -DHEADING=Sparse\ matrix\ ${TEST_INPUT}\ --\ s=${s},\ lwork=${l}
	    -P "${SuperLU_SOURCE_DIR}/TESTING/runtest.cmake")
      endforeach()
  endforeach()

endfunction(add_superlu_test)


if(enable_single)
  set(SLINTST sdrive.c sp_sconvert.c sgst01.c sgst02.c sgst04.c sgst07.c)

  add_executable(s_test ${ALINTST} ${SLINTST})
  target_link_libraries(s_test ${test_link_libs})

  add_superlu_test(s_test.out g20.rua s_test)

endif()


if(enable_double)
  set(DLINTST ddrive.c sp_dconvert.c dgst01.c dgst02.c dgst04.c dgst07.c)

  add_executable(d_test ${ALINTST} ${DLINTST})
  target_link_libraries(d_test ${test_link_libs})

  add_superlu_test(d_test.out g20.rua d_test)
endif()

if(enable_complex)
  set(CLINTST cdrive.c sp_cconvert.c cgst01.c cgst02.c cgst04.c cgst07.c)

  add_executable(c_test ${ALINTST} ${CLINTST})
  target_link_libraries(c_test ${test_link_libs})

  add_superlu_test(c_test.out cg20.cua c_test)
endif()


if(enable_complex16)
  set(ZLINTST zdrive.c sp_zconvert.c zgst01.c zgst02.c zgst04.c zgst07.c)

  add_executable(z_test ${ALINTST} ${ZLINTST})
  target_link_libraries(z_test ${test_link_libs})

  add_superlu_test(z_test.out cg20.cua z_test)
endif()
