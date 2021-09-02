# execute the test command that was added earlier.
execute_process( COMMAND "${TEST}" 
                 OUTPUT_FILE "${OUTPUT}"
		 RESULT_VARIABLE RET )
file(APPEND ${ALL_OUTPUT} ${HEADING})
file(APPEND ${ALL_OUTPUT} "\n")
file(READ "${OUTPUT}" SINGLE_OUTPUT)
file(APPEND ${ALL_OUTPUT} "${SINGLE_OUTPUT}\n")
file(REMOVE ${OUTPUT})   # remove the individual output file.
