function(zeno_prepare_sharc_cuda_headers sharc_include_dir output_dir output_files_var)
    set(_required_headers
        HashGridCommon.h
        HashGridTypes.h
        SharcCommon.h
        SharcTypes.h)

    foreach(_header IN LISTS _required_headers)
        if(NOT EXISTS "${sharc_include_dir}/${_header}")
            message(FATAL_ERROR
                "The required SHARC SDK header is missing: "
                "${sharc_include_dir}/${_header}")
        endif()
    endforeach()

    file(READ "${sharc_include_dir}/SharcCommon.h" _sharc_common_original)
    foreach(_component IN ITEMS MAJOR MINOR BUILD REVISION)
        string(REGEX MATCH
            "#define[ \t]+SHARC_VERSION_${_component}[ \t]+([0-9]+)"
            _version_match "${_sharc_common_original}")
        if(NOT _version_match)
            message(FATAL_ERROR
                "Unable to read SHARC_VERSION_${_component} from SharcCommon.h")
        endif()
        set(_version_${_component} "${CMAKE_MATCH_1}")
    endforeach()
    if(NOT _version_MAJOR EQUAL 1 OR
       NOT _version_MINOR EQUAL 8 OR
       NOT _version_BUILD EQUAL 3 OR
       NOT _version_REVISION EQUAL 0)
        message(FATAL_ERROR
            "The CUDA compatibility transform is pinned to SHARC 1.8.3.0; "
            "found ${_version_MAJOR}.${_version_MINOR}.${_version_BUILD}."
            "${_version_REVISION}")
    endif()

    file(MAKE_DIRECTORY "${output_dir}")

    # The SDK remains untouched. These build-tree copies only translate HLSL
    # parameter directions and vector swizzles into CUDA-compatible forms.
    foreach(_header IN ITEMS HashGridCommon.h SharcCommon.h)
        file(READ "${sharc_include_dir}/${_header}" _cuda_header)

        string(REPLACE "inout SharcState sharcState"
                       "SharcState& sharcState" _cuda_header "${_cuda_header}")

        foreach(_out_parameter IN ITEMS
                "float;voxelSize"
                "HASH_GRID_KEY_TYPE;originalValue"
                "HashGridIndex;cacheIndex"
                "uint;bucketOffset"
                "HASH_GRID_KEY_TYPE;hashKey"
                "int;responsiveIndexOffset"
                "bool;isNewSample"
                "float3;radiance")
            string(REPLACE ";" " " _out_signature "${_out_parameter}")
            string(REPLACE ";" "& " _out_reference "${_out_parameter}")
            string(REPLACE "out ${_out_signature}" "${_out_reference}"
                           _cuda_header "${_cuda_header}")
        endforeach()

        foreach(_in_type IN ITEMS
                "HashGrid_(Data)"
                "uint"
                "HASH_GRID_KEY_TYPE"
                "SharcParameters"
                "SharcState"
                "SharcHitData")
            string(REPLACE "in ${_in_type} " "${_in_type} "
                           _cuda_header "${_cuda_header}")
        endforeach()

        # Preserve the narrowing swizzles used by directional SH encoding
        # before removing the identity swizzles that CUDA vector types lack.
        string(REPLACE "ycocg.yz"
                       "make_float2(ycocg.y, ycocg.z)"
                       _cuda_header "${_cuda_header}")
        string(REPLACE "radianceData.luminanceSH.xyz"
                       "make_float3(radianceData.luminanceSH)"
                       _cuda_header "${_cuda_header}")
        string(REPLACE "accumulatedData.dataExt.xy"
                       "make_int2(accumulatedData.dataExt.x, accumulatedData.dataExt.y)"
                       _cuda_header "${_cuda_header}")
        string(REPLACE "radianceAndSampleNum.xyz"
                       "make_float3(radianceAndSampleNum)"
                       _cuda_header "${_cuda_header}")
        string(REPLACE ".xyz" "" _cuda_header "${_cuda_header}")
        string(REPLACE ".xy" "" _cuda_header "${_cuda_header}")
        string(REPLACE ".yz" "" _cuda_header "${_cuda_header}")

        # SHARC is compiled into both OptiX raygen modules. Its HLSL headers
        # define functions with external linkage, which would otherwise make
        # the OptiX pipeline linker see every helper twice. Keep those helper
        # definitions private to each generated CUDA translation unit.
        string(REGEX REPLACE
            "(\n)([A-Za-z_][A-Za-z0-9_]*)([ \t]+)(HashGrid_\\(|Sharc[A-Za-z0-9_]+\\()"
            "\\1static __forceinline__ __device__ \\2\\3\\4"
            _cuda_header "${_cuda_header}")
        string(REPLACE
            "static __forceinline__ __device__ struct HashGrid_("
            "struct HashGrid_(" _cuda_header "${_cuda_header}")

        string(REGEX MATCH
            "[(,][ \t\r\n]*(in|out|inout)[ \t]+"
            _untranslated_qualifier "${_cuda_header}")
        if(_untranslated_qualifier)
            message(FATAL_ERROR
                "Untranslated HLSL parameter qualifier in ${_header}: "
                "${_untranslated_qualifier}")
        endif()

        file(WRITE "${output_dir}/${_header}"
            "// Generated from unmodified SHARC 1.8.3 ${_header}.\n${_cuda_header}")
    endforeach()

    foreach(_header IN ITEMS HashGridTypes.h SharcTypes.h)
        configure_file(
            "${sharc_include_dir}/${_header}"
            "${output_dir}/${_header}"
            COPYONLY)
    endforeach()

    set(${output_files_var}
        "${output_dir}/HashGridCommon.h"
        "${output_dir}/HashGridTypes.h"
        "${output_dir}/SharcCommon.h"
        "${output_dir}/SharcTypes.h"
        PARENT_SCOPE)
endfunction()
