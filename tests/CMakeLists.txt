file(GLOB TEST_SOURCES CONFIGURE_DEPENDS *.cpp "*.cu")

if (TEST_SOURCES)
    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)

        add_executable(${test_name} ${test_src})

        target_link_libraries(${test_name}
            PRIVATE hvlib
        )

        target_compile_features(${test_name}
            PRIVATE cxx_std_20
        )

        target_compile_options(${test_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Debug>:-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer>>
                $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Release>:-O3 -march=native -ffast-math>>
                $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:RelWithDebInfo>:-O3 -march=native>>
                $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Profile>:-O0 -g -fno-inline -fno-omit-frame-pointer>>
                $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>:-O3 --use_fast_math>>
        )
        target_link_libraries(${test_name}
            PRIVATE CUDA::cudart
        )
        set_target_properties(${test_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_link_options(${test_name}
            PRIVATE
                $<$<CONFIG:Debug>:-fsanitize=address,undefined>
        )
    endforeach()
endif()