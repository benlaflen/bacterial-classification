file(GLOB TARGET_SOURCES CONFIGURE_DEPENDS *.cpp "*.cu")

if(TARGET_SOURCES)
    foreach(target_src ${TARGET_SOURCES})
        get_filename_component(target_name ${target_src} NAME_WE)

        add_executable(${target_name} ${target_src})

        target_link_libraries(${target_name}
            PRIVATE hvlib
        )

        target_compile_features(${target_name}
            PRIVATE cxx_std_20
        )

        set_target_properties(${target_name} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )

        target_compile_options(${target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Debug>:-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer>>
                $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Release>:-O3 -march=native -ffast-math>>
                $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:RelWithDebInfo>:-O3 -march=native>>
                $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Profile>:-O0 -g -fno-inline -fno-omit-frame-pointer>>
                $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>:-O3 --use_fast_math>>
                $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Profile>:-lineinfo>>
        )
        target_link_libraries(${target_name}
            PRIVATE CUDA::cudart
        )
        set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        target_link_options(${target_name}
            PRIVATE
                $<$<CONFIG:Debug>:-fsanitize=address,undefined>
        )
    endforeach()
endif()