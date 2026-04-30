file(GLOB HVLIB_SOURCES CONFIGURE_DEPENDS "*.cpp" "*.cu")
add_library(hvlib ${HVLIB_SOURCES})

target_include_directories(hvlib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(hvlib
    PUBLIC
        $<$<BOOL:${NORM_COSINE}>:NORM_COSINE>
)

target_compile_features(hvlib
    PUBLIC
        cxx_std_20
)

set_target_properties(hvlib PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(hvlib
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Debug>:-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Release>:-O3 -march=native -ffast-math>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:RelWithDebInfo>:-O3 -march=native>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Profile>:-O0 -g -fno-inline -fno-omit-frame-pointer>>
        $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>:-O3 --use_fast_math>>
)
target_link_libraries(hvlib
    PRIVATE CUDA::cudart
)
set_target_properties(hvlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_options(hvlib
    PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)