file(GLOB HVLIB_SOURCES CONFIGURE_DEPENDS "*.cpp")
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

target_compile_options(hvlib
    PRIVATE
        $<$<CONFIG:Debug>:-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer>
        $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
        $<$<CONFIG:RelWithDebInfo>:-O3 -march=native>
)

target_link_options(hvlib
    PRIVATE
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)