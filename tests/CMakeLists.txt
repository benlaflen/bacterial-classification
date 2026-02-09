file(GLOB TEST_SOURCES CONFIGURE_DEPENDS *.cpp)

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
                $<$<CONFIG:Debug>:-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer>
                $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
                $<$<CONFIG:RelWithDebInfo>:-O3 -march=native>
        )
    endforeach()
endif()