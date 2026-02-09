file(GLOB TARGET_SOURCES CONFIGURE_DEPENDS *.cpp)

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

        target_compile_options(${target_name}
            PRIVATE
                $<$<CONFIG:Debug>:-g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer>
                $<$<CONFIG:Release>:-O3 -march=native -ffast-math>
                $<$<CONFIG:RelWithDebInfo>:-O3 -march=native>
        )

        target_link_options(${target_name}
            PRIVATE
                $<$<CONFIG:Debug>:-fsanitize=address,undefined>
        )
    endforeach()
endif()