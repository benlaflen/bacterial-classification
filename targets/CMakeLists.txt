file(GLOB TARGET_SOURCES CONFIGURE_DEPENDS *.cpp)

foreach(target_src ${TARGET_SOURCES})
    get_filename_component(target_name ${target_src} NAME_WE)

    add_executable(${target_name} ${target_src})

    target_link_libraries(${target_name}
        PRIVATE hvlib
    )

    target_compile_features(${target_name}
        PRIVATE cxx_std_20
    )
endforeach()