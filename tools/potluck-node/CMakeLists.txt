set(TARGET potluck-node)
add_executable(${TARGET} main.cpp)

target_include_directories(${TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(${TARGET} PRIVATE potluck-discovery)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
