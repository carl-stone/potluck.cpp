if (NOT LLAMA_BUILD_COMMON OR NOT TARGET potluck-core)
    message(WARNING "potluck-cli requires LLAMA_BUILD_COMMON=ON; skipping")
    return()
endif()

add_executable(potluck-cli main.cpp)
target_include_directories(potluck-cli PRIVATE
    ${PROJECT_SOURCE_DIR}/tools/potluck-server
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/vendor)
target_compile_features(potluck-cli PRIVATE cxx_std_17)
target_link_libraries(potluck-cli PRIVATE potluck-core)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS potluck-cli RUNTIME)
endif()
