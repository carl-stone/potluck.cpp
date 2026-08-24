if (NOT LLAMA_BUILD_COMMON)
    message(WARNING "potluck-server requires LLAMA_BUILD_COMMON=ON; skipping")
    return()
endif()

add_executable(potluck-server
    main.cpp
    http_api.cpp)

target_include_directories(potluck-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/vendor
)
target_compile_features(potluck-server PRIVATE cxx_std_17)

target_link_libraries(potluck-server PRIVATE potluck-core)
