add_executable(potluck-worker
    main.cpp
)

target_include_directories(potluck-worker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${PROJECT_SOURCE_DIR}/common
)
target_compile_features(potluck-worker PRIVATE cxx_std_17)

target_link_libraries(potluck-worker PRIVATE potluck-distributed llama)
