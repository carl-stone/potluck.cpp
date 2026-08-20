add_executable(potluck-worker
    main.cpp
    ../../src/potluck-protocol.cpp
    ../../src/potluck-transport.cpp
)

target_include_directories(potluck-worker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${PROJECT_SOURCE_DIR}/common
)

target_link_libraries(potluck-worker PRIVATE llama)
