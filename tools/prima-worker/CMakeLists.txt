add_executable(prima-worker
    main.cpp
    ../../src/prima-distributed-protocol.cpp
    ../../src/prima-distributed-transport.cpp
)

target_include_directories(prima-worker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../prima-stage
)

target_link_libraries(prima-worker PRIVATE llama)
