add_executable(prima-stage
    main.cpp
    ../../src/prima-distributed-protocol.cpp
    ../../src/prima-distributed-transport.cpp
)

target_include_directories(prima-stage PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

target_link_libraries(prima-stage PRIVATE llama)
