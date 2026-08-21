# dependencies

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
if (NOT TARGET PkgConfig::ZMQ)
    pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)
endif()

if (NOT TARGET potluck-distributed)
    add_library(potluck-distributed STATIC
        ${PROJECT_SOURCE_DIR}/src/potluck-protocol.cpp
        ${PROJECT_SOURCE_DIR}/src/potluck-transport.cpp)
    target_include_directories(potluck-distributed PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(potluck-distributed PUBLIC PkgConfig::ZMQ)
endif()

if (NOT TARGET potluck-discovery)
    add_library(potluck-discovery STATIC
        ${PROJECT_SOURCE_DIR}/src/potluck-discovery.cpp
        ${PROJECT_SOURCE_DIR}/src/potluck-discovery.h)
    target_include_directories(potluck-discovery PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_compile_features(potluck-discovery PUBLIC cxx_std_17)
    if (APPLE)
        # Bonjour is part of libSystem on macOS.
    elseif (UNIX)
        if (NOT TARGET PkgConfig::AVAHI_COMPAT_LIBDNSSD)
            pkg_check_modules(AVAHI_COMPAT_LIBDNSSD REQUIRED IMPORTED_TARGET avahi-compat-libdns_sd)
        endif()
        target_link_libraries(potluck-discovery PUBLIC PkgConfig::AVAHI_COMPAT_LIBDNSSD)
    else()
        message(FATAL_ERROR "Potluck DNS-SD discovery requires macOS or Linux")
    endif()
endif()

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(potluck-shard)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    add_subdirectory(completion)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(ui)
        add_subdirectory(cli)
        add_subdirectory(server)
    endif()
    add_subdirectory(tokenize)
    add_subdirectory(parser)
    add_subdirectory(tts)
    add_subdirectory(mtmd)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        # these tools use backends directly (no dynamic loading) and depend on CPU backend symbols
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
    endif()
    add_subdirectory(fit-params)
    add_subdirectory(results)
    add_subdirectory(potluck-node)
    add_subdirectory(potluck-worker)
    add_subdirectory(potluck-server)
endif()
