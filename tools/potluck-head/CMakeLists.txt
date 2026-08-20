option(POTLUCK_HIGHS "Build potluck-head with the HiGHS LP scheduler (--lp)" ON)

add_executable(potluck-head
    main.cpp
    ../../src/potluck-protocol.cpp
    ../../src/potluck-transport.cpp
)

target_include_directories(potluck-head PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${PROJECT_SOURCE_DIR}/common
)

if (POTLUCK_HIGHS)
    set(HIGHS_BUILD_TESTS OFF)
    set(HIGHS_BUILD_EXAMPLES OFF)
    set(HIGHS_BUILD_SHARED OFF)
    include(FetchContent)
    FetchContent_Declare(highs
        GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS.git
        GIT_TAG v1.7.2
        GIT_SHALLOW TRUE
        # Idempotent patch: FetchContent re-runs PATCH_COMMAND whenever CMake
        # regenerates, and a plain `git apply` fails on the already-patched
        # tree. A reverse-check succeeds when the patch is already applied.
        PATCH_COMMAND sh -c "git apply --reverse --check ${CMAKE_CURRENT_SOURCE_DIR}/highs-zstr-cstdint.patch || git apply ${CMAKE_CURRENT_SOURCE_DIR}/highs-zstr-cstdint.patch")
    FetchContent_MakeAvailable(highs)
    target_compile_definitions(potluck-head PRIVATE POTLUCK_HIGHS=1)
    target_link_libraries(potluck-head PRIVATE llama highs::highs)
else()
    target_link_libraries(potluck-head PRIVATE llama)
endif()
