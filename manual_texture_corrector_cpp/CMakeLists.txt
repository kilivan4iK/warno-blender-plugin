cmake_minimum_required(VERSION 3.20)

project(manual_texture_corrector LANGUAGES CXX)

add_executable(manual_texture_corrector WIN32
    src/main.cpp
)

target_compile_features(manual_texture_corrector PRIVATE cxx_std_17)
target_compile_definitions(manual_texture_corrector PRIVATE UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)

if(MSVC)
    target_compile_options(manual_texture_corrector PRIVATE /W4 /permissive-)
else()
    target_compile_options(manual_texture_corrector PRIVATE -Wall -Wextra -Wpedantic)
endif()

target_link_libraries(manual_texture_corrector PRIVATE
    gdiplus
    comctl32
    shell32
)
