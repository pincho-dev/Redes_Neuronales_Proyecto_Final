cmake_minimum_required(VERSION 3.16)
project(LogRegEigen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Por defecto compilar en Release: sin optimizaciones el benchmark no tiene sentido
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de compilacion" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

find_package(Eigen3 REQUIRED NO_MODULE)

# Ruta raiz del proyecto: permite ejecutar desde Qt Creator sin pasar argumentos
get_filename_component(PROJECT_ROOT_DIR "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)

# Biblioteca con el nucleo matematico (compartida por ambos ejecutables)
add_library(logreg_core
  src/dataset.cpp
  src/logistic_regression.cpp
  src/metrics.cpp
  src/utils.cpp
)
target_include_directories(logreg_core PUBLIC include)
target_link_libraries(logreg_core PUBLIC Eigen3::Eigen)
target_compile_definitions(logreg_core PUBLIC
  PROJECT_ROOT="${PROJECT_ROOT_DIR}"
  EIGEN_DONT_PARALLELIZE)          # 1 solo hilo, igual que en Python

target_compile_options(logreg_core PUBLIC -Wall -Wextra)

# Ejecutable 1: entrenamiento + evaluacion + benchmark
add_executable(logreg src/main.cpp)
target_link_libraries(logreg PRIVATE logreg_core)

# Ejecutable 2: estudio de localidad espacial/temporal (Fase C)
add_executable(locality_benchmark src/locality_benchmark.cpp)
target_link_libraries(locality_benchmark PRIVATE logreg_core)
