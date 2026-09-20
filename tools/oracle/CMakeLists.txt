cmake_minimum_required(VERSION 3.16)
project(imorph_oracle LANGUAGES CXX)

# Racine d'une arborescence iMorph 3.2 (celle qui contient le .pro)
set(IMORPH_ROOT "" CACHE PATH "Racine des sources iMorph 3.2")
if(NOT IMORPH_ROOT)
  message(FATAL_ERROR "Passer -DIMORPH_ROOT=/chemin/vers/iMorph3.2")
endif()

set(CMAKE_CXX_STANDARD 11)          # le code de 2017 n'est pas C++17-propre
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# QtCore uniquement : aucune dependance a QtWidgets ni a OpenGL.
# Si le lien reclame QtGui ou QtWidgets, c'est qu'un en-tete d'interface s'est
# reintroduit — corriger utility_min.h plutot que d'ajouter le composant.
find_package(Qt5 COMPONENTS Core REQUIRED)

set(S "${IMORPH_ROOT}/Sources")
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${S}/DataType
  ${S}/Thread/Granulometry
  ${S}/Thread/ShapeClassif
  ${S}/Thread/Skeleton
  ${S}/Thread/Mesh
)

# Fichiers de calcul repris tels quels — ne jamais les modifier : l'oracle doit
# rester l'implementation d'origine, sinon il ne prouve rien.
add_executable(imorph_oracle
  oracle_main.cpp
  ${S}/Thread/Granulometry/morphology.cpp
  ${S}/Thread/Granulometry/fastMarchManu.cpp
  ${S}/Thread/Granulometry/heapVoxel.cpp
  ${S}/Thread/Granulometry/ball.cpp
)
target_link_libraries(imorph_oracle PRIVATE Qt5::Core)
target_compile_options(imorph_oracle PRIVATE -w)   # 2017 : beaucoup d'avertissements
