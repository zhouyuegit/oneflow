
IF(NOT ${WITH_MKLDNN})
return()
ENDIF(NOT ${WITH_MKLDNN})

include (ExternalProject)

set(MKLDNN_INCLUDE_DIR ${THIRD_PARTY_DIR}/mkldnn/include)
set(MKLDNN_LIBRARY_DIR ${THIRD_PARTY_DIR}/mkldnn/lib)

set(MKLDNN_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/mkldnn/src/mkldnn)
#set(MKLDNN_URL "https://github.com/01org/mkl-dnn.git")
set(MKLDNN_URL "/home/qiaojing/git_repo/mkl-dnn")
#set(MKLDNN_TAG "64e03a1939e0d526aa8e9f2e3f7dc0ad8d372944")

set(MKLML_ROOT ${THIRD_PARTY_DIR}/mklml)

if(WIN32)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
else()
    set(MKLDNN_BUILD_INCLUDE_DIR ${MKLDNN_INSTALL_DIR}/include)
    set(MKLDNN_BUILD_LIBRARY_DIR ${MKLDNN_INSTALL_DIR}/src)
    set(MKLDNN_LIBRARY_NAMES libmkldnn.so)
endif()

foreach(LIBRARY_NAME ${MKLDNN_LIBRARY_NAMES})
    list(APPEND MKLDNN_SHARED_LIBRARIES ${MKLDNN_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND MKLDNN_BUILD_SHARED_LIBRARIES ${MKLDNN_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()


if (BUILD_THIRD_PARTY)

ExternalProject_Add(mkldnn 
    DEPENDS mklml_copy_headers_to_destination mklml_copy_libs_to_destination
    PREFIX mkldnn
    GIT_REPOSITORY ${MKLDNN_URL}
    #GIT_TAG ${MKLDNN_TAG}
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
    CMAKE_ARGS
        -DWITH_TEST=OFF -DWITH_EXAMPLE=OFF
        -DMKLROOT=${MKLML_ROOT}
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
)


# put mkldnn includes in the 'THIRD_PARTY_DIR'
add_custom_target(mkldnn_create_header_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MKLDNN_INCLUDE_DIR}
  DEPENDS mkldnn)

add_custom_target(mkldnn_copy_headers_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${MKLDNN_BUILD_INCLUDE_DIR} ${MKLDNN_INCLUDE_DIR}
  DEPENDS mkldnn_create_header_dir)

# put mkldnn librarys in the 'THIRD_PARTY_DIR'
add_custom_target(mkldnn_create_library_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MKLDNN_LIBRARY_DIR}
  DEPENDS mkldnn)

add_custom_target(mkldnn_copy_libs_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MKLDNN_BUILD_SHARED_LIBRARIES} ${MKLDNN_LIBRARY_DIR}
  DEPENDS mkldnn_create_library_dir)

endif(BUILD_THIRD_PARTY)