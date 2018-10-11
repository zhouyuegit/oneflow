IF(NOT ${WITH_MKLML})
  return()
ENDIF(NOT ${WITH_MKLML})

include (ExternalProject)

set(MKLML_INCLUDE_DIR ${THIRD_PARTY_DIR}/mklml/include)
set(MKLML_LIBRARY_DIR ${THIRD_PARTY_DIR}/mklml/lib)


SET(MKLML_DOWNLOAD_DIR  "${CMAKE_CURRENT_BINARY_DIR}/mklml")
IF((NOT DEFINED MKLML_VER) OR (NOT DEFINED MKLML_URL))
  SET(MKLML_VER "mklml_lnx_2019.0.20180710" CACHE STRING "" FORCE)
  #SET(MKLML_URL "https://github.com/intel/mkl-dnn/releases/download/v0.16/${MKLML_VER}.tgz" CACHE STRING "" FORCE)
  SET("/home/qiaojing/git_repo/mkl-dnn/external/${MKLML_VER}.tgz" CACHE STRING "" FORCE)
ENDIF()

if(WIN32)
elseif(APPLE AND ("${CMAKE_GENERATOR}" STREQUAL "Xcode"))
else()
    set(MKLML_BUILD_INCLUDE_DIR ${MKLML_DOWNLOAD_DIR}/${MKLML_VER}/include)
    set(MKLML_BUILD_LIBRARY_DIR ${MKLML_DOWNLOAD_DIR}/${MKLML_VER}/lib)
    set(MKLML_LIBRARY_NAMES libmklml_intel.so libiomp5.so libmklml_gnu.so)
endif()

foreach(LIBRARY_NAME ${MKLML_LIBRARY_NAMES})
    list(APPEND MKLML_SHARED_LIBRARIES ${MKLML_LIBRARY_DIR}/${LIBRARY_NAME})
    list(APPEND MKLML_BUILD_SHARED_LIBRARIES ${MKLML_BUILD_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()



if (BUILD_THIRD_PARTY)

ExternalProject_Add(mklml
    PREFIX                mklml
    DOWNLOAD_DIR          ${MKLML_DOWNLOAD_DIR}
    DOWNLOAD_COMMAND      wget --no-check-certificate ${MKLML_URL} -c -q -O ${MKLML_VER}.tgz 
                          && tar zxf ${MKLML_VER}.tgz
    DOWNLOAD_NO_PROGRESS  1
    UPDATE_COMMAND        ""
    CONFIGURE_COMMAND     ""
    BUILD_COMMAND         ""
    INSTALL_COMMAND       ""
)

# put mklml includes in the 'THIRD_PARTY_DIR'
add_custom_target(mklml_create_header_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MKLML_INCLUDE_DIR}
  DEPENDS mklml)

add_custom_target(mklml_copy_headers_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${MKLML_BUILD_INCLUDE_DIR} ${MKLML_INCLUDE_DIR}
  DEPENDS mklml_create_header_dir)

# put mklml librarys in the 'THIRD_PARTY_DIR'
add_custom_target(mklml_create_library_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${MKLML_LIBRARY_DIR}
  DEPENDS mklml)

add_custom_target(mklml_copy_libs_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MKLML_BUILD_SHARED_LIBRARIES} ${MKLML_LIBRARY_DIR}
  DEPENDS mklml_create_library_dir)

endif(BUILD_THIRD_PARTY)