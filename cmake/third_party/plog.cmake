include (ExternalProject)

set(PLOG_INCLUDE_DIR ${THIRD_PARTY_DIR}/plog/include)
set(PLOG_BUILD_INCLUDE ${CMAKE_CURRENT_BINARY_DIR}/plog/src/plog)

set(PLOG_URL ${CMAKE_CURRENT_BINARY_DIR}/third_party/plog/include/plog)

if(THIRD_PARTY)

ExternalProject_Add(plog
    PREFIX plog
    URL ${PLOG_URL}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

add_custom_target(plog_create_header_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory ${PLOG_INCLUDE_DIR}/plog
    DEPENDS plog)
add_custom_target(plog_copy_headers_to_destination
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PLOG_BUILD_INCLUDE} ${PLOG_INCLUDE_DIR}/plog
    DEPENDS plog_create_header_dir)

endif(THIRD_PARTY)
