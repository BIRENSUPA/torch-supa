# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

###############################################################################
# FindSUPABRCC.cmake
###############################################################################
include(CheckCXXCompilerFlag)
###############################################################################
# SET: Variable defaults
###############################################################################
# User defined flags
set(SUPA_BRCC_FLAGS "" CACHE STRING "Semicolon delimited flags for BRCC")
set(SUPA_CLANG_FLAGS "" CACHE STRING "Semicolon delimited flags for CLANG")
mark_as_advanced(SUPA_BRCC_FLAGS SUPA_CLANG_FLAGS)

set(_supa_configuration_types ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE} Debug MinSizeRel Release RelWithDebInfo)
list(REMOVE_DUPLICATES _supa_configuration_types)
foreach(config ${_supa_configuration_types})
    string(TOUPPER ${config} config_upper)
    set(SUPA_BRCC_FLAGS_${config_upper} "" CACHE STRING "Semicolon delimited flags for BRCC")
    set(SUPA_CLANG_FLAGS_${config_upper} "" CACHE STRING "Semicolon delimited flags for CLANG")
    mark_as_advanced(SUPA_BRCC_FLAGS_${config_upper} SUPA_CLANG_FLAGS_${config_upper})
endforeach()
option(SUPA_HOST_COMPILATION_CPP "Host code compilation mode" ON)
option(SUPA_VERBOSE_BUILD "Print out the commands run while compiling the SUPA source file.  With the Makefile generator this defaults to VERBOSE variable specified on the command line, but can be forced on with this option." OFF)
mark_as_advanced(SUPA_HOST_COMPILATION_CPP)

###############################################################################
# FIND: SUPA and associated helper binaries
###############################################################################

set(SUPA_FOUND TRUE)

get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../" REALPATH)

# SUPA is currently not supported for apple
if(NOT APPLE)
    # Search for brcc installation
    if(NOT SUPA_ROOT_DIR)
        # Search in user specified path first
        find_path(
            SUPA_ROOT_DIR
            NAMES bin/brcc
            PATHS
            "$ENV{BRCC_PATH}"
            ENV BRCC_PATH
            ${_IMPORT_PREFIX}
            ${BRCC_PATH}
            DOC "brcc installed location"
            NO_DEFAULT_PATH
            )
        if(NOT EXISTS ${SUPA_ROOT_DIR})
            if(SUPA_FIND_REQUIRED)
                message(FATAL_ERROR "Specify SUPA_ROOT_DIR")
            elseif(NOT SUPA_FIND_QUIETLY)
                message("SUPA_ROOT_DIR not found or specified")
            endif()
        endif()
        # And push it back to the cache
        set(SUPA_ROOT_DIR ${SUPA_ROOT_DIR} CACHE PATH "SUPA installed location" FORCE)
    endif()

    # Find BRCC executable
    find_program(
        SUPA_BRCC_EXECUTABLE
        NAMES brcc
        PATHS
        "${SUPA_ROOT_DIR}"
        ENV BRCC_PATH
        PATH_SUFFIXES bin
        NO_DEFAULT_PATH
        )
    if(NOT SUPA_BRCC_EXECUTABLE)
        # Now search in default paths
        find_program(SUPA_BRCC_EXECUTABLE brcc)
    endif()

    mark_as_advanced(SUPA_BRCC_EXECUTABLE)

    if(SUPA_VERSION)
        string(REPLACE "." ";" _supa_version_list "${SUPA_VERSION}")
        list(GET _supa_version_list 0 SUPA_VERSION_MAJOR)
        list(GET _supa_version_list 1 SUPA_VERSION_MINOR)
        list(GET _supa_version_list 2 SUPA_VERSION_PATCH)
        set(SUPA_VERSION_STRING "${SUPA_VERSION}")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    SUPA
    REQUIRED_VARS
    SUPA_ROOT_DIR
    SUPA_BRCC_EXECUTABLE
    VERSION_VAR SUPA_VERSION
    )

###############################################################################
# Set SUPA CMAKE Flags
###############################################################################
# Copy the invocation styles from CXX to SUPA
set(CMAKE_SUPA_ARCHIVE_CREATE ${CMAKE_CXX_ARCHIVE_CREATE})
set(CMAKE_SUPA_ARCHIVE_APPEND ${CMAKE_CXX_ARCHIVE_APPEND})
set(CMAKE_SUPA_ARCHIVE_FINISH ${CMAKE_CXX_ARCHIVE_FINISH})
set(CMAKE_SHARED_LIBRARY_SONAME_SUPA_FLAG ${CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG})
set(CMAKE_SHARED_LIBRARY_CREATE_SUPA_FLAGS ${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS})
set(CMAKE_SHARED_LIBRARY_SUPA_FLAGS ${CMAKE_SHARED_LIBRARY_CXX_FLAGS})
#set(CMAKE_SHARED_LIBRARY_LINK_supa_flags ${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS})
set(CMAKE_SHARED_LIBRARY_RUNTIME_SUPA_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG})
set(CMAKE_SHARED_LIBRARY_RUNTIME_SUPA_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG_SEP})
set(CMAKE_SHARED_LIBRARY_LINK_STATIC_SUPA_FLAGS ${CMAKE_SHARED_LIBRARY_LINK_STATIC_CXX_FLAGS})
set(CMAKE_SHARED_LIBRARY_LINK_DYNAMIC_SUPA_FLAGS ${CMAKE_SHARED_LIBRARY_LINK_DYNAMIC_CXX_FLAGS})

set(SUPA_CLANG_PARALLEL_BUILD_COMPILE_OPTIONS "")
set(SUPA_CLANG_PARALLEL_BUILD_LINK_OPTIONS "")

###############################################################################
# MACRO: Locate helper files
###############################################################################
macro(SUPA_FIND_HELPER_FILE _name _extension)
    set(_supa_full_name "${_name}.${_extension}")
    get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
    set(SUPA_${_name} "${CMAKE_CURRENT_LIST_DIR}/SUPAHelpers/${_supa_full_name}")
    if(NOT EXISTS "${SUPA_${_name}}")
        set(error_message "${_supa_full_name} not found in ${CMAKE_CURRENT_LIST_DIR}/SUPAHelpers")
        if(SUPA_FIND_REQUIRED)
            message(FATAL_ERROR "${error_message}")
        else()
            if(NOT SUPA_FIND_QUIETLY)
                message(STATUS "${error_message}")
            endif()
        endif()
    endif()
    # Set this variable as internal, so the user isn't bugged with it.
    set(SUPA_${_name} ${SUPA_${_name}} CACHE INTERNAL "Location of ${_full_name}" FORCE)
endmacro()

###############################################################################
supa_find_helper_file(run_brcc cmake)
###############################################################################

###############################################################################
# MACRO: Reset compiler flags
###############################################################################
macro(SUPA_RESET_FLAGS)
    unset(SUPA_BRCC_FLAGS)
    unset(SUPA_CLANG_FLAGS)
    foreach(config ${_supa_configuration_types})
        string(TOUPPER ${config} config_upper)
        unset(SUPA_BRCC_FLAGS_${config_upper})
        unset(SUPA_CLANG_FLAGS_${config_upper})
    endforeach()
endmacro()

###############################################################################
# MACRO: Separate the options from the sources
###############################################################################
macro(SUPA_GET_SOURCES_AND_OPTIONS _sources _cmake_options _brcc_options _clang_options)
    set(${_sources})
    set(${_cmake_options})
    set(${_brcc_options})
    set(${_clang_options})
    set(_brcc_found_options FALSE)
    set(_clang_found_options FALSE)
    foreach(arg ${ARGN})
        if("x${arg}" STREQUAL "xBRCC_OPTIONS")
            set(_brcc_found_options TRUE)
            set(_clang_found_options FALSE)
        elseif("x${arg}" STREQUAL "xCLANG_OPTIONS")
            set(_brcc_found_options FALSE)
            set(_clang_found_options TRUE)
        elseif(
                "x${arg}" STREQUAL "xEXCLUDE_FROM_ALL" OR
                "x${arg}" STREQUAL "xSTATIC" OR
                "x${arg}" STREQUAL "xSHARED" OR
                "x${arg}" STREQUAL "xMODULE"
                )
            list(APPEND ${_cmake_options} ${arg})
        else()
            if(_brcc_found_options)
                list(APPEND ${_brcc_options} ${arg})
            elseif(_clang_found_options)
                list(APPEND ${_clang_options} ${arg})
            else()
                # Assume this is a file
                list(APPEND ${_sources} ${arg})
            endif()
        endif()
    endforeach()
endmacro()

###############################################################################
# MACRO: Add include directories to pass to the brcc command
###############################################################################
set(SUPA_BRCC_INCLUDE_ARGS_USER "")
macro(SUPA_INCLUDE_DIRECTORIES)
    foreach(dir ${ARGN})
        list(APPEND SUPA_BRCC_INCLUDE_ARGS_USER $<$<BOOL:${dir}>:-I${dir}>)
    endforeach()
endmacro()

###############################################################################
# FUNCTION: Helper to avoid clashes of files with the same basename but different paths
###############################################################################
function(SUPA_COMPUTE_BUILD_PATH path build_path)
    # Convert to cmake style paths
    file(TO_CMAKE_PATH "${path}" bpath)
    if(IS_ABSOLUTE "${bpath}")
        string(FIND "${bpath}" "${CMAKE_CURRENT_BINARY_DIR}" _binary_dir_pos)
        if(_binary_dir_pos EQUAL 0)
            file(RELATIVE_PATH bpath "${CMAKE_CURRENT_BINARY_DIR}" "${bpath}")
        else()
            file(RELATIVE_PATH bpath "${CMAKE_CURRENT_SOURCE_DIR}" "${bpath}")
        endif()
    endif()

    # Remove leading /
    string(REGEX REPLACE "^[/]+" "" bpath "${bpath}")
    # Avoid absolute paths by removing ':'
    string(REPLACE ":" "_" bpath "${bpath}")
    # Avoid relative paths that go up the tree
    string(REPLACE "../" "__/" bpath "${bpath}")
    # Avoid spaces
    string(REPLACE " " "_" bpath "${bpath}")
    # Strip off the filename
    get_filename_component(bpath "${bpath}" PATH)

    set(${build_path} "${bpath}" PARENT_SCOPE)
endfunction()

###############################################################################
# MACRO: Parse OPTIONS from ARGN & set variables prefixed by _option_prefix
###############################################################################
macro(SUPA_PARSE_BRCC_OPTIONS _option_prefix)
    set(_supa_found_config)
    foreach(arg ${ARGN})
        # Determine if we are dealing with a per-configuration flag
        foreach(config ${_supa_configuration_types})
            string(TOUPPER ${config} config_upper)
            if(arg STREQUAL "${config_upper}")
                set(_supa_found_config _${arg})
                # Clear arg to prevent it from being processed anymore
                set(arg)
            endif()
        endforeach()
        if(arg)
            list(APPEND ${_option_prefix}${_supa_found_config} "${arg}")
        endif()
    endforeach()
endmacro()

###############################################################################
# MACRO: Prepare cmake commands for the target
###############################################################################
macro(SUPA_PREPARE_TARGET_COMMANDS _target _format _generated_files _source_files)
    set(_supa_flags "")
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _supa_build_configuration)
    if(SUPA_HOST_COMPILATION_CPP)
        set(SUPA_C_OR_CXX CXX)
    else()
        set(SUPA_C_OR_CXX C)
    endif()
    set(generated_extension ${CMAKE_${SUPA_C_OR_CXX}_OUTPUT_EXTENSION})

    # Initialize list of includes with those specified by the user. Append with
    # ones specified to cmake directly.
    set(SUPA_BRCC_INCLUDE_ARGS ${SUPA_BRCC_INCLUDE_ARGS_USER})

    # Add the include directories
    set(include_directories_generator "$<TARGET_PROPERTY:${_target},INCLUDE_DIRECTORIES>")
    list(APPEND SUPA_BRCC_INCLUDE_ARGS "$<$<BOOL:${include_directories_generator}>:-I$<JOIN:${include_directories_generator}, -I>>")

    # get_directory_property(_supa_include_directories INCLUDE_DIRECTORIES)
    # list(REMOVE_DUPLICATES _supa_include_directories)
    # if(_supa_include_directories)
    #     foreach(dir ${_supa_include_directories})
    #         list(APPEND SUPA_BRCC_INCLUDE_ARGS $<$<BOOL:${dir}>:-I${dir}>)
    #     endforeach()
    # endif()

    SUPA_GET_SOURCES_AND_OPTIONS(_supa_sources _supa_cmake_options _brcc_options _clang_options ${ARGN})

    # use local copy of FLAGS in order not to pollute global variable.
    set(LOCAL_SUPA_BRCC_FLAGS ${SUPA_BRCC_FLAGS})
    set(LOCAL_SUPA_CLANG_FLAGS ${SUPA_CLANG_FLAGS})

    SUPA_PARSE_BRCC_OPTIONS(LOCAL_SUPA_BRCC_FLAGS ${_brcc_options})
    SUPA_PARSE_BRCC_OPTIONS(LOCAL_SUPA_CLANG_FLAGS ${_clang_options})

    # Add the compile definitions
    set(compile_definition_generator "$<TARGET_PROPERTY:${_target},COMPILE_DEFINITIONS>")
    list(APPEND LOCAL_SUPA_BRCC_FLAGS "$<$<BOOL:${compile_definition_generator}>:-D$<JOIN:${compile_definition_generator}, -D>>")
    # Check if we are building shared library.
    set(_supa_build_shared_libs FALSE)
    list(FIND _supa_cmake_options SHARED _supa_found_SHARED)
    list(FIND _supa_cmake_options MODULE _supa_found_MODULE)
    if(_supa_found_SHARED GREATER -1 OR _supa_found_MODULE GREATER -1)
        set(_supa_build_shared_libs TRUE)
    endif()
    list(FIND _supa_cmake_options STATIC _supa_found_STATIC)
    if(_supa_found_STATIC GREATER -1)
        set(_supa_build_shared_libs FALSE)
    endif()

    # If we are building a shared library, add extra flags to SUPA_BRCC_FLAGS
    if(_supa_build_shared_libs)
        list(APPEND LOCAL_SUPA_CLANG_FLAGS "-fPIC")
    endif()

    # Set host compiler
    set(SUPA_HOST_COMPILER "${CMAKE_${SUPA_C_OR_CXX}_COMPILER}")

    # Set compiler flags
    set(_SUPA_HOST_FLAGS "set(CMAKE_HOST_FLAGS ${CMAKE_${SUPA_C_OR_CXX}_FLAGS})")
    set(_SUPA_BRCC_FLAGS "set(SUPA_BRCC_FLAGS ${LOCAL_SUPA_BRCC_FLAGS};\${SUPA_BRCC_SPECIFIC_FLAGS})")
    set(_SUPA_CLANG_FLAGS "set(SUPA_CLANG_FLAGS ${LOCAL_SUPA_CLANG_FLAGS})")
    foreach(config ${_supa_configuration_types})
        string(TOUPPER ${config} config_upper)
        set(_SUPA_HOST_FLAGS "${_SUPA_HOST_FLAGS}\nset(CMAKE_HOST_FLAGS_${config_upper} ${CMAKE_${SUPA_C_OR_CXX}_FLAGS_${config_upper}})")
        set(_SUPA_BRCC_FLAGS "${_SUPA_BRCC_FLAGS}\nset(SUPA_BRCC_FLAGS_${config_upper} ${SUPA_BRCC_FLAGS_${config_upper}})")
        set(_SUPA_CLANG_FLAGS "${_SUPA_CLANG_FLAGS}\nset(SUPA_CLANG_FLAGS_${config_upper} ${SUPA_CLANG_FLAGS_${config_upper}})")
    endforeach()

    # Reset the output variable
    set(_supa_generated_files "")
    set(_supa_source_files "")

    list(REMOVE_DUPLICATES SUPA_BRCC_INCLUDE_ARGS)
    set(ORIGIN_SUPA_BRCC_INCLUDE_ARGS ${SUPA_BRCC_INCLUDE_ARGS})

    # Iterate over all arguments and create custom commands for all source files
    foreach(file ${ARGN})
        set(SUPA_BRCC_INCLUDE_ARGS ${ORIGIN_SUPA_BRCC_INCLUDE_ARGS})

        # Ignore any file marked as a HEADER_FILE_ONLY
        get_source_file_property(_is_header ${file} HEADER_FILE_ONLY)
        # Allow per source file overrides of the format. Also allows compiling non .su files.
        get_source_file_property(_supa_source_format ${file} SUPA_SOURCE_PROPERTY_FORMAT)
	get_source_file_property(_supa_brcc_specific_flags ${file} COMPILE_OPTIONS)
	if(_supa_brcc_specific_flags STREQUAL "NOTFOUND")
	    set(_SUPA_BRCC_SPECIFIC_FLAGS "set(SUPA_BRCC_SPECIFIC_FLAGS \"\")")
	else()
	    set(_SUPA_BRCC_SPECIFIC_FLAGS "set(SUPA_BRCC_SPECIFIC_FLAGS ${_supa_brcc_specific_flags})")
	endif()

        if((${file} MATCHES "\\.su$" OR _supa_source_format) AND NOT _is_header)
            set(host_flag FALSE)
        else()
            set(host_flag TRUE)
        endif()

        if(NOT host_flag)
            # Determine output directory
            SUPA_COMPUTE_BUILD_PATH("${file}" supa_build_path)
            set(supa_compile_output_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_target}.dir/${supa_build_path}")

            get_filename_component(basename ${file} NAME)
            set(generated_file_path "${supa_compile_output_dir}/${CMAKE_CFG_INTDIR}")
            set(generated_file_basename "${_target}_generated_${basename}${generated_extension}")

            # Set file names
            set(generated_file "${generated_file_path}/${generated_file_basename}")
            set(cmake_dependency_file "${supa_compile_output_dir}/${generated_file_basename}.depend")
            set(custom_target_script_pregen "${supa_compile_output_dir}/${generated_file_basename}.cmake.pre-gen")
            set(custom_target_script "${supa_compile_output_dir}/${generated_file_basename}.cmake")

            # Set properties for object files
            set_source_files_properties("${generated_file}"
                PROPERTIES
                EXTERNAL_OBJECT true # This is an object file not to be compiled, but only be linked
                )

            get_source_file_property(_include_dirs ${file} INCLUDE_DIRECTORIES)
            if(_include_dirs)
                foreach(dir ${_include_dirs})
                    list(PREPEND SUPA_BRCC_INCLUDE_ARGS $<$<BOOL:${dir}>:-I${dir}>)
                endforeach()
            endif()
            list(REMOVE_DUPLICATES SUPA_BRCC_INCLUDE_ARGS)

            # Don't add CMAKE_CURRENT_SOURCE_DIR if the path is already an absolute path
            get_filename_component(file_path "${file}" PATH)
            if(IS_ABSOLUTE "${file_path}")
                set(source_file "${file}")
            else()
                set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
            endif()

            if(NOT EXISTS ${cmake_dependency_file})
                file(WRITE ${cmake_dependency_file} "# Generated by: FindSUPABRCC.cmake. Do not edit.\n")
            endif()

            # Create up the comment string
            file(RELATIVE_PATH generated_file_relative_path "${CMAKE_BINARY_DIR}" "${generated_file}")
            set(supa_build_comment_string "Building BRCC object ${generated_file_relative_path}")

            # Configure the build script
            configure_file("${SUPA_run_brcc}" "${custom_target_script_pregen}" @ONLY)
            file(GENERATE
                OUTPUT "${custom_target_script}"
                INPUT "${custom_target_script_pregen}"
                )
            set(main_dep DEPENDS ${source_file})
            if(CMAKE_GENERATOR MATCHES "Makefiles")
                set(verbose_output "$(VERBOSE)")
            elseif(SUPA_VERBOSE_BUILD)
                set(verbose_output ON)
            else()
                set(verbose_output OFF)
            endif()

            # Build the generated file and dependency file
            add_custom_command(
                OUTPUT ${generated_file}
                # These output files depend on the source_file and the contents of cmake_dependency_file
                ${main_dep}
                DEPENDS ${custom_target_script}
                DEPFILE ${cmake_dependency_file}
                # Make sure the output directory exists before trying to write to it.
                COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_file_path}"
                COMMAND ${CMAKE_COMMAND} ARGS
                -D verbose:BOOL=${verbose_output}
                -D build_configuration:STRING=${_supa_build_configuration}
                -D "generated_file:STRING=${generated_file}"
                -P "${custom_target_script}"
                WORKING_DIRECTORY "${supa_compile_output_dir}"
                COMMENT "${supa_build_comment_string}"
                )

            # Make sure the build system knows the file is generated
            set_source_files_properties(${generated_file} PROPERTIES GENERATED TRUE)
            list(APPEND _supa_generated_files ${generated_file})
            list(APPEND _supa_source_files ${file})
        endif()
    endforeach()

    # Set the return parameter
    set(${_generated_files} ${_supa_generated_files})
    set(${_source_files} ${_supa_source_files})
endmacro()

###############################################################################
# SUPA_ADD_EXECUTABLE
###############################################################################
macro(SUPA_ADD_EXECUTABLE supa_target)
    # Separate the sources from the options
    SUPA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _brcc_options _clang_options ${ARGN})
    SUPA_PREPARE_TARGET_COMMANDS(${supa_target} OBJ _generated_files _source_files ${_sources} ${_cmake_options} BRCC_OPTIONS ${_brcc_options} CLANG_OPTIONS ${_clang_options})
    if(_source_files)
        list(REMOVE_ITEM _sources ${_source_files})
    endif()
    if("${SUPA_COMPILER}" STREQUAL "clang")
        if("x${SUPA_CLANG_PATH}" STREQUAL "x")
            if(DEFINED ENV{SUPA_CLANG_PATH})
                set(SUPA_CLANG_PATH "$ENV{SUPA_CLANG_PATH}")
            elseif(DEFINED ENV{SUPA_PATH})
                set(SUPA_CLANG_PATH "$ENV{SUPA_PATH}/bin")
            else()
                message(FATAL_ERROR "Unable to find the clang compiler path. Set SUPA_PATH or SUPA_PATH in env")
            endif()
        endif()
        set(CMAKE_SUPA_LINK_EXECUTABLE "${SUPA_BRCC_CMAKE_LINKER_HELPER} ${SUPA_CLANG_PATH}/clang --supa-link ${SUPA_CLANG_PARALLEL_BUILD_LINK_OPTIONS} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
    else()
        set(CMAKE_SUPA_LINK_EXECUTABLE "${SUPA_BRCC_CMAKE_LINKER_HELPER} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
    endif()
    if ("${_sources}" STREQUAL "")
        add_executable(${supa_target} ${_cmake_options} ${_generated_files} "")
    else()
        add_executable(${supa_target} ${_cmake_options} ${_generated_files} ${_sources})
    endif()
    set_target_properties(${supa_target} PROPERTIES LINKER_LANGUAGE SUPA)
    # Link with host
    if (SUPA_HOST_INTERFACE)
        target_link_libraries(${supa_target} ${SUPA_HOST_INTERFACE})
    endif()
endmacro()

###############################################################################
# SUPA_ADD_LIBRARY
###############################################################################
macro(SUPA_ADD_LIBRARY supa_target)
    # Separate the sources from the options
    SUPA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _brcc_options _clang_options ${ARGN})
    SUPA_PREPARE_TARGET_COMMANDS(${supa_target} OBJ _generated_files _source_files ${_sources} ${_cmake_options} BRCC_OPTIONS ${_brcc_options} CLANG_OPTIONS ${_clang_options})
    if(_source_files)
        list(REMOVE_ITEM _sources ${_source_files})
    endif()

    set(CMAKE_LINKER ${SUPA_BRCC_EXECUTABLE})
    set(CMAKE_SUPA_CREATE_SHARED_LIBRARY
        "<CMAKE_LINKER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
    if ("${_sources}" STREQUAL "")
        add_library(${supa_target} ${_cmake_options} ${_generated_files} "")
    else()
        add_library(${supa_target} ${_cmake_options} ${_generated_files} ${_sources})
    endif()

    if (ENABLE_COVERAGE)
      set_target_properties(${supa_target} PROPERTIES LINK_FLAGS_DEBUG  "-g --coverage")
    endif()
    set_target_properties(${supa_target} PROPERTIES LINK_FLAGS_RELEASE  "-s")
    set_target_properties(${supa_target} PROPERTIES LINKER_LANGUAGE SUPA)
    set_target_properties(${supa_target} PROPERTIES LINK_FLAGS "--supa-link -O2 --supa-gpu-arch=${SUPA_ARCH} ${OpenMP_CXX_FLAGS}")
    # Link with host
    if (SUPA_HOST_INTERFACE)
        target_link_libraries(${supa_target} ${SUPA_HOST_INTERFACE})
    endif()
    # set(CMAKE_LINKER ${CMAKE_CXX_COMPILER})
endmacro()

###############################################################################
# SUPA_SPECIAL_ADD_LIBRARY
###############################################################################
macro(SUPA_SPECIAL_ADD_LIBRARY supa_target)
    # Separate the sources from the options
    SUPA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _brcc_options _clang_options ${ARGN})
    SUPA_PREPARE_TARGET_COMMANDS(${supa_target} OBJ _generated_files _source_files ${_sources} ${_cmake_options} BRCC_OPTIONS ${_brcc_options} CLANG_OPTIONS ${_clang_options})
    if(_source_files)
        list(REMOVE_ITEM _sources ${_source_files})
    endif()

    set(CMAKE_LINKER ${SUPA_BRCC_EXECUTABLE})
    set(CMAKE_SUPA_CREATE_SHARED_LIBRARY
        "<CMAKE_LINKER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
    if ("${_sources}" STREQUAL "")
        add_library(${supa_target} ${_cmake_options} ${_generated_files} "")
    else()
        add_library(${supa_target} ${_cmake_options} ${_generated_files} ${_sources})
    endif()

    if (ENABLE_COVERAGE)
      set_target_properties(${supa_target} PROPERTIES LINK_FLAGS_DEBUG  "-g --coverage")
    endif()

    set_target_properties(${supa_target} PROPERTIES LINK_FLAGS_RELEASE  "-s")
    set_target_properties(${supa_target} PROPERTIES LINKER_LANGUAGE SUPA)
    set_target_properties(${supa_target} PROPERTIES LINK_FLAGS "--supa-link -O2 -ping-pong-sync --supa-gpu-arch=${SUPA_ARCH}")
endmacro()

# vim: ts=4:sw=4:expandtab:smartindent
