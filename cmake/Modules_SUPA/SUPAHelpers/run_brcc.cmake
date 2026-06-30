# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

###############################################################################
# Runs commands using BRCC
###############################################################################

###############################################################################
# This file runs the brcc commands to produce the desired output file
# along with the dependency file needed by CMake to compute dependencies.
#
# Input variables:
#
# verbose:BOOL=<>               OFF: Be as quiet as possible (default)
#                               ON : Describe each step
# build_configuration:STRING=<> Build configuration. Defaults to Debug.
# generated_file:STRING=<>      File to generate. Mandatory argument.

if(NOT build_configuration)
    set(build_configuration Debug)
endif()
if(NOT generated_file)
    message(FATAL_ERROR "You must specify generated_file on the command line")
endif()

# Set these up as variables to make reading the generated file easier
set(SUPA_BRCC_EXECUTABLE "@SUPA_BRCC_EXECUTABLE@") # path
set(SUPA_HOST_COMPILER "@SUPA_HOST_COMPILER@") # path
set(CMAKE_COMMAND "@CMAKE_COMMAND@") # path
set(SUPA_CLANG_PATH "@SUPA_CLANG_PATH@") #path
set(SUPA_CLANG_PARALLEL_BUILD_COMPILE_OPTIONS "@SUPA_CLANG_PARALLEL_BUILD_COMPILE_OPTIONS@")

@_SUPA_HOST_FLAGS@
@_SUPA_BRCC_SPECIFIC_FLAGS@
@_SUPA_BRCC_FLAGS@
@_SUPA_CLANG_FLAGS@
#Needed to bring the SUPA_BRCC_INCLUDE_ARGS variable in scope
set(SUPA_BRCC_INCLUDE_ARGS @SUPA_BRCC_INCLUDE_ARGS@) # list

set(cmake_dependency_file "@cmake_dependency_file@") # path
set(source_file "@source_file@") # path
set(host_flag "@host_flag@") # bool
set(generated_file_relative_path "@generated_file_relative_path@") # path, target name in makefile.

if(NOT host_flag)
    set(__CC ${SUPA_BRCC_EXECUTABLE})
    if("${SUPA_PLATFORM}" STREQUAL "biren")
        if("${BRCC_COMPILER}" STREQUAL "clang")
            if(NOT "x${SUPA_CLANG_PATH}" STREQUAL "x")
                set(ENV{SUPA_CLANG_PATH} ${SUPA_CLANG_PATH})
            endif()
            set(__CC_FLAGS ${SUPA_CLANG_PARALLEL_BUILD_COMPILE_OPTIONS} ${SUPA_BRCC_FLAGS} ${SUPA_CLANG_FLAGS} ${SUPA_BRCC_FLAGS_${build_configuration}} ${SUPA_CLANG_FLAGS_${build_configuration}})
        endif()
    else()
        set(__CC_FLAGS ${SUPA_BRCC_FLAGS} ${SUPA_BRCC_FLAGS_${build_configuration}})
    endif()
else()
    set(__CC ${SUPA_HOST_COMPILER})
    set(__CC_FLAGS ${CMAKE_HOST_FLAGS} ${CMAKE_HOST_FLAGS_${build_configuration}})
endif()
set(__CC_INCLUDES ${SUPA_BRCC_INCLUDE_ARGS})

# supa_execute_process - Executes a command with optional command echo and status message.
#   status     - Status message to print if verbose is true
#   command    - COMMAND argument from the usual execute_process argument structure
#   ARGN       - Remaining arguments are the command with arguments
#   SUPA_result - Return value from running the command
macro(supa_execute_process status command)
    set(_command ${command})
    if(NOT "x${_command}" STREQUAL "xCOMMAND")
        message(FATAL_ERROR "Malformed call to supa_execute_process.  Missing COMMAND as second argument. (command = ${command})")
    endif()
    if(verbose)
        execute_process(COMMAND "${CMAKE_COMMAND}" -E echo -- ${status})
        # Build command string to print
        set(supa_execute_process_string)
        foreach(arg ${ARGN})
            # Escape quotes if any
            string(REPLACE "\"" "\\\"" arg ${arg})
            # Surround args with spaces with quotes
            if(arg MATCHES " ")
                list(APPEND supa_execute_process_string "\"${arg}\"")
            else()
                list(APPEND supa_execute_process_string ${arg})
            endif()
        endforeach()
        # Echo the command
        execute_process(COMMAND ${CMAKE_COMMAND} -E echo ${supa_execute_process_string})
    endif()
    # Run the command
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE SUPA_result)
endmacro()

# Delete the target file
supa_execute_process(
    "Removing ${generated_file}"
    COMMAND "${CMAKE_COMMAND}" -E remove "${generated_file}"
    )

# Generate the dependency file
supa_execute_process(
    "Generating dependency file: ${cmake_dependency_file}"
    COMMAND "${__CC}"
    --supa-host-only
    -Wno-unused-command-line-argument
    -M
    "${source_file}"
    -o "${cmake_dependency_file}"
    ${__CC_FLAGS}
    ${__CC_INCLUDES}
    )

if(SUPA_result)
    message(FATAL_ERROR "Error generating deps for ${generated_file}")
endif()

# replace target name in depend file in order to match name in build.make
FILE(READ "${cmake_dependency_file}" TEXT)
string(REGEX REPLACE "^.*\\.o: " "${generated_file_relative_path}: " TEXT ${TEXT})
FILE(WRITE "${cmake_dependency_file}" "${TEXT}")

# Generate the output file
supa_execute_process(
    "Generating ${generated_file}"
    COMMAND "${__CC}"
    -c
    "${source_file}"
    -o "${generated_file}"
    ${__CC_FLAGS}
    ${__CC_INCLUDES}
    )

if(SUPA_result)
    # Make sure that we delete the output file
    supa_execute_process(
        "Removing ${generated_file}"
        COMMAND "${CMAKE_COMMAND}" -E remove "${generated_file}"
        )
    message(FATAL_ERROR "Error generating file ${generated_file}")
else()
    if(verbose)
        message("Generated ${generated_file} successfully.")
    endif()
endif()
# vim: ts=4:sw=4:expandtab:smartindent
