# Build the pyquick Python extension via f2py.
# Enabled with -DPYTHON=TRUE at the cmake configure step.

# ---------------------------------------------------------------------------
# 1. Locate the Python interpreter.
#
#    Honour an explicit -DPython3_EXECUTABLE=... hint if provided.
#    Otherwise resolve "python3" from PATH so that conda/virtualenv
#    environments are found before any system Python.
# ---------------------------------------------------------------------------
if(NOT Python3_EXECUTABLE)
    find_program(Python3_EXECUTABLE
        NAMES python3 python
        DOC "Python interpreter used to run f2py"
        NO_CMAKE_SYSTEM_PATH   # skip CMake's built-in system search first
    )
    if(NOT Python3_EXECUTABLE)
        find_program(Python3_EXECUTABLE NAMES python3 python
            DOC "Python interpreter used to run f2py")
    endif()
endif()

if(NOT Python3_EXECUTABLE)
    message(FATAL_ERROR
        "Python interpreter not found. Set -DPython3_EXECUTABLE=/path/to/python3 "
        "or activate the correct conda/virtualenv environment before running cmake.")
endif()

message(STATUS "pyquick: using Python interpreter: ${Python3_EXECUTABLE}")

# ---------------------------------------------------------------------------
# 2. Verify numpy and f2py are available via the located interpreter
# ---------------------------------------------------------------------------
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy"
    RESULT_VARIABLE _numpy_result
    OUTPUT_QUIET ERROR_QUIET)
if(NOT _numpy_result EQUAL 0)
    message(FATAL_ERROR
        "numpy not importable by ${Python3_EXECUTABLE}. "
        "Install numpy or reconfigure without -DPYTHON=TRUE.")
endif()

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m numpy.f2py -h
    RESULT_VARIABLE _f2py_result
    OUTPUT_QUIET
    ERROR_QUIET)
if(NOT _f2py_result EQUAL 0)
    message(FATAL_ERROR
        "f2py not found (numpy.f2py unavailable for ${Python3_EXECUTABLE}). "
        "Install numpy to build the Python extension, or reconfigure without -DPYTHON=TRUE.")
endif()

# ---------------------------------------------------------------------------
# 3. Compute the platform-specific extension suffix
#    e.g. .cpython-313-x86_64-linux-gnu.so
# ---------------------------------------------------------------------------
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'), end='')"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX)

set(_PYQUICK_SO "${CMAKE_CURRENT_BINARY_DIR}/_pyquick${PYTHON_EXT_SUFFIX}")

# ---------------------------------------------------------------------------
# 4. Build the .so via f2py as a custom command
#
#    Include path: build-tree .mod files for the serial libquick target.
#    The directory is ${CMAKE_BINARY_DIR}/amber-modules/quick/serial, which
#    is set by config_module_dirs(libquick quick/serial ...) in src/CMakeLists.txt
#    and is fully populated after libquick has been compiled (DEPENDS libquick).
#
#    Link path: resolved at build time via $<TARGET_FILE_DIR:libquick>, so
#    -L always points to the actual location of libquick.so regardless of
#    CMAKE_INSTALL_PREFIX.
# ---------------------------------------------------------------------------
add_custom_command(
    OUTPUT  "${_PYQUICK_SO}"
    COMMAND ${Python3_EXECUTABLE} -m numpy.f2py
                -c "${CMAKE_SOURCE_DIR}/src/pyquick.f90"
                "-I${CMAKE_BINARY_DIR}/amber-modules/quick/serial"
                "-L$<TARGET_FILE_DIR:libquick>"
                -lquick
                -m _pyquick
    DEPENDS libquick "${CMAKE_SOURCE_DIR}/src/pyquick.f90"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Building _pyquick Python extension via f2py"
    VERBATIM)

# ---------------------------------------------------------------------------
# 5. Named target included in the default ALL build
# ---------------------------------------------------------------------------
add_custom_target(pyquick ALL DEPENDS "${_PYQUICK_SO}")

# ---------------------------------------------------------------------------
# 6. Install rules
#    Both the compiled extension and the Python package glue go into
#    <prefix>/lib/pyquick/ so that after "source quick.rc" (which prepends
#    <prefix>/lib to PYTHONPATH) the package is importable as "import pyquick".
# ---------------------------------------------------------------------------
install(FILES "${_PYQUICK_SO}"
        DESTINATION lib/pyquick)
install(FILES "${CMAKE_SOURCE_DIR}/src/pyquick/__init__.py"
        DESTINATION lib/pyquick)
