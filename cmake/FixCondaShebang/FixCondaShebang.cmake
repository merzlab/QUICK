# run this script from an install script to fix Python scripts with shebangs like this:
# "#!/home/jamie/rehs/build/CMakeFiles/miniconda/install/bin/python"
# that point to Miniconda Python in the build tree.

# arguments:
# MINICONDA_INSTALL_DIR      - miniconda install dir inside build tree
# AMBER_INSTALL_DIR	         - Amber install prefix (don't forget to use ENV{DESTDIR})
# PREFIX_RELATIVE_PYTHONPATH - prefix-relative Pythonpath, as determined by InterpreterConfig

# --------------------------------------------------------------------

# figure out bad shebang to replace, and replacement
set(BUILD_TREE_SHEBANG "#!${MINICONDA_INSTALL_DIR}/bin/python")
set(INSTALL_TREE_SHEBANG "#!${AMBER_INSTALL_DIR}/miniconda/bin/python")

# get files to fix
# --------------------------------------------------------------------

# get all files in Amber and Miniconda bin dirs
file(GLOB AMBER_BIN_FILES "${AMBER_INSTALL_DIR}/bin/*")
file(GLOB MINICONDA_BIN_FILES "${AMBER_INSTALL_DIR}/miniconda/bin/*")

# find all files starting with a shebang, to filter out executables
set(FILES_NEEDING_REPLACEMENT "")
foreach(FILE ${AMBER_BIN_FILES} ${MINICONDA_BIN_FILES})
	file(READ ${FILE} FILE_FIRST_CHARS LIMIT 3)
		
	if("${FILE_FIRST_CHARS}" MATCHES "#!")
		list(APPEND FILES_NEEDING_REPLACEMENT ${FILE})
	endif()
endforeach()

# get one last script that needs fixing
list(APPEND FILES_NEEDING_REPLACEMENT "${AMBER_INSTALL_DIR}/miniconda${PREFIX_RELATIVE_PYTHONPATH}/conda_build/convert.py")

#message("FILES_NEEDING_REPLACEMENT: ${FILES_NEEDING_REPLACEMENT}")

# now call the replacer script
# --------------------------------------------------------------------
include(${CMAKE_CURRENT_LIST_DIR}/../Replace-function.cmake)

foreach(FILE ${FILES_NEEDING_REPLACEMENT})
	configuretime_file_replace(${FILE} ${FILE} TO_REPLACE ${BUILD_TREE_SHEBANG} REPLACEMENT ${INSTALL_TREE_SHEBANG})
endforeach()