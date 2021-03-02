# Conda shebang fixer script helper buildfile.

# The Miniconda shebang fixer script neds to run after all other install rules.
# Unfortuantely, CMake does not have any way to make a custom install rule be installed after subdirectories.
# So, this is a dummy subdirectory for CMake to process as the last top-level subdirectory.

# All we have to do is
# create an install rule to invoke FixCondaShebang.cmake
install(CODE "
	message(STATUS \"Fixing Miniconda script shebangs\")
	execute_process(COMMAND ${CMAKE_COMMAND} 
	-DMINICONDA_INSTALL_DIR=${MINICONDA_INSTALL_DIR} 
	-DAMBER_INSTALL_DIR=\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_POSTFIX}
	-DPREFIX_RELATIVE_PYTHONPATH=${PREFIX_RELATIVE_PYTHONPATH}
	-P ${CMAKE_SOURCE_DIR}/cmake/FixCondaShebang/FixCondaShebang.cmake)
		" COMPONENT Python)