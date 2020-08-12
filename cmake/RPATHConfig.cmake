# Configures the RPATH.
# Must be included after CompilerOptions.cmake


if(TARGET_OSX)
	# recent macOS versions have disabled DYLD_LIBRARY_PATH for security reasons.
	# so, we set the RPATH to '<executable path>/../lib'
	
	# NOTE: if you change this here, you also have to change BundleOSXDependencies.cmake to account for it
	set(CMAKE_INSTALL_RPATH "@loader_path/../${LIBDIR}")
	set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
else()
	
	# set the RPATH to the absolute install dir.  This enables using many Amber programs without sourcing amber.sh.
	# If you do move the install tree, then amber.sh will still set LD_LIBRARY_PATH, and it'll will work fine once you've sourced it.
	set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIBDIR}")
	set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()