# Provides the copy_target() function, which is used to make duplicate MPI and OpenMP versions of targets.

# copies the value of a property from one target to another.  Does nothing if the property is not defined on the SOURCE.
function(copy_property PROPERTY SOURCE DESTINATION)
	get_property(PROP_IS_DEFINED TARGET ${SOURCE} PROPERTY ${PROPERTY} SET)
	
	if(PROP_IS_DEFINED)
		get_property(PROP_VALUE TARGET ${SOURCE} PROPERTY ${PROPERTY})
		set_property(TARGET ${DESTINATION} PROPERTY ${PROPERTY} "${PROP_VALUE}")
	endif()
	
endfunction(copy_property)

# Copies a target.  Creates a new target with a different name and the same sources and (important) properties.  Doesn't work on aliases or imported libraries.
# Allows switching out sources if needed.
# usage: copy_target(<target> <new name> [SWAP_SOURCES <source 1...> TO <replacement source 1...>])

function(copy_target SOURCE DESTINATION)

	#message("Copying target ${SOURCE} -> ${DESTINATION} --------------------------------------")
	# parse arguments
	# --------------------------------------------------------------------	
	cmake_parse_arguments(COPY_TARGET "" "" "LANGUAGES;SWAP_SOURCES;TO" ${ARGN})
	
	# set up source list
	# --------------------------------------------------------------------
	get_property(TARGET_SOURCES TARGET ${SOURCE} PROPERTY SOURCES)
	
	if(NOT "${COPY_TARGET_SWAP_SOURCES}" STREQUAL "")
		list(REMOVE_ITEM TARGET_SOURCES ${COPY_TARGET_SWAP_SOURCES})
	endif()
	
	if(NOT "${COPY_TARGET_TO}" STREQUAL "")
		list(APPEND TARGET_SOURCES ${COPY_TARGET_TO})
	endif()
	
	# create target according to type
	# --------------------------------------------------------------------
	get_property(TARGET_TYPE TARGET ${SOURCE} PROPERTY TYPE)
	
	if("${TARGET_TYPE}" STREQUAL "SHARED_LIBRARY")
		add_library(${DESTINATION} SHARED ${TARGET_SOURCES})
		
	elseif("${TARGET_TYPE}" STREQUAL "STATIC_LIBRARY")
		add_library(${DESTINATION} STATIC ${TARGET_SOURCES})
		
	elseif("${TARGET_TYPE}" STREQUAL "MODULE_LIBRARY")
		add_library(${DESTINATION} MODULE ${TARGET_SOURCES})
		
	elseif("${TARGET_TYPE}" STREQUAL "OBJECT_LIBRARY")
		add_library(${DESTINATION} OBJECT ${TARGET_SOURCES})
		
	elseif("${TARGET_TYPE}" STREQUAL "INTERFACE_LIBRARY")
		add_library(${DESTINATION} INTERFACE)
		
	elseif("${TARGET_TYPE}" STREQUAL "EXECUTABLE")
		add_executable(${DESTINATION} ${TARGET_SOURCES})
		
	else()
		message(FATAL_ERROR "copy_target(): cannot copy target ${SOURCE}: unknown target type ${TARGET_TYPE}")
	endif()
	
	# copy properties (feel free to add more if some are missing from this list)
	# this *should* be every target property that isn't obscure (ex: only used for QT support) or deprecated
	# --------------------------------------------------------------------
	copy_property(BUILD_WITH_INSTALL_RPATH ${SOURCE} ${DESTINATION})
	copy_property(C_EXTENSIONS ${SOURCE} ${DESTINATION})
	copy_property(C_STANDARD ${SOURCE} ${DESTINATION})
	copy_property(COMPILE_DEFINITIONS ${SOURCE} ${DESTINATION})
	copy_property(COMPILE_FEATURES ${SOURCE} ${DESTINATION})
	copy_property(COMPILE_FLAGS ${SOURCE} ${DESTINATION})
	copy_property(COMPILE_OPTIONS ${SOURCE} ${DESTINATION})
	copy_property(CXX_EXTENSIONS ${SOURCE} ${DESTINATION})
	copy_property(CXX_STANDARD ${SOURCE} ${DESTINATION})
	copy_property(DEBUG_POSTFIX ${SOURCE} ${DESTINATION})
	copy_property(DEFINE_SYMBOL ${SOURCE} ${DESTINATION})
	copy_property(ENABLE_EXPORTS ${SOURCE} ${DESTINATION})
	copy_property(EXCLUDE_FROM_ALL ${SOURCE} ${DESTINATION})
	copy_property(FRAMEWORK ${SOURCE} ${DESTINATION})
	copy_property(FOLDER ${SOURCE} ${DESTINATION})
	copy_property(Fortran_FORMAT ${SOURCE} ${DESTINATION})
	copy_property(Fortran_MODULE_DIRECTORY ${SOURCE} ${DESTINATION})
	copy_property(GNUtoMS ${SOURCE} ${DESTINATION})
	copy_property(IMPORT_PREFIX ${SOURCE} ${DESTINATION})
	copy_property(INCLUDE_DIRECTORIES ${SOURCE} ${DESTINATION})
	copy_property(INSTALL_NAME_DIR ${SOURCE} ${DESTINATION})
	copy_property(INSTALL_RPATH ${SOURCE} ${DESTINATION})
	copy_property(INSTALL_RPATH_USE_LINK_PATH ${SOURCE} ${DESTINATION})
	copy_property(INTERFACE_COMPILE_DEFINITIONS ${SOURCE} ${DESTINATION})
	copy_property(INTERFACE_COMPILE_FEATURES ${SOURCE} ${DESTINATION})
	copy_property(INTERFACE_COMPILE_OPTIONS ${SOURCE} ${DESTINATION})
	copy_property(INTERFACE_INCLUDE_DIRECTORIES ${SOURCE} ${DESTINATION})
	copy_property(INTERFACE_LINK_LIBRARIES ${SOURCE} ${DESTINATION})
	copy_property(INTERFACE_POSITION_INDEPENDENT_CODE ${SOURCE} ${DESTINATION})
	copy_property(INTERFACE_SOURCES ${SOURCE} ${DESTINATION})
	copy_property(INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${SOURCE} ${DESTINATION})
	copy_property(INTERPROCEDURAL_OPTIMIZATION ${SOURCE} ${DESTINATION})
	copy_property(LABELS ${SOURCE} ${DESTINATION})
	copy_property(LIBRARY_OUTPUT_DIRECTORY ${SOURCE} ${DESTINATION})
	copy_property(LIBRARY_OUTPUT_NAME ${SOURCE} ${DESTINATION})
	copy_property(LINKER_LANGUAGE ${SOURCE} ${DESTINATION})
	copy_property(LINK_FLAGS ${SOURCE} ${DESTINATION})
	copy_property(LINK_LIBRARIES ${SOURCE} ${DESTINATION})
	copy_property(MACOSX_BUNDLE_INFO_PLIST ${SOURCE} ${DESTINATION})
	copy_property(MACOSX_BUNDLE ${SOURCE} ${DESTINATION})
	copy_property(MACOSX_FRAMEWORK_INFO_PLIST ${SOURCE} ${DESTINATION})
	copy_property(MACOSX_RPATH ${SOURCE} ${DESTINATION})
	copy_property(NO_SONAME ${SOURCE} ${DESTINATION})
	copy_property(NO_SYSTEM_FROM_IMPORTED ${SOURCE} ${DESTINATION})
	copy_property(OSX_ARCHITECTURES ${SOURCE} ${DESTINATION})
	copy_property(OUTPUT_NAME ${SOURCE} ${DESTINATION})
	copy_property(POSITION_INDEPENDENT_CODE ${SOURCE} ${DESTINATION})
	copy_property(PREFIX ${SOURCE} ${DESTINATION})
	copy_property(PRIVATE_HEADER ${SOURCE} ${DESTINATION})
	copy_property(PROJECT_LABEL ${SOURCE} ${DESTINATION})
	copy_property(PUBLIC_HEADER ${SOURCE} ${DESTINATION})
	copy_property(RESOURCE ${SOURCE} ${DESTINATION})
	copy_property(RUNTIME_OUTPUT_DIRECTORY ${SOURCE} ${DESTINATION})
	copy_property(RUNTIME_OUTPUT_NAME ${SOURCE} ${DESTINATION})
	copy_property(SKIP_BUILD_RPATH ${SOURCE} ${DESTINATION})
	copy_property(SOVERSION ${SOURCE} ${DESTINATION})
	copy_property(STATIC_LIBRARY_FLAGS ${SOURCE} ${DESTINATION})
	copy_property(SUFFIX ${SOURCE} ${DESTINATION})
	copy_property(VERSION ${SOURCE} ${DESTINATION})
	copy_property(VISIBILITY_INLINES_HIDDEN ${SOURCE} ${DESTINATION})
	copy_property(VS_GLOBAL_KEYWORD ${SOURCE} ${DESTINATION})
	copy_property(VS_GLOBAL_PROJECT_TYPES ${SOURCE} ${DESTINATION})
	copy_property(VS_GLOBAL_ROOTNAMESPACE ${SOURCE} ${DESTINATION})
	copy_property(VS_KEYWORD ${SOURCE} ${DESTINATION})
	copy_property(VS_WINRT_EXTENSIONS ${SOURCE} ${DESTINATION})
	copy_property(WIN32_EXECUTABLE ${SOURCE} ${DESTINATION})
endfunction(copy_target)

# useful function when duplicating targets:
function(remove_link_libraries TARGET) # ARGN: LIBRARIES...
	get_property(TARGET_LINK_LIBRARIES TARGET ${TARGET} PROPERTY LINK_LIBRARIES)
	get_property(TARGET_INTERFACE_LINK_LIBRARIES TARGET ${TARGET} PROPERTY INTERFACE_LINK_LIBRARIES)
	
	list(REMOVE_ITEM TARGET_LINK_LIBRARIES ${ARGN})
	list(REMOVE_ITEM TARGET_INTERFACE_LINK_LIBRARIES ${ARGN})
	
	set_property(TARGET ${TARGET} PROPERTY LINK_LIBRARIES ${TARGET_LINK_LIBRARIES})
	set_property(TARGET ${TARGET} PROPERTY INTERFACE_LINK_LIBRARIES ${TARGET_INTERFACE_LINK_LIBRARIES})
endfunction(remove_link_libraries)
