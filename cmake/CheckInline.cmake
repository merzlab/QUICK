# Inspired from /usr/share/autoconf/autoconf/c.m4

# Checks if the compiler supports the inline keyword, and fixes it with the preprocessor if the keyword is not "inline"

set(C_INLINE FALSE)

foreach(KEYWORD "inline" "__inline__" "__inline")
	if(NOT C_INLINE)
		try_compile(C_HAS_${KEYWORD} "${CMAKE_BINARY_DIR}/CMakeFiles" "${CMAKE_SOURCE_DIR}/cmake/test_inline.c" COMPILE_DEFINITIONS "-Dinline=${KEYWORD}")
		if(C_HAS_${KEYWORD})
			set(C_INLINE TRUE)
	       
			if(NOT ${KEYWORD} STREQUAL inline)
				add_definitions("-Dinline=${KEYWORD}")
			endif()
		endif()
	endif()
endforeach()
