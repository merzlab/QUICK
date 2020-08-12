#
# AMBER: Taken from the Boost source (where it had the below comment)
# A big shout out to the cmake gurus @ compiz
#
# colormsg("Colors:"  
#   WHITE "white" GRAY "gray" GREEN "green" 
#   RED "red" YELLOW "yellow" BLUE "blue" MAG "mag" CYAN "cyan" 
#   _WHITE_ "white" _GRAY_ "gray" _GREEN_ "green" 
#   _RED_ "red" _YELLOW_ "yellow" _BLUE_ "blue" _MAG_ "mag" _CYAN_ "cyan" 
#   _HIWHITE_ "white" _HIGRAY_ "gray" _HIGREEN_ "green" 
#   _HIRED_ "red" _HIYELLOW_ "yellow" _HIBLUE_ "blue" _HIMAG_ "mag" _HICYAN_ "cyan" 
#   HIWHITE "white" HIGRAY "gray" HIGREEN "green" 
#   HIRED "red" HIYELLOW "yellow" HIBLUE "blue" HIMAG "mag" HICYAN "cyan" 
#   "right?")

# figure out if color messaging is supported
test(COLOR_MSG_SUPPORTED (NOT WIN32))

option(COLOR_CMAKE_MESSAGES "Colorize output from the configuration script.  This comes out all wrong if you use the GUI, so make sure to set this to false if you do." ${COLOR_MSG_SUPPORTED})

# AMBER: we put this stuff out here.  Yes, it pollutes the global namespace, but it also saves having to recalculate this each invocation
# (which is what the original function did)
string (ASCII 27 _escape)
set(WHITE "29")
set(GRAY "30")
set(RED "31")
set(GREEN "32")
set(YELLOW "33")
set(BLUE "34")
set(MAG "35")
set(CYAN "36")

foreach (color WHITE GRAY RED GREEN YELLOW BLUE MAG CYAN)
	set(HI${color} "1\;${${color}}")
	set(LO${color} "2\;${${color}}")
	set(_${color}_ "4\;${${color}}")
	set(_HI${color}_ "1\;4\;${${color}}")
	set(_LO${color}_ "2\;4\;${${color}}")
endforeach()

function (colormsg)

	if(COLOR_CMAKE_MESSAGES)
		set(str "")
		set(coloron FALSE)
		foreach(arg ${ARGV})
			if (DEFINED ${arg})
				if (CMAKE_COLOR_MAKEFILE)
					set(str "${str}${_escape}[${${arg}}m")
					set(coloron TRUE)
				endif()
			else()
				set(str "${str}${arg}")
				if (coloron)
					set(str "${str}${_escape}[0m")
					set(coloron FALSE)
				endif()
				set(str "${str} ")
			endif()
		endforeach()
		message(STATUS ${str})
	else()
		# just get the color words out of the arguments, then print the string
		set(str "")
	
		foreach(arg ${ARGV})
			if (DEFINED ${arg})
				# do nothing
			else()
				set(str "${str}${arg} ")
			endif()
		endforeach()
		message(STATUS ${str})
	endif()
endfunction()

# Print a boolean variable with a colored "ON" or "OFF" indicator
# Mainly used by the build report
function(color_print_bool MESSAGE VALUE)
	if("${VALUE}")
		colormsg(${MESSAGE} GREEN ON)
	else()
		colormsg(${MESSAGE} HIRED OFF)
	endif()
endfunction(color_print_bool)
