# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/duka/Projects/Talk2Car_Path/baselines/end_position/MDN/wemd

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/duka/Projects/Talk2Car_Path/baselines/end_position/MDN/wemd/build

# Include any dependencies generated for this target.
include CMakeFiles/wemd.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/wemd.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/wemd.dir/flags.make

CMakeFiles/wemd.dir/wemd_impl.cpp.o: CMakeFiles/wemd.dir/flags.make
CMakeFiles/wemd.dir/wemd_impl.cpp.o: ../wemd_impl.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/duka/Projects/Talk2Car_Path/baselines/end_position/MDN/wemd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/wemd.dir/wemd_impl.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/wemd.dir/wemd_impl.cpp.o -c /home/duka/Projects/Talk2Car_Path/baselines/end_position/MDN/wemd/wemd_impl.cpp

CMakeFiles/wemd.dir/wemd_impl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/wemd.dir/wemd_impl.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/duka/Projects/Talk2Car_Path/baselines/end_position/MDN/wemd/wemd_impl.cpp > CMakeFiles/wemd.dir/wemd_impl.cpp.i

CMakeFiles/wemd.dir/wemd_impl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/wemd.dir/wemd_impl.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/duka/Projects/Talk2Car_Path/baselines/end_position/MDN/wemd/wemd_impl.cpp -o CMakeFiles/wemd.dir/wemd_impl.cpp.s

CMakeFiles/wemd.dir/wemd_impl.cpp.o.requires:

.PHONY : CMakeFiles/wemd.dir/wemd_impl.cpp.o.requires

CMakeFiles/wemd.dir/wemd_impl.cpp.o.provides: CMakeFiles/wemd.dir/wemd_impl.cpp.o.requires
	$(MAKE) -f CMakeFiles/wemd.dir/build.make CMakeFiles/wemd.dir/wemd_impl.cpp.o.provides.build
.PHONY : CMakeFiles/wemd.dir/wemd_impl.cpp.o.provides

CMakeFiles/wemd.dir/wemd_impl.cpp.o.provides.build: CMakeFiles/wemd.dir/wemd_impl.cpp.o


# Object files for target wemd
wemd_OBJECTS = \
"CMakeFiles/wemd.dir/wemd_impl.cpp.o"

# External object files for target wemd
wemd_EXTERNAL_OBJECTS =

../lib/libwemd.so: CMakeFiles/wemd.dir/wemd_impl.cpp.o
../lib/libwemd.so: CMakeFiles/wemd.dir/build.make
../lib/libwemd.so: CMakeFiles/wemd.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/duka/Projects/Talk2Car_Path/baselines/end_position/MDN/wemd/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ../lib/libwemd.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/wemd.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/wemd.dir/build: ../lib/libwemd.so

.PHONY : CMakeFiles/wemd.dir/build

CMakeFiles/wemd.dir/requires: CMakeFiles/wemd.dir/wemd_impl.cpp.o.requires

.PHONY : CMakeFiles/wemd.dir/requires

CMakeFiles/wemd.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/wemd.dir/cmake_clean.cmake
.PHONY : CMakeFiles/wemd.dir/clean

CMakeFiles/wemd.dir/depend:
	cd /home/duka/Projects/Talk2Car_Path/baselines/end_position/MDN/wemd/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/duka/Projects/Talk2Car_Path/baselines/end_position/MDN/wemd /home/duka/Projects/Talk2Car_Path/baselines/end_position/MDN/wemd /home/duka/Projects/Talk2Car_Path/baselines/end_position/MDN/wemd/build /home/duka/Projects/Talk2Car_Path/baselines/end_position/MDN/wemd/build /home/duka/Projects/Talk2Car_Path/baselines/end_position/MDN/wemd/build/CMakeFiles/wemd.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/wemd.dir/depend

