# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

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
CMAKE_SOURCE_DIR = /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/bin

# Include any dependencies generated for this target.
include CMakeFiles/dummy.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dummy.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dummy.dir/flags.make

CMakeFiles/dummy.dir/main.cpp.o: CMakeFiles/dummy.dir/flags.make
CMakeFiles/dummy.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dummy.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dummy.dir/main.cpp.o -c /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/main.cpp

CMakeFiles/dummy.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dummy.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/main.cpp > CMakeFiles/dummy.dir/main.cpp.i

CMakeFiles/dummy.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dummy.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/main.cpp -o CMakeFiles/dummy.dir/main.cpp.s

CMakeFiles/dummy.dir/src/discrete_environment.cpp.o: CMakeFiles/dummy.dir/flags.make
CMakeFiles/dummy.dir/src/discrete_environment.cpp.o: ../src/discrete_environment.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/dummy.dir/src/discrete_environment.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dummy.dir/src/discrete_environment.cpp.o -c /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/discrete_environment.cpp

CMakeFiles/dummy.dir/src/discrete_environment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dummy.dir/src/discrete_environment.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/discrete_environment.cpp > CMakeFiles/dummy.dir/src/discrete_environment.cpp.i

CMakeFiles/dummy.dir/src/discrete_environment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dummy.dir/src/discrete_environment.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/discrete_environment.cpp -o CMakeFiles/dummy.dir/src/discrete_environment.cpp.s

CMakeFiles/dummy.dir/src/random_utils.cpp.o: CMakeFiles/dummy.dir/flags.make
CMakeFiles/dummy.dir/src/random_utils.cpp.o: ../src/random_utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/dummy.dir/src/random_utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dummy.dir/src/random_utils.cpp.o -c /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/random_utils.cpp

CMakeFiles/dummy.dir/src/random_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dummy.dir/src/random_utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/random_utils.cpp > CMakeFiles/dummy.dir/src/random_utils.cpp.i

CMakeFiles/dummy.dir/src/random_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dummy.dir/src/random_utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/random_utils.cpp -o CMakeFiles/dummy.dir/src/random_utils.cpp.s

CMakeFiles/dummy.dir/src/envs/frozen_lake.cpp.o: CMakeFiles/dummy.dir/flags.make
CMakeFiles/dummy.dir/src/envs/frozen_lake.cpp.o: ../src/envs/frozen_lake.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/dummy.dir/src/envs/frozen_lake.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dummy.dir/src/envs/frozen_lake.cpp.o -c /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/envs/frozen_lake.cpp

CMakeFiles/dummy.dir/src/envs/frozen_lake.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dummy.dir/src/envs/frozen_lake.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/envs/frozen_lake.cpp > CMakeFiles/dummy.dir/src/envs/frozen_lake.cpp.i

CMakeFiles/dummy.dir/src/envs/frozen_lake.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dummy.dir/src/envs/frozen_lake.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/envs/frozen_lake.cpp -o CMakeFiles/dummy.dir/src/envs/frozen_lake.cpp.s

CMakeFiles/dummy.dir/src/envs/frozen_lake_ext.cpp.o: CMakeFiles/dummy.dir/flags.make
CMakeFiles/dummy.dir/src/envs/frozen_lake_ext.cpp.o: ../src/envs/frozen_lake_ext.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/dummy.dir/src/envs/frozen_lake_ext.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dummy.dir/src/envs/frozen_lake_ext.cpp.o -c /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/envs/frozen_lake_ext.cpp

CMakeFiles/dummy.dir/src/envs/frozen_lake_ext.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dummy.dir/src/envs/frozen_lake_ext.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/envs/frozen_lake_ext.cpp > CMakeFiles/dummy.dir/src/envs/frozen_lake_ext.cpp.i

CMakeFiles/dummy.dir/src/envs/frozen_lake_ext.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dummy.dir/src/envs/frozen_lake_ext.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/envs/frozen_lake_ext.cpp -o CMakeFiles/dummy.dir/src/envs/frozen_lake_ext.cpp.s

CMakeFiles/dummy.dir/src/envs/copy.cpp.o: CMakeFiles/dummy.dir/flags.make
CMakeFiles/dummy.dir/src/envs/copy.cpp.o: ../src/envs/copy.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/dummy.dir/src/envs/copy.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dummy.dir/src/envs/copy.cpp.o -c /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/envs/copy.cpp

CMakeFiles/dummy.dir/src/envs/copy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dummy.dir/src/envs/copy.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/envs/copy.cpp > CMakeFiles/dummy.dir/src/envs/copy.cpp.i

CMakeFiles/dummy.dir/src/envs/copy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dummy.dir/src/envs/copy.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/src/envs/copy.cpp -o CMakeFiles/dummy.dir/src/envs/copy.cpp.s

# Object files for target dummy
dummy_OBJECTS = \
"CMakeFiles/dummy.dir/main.cpp.o" \
"CMakeFiles/dummy.dir/src/discrete_environment.cpp.o" \
"CMakeFiles/dummy.dir/src/random_utils.cpp.o" \
"CMakeFiles/dummy.dir/src/envs/frozen_lake.cpp.o" \
"CMakeFiles/dummy.dir/src/envs/frozen_lake_ext.cpp.o" \
"CMakeFiles/dummy.dir/src/envs/copy.cpp.o"

# External object files for target dummy
dummy_EXTERNAL_OBJECTS =

dummy: CMakeFiles/dummy.dir/main.cpp.o
dummy: CMakeFiles/dummy.dir/src/discrete_environment.cpp.o
dummy: CMakeFiles/dummy.dir/src/random_utils.cpp.o
dummy: CMakeFiles/dummy.dir/src/envs/frozen_lake.cpp.o
dummy: CMakeFiles/dummy.dir/src/envs/frozen_lake_ext.cpp.o
dummy: CMakeFiles/dummy.dir/src/envs/copy.cpp.o
dummy: CMakeFiles/dummy.dir/build.make
dummy: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
dummy: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
dummy: CMakeFiles/dummy.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable dummy"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dummy.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dummy.dir/build: dummy

.PHONY : CMakeFiles/dummy.dir/build

CMakeFiles/dummy.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dummy.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dummy.dir/clean

CMakeFiles/dummy.dir/depend:
	cd /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++ /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++ /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/bin /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/bin /home/tuandam/workspace/h-greedy-tree-search-icml2020/c++/bin/CMakeFiles/dummy.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dummy.dir/depend
