# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

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
CMAKE_COMMAND = /opt/cmake/bin/cmake

# The command to remove a file.
RM = /opt/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/mount/hanna/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/mount/hanna/code/build

# Include any dependencies generated for this target.
include CMakeFiles/MatterSim.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/MatterSim.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MatterSim.dir/flags.make

CMakeFiles/MatterSim.dir/src/lib/MatterSim.cpp.o: CMakeFiles/MatterSim.dir/flags.make
CMakeFiles/MatterSim.dir/src/lib/MatterSim.cpp.o: ../src/lib/MatterSim.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/mount/hanna/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MatterSim.dir/src/lib/MatterSim.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MatterSim.dir/src/lib/MatterSim.cpp.o -c /root/mount/hanna/code/src/lib/MatterSim.cpp

CMakeFiles/MatterSim.dir/src/lib/MatterSim.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MatterSim.dir/src/lib/MatterSim.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/mount/hanna/code/src/lib/MatterSim.cpp > CMakeFiles/MatterSim.dir/src/lib/MatterSim.cpp.i

CMakeFiles/MatterSim.dir/src/lib/MatterSim.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MatterSim.dir/src/lib/MatterSim.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/mount/hanna/code/src/lib/MatterSim.cpp -o CMakeFiles/MatterSim.dir/src/lib/MatterSim.cpp.s

CMakeFiles/MatterSim.dir/src/lib/NavGraph.cpp.o: CMakeFiles/MatterSim.dir/flags.make
CMakeFiles/MatterSim.dir/src/lib/NavGraph.cpp.o: ../src/lib/NavGraph.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/mount/hanna/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/MatterSim.dir/src/lib/NavGraph.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MatterSim.dir/src/lib/NavGraph.cpp.o -c /root/mount/hanna/code/src/lib/NavGraph.cpp

CMakeFiles/MatterSim.dir/src/lib/NavGraph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MatterSim.dir/src/lib/NavGraph.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/mount/hanna/code/src/lib/NavGraph.cpp > CMakeFiles/MatterSim.dir/src/lib/NavGraph.cpp.i

CMakeFiles/MatterSim.dir/src/lib/NavGraph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MatterSim.dir/src/lib/NavGraph.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/mount/hanna/code/src/lib/NavGraph.cpp -o CMakeFiles/MatterSim.dir/src/lib/NavGraph.cpp.s

CMakeFiles/MatterSim.dir/src/lib/Benchmark.cpp.o: CMakeFiles/MatterSim.dir/flags.make
CMakeFiles/MatterSim.dir/src/lib/Benchmark.cpp.o: ../src/lib/Benchmark.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/mount/hanna/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/MatterSim.dir/src/lib/Benchmark.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MatterSim.dir/src/lib/Benchmark.cpp.o -c /root/mount/hanna/code/src/lib/Benchmark.cpp

CMakeFiles/MatterSim.dir/src/lib/Benchmark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MatterSim.dir/src/lib/Benchmark.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/mount/hanna/code/src/lib/Benchmark.cpp > CMakeFiles/MatterSim.dir/src/lib/Benchmark.cpp.i

CMakeFiles/MatterSim.dir/src/lib/Benchmark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MatterSim.dir/src/lib/Benchmark.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/mount/hanna/code/src/lib/Benchmark.cpp -o CMakeFiles/MatterSim.dir/src/lib/Benchmark.cpp.s

CMakeFiles/MatterSim.dir/src/lib/cbf.cpp.o: CMakeFiles/MatterSim.dir/flags.make
CMakeFiles/MatterSim.dir/src/lib/cbf.cpp.o: ../src/lib/cbf.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/mount/hanna/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/MatterSim.dir/src/lib/cbf.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MatterSim.dir/src/lib/cbf.cpp.o -c /root/mount/hanna/code/src/lib/cbf.cpp

CMakeFiles/MatterSim.dir/src/lib/cbf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MatterSim.dir/src/lib/cbf.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/mount/hanna/code/src/lib/cbf.cpp > CMakeFiles/MatterSim.dir/src/lib/cbf.cpp.i

CMakeFiles/MatterSim.dir/src/lib/cbf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MatterSim.dir/src/lib/cbf.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/mount/hanna/code/src/lib/cbf.cpp -o CMakeFiles/MatterSim.dir/src/lib/cbf.cpp.s

# Object files for target MatterSim
MatterSim_OBJECTS = \
"CMakeFiles/MatterSim.dir/src/lib/MatterSim.cpp.o" \
"CMakeFiles/MatterSim.dir/src/lib/NavGraph.cpp.o" \
"CMakeFiles/MatterSim.dir/src/lib/Benchmark.cpp.o" \
"CMakeFiles/MatterSim.dir/src/lib/cbf.cpp.o"

# External object files for target MatterSim
MatterSim_EXTERNAL_OBJECTS =

libMatterSim.so: CMakeFiles/MatterSim.dir/src/lib/MatterSim.cpp.o
libMatterSim.so: CMakeFiles/MatterSim.dir/src/lib/NavGraph.cpp.o
libMatterSim.so: CMakeFiles/MatterSim.dir/src/lib/Benchmark.cpp.o
libMatterSim.so: CMakeFiles/MatterSim.dir/src/lib/cbf.cpp.o
libMatterSim.so: CMakeFiles/MatterSim.dir/build.make
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libOpenGL.so
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libEGL.so
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
libMatterSim.so: /usr/lib/x86_64-linux-gnu/libOpenGL.so
libMatterSim.so: CMakeFiles/MatterSim.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/mount/hanna/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX shared library libMatterSim.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MatterSim.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MatterSim.dir/build: libMatterSim.so

.PHONY : CMakeFiles/MatterSim.dir/build

CMakeFiles/MatterSim.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MatterSim.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MatterSim.dir/clean

CMakeFiles/MatterSim.dir/depend:
	cd /root/mount/hanna/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/mount/hanna/code /root/mount/hanna/code /root/mount/hanna/code/build /root/mount/hanna/code/build /root/mount/hanna/code/build/CMakeFiles/MatterSim.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MatterSim.dir/depend

