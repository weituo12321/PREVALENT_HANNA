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
include CMakeFiles/MatterSimPython.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/MatterSimPython.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MatterSimPython.dir/flags.make

CMakeFiles/MatterSimPython.dir/src/lib_python/MatterSimPython.cpp.o: CMakeFiles/MatterSimPython.dir/flags.make
CMakeFiles/MatterSimPython.dir/src/lib_python/MatterSimPython.cpp.o: ../src/lib_python/MatterSimPython.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/mount/hanna/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MatterSimPython.dir/src/lib_python/MatterSimPython.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MatterSimPython.dir/src/lib_python/MatterSimPython.cpp.o -c /root/mount/hanna/code/src/lib_python/MatterSimPython.cpp

CMakeFiles/MatterSimPython.dir/src/lib_python/MatterSimPython.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MatterSimPython.dir/src/lib_python/MatterSimPython.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/mount/hanna/code/src/lib_python/MatterSimPython.cpp > CMakeFiles/MatterSimPython.dir/src/lib_python/MatterSimPython.cpp.i

CMakeFiles/MatterSimPython.dir/src/lib_python/MatterSimPython.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MatterSimPython.dir/src/lib_python/MatterSimPython.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/mount/hanna/code/src/lib_python/MatterSimPython.cpp -o CMakeFiles/MatterSimPython.dir/src/lib_python/MatterSimPython.cpp.s

# Object files for target MatterSimPython
MatterSimPython_OBJECTS = \
"CMakeFiles/MatterSimPython.dir/src/lib_python/MatterSimPython.cpp.o"

# External object files for target MatterSimPython
MatterSimPython_EXTERNAL_OBJECTS =

MatterSim.cpython-36m-x86_64-linux-gnu.so: CMakeFiles/MatterSimPython.dir/src/lib_python/MatterSimPython.cpp.o
MatterSim.cpython-36m-x86_64-linux-gnu.so: CMakeFiles/MatterSimPython.dir/build.make
MatterSim.cpython-36m-x86_64-linux-gnu.so: libMatterSim.so
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libEGL.so
MatterSim.cpython-36m-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libOpenGL.so
MatterSim.cpython-36m-x86_64-linux-gnu.so: CMakeFiles/MatterSimPython.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/mount/hanna/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module MatterSim.cpython-36m-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MatterSimPython.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /root/mount/hanna/code/build/MatterSim.cpython-36m-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/MatterSimPython.dir/build: MatterSim.cpython-36m-x86_64-linux-gnu.so

.PHONY : CMakeFiles/MatterSimPython.dir/build

CMakeFiles/MatterSimPython.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MatterSimPython.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MatterSimPython.dir/clean

CMakeFiles/MatterSimPython.dir/depend:
	cd /root/mount/hanna/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/mount/hanna/code /root/mount/hanna/code /root/mount/hanna/code/build /root/mount/hanna/code/build /root/mount/hanna/code/build/CMakeFiles/MatterSimPython.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MatterSimPython.dir/depend

