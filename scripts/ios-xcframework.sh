#!/bin/sh
#
# Copyright 2026 Nope Forge
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

#
# Package static iOS builds (see `configure.py --host iOS`) into a single
# NopeGL.xcframework. Each prefix becomes one slice of the XCFramework: all its
# static libraries (nope.gl, nope.media, FFmpeg, BoringSSL, glslang,
# SPIRV-Cross, FreeType, HarfBuzz, FriBidi, ...) are merged into one libNopeGL.a
# and the public headers are exposed with a "NopeGL" Clang module, marked extern_c so
# C++ clients (and Swift with C++ interop) link against the C symbols.
#
# Usage: ios-xcframework.sh <output-dir> <ios-prefix> [<ios-prefix>...]
# Example:
#   ./configure.py --host iOS --ios-platform iphoneos && make -f Makefile.iOS.aarch64
#   ./configure.py --host iOS --ios-platform iphonesimulator && make -f Makefile.iOS.aarch64-simulator
#   scripts/ios-xcframework.sh dist-ios iOS/aarch64 iOS/aarch64-simulator
#

set -eu

if [ $# -lt 2 ]; then
    echo "Usage: $0 <output-dir> <ios-prefix> [<ios-prefix>...]" >&2
    exit 1
fi

outdir="$1"
shift

# Libraries installed in the prefix that nope.gl does not need
excluded_libs="libavdevice.a libharfbuzz-subset.a"

workdir="$outdir/.work"
rm -rf "$workdir" "$outdir/NopeGL.xcframework"
mkdir -p "$workdir"

xcf_args=""
i=0
for prefix in "$@"; do
    i=$((i + 1))
    slice="$workdir/slice$i"
    mkdir -p "$slice/Headers/nopegl"

    libs=""
    for lib in "$prefix"/lib/*.a; do
        name=$(basename "$lib")
        case " $excluded_libs " in
            *" $name "*) continue ;;
        esac
        libs="$libs $lib"
    done
    # shellcheck disable=SC2086
    libtool -static -no_warning_for_no_symbols -o "$slice/libNopeGL.a" $libs

    cp "$prefix/include/nopegl/nopegl.h" "$slice/Headers/nopegl/"
    cp "$prefix/include/nopemd.h" "$slice/Headers/"
    cat > "$slice/Headers/module.modulemap" <<EOF
module NopeGL [extern_c] {
    header "nopegl/nopegl.h"
    export *
}
EOF
    xcf_args="$xcf_args -library $slice/libNopeGL.a -headers $slice/Headers"
done

# shellcheck disable=SC2086
xcodebuild -create-xcframework $xcf_args -output "$outdir/NopeGL.xcframework"
rm -rf "$workdir"
