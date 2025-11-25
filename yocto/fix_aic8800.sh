#!/bin/bash

# -----------------------------------------------------------------------------
# FIX SCRIPT V4 (Return 1 Variant):
# 1. Cleans garbage binaries
# 2. Applies Debian Patches
# 3. Fixes Architecture hardcoding
# 4. Fixes C Syntax errors
# 5. NEW: Bypasses Power-on Check (Returns 1)
# -----------------------------------------------------------------------------

if [ ! -d "debian/patches" ]; then
    echo "Error: debian/patches not found. Are you in the aic8800-fix root?"
    exit 1
fi

# -----------------------------------------------------------------------------
# 1. CLEANUP: Remove pre-compiled binaries that cause Arch Mismatch
# -----------------------------------------------------------------------------
echo "-> 1. Cleaning pre-compiled garbage..."
find . -name "*.ko" -type f -delete -print
find . -name "*.o" -type f -delete
find . -name "*.cmd" -type f -delete
find . -name "modules.order" -type f -delete
find . -name "Module.symvers" -type f -delete
echo "   Cleanup complete."

# -----------------------------------------------------------------------------
# 2. PATCHES: Apply Debian patches for Kernel 6.x support
# -----------------------------------------------------------------------------
echo "-> 2. Applying Debian Patches..."
grep -v '^#' debian/patches/series | while read patch; do
    [ -z "$patch" ] && continue
    echo "   Applying $patch..."
    patch -p1 --forward --ignore-whitespace < "debian/patches/$patch" || echo "   Note: $patch might already be applied."
done

# -----------------------------------------------------------------------------
# 3. MAKEFILES: Remove hardcoded ARCH and Cross-Compiler paths
# -----------------------------------------------------------------------------
echo "-> 3. Fixing Makefiles..."
find src/SDIO/driver_fw/driver/aic8800 -name "Makefile" | while read -r makefile; do
    echo "   Scrubbing $makefile"
    sed -i '/^\s*ARCH\s*[?:]*=/d' "$makefile"
    sed -i '/^\s*CROSS_COMPILE\s*[?:]*=/d' "$makefile"
    sed -i '/^\s*KDIR\s*[?:]*=/d' "$makefile"
    sed -i '/\/home\/yaya/d' "$makefile"
    sed -i '1i EXTRA_CFLAGS += -Wno-error=missing-prototypes -Wno-error=incompatible-pointer-types' "$makefile"
done

# -----------------------------------------------------------------------------
# 4. HEADERS: Fix obsolete rfkill header
# -----------------------------------------------------------------------------
echo "-> 4. Fixing RFKILL headers..."
grep -rl "linux/rfkill-wlan.h" src/ | while read -r file; do
    sed -i 's|linux/rfkill-wlan.h|linux/rfkill.h|g' "$file"
done

# -----------------------------------------------------------------------------
# 5. SYNTAX: Fix MODULE_IMPORT_NS quoting
# -----------------------------------------------------------------------------
echo "-> 5. Fixing MODULE_IMPORT_NS syntax..."
grep -rl "MODULE_IMPORT_NS(VFS_internal" src/ | while read -r file; do
    sed -i 's/MODULE_IMPORT_NS(VFS_internal_I_am_really_a_filesystem_and_am_NOT_a_driver)/MODULE_IMPORT_NS("VFS_internal_I_am_really_a_filesystem_and_am_NOT_a_driver")/g' "$file"
done

# -----------------------------------------------------------------------------
# 6. POWER: Bypass Dummy Driver Check (Return 1)
# -----------------------------------------------------------------------------
echo "-> 6. Short-circuiting Power Check (Returning 1)..."
# Reverts any previous manual fix to avoid stacking "return 0; return 1;"
target_file="src/SDIO/driver_fw/driver/aic8800/aic8800_bsp/aicsdio.c"

# Reset the file if it was already patched with return 0 (optional safety)
# This assumes you haven't committed the previous 'return 0' patch yet, 
# or you want to overwrite it. Ideally run on a clean file.

# Find the function and inject 'return 1;' 
sed -i '/static int aicbsp_platform_power_on(void)/{n;s/{/{ return 1; \/\//;}' "$target_file"

echo "----------------------------------------------------------------"
echo "All fixes applied. Please run:"
echo "  git add ."
echo "  git commit -m 'Fix: Bypass power check with return 1'"
echo "----------------------------------------------------------------"
