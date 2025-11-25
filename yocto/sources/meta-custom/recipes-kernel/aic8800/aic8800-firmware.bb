# Recipe: meta-custom/recipes-kernel/aic8800/aic8800-firmware.bb

SUMMARY = "Firmware binaries for Aicsemi AIC8800 Wi-Fi Chipset"
LICENSE = "CLOSED"

# Use local repo
SRC_URI = "git:///home/ryu/aic8800-fix;protocol=file;branch=main"
SRCREV = "${AUTOREV}"

S = "${WORKDIR}/git"

# Firmware packages don't need configuration or compilation
do_configure[noexec] = "1"
do_compile[noexec] = "1"

# Define the installation path expected by the driver (from dmesg error)
# Path: /lib/firmware/aic8800_fw/SDIO/aic8800D80/
FIRMWARE_INSTALL_DIR = "${nonarch_base_libdir}/firmware/aic8800_fw/SDIO/aic8800D80"

do_install() {
    install -d ${D}${FIRMWARE_INSTALL_DIR}

    # Copy the patch table we found
    install -m 0644 ${S}/src/SDIO/driver_fw/fw/aic8800D80/fw_patch_table_8800d80_u02.bin ${D}${FIRMWARE_INSTALL_DIR}/
    
    # Copy the user config text file (often required for initialization)
    install -m 0644 ${S}/src/SDIO/driver_fw/fw/aic8800D80/aic_userconfig_8800d80.txt ${D}${FIRMWARE_INSTALL_DIR}/

    # Attempt to copy any other binaries in that folder (like the main firmware blob)
    # We use find here so it doesn't fail if the file names vary slightly
    find ${S}/src/SDIO/driver_fw/fw/aic8800D80 -name "*.bin" -exec install -m 0644 {} ${D}${FIRMWARE_INSTALL_DIR}/ \;
}

# Package the files
FILES:${PN} += "${FIRMWARE_INSTALL_DIR}/*"
