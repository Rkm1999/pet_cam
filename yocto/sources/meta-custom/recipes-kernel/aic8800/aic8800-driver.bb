# Recipe: meta-custom/recipes-kernel/aic8800/aic8800-driver.bb

SUMMARY = "Aicsemi AIC8800 Wi-Fi 6 SDIO Driver for Radxa Zero 3W"
LICENSE = "GPL-3.0-only"
LIC_FILES_CHKSUM = "file://LICENSE;md5=1ebbd3e34237af26da5dc08a4e440464"

# Use your local fixed repo
SRC_URI = "git:///home/ryu/aic8800-fix;protocol=file;branch=main"
SRCREV = "${AUTOREV}"

S = "${WORKDIR}/git"

inherit module

# Explicitly disable ALL non-Ubuntu platforms to prevent 32-bit logic from leaking in.
# We only enable UBUNTU (generic Linux) and pass necessary defines via CFLAGS.
EXTRA_OEMAKE += "CONFIG_PLATFORM_ROCKCHIP=n \
                 CONFIG_PLATFORM_ROCKCHIP2=n \
                 CONFIG_PLATFORM_ALLWINNER=n \
                 CONFIG_PLATFORM_AMLOGIC=n \
                 CONFIG_PLATFORM_INGENIC_T20=n \
                 CONFIG_PLATFORM_UBUNTU=y \
                 CONFIG_PLATFORM_ANDROID=n \
                 ARCH=arm64 \
                 CROSS_COMPILE=${TARGET_PREFIX} \
                 KERNEL_SRC=${STAGING_KERNEL_DIR} \
                 KDIR=${STAGING_KERNEL_DIR} \
                 USER_EXTRA_CFLAGS='-DCONFIG_PLATFORM_ROCKCHIP -DANDROID_PLATFORM'"

INSANE_SKIP:${PN}-dbg += "buildpaths"

MODULE_SRC_DIR = "${S}/src/SDIO/driver_fw/driver/aic8800"

do_compile() {
    oe_runmake -C ${STAGING_KERNEL_DIR} M=${MODULE_SRC_DIR} modules
}

do_install() {
    install -d ${D}${nonarch_base_libdir}/modules/${KERNEL_VERSION}/extra
    
    # Explicitly install the expected modules from their build locations.
    # If these fail, it means the build didn't produce them (easier to debug).
    install -m 0644 ${MODULE_SRC_DIR}/aic8800_bsp/aic8800_bsp.ko ${D}${nonarch_base_libdir}/modules/${KERNEL_VERSION}/extra/
    install -m 0644 ${MODULE_SRC_DIR}/aic8800_fdrv/aic8800_fdrv.ko ${D}${nonarch_base_libdir}/modules/${KERNEL_VERSION}/extra/
    
    # Optional: BT module if built
    # install -m 0644 ${MODULE_SRC_DIR}/aic8800_btlpm/aic8800_btlpm.ko ${D}${nonarch_base_libdir}/modules/${KERNEL_VERSION}/extra/
}

RDEPENDS:${PN} += "aic8800-firmware"