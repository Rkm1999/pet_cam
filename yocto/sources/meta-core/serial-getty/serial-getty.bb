SUMMARY = "Serial getty on ttyGS0"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

inherit systemd

SRC_URI = "file://serial-getty@ttyGS0.service"

SYSTEMD_SERVICE:${PN} = "serial-getty@ttyGS0.service"
SYSTEMD_AUTO_ENABLE:${PN} = "enable"

FILES:${PN} += "${systemd_system_unitdir}/serial-getty@ttyGS0.service"

do_install() {
    install -d ${D}${systemd_system_unitdir}
    install -m 0644 ${WORKDIR}/serial-getty@ttyGS0.service \
        ${D}${systemd_system_unitdir}/
}