SUMMARY = "USB gadget helper (script + systemd service)"
LICENSE = "MIT"
LIC_FILES_CHKSUM = "file://${COMMON_LICENSE_DIR}/MIT;md5=0835ade698e0bcf8506ecda2f7b4f302"

SRC_URI = " \
    file://usb-gadget.sh \
    file://usb-gadget.service \
"

S = "${WORKDIR}"

inherit systemd

do_install() {
    install -d ${D}${bindir}
    install -m 0755 ${WORKDIR}/usb-gadget.sh ${D}${bindir}/usb-gadget.sh

    install -d ${D}${systemd_system_unitdir}
    install -m 0644 ${WORKDIR}/usb-gadget.service ${D}${systemd_system_unitdir}/usb-gadget.service
}

SYSTEMD_SERVICE:${PN} = "usb-gadget.service"
SYSTEMD_AUTO_ENABLE:${PN} = "enable"