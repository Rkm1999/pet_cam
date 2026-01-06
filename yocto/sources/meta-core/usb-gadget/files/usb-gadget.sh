#!/bin/sh

set -e

modprobe libcomposite || true

mkdir -p /sys/kernel/config
mountpoint -q /sys/kernel/config || mount -t configfs none /sys/kernel/config

cd /sys/kernel/config/usb_gadget

[ -d g1 ] || mkdir g1
cd g1

echo 0x1d6b > idVendor
echo 0x0104 > idProduct

mkdir -p strings/0x409
echo "zero3w-serial" > strings/0x409/serialnumber
echo "Radxa" > strings/0x409/manufacturer
echo "Zero 3W USB Serial" > strings/0x409/product

mkdir -p functions/acm.usb0
mkdir -p configs/c.1
ln -sf functions/acm.usb0 configs/c.1/

ls /sys/class/udc > UDC