#!/bin/bash
set -e

# The Neutron NPU firmware is loaded by the host kernel, but it has to match
# this container's Neutron delegate (and the converter the models were built
# with); a host OS with a different firmware makes the NPU return garbage.
# Point the kernel's firmware search path at our own copy. /tmp is the host's
# /tmp (bind mount), so the kernel can read it. Resets on reboot.
mkdir -p /tmp/neutron-fw
cp /usr/lib/firmware/NeutronFirmware.elf /tmp/neutron-fw/
echo -n /tmp/neutron-fw > /sys/module/firmware_class/parameters/path \
    || echo "WARNING: could not set firmware path, NPU results may be wrong" >&2

. .venv/bin/activate

exec python main-f.py
