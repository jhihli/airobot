#!/usr/bin/env python3
from pymodbus.client import ModbusSerialClient
import sys
import select
import termios
import tty

print("=== EPG2 Gripper Control ===")

# 连接夹爪
gripper = ModbusSerialClient(
    port="/dev/ttyUSB0",
    baudrate=115200,
    bytesize=8,
    parity="N",
    stopbits=1,
    timeout=1,
)

if not gripper.connect():
    print("❌ Failed to connect to gripper on /dev/ttyUSB0")
    print("Make sure:")
    print("  1. USB-RS485 cable is connected")
    print("  2. Run: sudo chmod 666 /dev/ttyUSB0")
    sys.exit(1)

print("✅ Connected to EPG2 gripper")

def open_gripper():
    print("→ Opening gripper (26mm)...")
    gripper.write_registers(0x0020, [10000,0,0,10000,10000,0], device_id=1)

def close_gripper():
    print("→ Closing gripper (0mm)...")
    gripper.write_registers(0x0020, [10000,0,0,10000,0,0], device_id=1)

def half_open():
    print("→ Half opening gripper (13mm)...")
    gripper.write_registers(0x0020, [10000,0,0,10000,5000,0], device_id=1)

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

print("\n📋 Controls:")
print("  o - OPEN gripper (26mm)")
print("  c - CLOSE gripper (0mm)")
print("  h - HALF open (13mm)")
print("  q - QUIT")
print("\n💡 Tip: Use UR5 freedrive button to move the robot")
print("       Use this script to control gripper\n")

old_settings = termios.tcgetattr(sys.stdin)
try:
    tty.setcbreak(sys.stdin.fileno())
    
    print("Ready! Press keys to control gripper...\n")
    
    while True:
        if isData():
            c = sys.stdin.read(1)
            
            if c == 'o':
                open_gripper()
            elif c == 'c':
                close_gripper()
            elif c == 'h':
                half_open()
            elif c == 'q':
                print("\nExiting...")
                break
                
except KeyboardInterrupt:
    print("\n\nInterrupted by user")
finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    gripper.close()
    print("✅ Gripper disconnected")
    print("Done!")
