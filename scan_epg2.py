# 保存为 ~/lerobot/scan_coils.py
from pymodbus.client import ModbusSerialClient
import time

PORT = "/dev/ttyUSB0"
BAUD = 115200
SLAVE_ID = 1

client = ModbusSerialClient(
    port=PORT, baudrate=BAUD,
    bytesize=8, parity='N', stopbits=1, timeout=1
)
client.connect()

# 方法1：扫描线圈（Coils）0x01
print("=== 扫描线圈 Coils (地址 0-100) ===")
for addr in range(0, 100):
    try:
        r = client.read_coils(address=addr, count=1, device_id=SLAVE_ID)
        if hasattr(r, 'bits'):
            print(f"线圈 地址 {addr:3d} = {r.bits[0]}")
    except:
        pass
    time.sleep(0.02)

# 方法2：扫描离散输入（Discrete Inputs）0x02
print("\n=== 扫描离散输入 Discrete Inputs (地址 0-100) ===")
for addr in range(0, 100):
    try:
        r = client.read_discrete_inputs(address=addr, count=1, device_id=SLAVE_ID)
        if hasattr(r, 'bits'):
            print(f"离散输入 地址 {addr:3d} = {r.bits[0]}")
    except:
        pass
    time.sleep(0.02)

# 方法3：扫描输入寄存器（Input Registers）0x04
print("\n=== 扫描输入寄存器 Input Registers (地址 0-300) ===")
for addr in range(0, 300):
    try:
        r = client.read_input_registers(address=addr, count=1, device_id=SLAVE_ID)
        if hasattr(r, 'registers'):
            print(f"输入寄存器 地址 {addr:3d} = {r.registers[0]}")
    except:
        pass
    time.sleep(0.02)

client.close()
print("\n扫描完成！")
