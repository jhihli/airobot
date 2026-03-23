# 保存为 ~/lerobot/test_epg2_v4.py
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

def send_raw(hex_str):
    """直接发送原始 Modbus 字节（不含 CRC，pymodbus 自动加）"""
    pass

def write_registers_raw(address, values):
    """写多个寄存器"""
    result = client.write_registers(
        address=address,
        values=values,
        device_id=SLAVE_ID
    )
    time.sleep(0.1)
    return result

def reset():
    """复位: 01 10 00 E4 00 01 02 00 02"""
    print(">>> 复位...")
    write_registers_raw(0x00E4, [0x0002])
    time.sleep(1)

def move_to_position(pos_um):
    """
    移动到指定位置
    pos_um: 微米单位, 0=闭合, 26000=全开(26mm)
    命令格式: 01 10 00 20 00 06 0C [speed_H speed_L] 00 00 00 00 [speed2_H speed2_L] [pos_H pos_L] 00 00
    从 sscom 数据分析:
    N1(0mm):   27 10 00 00 00 00 27 10 00 00 00 00  → pos=0x0000=0
    N2(2.6mm): 27 10 00 00 00 00 27 10 03 E8 00 00  → pos=0x03E8=1000
    N4(26mm):  27 10 00 00 00 00 27 10 27 10 00 00  → pos=0x2710=10000
    规律: 26mm=10000, 1mm≈384.6, pos_um/2.6 * 1000
    """
    pos_val = int(pos_um / 26000 * 10000)
    pos_val = max(0, min(10000, pos_val))
    
    speed = 0x2710  # 10000 (从 sscom 数据)
    
    values = [
        speed,      # 寄存器32: 速度1
        0x0000,     # 寄存器33
        0x0000,     # 寄存器34
        speed,      # 寄存器35: 速度2
        pos_val,    # 寄存器36: 目标位置
        0x0000      # 寄存器37
    ]
    print(f">>> 移动到 {pos_um/1000:.1f}mm (pos_val={pos_val})")
    result = write_registers_raw(0x0020, values)
    print(f"    写入结果: {result}")

def read_status():
    """读取状态: 01 03 00 20 00 06"""
    r = client.read_holding_registers(address=0x0020, count=6, device_id=SLAVE_ID)
    if hasattr(r, 'registers'):
        print(f"  状态寄存器32-37: {[hex(v) for v in r.registers]}")
        print(f"  位置值: {r.registers[4]} → {r.registers[4]/10000*26:.2f}mm")

print("=== 初始状态 ===")
read_status()

print("\n>>> 步骤1: 复位")
reset()
read_status()

print("\n>>> 步骤2: 全开 (26mm)")
move_to_position(26000)
time.sleep(3)
read_status()

print("\n>>> 步骤3: 闭合 (0mm)")
move_to_position(0)
time.sleep(3)
read_status()

print("\n>>> 步骤4: 半开 (13mm)")
move_to_position(13000)
time.sleep(3)
read_status()

client.close()
print("\n完成！")
