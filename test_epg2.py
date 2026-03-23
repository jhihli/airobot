# 保存为 ~/lerobot/test_epg2.py
from pymodbus.client import ModbusSerialClient
import time

PORT = "/dev/ttyUSB0"
SLAVE_ID = 1
BAUDRATES = [115200, 9600, 19200, 38400, 57600]

for baud in BAUDRATES:
    print(f"\n尝试波特率: {baud}")
    client = ModbusSerialClient(
        port=PORT,
        baudrate=baud,
        bytesize=8,
        parity='N',
        stopbits=1,
        timeout=1
    )
    
    try:
        client.connect()
        # pymodbus 3.12 正确参数名是 device_id
        result = client.read_holding_registers(address=260, count=4, device_id=SLAVE_ID)
        
        if hasattr(result, 'registers'):
            print(f"✅ 连接成功！波特率={baud}, 从机ID={SLAVE_ID}")
            print(f"   当前位置: {result.registers[0]}")
            print(f"   当前力:   {result.registers[1]}")
            print(f"   当前速度: {result.registers[2]}")
            print(f"   状态字:   {result.registers[3]}")
            client.close()
            break
        else:
            print(f"   读取失败: {result}")
            
    except Exception as e:
        print(f"   异常: {e}")
    finally:
        client.close()
        time.sleep(0.5)
