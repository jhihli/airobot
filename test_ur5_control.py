# 保存为 ~/lerobot/test_ur5.py
import rtde_receive

ROBOT_IP = "192.168.0.102"

print("正在连接 UR5...")

rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
print("✅ 接收接口连接成功")

joints = rtde_r.getActualQ()
tcp = rtde_r.getActualTCPPose()
mode = rtde_r.getRobotMode()
safety_mode = rtde_r.getSafetyMode()
is_connected = rtde_r.isConnected()

print(f"\n--- UR5 当前状态 ---")
print(f"已连接: {is_connected}")
print(f"机器人模式: {mode}")
print(f"安全模式: {safety_mode}")
print(f"关节角度 (rad): {[round(j,3) for j in joints]}")
print(f"TCP 位姿 [x,y,z,rx,ry,rz]: {[round(v,4) for v in tcp]}")

rtde_r.disconnect()
print("\n✅ UR5 状态读取完成！")
