from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from ..config import RobotConfig

@dataclass
class UR5HDDConfig:
    """Base configuration class for UR5 HDD disassembly robot."""

    # UR5 robot IP address
    robot_ip: str = "192.168.0.102"

    # EPG2 gripper serial port
    gripper_port: str = "/dev/ttyUSB0"

    # EPG2 gripper baud rate
    gripper_baudrate: int = 115200

    # EPG2 Modbus slave ID
    gripper_slave_id: int = 1

    # UR5 move speed (rad/s) for joint moves
    move_speed: float = 0.5

    # UR5 move acceleration (rad/s²)
    move_acceleration: float = 0.5

    # End-effector linear speed (m/s) for Cartesian moves
    ee_speed: float = 0.05

    # End-effector linear acceleration (m/s²)
    ee_acceleration: float = 0.05

    # Step size for keyboard teleoperation (meters per step)
    ee_step_size: float = 0.005

    # Gripper step size for keyboard teleoperation (normalized 0-1 per keypress)
    gripper_step_size: float = 1.0

    # Cameras — 3x Intel RealSense D455, 640x480 @ 5fps
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {
        "top": CameraConfig(
            fps=5,
            width=640,
            height=480
        ),
        "side_left": CameraConfig(
            fps=5,
            width=640,
            height=480
        ),
        "side_right": CameraConfig(
            fps=5,
            width=640,
            height=480
        ),
    })

@RobotConfig.register_subclass("ur5_hdd")
@dataclass
class UR5HDDRobotConfig(RobotConfig, UR5HDDConfig):
    pass
