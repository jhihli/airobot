import time
import numpy as np
from pathlib import Path
import rtde_control
import rtde_receive
from pymodbus.client import ModbusSerialClient

from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

# Constants
UR5_JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']

# Camera serial number mapping
CAMERA_SERIAL_NUMBERS = {
    "top": "333422305051",
    "side_left": "333422303023",
    "side_right": "338122302446"
}

class RobotDeviceNotConnectedError(Exception):
    pass

class RobotDeviceAlreadyConnectedError(Exception):
    pass

class UR5HDDRobot:
    """
    UR5 Robot with EPG2 Gripper and 3x RealSense D455 cameras.
    For HDD disassembly task.
    """
    
    def __init__(self, config):
        self.config = config
        self.is_connected = False
        
        # Robot controllers
        self._rtde_control = None
        self._rtde_receive = None
        
        # Gripper
        self._gripper_client = None
        
        # Cameras
        self._cameras = {}
        self._camera_configs = config.cameras if hasattr(config, 'cameras') else {}
        
        # Last action tracking
        self._last_action = {}
    
    @property
    def name(self):
        return "ur5_hdd"
    
    @property
    def action_features(self):
        """Action features: delta_x, delta_y, delta_z, gripper"""
        return ["action"]
    
    @property
    def observation_features(self):
        """Observation features: state + 3 camera images"""
        features = ["observation.state"]
        for camera_name in self._camera_configs.keys():
            features.append(f"observation.images.{camera_name}")
        return features
    
    def connect(self):
        """Connect to UR5, gripper, and cameras."""
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"{self.name} UR5HDDRobot is already connected."
            )
        
        # Connect UR5
        print(f"Connecting to UR5 at {self.config.robot_ip} ...")
        self._rtde_control = rtde_control.RTDEControlInterface(self.config.robot_ip)
        self._rtde_receive = rtde_receive.RTDEReceiveInterface(self.config.robot_ip)
        print("✅ UR5 connected")
        
        # Connect EPG2 gripper
        print(f"Connecting to EPG2 gripper on {self.config.gripper_port} ...")
        self._gripper_client = ModbusSerialClient(
            port=self.config.gripper_port,
            baudrate=self.config.gripper_baudrate,
            parity='N',
            stopbits=1,
            bytesize=8,
            timeout=1
        )
        self._gripper_client.connect()
        print("✅ EPG2 gripper connected")
        
        # Connect cameras
        for camera_name, camera_config in self._camera_configs.items():
            serial_num = CAMERA_SERIAL_NUMBERS.get(camera_name, '')
            fps = camera_config.fps if camera_config.fps else 30
            width = camera_config.width if camera_config.width else 640
            height = camera_config.height if camera_config.height else 480

            print(f"Connecting camera: {camera_name} (SN: {serial_num}, {width}x{height}@{fps}fps)")

            rs_config = RealSenseCameraConfig(
                serial_number_or_name=serial_num,
                fps=fps,
                width=width,
                height=height
            )

            camera = RealSenseCamera(config=rs_config)
            camera.connect()
            self._cameras[camera_name] = camera
            print(f"Camera {camera_name} connected")

        # Warmup: wait for auto-exposure to stabilize (skill requirement: warmup_s=3)
        if self._cameras:
            print("Waiting 3s for camera auto-exposure to stabilize...")
            time.sleep(3)
        
        # Set connected flag
        self.is_connected = True
        
        # Configure robot
        self.configure()
        
        print(f"{self.name} UR5HDDRobot connected successfully.")
    
    def disconnect(self):
        """Disconnect from all devices."""
        if not self.is_connected:
            return
        
        # Disconnect UR5
        if self._rtde_control:
            try:
                print("Stopping robot motion...")
                self._rtde_control.speedStop()
                self._rtde_control.stopJ(1.0)
                print("Robot stopped")
            except Exception as e:
                print(f"Warning: error stopping robot: {e}")
            
            try:
                self._rtde_control.disconnect()
            except:
                pass
            self._rtde_control = None
        
        if self._rtde_receive:
            try:
                self._rtde_receive.disconnect()
            except:
                pass
            self._rtde_receive = None
        
        # Disconnect gripper
        if self._gripper_client:
            try:
                self._gripper_client.close()
            except:
                pass
            self._gripper_client = None
        
        # Disconnect cameras
        for camera_name, camera in self._cameras.items():
            try:
                camera.disconnect()
                print(f"Camera {camera_name} disconnected")
            except:
                pass
        self._cameras.clear()
        
        self.is_connected = False
        print(f"{self.name} UR5HDDRobot disconnected.")
    
    def configure(self):
        """Configure robot - Initialize last action state."""
        if self._rtde_receive is None:
            raise RobotDeviceNotConnectedError("Robot receive interface not initialized")
        
        # Initialize last action
        joints = self._rtde_receive.getActualQ()
        self._last_action = {}
        for i, name in enumerate(UR5_JOINT_NAMES):
            self._last_action[f"{name}.pos"] = float(joints[i])
        self._last_action["gripper.pos"] = 0.5
        
        print("UR5 configured for position control.")
        print("Initialized action with current joint positions")
    
    def calibrate(self):
        """Calibration not required for UR5."""
        pass
    
    @property
    def is_calibrated(self):
        """UR5 doesn't require calibration."""
        return True
    
    def get_observation(self) -> dict:
        """Get current robot observation including camera images."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "UR5HDDRobot is not connected. Try running `robot.connect()` first."
            )
        
        observation = {}
        
        # ===== State data =====
        joint_positions = self._rtde_receive.getActualQ()
        tcp_pose = self._rtde_receive.getActualTCPPose()
        gripper_pos = self._last_action.get("gripper.pos", 0.5)
        
        # Build state array (13 dimensions)
        state = []
        for i in range(6):  # 6 joints
            state.append(float(joint_positions[i]))
        for i in range(6):  # TCP x,y,z,rx,ry,rz
            state.append(float(tcp_pose[i]))
        state.append(float(gripper_pos))  # gripper
        
        observation["observation.state"] = np.array(state, dtype=np.float32)
        
        # ===== Camera images =====
        # Return HWC uint8 — lerobot handles CHW conversion and normalization internally
        for camera_name, camera in self._cameras.items():
            image = camera.read()
            if image is not None and image.ndim == 3:
                observation[f"observation.images.{camera_name}"] = image.astype(np.uint8)
        
        return observation
    
    def send_action(self, action: dict):
        """Send action to robot."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Robot not connected")
        
        # Extract action
        if "action" in action:
            act = action["action"]
        else:
            raise ValueError("Action dict must contain 'action' key")
        
        # Action should be [delta_x, delta_y, delta_z, gripper]
        if len(act) != 4:
            raise ValueError(f"Expected 4-dim action, got {len(act)}")
        
        delta_x, delta_y, delta_z, gripper = act
        
        # Get current TCP pose
        current_tcp = self._rtde_receive.getActualTCPPose()
        current_q = self._rtde_receive.getActualQ()
        
        # Compute target TCP pose
        target_tcp = list(current_tcp)
        target_tcp[0] += delta_x * self.config.ee_step_size
        target_tcp[1] += delta_y * self.config.ee_step_size
        target_tcp[2] += delta_z * self.config.ee_step_size
        
        # Get target joint positions via IK
        target_q = self._rtde_control.getInverseKinematics(target_tcp)

        if target_q is not None:
            # Calculate joint velocities: velocity = (target - current) * gain
            qd = []
            for i in range(6):
                vel = (target_q[i] - current_q[i]) * 3.0  # Gain=3.0
                vel = max(-1.0, min(1.0, vel))
                qd.append(vel)

            # speedJ: velocity control for smooth motion
            self._rtde_control.speedJ(qd, 0.5, 0.1)
        else:
            # IK failed — hold current position
            self._rtde_control.speedJ([0, 0, 0, 0, 0, 0], 0.5, 0.1)
        
        # Control gripper
        self._control_gripper(gripper)
        
        # Update last action
        self._last_action["gripper.pos"] = float(gripper)
    
    def _control_gripper(self, gripper_value: float):
        """Control EPG2 gripper via Modbus."""
        if gripper_value < 0.3:
            position = 0  # mm (open)
        elif gripper_value > 0.7:
            position = 50  # mm (closed)
        else:
            position = int((gripper_value - 0.3) / 0.4 * 50)
        
        try:
            self._gripper_client.write_register(
                address=0x0100,
                value=position * 100,
                device_id=self.config.gripper_slave_id
            )
        except Exception as e:
            print(f"Gripper control error: {e}")
    
    def get_joint_positions(self):
        """Get current joint positions."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Robot not connected")
        return self._rtde_receive.getActualQ()
    
    def get_tcp_pose(self):
        """Get current TCP pose."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Robot not connected")
        return self._rtde_receive.getActualTCPPose()
    
    def freedrive_mode(self, enable: bool = True):
        """Enable/disable freedrive mode."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Robot not connected")
        
        if enable:
            self._rtde_control.freedriveMode()
            print("Freedrive mode enabled")
        else:
            self._rtde_control.endFreedriveMode()
            print("Freedrive mode disabled")

    @property
    def config_class(self):
        from lerobot.robots.ur5_hdd.config_ur5_hdd import UR5HDDRobotConfig
        return UR5HDDRobotConfig