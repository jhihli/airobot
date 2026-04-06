import socket
import time
import numpy as np
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

        # Incremental tcp.rx unwrapping — tracks cumulative offset so ±π crossings
        # are resolved in real-time, avoiding the need for post-hoc np.unwrap on the dataset.
        self._tcp_rx_last: float | None = None
        self._tcp_rx_offset: float = 0.0

    @property
    def name(self):
        return "ur5_hdd"

    @property
    def robot_type(self):
        return "ur5_hdd"

    @property
    def cameras(self):
        return self._cameras

    @property
    def action_features(self) -> dict:
        """Action: absolute TCP pose (6D) + gripper state (1D) = 7D total.
        Recorded directly from robot state during freedrive; policy predicts target TCP.
        """
        return {
            "tcp.x": float,
            "tcp.y": float,
            "tcp.z": float,
            "tcp.rx": float,
            "tcp.ry": float,
            "tcp.rz": float,
            "gripper": float,
        }

    @property
    def observation_features(self) -> dict:
        """Observation features: joint positions, TCP pose, and camera images.
        gripper.pos is intentionally excluded so the policy learns gripper timing
        from visual cues rather than feeding back the commanded gripper state.
        """
        feats = {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            "elbow.pos": float,
            "wrist_1.pos": float,
            "wrist_2.pos": float,
            "wrist_3.pos": float,
            "tcp.x": float,
            "tcp.y": float,
            "tcp.z": float,
            "tcp.rx": float,
            "tcp.ry": float,
            "tcp.rz": float,
        }
        for camera_name, camera_cfg in self._camera_configs.items():
            h = camera_cfg.height if camera_cfg.height else 480
            w = camera_cfg.width if camera_cfg.width else 640
            # "images.{cam}" → with "observation." prefix → "observation.images.{cam}"
            feats[f"images.{camera_name}"] = (h, w, 3)
        return feats

    def connect(self):
        """Connect to UR5, gripper, and cameras."""
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"{self.name} UR5HDDRobot is already connected."
            )
        self._tcp_rx_last = None
        self._tcp_rx_offset = 0.0

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

    def _dashboard_stop(self):
        """Send 'stop' to the UR5 dashboard server (port 29999).

        This is the most reliable way to stop the URScript running on the
        controller. stopScript() via RTDE can fail silently (e.g. when the
        robot is in freedrive or the RTDE interface is in a bad state), leaving
        the script running and blocking the next RTDEControlInterface() call.
        The dashboard server is always reachable and always honours 'stop'.
        """
        try:
            ip = self.config.robot_ip
            with socket.create_connection((ip, 29999), timeout=2.0) as sock:
                sock.recv(1024)          # discard welcome banner
                sock.sendall(b"stop\n")
                sock.recv(1024)          # discard response
        except Exception as e:
            print(f"Warning: dashboard stop failed: {e}")

    def disconnect(self):
        """Disconnect from all devices."""
        if not self.is_connected:
            return

        # Stop the URScript on the controller via the dashboard server first.
        # This is more reliable than stopScript() over RTDE, which can fail
        # silently when the robot is in freedrive or the RTDE state is bad.
        self._dashboard_stop()

        # Disconnect UR5 — stop motion, wait to settle, then disconnect.
        # Only speedStop() here; the caller is responsible for any prior stopJ().
        # Calling stopJ() a second time on an already-stopped arm causes a servo jerk.
        if self._rtde_control:
            try:
                self._rtde_control.speedStop()
                time.sleep(0.3)   # let arm settle to zero velocity before RTDE closes
            except Exception as e:
                print(f"Warning: error stopping robot: {e}")

            try:
                self._rtde_control.stopScript()
            except Exception as e:
                print(f"Warning: error stopping RTDE script: {e}")

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

        # Disconnect gripper — give EPG2 time to process the stop command before
        # closing the Modbus connection (EPG2 firmware can take ~200ms to respond).
        if self._gripper_client:
            time.sleep(0.2)
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
        # Initialize gripper to "unknown" sentinel so the first send_action call
        # always issues a physical command regardless of what the policy predicts.
        # Use -1 (never a valid gripper_cmd) so both 0 and 2 trigger on first call.
        self._last_action["gripper.pos"] = -1

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
        """Get current robot observation including camera images.

        Returns individual scalar keys matching observation_features so the LeRobot
        framework can group them into the dataset correctly:
          - state scalars: "shoulder_pan.pos", "tcp.x", "gripper.pos", ...
          - camera images: "images.top", "images.side_left", "images.side_right"

        The framework adds the "observation." prefix when writing to the dataset,
        producing "observation.state" and "observation.images.{cam}".
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "UR5HDDRobot is not connected. Try running `robot.connect()` first."
            )

        obs = {}

        # ===== State: individual scalar keys =====
        joint_positions = self._rtde_receive.getActualQ()
        tcp_pose = self.get_tcp_pose()   # uses _unwrap_rx internally

        for i, name in enumerate(UR5_JOINT_NAMES):
            obs[f"{name}.pos"] = float(joint_positions[i])
        obs["tcp.x"]  = float(tcp_pose[0])
        obs["tcp.y"]  = float(tcp_pose[1])
        obs["tcp.z"]  = float(tcp_pose[2])
        obs["tcp.rx"] = float(tcp_pose[3])
        obs["tcp.ry"] = float(tcp_pose[4])
        obs["tcp.rz"] = float(tcp_pose[5])

        # ===== Camera images: plain camera name keys (HWC uint8) =====
        # build_dataset_frame strips "observation.images." from the dataset key to get the
        # lookup key in values(), so we must use the bare camera name here (e.g. "top").
        for camera_name, camera in self._cameras.items():
            image = camera.read_latest()
            if image is not None and image.ndim == 3:
                obs[camera_name] = image.astype(np.uint8)

        return obs

    def send_action(self, action: dict):
        """Send action: move to absolute TCP pose, optionally open/close gripper.

        Action keys: tcp.x/y/z/rx/ry/rz (metres/radians), gripper (0=close, 1=stay, 2=open).
        Uses moveL(asynchronous=True) so the call returns immediately; each new call redirects
        the robot smoothly to the latest target at ee_speed (m/s).
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Robot not connected")

        target_tcp = [
            float(action.get("tcp.x", 0.0)),
            float(action.get("tcp.y", 0.0)),
            float(action.get("tcp.z", 0.0)),
            float(action.get("tcp.rx", 0.0)),
            float(action.get("tcp.ry", 0.0)),
            float(action.get("tcp.rz", 0.0)),
        ]
        gripper = float(action.get("gripper", 1.0))

        # Only call moveL if the target differs meaningfully from current pose (> 0.1 mm)
        current_tcp = self._rtde_receive.getActualTCPPose()
        displacement = sum((target_tcp[i] - float(current_tcp[i])) ** 2 for i in range(3)) ** 0.5
        if displacement > 0.0001:
            try:
                self._rtde_control.moveL(
                    target_tcp,
                    self.config.ee_speed,
                    self.config.ee_acceleration,
                    asynchronous=True,
                )
            except Exception as e:
                # During data collection this is expected (freedrive mode).
                # During deployment this is an error — log it so it's not silently lost.
                import logging as _logging
                _logging.getLogger(__name__).warning(f"moveL failed: {e}")

        # Threshold continuous policy output → discrete command {0=close, 1=stay, 2=open}.
        # During data collection, gripper is always integer {0,1,2}.
        # At inference, ACT predicts continuous values — map to nearest command:
        #   < 0.5  → 0 (close)
        #   0.5–1.5 → 1 (stay, no Modbus write)
        #   > 1.5  → 2 (open)
        if gripper < 0.5:
            gripper_cmd = 0
        elif gripper > 1.5:
            gripper_cmd = 2
        else:
            gripper_cmd = 1  # stay

        last_gripper = self._last_action.get("gripper.pos", 1)
        if gripper_cmd != 1 and gripper_cmd != last_gripper:
            self._control_gripper(gripper_cmd)
            self._last_action["gripper.pos"] = gripper_cmd

    def _control_gripper(self, gripper_value: float):
        """Control EPG2 gripper via Modbus.

        gripper_value: 0 = close, 2 = open (1 = stay, never passed here)
        Uses same 6-register command as freedrive_with_gripper.py:
          registers 0x0020..0x0025: [speed, 0, 0, speed, position, 0]
          position: 0 = 0mm (closed), 10000 = 26mm (open)
        """
        if gripper_value < 1:
            pos_val = 0       # 0mm — closed
        else:
            pos_val = 10000   # 26mm — open

        try:
            self._gripper_client.write_registers(
                address=0x0020,
                values=[10000, 0, 0, 10000, pos_val, 0],
                device_id=self.config.gripper_slave_id
            )
        except Exception as e:
            print(f"Gripper control error: {e}")

    def get_joint_positions(self):
        """Get current joint positions."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Robot not connected")
        return self._rtde_receive.getActualQ()

    def _unwrap_rx(self, raw_rx: float) -> float:
        """Return tcp.rx with incremental unwrapping to avoid ±π discontinuities.

        Tracks the cumulative 2π offset across calls so the returned value is
        always continuous — no post-hoc np.unwrap needed on the recorded dataset.
        Resets automatically when connect() is called (offset/last are re-initialised).
        """
        if self._tcp_rx_last is not None:
            delta = raw_rx - self._tcp_rx_last
            if delta > np.pi:
                self._tcp_rx_offset -= 2 * np.pi
            elif delta < -np.pi:
                self._tcp_rx_offset += 2 * np.pi
        self._tcp_rx_last = raw_rx
        return raw_rx + self._tcp_rx_offset

    def get_tcp_pose(self):
        """Get current TCP pose with tcp.rx continuously unwrapped."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Robot not connected")
        pose = list(self._rtde_receive.getActualTCPPose())
        pose[3] = self._unwrap_rx(pose[3])
        return pose

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
