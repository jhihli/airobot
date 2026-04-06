import logging
import time
import threading
import numpy as np
import torch
from pathlib import Path
import sys
import signal

sys.path.insert(0, str(Path.home() / "lerobot" / "src"))

from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.ur5_hdd.ur5_hdd_robot import UR5HDDRobot
from lerobot.robots.ur5_hdd.config_ur5_hdd import UR5HDDRobotConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = Path.home() / "lerobot/outputs/train/ur5_ab_grip_test_act_v4/checkpoints/last/pretrained_model"
DATASET_REPO_ID = "toyoshima/ur5_ab_grip_test"
DATASET_ROOT    = "/mnt/data/lerobot_datasets_ab_grip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONTROL_FREQUENCY = 5    # Match training fps
NUM_REPEATS = 1          # How many task cycles to run

EE_SPEED = 0.05          # target: robot arrives at B at step ~38 (= training frame 38); raise if B arrival > 38, lower if < 38
EE_ACCELERATION = 1.0   # must be high enough for 5Hz: at 0.05 robot moves <1mm/step; 1.0 = ~15mm/step
MAX_DISPLACEMENT = 0.3


def load_deploy_params(repo_id: str, root: str):
    """Derive HOME_JOINTS and MAX_STEPS directly from the training dataset.

    HOME_JOINTS: mean joint positions at frame 0 across all episodes (std < 0.0001 rad).
    MAX_STEPS:   max episode frame count in the dataset + 5 frame buffer.

    Never hardcode these — they must always reflect the actual training data.
    See CLAUDE.md for the project rule.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logger.info(f"加载数据集参数: {repo_id}")
    ds = LeRobotDataset(repo_id, root=root)

    hf = ds.hf_dataset
    frame0 = hf.filter(lambda x: x["frame_index"] == 0)
    states = np.array(frame0["observation.state"])   # (n_episodes, state_dim)
    home_joints = states[:, :6].mean(axis=0).tolist()
    home_tcp    = states[:, 6:].mean(axis=0).tolist()

    max_frame = int(max(hf["frame_index"]))
    max_steps = max_frame + 5   # small buffer above the longest episode

    logger.info(f"HOME_JOINTS (from dataset frame 0): {[round(v, 6) for v in home_joints]}")
    logger.info(f"Home TCP: x={home_tcp[0]:.4f} y={home_tcp[1]:.4f} z={home_tcp[2]:.4f}")
    logger.info(f"MAX_STEPS (max_frame={max_frame} + 5): {max_steps}")

    return home_joints, max_steps

# State key order must match training observation_features order
STATE_KEYS = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow.pos",
    "wrist_1.pos", "wrist_2.pos", "wrist_3.pos",
    "tcp.x", "tcp.y", "tcp.z", "tcp.rx", "tcp.ry", "tcp.rz",
]
CAMERA_NAMES = ["top", "side_left", "side_right"]

# Ctrl+C: clear this flag to stop the loop cleanly
running = threading.Event()
running.set()

robot_instance = None

def _stop_gripper(robot):
    """Send speed=0 to EPG2 gripper to cancel any ongoing motion.
    Prevents the EPG2 watchdog from triggering a cycle when Modbus closes."""
    try:
        if robot._gripper_client and robot._gripper_client.connected:
            robot._gripper_client.write_registers(
                address=0x0020,
                values=[0, 0, 0, 0, 0, 0],  # speed=0 across all registers = stop
                device_id=robot.config.gripper_slave_id
            )
    except Exception:
        pass

def signal_handler(_sig, _frame):
    logger.warning("\n⚠️  Ctrl+C - 停止!\n")
    running.clear()
    if robot_instance is not None:
        _stop_gripper(robot_instance)
        if robot_instance._rtde_control:
            try:
                robot_instance._rtde_control.stopL(2.0)  # Cartesian stop
            except Exception:
                pass

def load_policy_and_processors(checkpoint_path: Path, device: str):
    logger.info("加载策略...")
    policy = ACTPolicy.from_pretrained(str(checkpoint_path))

    # Enable temporal ensembling: re-query the model every step and average
    # overlapping predictions with exponential weighting (coeff=0.01 per ACT paper).
    # Without this, the model re-queries every n_action_steps=15 steps, and if the
    # robot hasn't matched the expected position by then, predictions phase-shift to
    # "return to A" causing incoherent behavior. Temporal ensembling is robust to
    # timing mismatches without retraining.
    # Must create temporal_ensembler manually — from_pretrained() only creates it
    # in __init__ when the config already has temporal_ensemble_coeff set.
    # temporal_ensemble_coeff controls how fast old predictions decay.
    # coeff=0.01 (ACT paper default): weight 15 steps ago = 0.86 (almost equal to now) → very smooth but slow to respond
    # coeff=0.05: weight 15 steps ago = 0.47 → recent predictions dominate → faster phase transitions
    # Increased from 0.01 to 0.05: robot was reaching within 12mm of C then reversing because
    # the ensemble was too slow to commit to the final approach. Higher coeff makes "go to C"
    # predictions dominate before "return" predictions accumulate.
    TEMPORAL_ENSEMBLE_COEFF = 0.05
    policy.config.temporal_ensemble_coeff = TEMPORAL_ENSEMBLE_COEFF
    policy.temporal_ensembler = ACTTemporalEnsembler(TEMPORAL_ENSEMBLE_COEFF, policy.config.chunk_size)

    policy = policy.to(device)
    policy.eval()

    logger.info("加载 pre/post processors (含 normalization stats)...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_path),
    )
    logger.info("✅ 策略和处理器加载成功")
    return policy, preprocessor, postprocessor

def prepare_observation(obs: dict) -> dict:
    """
    Convert raw robot observation to tensors (no batch dim, no normalization).
    The preprocessor pipeline will add batch dim, normalize, and move to device.
    - observation.state: individual scalar keys → packed tensor float32
    - observation.images.*: numpy uint8 HWC → tensor float32 CHW [0,1]
    """
    prepared = {}

    state = np.array([obs[k] for k in STATE_KEYS], dtype=np.float32)
    prepared["observation.state"] = torch.from_numpy(state)

    for cam in CAMERA_NAMES:
        if cam not in obs:
            continue
        image = obs[cam]
        if isinstance(image, np.ndarray) and image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        prepared[f"observation.images.{cam}"] = torch.from_numpy(image).float() / 255.0

    return prepared

def check_safety(current_tcp, initial_tcp, max_displacement):
    displacement = np.linalg.norm(np.array(current_tcp[:3]) - np.array(initial_tcp[:3]))
    return displacement < max_displacement, displacement

def main():
    global robot_instance

    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=" * 70)
    logger.info("UR5 抓取任务部署")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"控制频率: {CONTROL_FREQUENCY} Hz")
    logger.info("=" * 70)

    if not CHECKPOINT_PATH.exists():
        logger.error(f"❌ Checkpoint不存在: {CHECKPOINT_PATH}")
        return

    # Derive HOME_JOINTS and MAX_STEPS from training dataset — never hardcode.
    HOME_JOINTS, MAX_STEPS = load_deploy_params(DATASET_REPO_ID, DATASET_ROOT)

    policy, preprocessor, postprocessor = load_policy_and_processors(CHECKPOINT_PATH, DEVICE)

    config = UR5HDDRobotConfig(
        robot_ip="192.168.0.102",
        gripper_port="/dev/ttyUSB0",
        ee_speed=EE_SPEED,
        ee_acceleration=EE_ACCELERATION,
    )

    robot = UR5HDDRobot(config)
    robot_instance = robot

    logger.info("\n连接机器人...")
    robot.connect()

    # Move to training home position before starting (environment reset).
    # All training episodes start at HOME_JOINTS — starting elsewhere puts the
    # robot in an out-of-distribution state and the model will behave incorrectly.
    logger.info(f"移动到训练 Home 位置: {HOME_JOINTS}")
    robot._rtde_control.moveJ(HOME_JOINTS, speed=0.3, acceleration=0.5)
    logger.info("✅ 已到达 Home 位置")

    initial_tcp = robot.get_tcp_pose()
    logger.info(f"初始TCP: {[f'{v:.4f}' for v in initial_tcp]}")

    # Reset ACT temporal ensembling queue before starting
    policy.reset()
    logger.info("✅ Policy reset")

    logger.info("\n⭐ 开始执行抓取任务!\n")

    try:
        total_start = time.time()

        with torch.no_grad():
            for repeat in range(NUM_REPEATS):
                if not running.is_set():
                    break

                logger.info(f"\n{'='*40}")
                logger.info(f"▶ 第 {repeat+1}/{NUM_REPEATS} 次: 0 → A → B → C → 0")
                logger.info(f"{'='*40}")

                # Reset to home position before each run (environment reset)
                robot._rtde_control.moveJ(HOME_JOINTS, speed=0.3, acceleration=0.5)
                initial_tcp = robot.get_tcp_pose()

                # Open gripper before every run as part of environment reset.
                # Ensures the physical start state matches training (gripper open at frame 0).
                robot._control_gripper(2)
                time.sleep(0.5)   # let gripper fully open

                policy.reset()

                step = 0
                last_log_time = time.time()
                # Timing landmarks — log once when each phase is first reached.
                # Compare these step numbers vs training (B close: frame ~38, C open: frame ~83).
                _logged_b = False
                _logged_c = False

                while step < MAX_STEPS and running.is_set():
                    step_start = time.time()

                    # 安全检查
                    current_tcp = robot.get_tcp_pose()
                    is_safe, displacement = check_safety(current_tcp, initial_tcp, MAX_DISPLACEMENT)

                    if not is_safe:
                        logger.warning(f"\n⚠️  超出安全范围! {displacement:.3f}m")
                        running.clear()
                        break

                    # 获取 observation
                    obs = robot.get_observation()
                    policy_input = prepare_observation(obs)
                    policy_input = preprocessor(policy_input)

                    # 推理
                    action = policy.select_action(policy_input)
                    action = postprocessor(action)
                    action_array = action.numpy()
                    if action_array.ndim == 2:
                        action_array = action_array[0]

                    # Log once when B (gripper closing) and C (gripper opening) are first reached.
                    # Compare these step numbers vs training averages: B~frame38, C~frame83.
                    # A large gap (e.g. B at step 60 instead of 38) = robot is slow → increase EE_SPEED.
                    grip_pred = float(action_array[6])
                    if not _logged_b and grip_pred < 0.5:
                        logger.info(
                            f"[PHASE] gripper CLOSE at step {step} "
                            f"(training avg: frame 38) — "
                            f"tcp=({current_tcp[0]:.4f},{current_tcp[2]:.4f})"
                        )
                        _logged_b = True
                    if _logged_b and not _logged_c and grip_pred > 1.5:
                        logger.info(
                            f"[PHASE] gripper OPEN at step {step} "
                            f"(training avg: frame 83) — "
                            f"tcp=({current_tcp[0]:.4f},{current_tcp[2]:.4f})"
                        )
                        _logged_c = True

                    # Debug: log predicted action vs actual robot state
                    if step < 5 or step % 30 == 0:
                        actual_rx = float(obs.get("tcp.rx", float("nan")))
                        logger.info(
                            f"[run {repeat+1} step {step}] "
                            f"actual=({current_tcp[0]:.4f},{current_tcp[1]:.4f},{current_tcp[2]:.4f}) "
                            f"actual_rx={actual_rx:.4f} "
                            f"pred=({action_array[0]:.4f},{action_array[1]:.4f},{action_array[2]:.4f}) "
                            f"pred_rx={action_array[3]:.4f} grip={action_array[6]:.4f}"
                        )

                    # Always pass the predicted TCP directly to send_action.
                    # send_action has its own 0.1mm threshold internally and only
                    # issues moveL when the target differs meaningfully from actual
                    # position — no need for a deploy-level gate.
                    # A 3mm gate here blocked sub-3mm corrections (e.g. final 2mm
                    # descent to B), preventing phase transitions to grip/C/return.
                    robot.send_action({
                        "tcp.x":  float(action_array[0]),
                        "tcp.y":  float(action_array[1]),
                        "tcp.z":  float(action_array[2]),
                        "tcp.rx": float(action_array[3]),
                        "tcp.ry": float(action_array[4]),
                        "tcp.rz": float(action_array[5]),
                        "gripper": float(action_array[6]),
                    })

                    # 记录
                    current_time = time.time()
                    if current_time - last_log_time >= 5.0:
                        elapsed = current_time - total_start
                        logger.info(
                            f"步骤 {step:3d}/{MAX_STEPS} | "
                            f"时间: {elapsed:5.1f}s | "
                            f"位移: {displacement:.3f}m | "
                            f"夹爪: {action_array[6]:.2f}"
                        )
                        last_log_time = current_time

                    step += 1

                    # 控制频率
                    step_duration = time.time() - step_start
                    sleep_time = max(0, 1.0/CONTROL_FREQUENCY - step_duration)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                logger.info(f"✅ 第 {repeat+1} 次完成 ({step} steps)")

                # Snap back to exact home after each run so repeated runs always
                # start from the same position regardless of policy return accuracy.
                logger.info("归位到 Home 位置...")
                robot._rtde_control.moveJ(HOME_JOINTS, speed=0.3, acceleration=0.5)
                logger.info("✅ 归位完成")

        if not running.is_set():
            logger.info("已收到停止信号，退出循环")
        else:
            total_time = time.time() - total_start
            logger.info("=" * 70)
            logger.info(f"✅ 全部 {NUM_REPEATS} 次任务完成!")
            logger.info(f"  总时间: {total_time:.1f}秒")
            logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\n❌ 错误: {e}\n")
        import traceback
        traceback.print_exc()

    finally:
        logger.info("\n停止中...")
        # 1. Stop gripper first so EPG2 has time to process before Modbus closes
        _stop_gripper(robot)
        # 2. Stop any in-flight Cartesian motion, wait for arm to settle
        if robot._rtde_control:
            try:
                robot._rtde_control.stopL(2.0)   # Cartesian deceleration stop
                time.sleep(0.5)                   # wait for arm to reach zero velocity
            except Exception:
                pass
        # 3. Disconnect everything (speedStop + RTDE close + gripper Modbus + cameras)
        robot.disconnect()
        logger.info("✅ 已断开")

if __name__ == "__main__":
    main()
