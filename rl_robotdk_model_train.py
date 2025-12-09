# rl_robotdk_model_train_safe.py
# Single-file safe Q-learning for RoboDK (Python 3.13)
# - Auto-detect 2 robot arms (>=6 joints)
# - Safe randomized start poses
# - Safe MoveJ with rollback on collision
# - Reward shaping + min steps before termination
# - Save/load Q-table
# Test only in RoboDK simulator!

from robodk import robolink, robomath
import numpy as np
import random
import time
import os

# ---------------- Config ----------------
Q_TABLE_PATH = 'q_table_robotdk_safe.npy'
NUM_EPISODES = 600
STEP_DEG = 10.0                 # degrees per discrete action (larger so movement is visible)
EPSILON = 0.25
LEARNING_RATE = 0.08
DISCOUNT = 0.9
JOINT_BINS = 5
MIN_STEPS_BEFORE_TERMINATE = 8  # allow a few steps before terminating on collision
SAFE_DISTANCE_M = 0.08
REACH_THRESHOLD_M = 0.06
MAX_STEPS_PER_EP = 120
RANDOM_START_NOISE = 20.0       # degrees noise from home

# ---------------- RoboDK init ----------------
RDK = robolink.Robolink()
RDK.setRunMode(robolink.RUNMODE_SIMULATE)

# ---------------- Auto-detect robots ----------------
all_robots = RDK.ItemList(robolink.ITEM_TYPE_ROBOT)
real_robots = []
for r in all_robots:
    try:
        joints = r.Joints().list()
        if len(joints) >= 6:
            real_robots.append(r)
    except Exception:
        continue

if len(real_robots) < 2:
    raise Exception("Not enough robot arms (>=6 joints) found in station. Please add 2 robot arms.")

robot_a, robot_b = real_robots[:2]
print("Robot A:", robot_a.Name())
print("Robot B:", robot_b.Name())

# ---------------- Auto-detect targets ----------------
all_targets = RDK.ItemList(robolink.ITEM_TYPE_TARGET)
if len(all_targets) < 2:
    raise Exception("Require at least 2 targets in the station.")

target_a, target_b = all_targets[:2]
# optional start target
start_target = None
for t in all_targets:
    if 'start' in t.Name().lower():
        start_target = t
        break

# ---------------- Helpers ----------------
def joints_list(item):
    try:
        return [float(x) for x in item.Joints().list()]
    except Exception:
        return [float(x) for x in item.Joints()]

def clamp_joints(jlist, low=-180.0, high=180.0):
    return [float(np.clip(x, low, high)) for x in jlist]

def safe_movej(robot, joints_target, wait=0.03):
    """MoveJ then check global collisions; rollback on collision.
       Returns: moved (bool), collided (bool)"""
    prev = joints_list(robot)
    try:
        robot.MoveJ(joints_target)
        time.sleep(wait)
    except Exception:
        # move failed; return not moved
        return False, False
    # collision check
    if RDK.Collisions() > 0:
        # rollback
        try:
            robot.MoveJ(prev)
            time.sleep(wait)
        except Exception:
            pass
        return False, True
    return True, False

def random_start_pose(robot, attempts=6, noise_deg=RANDOM_START_NOISE):
    """Move robot to JointsHome then random small noise; return True if placed without collision"""
    tried = 0
    while tried < attempts:
        try:
            # move to home
            try:
                robot.MoveJ(robot.JointsHome())
                time.sleep(0.03)
            except Exception:
                pass
            base = joints_list(robot)
            # apply random noise
            new = [base[i] + random.uniform(-noise_deg, noise_deg) for i in range(len(base))]
            new = clamp_joints(new)
            moved, collided = safe_movej(robot, new)
            # if no collision return success
            if not collided and RDK.Collisions() == 0:
                return True
            # otherwise rollback already done, try again
            tried += 1
        except Exception:
            tried += 1
    # if all attempts failed, leave robot at home (may be colliding)
    return False

def tcp_xyz(item):
    try:
        p = item.Pose()
        t = robomath.Transl(p)
        return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
    except Exception:
        try:
            P = item.Pose()
            pos = P.Pos()
            return np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)
        except Exception:
            return np.zeros(3, dtype=np.float64)

def dist_items(a, b):
    return float(np.linalg.norm(tcp_xyz(a) - tcp_xyz(b)))

# discretize first 6 joints into bins
def discretize6(joints, bins=JOINT_BINS):
    arr = []
    for i in range(6):
        angle = float(joints[i]) if i < len(joints) else 0.0
        # simple mapping -180..180 into bins
        edges = np.linspace(-180.0, 180.0, bins-1) if bins > 1 else [-180.0]
        idx = int(np.digitize(angle, edges))
        arr.append(idx)
    return tuple(arr)

# ---------------- Actions & Q-table setup ----------------
nj_a = len(joints_list(robot_a))
nj_b = len(joints_list(robot_b))
ACTIONS_A = nj_a * 2
ACTIONS_B = nj_b * 2
print(f"Robot A joints: {nj_a}, actions: {ACTIONS_A}")
print(f"Robot B joints: {nj_b}, actions: {ACTIONS_B}")

Q_shape = tuple([JOINT_BINS]*6) + (max(ACTIONS_A, ACTIONS_B),)
if os.path.exists(Q_TABLE_PATH):
    Q_table = np.load(Q_TABLE_PATH, allow_pickle=True)
    print("[Loaded Q-table]")
else:
    Q_table = np.random.uniform(low=-1.0, high=0.0, size=Q_shape)
    print("[Created new Q-table]")

# ---------------- Reward ----------------
def compute_reward_single(robot, target_item, other_robot):
    # collision check
    if RDK.Collisions() > 0:
        return -200.0
    # distance to target
    d_target = dist_items(robot, target_item)
    r_target = -d_target * 8.0  # closer -> higher reward (less negative)
    # safety to other robot
    d_other = dist_items(robot, other_robot)
    r_safe = 0.0
    if d_other < SAFE_DISTANCE_M:
        r_safe = -50.0 * (SAFE_DISTANCE_M - d_other)
    # reach bonus
    bonus = 0.0
    if d_target < REACH_THRESHOLD_M:
        bonus = 400.0
    return float(r_target + r_safe + bonus)

# ---------------- Training loop ----------------
print("Start training...")
for ep in range(NUM_EPISODES):
    # reset start poses (randomized safe)
    ok_a = random_start_pose(robot_a)
    ok_b = random_start_pose(robot_b)
    # if start_target exists, prefer moving there then randomize small around it
    if start_target:
        try:
            robot_a.MoveJ(start_target); robot_b.MoveJ(start_target); time.sleep(0.03)
            # small perturbation
            random_start_pose(robot_a, attempts=3, noise_deg=8.0)
            random_start_pose(robot_b, attempts=3, noise_deg=8.0)
        except Exception:
            pass

    # read discrete states (first 6 joints)
    sj_a = joints_list(robot_a)[:6]
    sj_b = joints_list(robot_b)[:6]
    state_a = discretize6(sj_a)
    state_b = discretize6(sj_b)

    total_reward = 0.0
    done = False
    step = 0
    # debug start info occasionally
    if ep % 100 == 0:
        print(f"[EP {ep}] start_A={sj_a} start_B={sj_b} collisions={RDK.Collisions()} ok_a={ok_a} ok_b={ok_b}")

    while not done and step < MAX_STEPS_PER_EP:
        step += 1
        # choose action (epsilon greedy)
        if random.random() < EPSILON:
            act_a = random.randrange(ACTIONS_A)
            act_b = random.randrange(ACTIONS_B)
        else:
            act_a = int(np.argmax(Q_table[state_a][:ACTIONS_A]))
            act_b = int(np.argmax(Q_table[state_b][:ACTIONS_B]))

        # form proposed joint targets
        curr_a = joints_list(robot_a)
        curr_b = joints_list(robot_b)
        ji_a = min(act_a // 2, len(curr_a)-1)
        dir_a = 1 if (act_a % 2 == 0) else -1
        ji_b = min(act_b // 2, len(curr_b)-1)
        dir_b = 1 if (act_b % 2 == 0) else -1

        prop_a = curr_a.copy()
        prop_b = curr_b.copy()
        prop_a[ji_a] = float(np.clip(prop_a[ji_a] + dir_a * STEP_DEG, -180.0, 180.0))
        prop_b[ji_b] = float(np.clip(prop_b[ji_b] + dir_b * STEP_DEG, -180.0, 180.0))

        # safe move sequentially (A then B)
        moved_a, col_a = safe_movej(robot_a, prop_a)
        moved_b, col_b = safe_movej(robot_b, prop_b)

        # compute reward after moves
        r_a = compute_reward_single(robot_a, target_a, robot_b)
        r_b = compute_reward_single(robot_b, target_b, robot_a)
        reward = r_a + r_b

        # observe new states
        new_state_a = discretize6(joints_list(robot_a)[:6])
        new_state_b = discretize6(joints_list(robot_b)[:6])

        # Q-update
        old_q_a = Q_table[state_a + (act_a,)]
        old_q_b = Q_table[state_b + (act_b,)]
        max_future_a = np.max(Q_table[new_state_a][:ACTIONS_A])
        max_future_b = np.max(Q_table[new_state_b][:ACTIONS_B])
        Q_table[state_a + (act_a,)] = old_q_a + LEARNING_RATE * (reward + DISCOUNT * max_future_a - old_q_a)
        Q_table[state_b + (act_b,)] = old_q_b + LEARNING_RATE * (reward + DISCOUNT * max_future_b - old_q_b)

        total_reward += reward
        state_a, state_b = new_state_a, new_state_b

        # termination: collision or big positive reward, but allow some steps first
        if (RDK.Collisions() > 0 and step >= MIN_STEPS_BEFORE_TERMINATE) or reward > 500.0:
            done = True

    if ep % 50 == 0:
        print(f"EP {ep} | Total Reward={total_reward:.2f} | Steps={step} | Collisions={RDK.Collisions()}")

# Save Q-table
np.save(Q_TABLE_PATH, Q_table)
print("Q-table saved ->", Q_TABLE_PATH)

# ---------------- Run trained policy ----------------
print("\nRun trained policy (watch in RoboDK)...")
try:
    robot_a.MoveJ(robot_a.JointsHome()); robot_b.MoveJ(robot_b.JointsHome()); time.sleep(0.05)
except Exception:
    pass

for t in range(80):
    s_a = discretize6(joints_list(robot_a)[:6])
    s_b = discretize6(joints_list(robot_b)[:6])
    act_a = int(np.argmax(Q_table[s_a][:ACTIONS_A]))
    act_b = int(np.argmax(Q_table[s_b][:ACTIONS_B]))

    curr_a = joints_list(robot_a); curr_b = joints_list(robot_b)
    ji_a = min(act_a // 2, len(curr_a)-1); dir_a = 1 if (act_a % 2 == 0) else -1
    ji_b = min(act_b // 2, len(curr_b)-1); dir_b = 1 if (act_b % 2 == 0) else -1
    prop_a = curr_a.copy(); prop_b = curr_b.copy()
    prop_a[ji_a] = float(np.clip(prop_a[ji_a] + dir_a * STEP_DEG, -180.0, 180.0))
    prop_b[ji_b] = float(np.clip(prop_b[ji_b] + dir_b * STEP_DEG, -180.0, 180.0))

    safe_movej(robot_a, prop_a)
    safe_movej(robot_b, prop_b)
    print(f"t={t} actA={act_a} actB={act_b} coll={RDK.Collisions()}")
    time.sleep(0.05)

print("Done.")
