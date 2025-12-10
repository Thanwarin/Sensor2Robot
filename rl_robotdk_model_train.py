# rl_robotdk_model_train_tuned.py
# Safe Q-learning tuned for RoboDK (single-file, no torch)
# - auto-detect two 6+ DOF robots
# - set safer home poses (override default home)
# - reduced collision penalty, stronger target reward
# - randomized safe starts, safe MoveJ with rollback
# - save/load Q-table
# Run only in RoboDK simulator

from robodk import robolink, robomath
import numpy as np
import random
import time
import os

# ---------------- Config (tuned) ----------------
Q_TABLE_PATH = 'q_table_robotdk_tuned.npy'

NUM_EPISODES = 800
STEP_DEG = 5.0                 # moderate step so motion moves off table but not huge
EPSILON = 0.35                 # more exploration
LEARNING_RATE = 0.08
DISCOUNT = 0.80
JOINT_BINS = 5

MIN_STEPS_BEFORE_TERMINATE = 15
SAFE_DISTANCE_M = 0.12         # 120 mm safe distance
REACH_THRESHOLD_M = 0.06       # 60 mm reach threshold
MAX_STEPS_PER_EP = 150
RANDOM_START_NOISE = 12.0      # degrees perturbation around safe home

# Reward tuning
COLLISION_PENALTY = -50.0      # reduced from -200 to encourage risk-taking
SAFE_PENALTY_SCALE = -10.0     # penalty scale when too close to other robot
TARGET_WEIGHT = 50.0           # multiply distance to target -> stronger incentive

# Safer home poses (values in degrees) - adjust to your cell if needed
# These are typical "up" poses. Edit if your robot uses different axes sign.
SAFE_HOME_A = [0.0, -60.0, 60.0, 0.0, 45.0, 0.0]
SAFE_HOME_B = [0.0, -60.0, 60.0, 0.0, 45.0, 0.0]

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
    raise Exception("Require at least 2 robot arms (>=6 joints) in station")

robot_a, robot_b = real_robots[:2]
print("Robot A:", robot_a.Name())
print("Robot B:", robot_b.Name())

# ---------------- Auto-detect targets ----------------
all_targets = RDK.ItemList(robolink.ITEM_TYPE_TARGET)
if len(all_targets) < 2:
    raise Exception("Require at least 2 targets in station")
target_a, target_b = all_targets[:2]

# optional start target name match
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

def clamp_joints(lst, low=-180.0, high=180.0):
    return [float(np.clip(x, low, high)) for x in lst]

def safe_movej(robot, joints_target, wait=0.03):
    """MoveJ then check collisions using RDK.Collisions(); rollback if collision.
       Returns moved(bool), collided(bool)."""
    prev = joints_list(robot)
    try:
        robot.MoveJ(joints_target)
        time.sleep(wait)
    except Exception:
        return False, False
    if RDK.Collisions() > 0:
        try:
            robot.MoveJ(prev)
            time.sleep(wait)
        except Exception:
            pass
        return False, True
    return True, False

def set_safe_home(robot, home_pose):
    """Try to move robot to home_pose (list) safely (MoveJ)."""
    try:
        home = clamp_joints(home_pose)
        moved, coll = safe_movej(robot, home)
        if coll:
            # if collision, try a smaller offset upward on joint 2
            alt = home.copy()
            if len(alt) >= 2:
                alt[1] -= 15.0
            try:
                safe_movej(robot, alt)
            except Exception:
                pass
        return
    except Exception:
        # fallback to MoveJ(JointsHome)
        try:
            robot.MoveJ(robot.JointsHome())
            time.sleep(0.03)
        except Exception:
            pass

def randomize_around_home(robot, home_pose, noise_deg=RANDOM_START_NOISE):
    base = clamp_joints(home_pose)
    new = [base[i] + random.uniform(-noise_deg, noise_deg) for i in range(len(base))]
    new = clamp_joints(new)
    ok, coll = safe_movej(robot, new)
    return ok and not coll

def tcp_xyz(item):
    try:
        p = item.Pose()
        t = robomath.Transl(p)
        return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
    except Exception:
        try:
            P = item.Pose(); pos = P.Pos(); return np.array([float(pos[0]),float(pos[1]),float(pos[2])], dtype=np.float64)
        except Exception:
            return np.zeros(3, dtype=np.float64)

def dist_items(a,b):
    return float(np.linalg.norm(tcp_xyz(a) - tcp_xyz(b)))

def discretize6(joints, bins=JOINT_BINS):
    arr=[]
    for i in range(6):
        angle = float(joints[i]) if i < len(joints) else 0.0
        edges = np.linspace(-180.0, 180.0, bins-1) if bins>1 else [-180.0]
        idx = int(np.digitize(angle, edges))
        arr.append(idx)
    return tuple(arr)

# ---------------- Action & Q-table ----------------
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

# ---------------- Reward function (tuned) ----------------
def compute_reward_single(robot, target_item, other_robot):
    # collision immediate penalty (reduced)
    if RDK.Collisions() > 0:
        return COLLISION_PENALTY
    # target proximity (stronger incentive)
    d_target = dist_items(robot, target_item)
    r_target = -d_target * TARGET_WEIGHT
    # safety penalty if too close to other robot (linear)
    d_other = dist_items(robot, other_robot)
    r_safe = 0.0
    if d_other < SAFE_DISTANCE_M:
        r_safe = SAFE_PENALTY_SCALE * (SAFE_DISTANCE_M - d_other)
    # reach bonus
    bonus = 0.0
    if d_target < REACH_THRESHOLD_M:
        bonus = 400.0
    return float(r_target + r_safe + bonus)

# ---------------- Prepare safe homes ----------------
# Try setting the safer homes first (so Home is not colliding)
set_safe_home(robot_a, SAFE_HOME_A)
set_safe_home(robot_b, SAFE_HOME_B)
time.sleep(0.05)

# ---------------- Training loop ----------------
print("Start training (tuned)...")
for ep in range(NUM_EPISODES):
    # Reset: prefer safe home then randomize
    set_safe_home(robot_a, SAFE_HOME_A)
    set_safe_home(robot_b, SAFE_HOME_B)
    # small random perturb to encourage exploration and avoid deterministic collision
    ok_a = randomize_around_home(robot_a, SAFE_HOME_A, noise_deg=RANDOM_START_NOISE)
    ok_b = randomize_around_home(robot_b, SAFE_HOME_B, noise_deg=RANDOM_START_NOISE)
    # optionally, if start_target exists, bias around it
    if start_target:
        try:
            robot_a.MoveJ(start_target); robot_b.MoveJ(start_target); time.sleep(0.03)
            randomize_around_home(robot_a, SAFE_HOME_A, noise_deg=8.0)
            randomize_around_home(robot_b, SAFE_HOME_B, noise_deg=8.0)
        except Exception:
            pass

    # read states
    sj_a = joints_list(robot_a)[:6]
    sj_b = joints_list(robot_b)[:6]
    state_a = discretize6(sj_a)
    state_b = discretize6(sj_b)

    total_reward = 0.0
    done = False
    step = 0
    if ep % 100 == 0:
        print(f"[EP {ep}] startA={sj_a} startB={sj_b} collisions={RDK.Collisions()} ok_a={ok_a} ok_b={ok_b}")

    while not done and step < MAX_STEPS_PER_EP:
        step += 1
        # choose actions: epsilon-greedy
        if random.random() < EPSILON:
            act_a = random.randrange(ACTIONS_A)
            act_b = random.randrange(ACTIONS_B)
        else:
            act_a = int(np.argmax(Q_table[state_a][:ACTIONS_A]))
            act_b = int(np.argmax(Q_table[state_b][:ACTIONS_B]))

        # current joints
        curr_a = joints_list(robot_a)
        curr_b = joints_list(robot_b)

        # compute joint index and direction
        ji_a = min(act_a // 2, len(curr_a)-1); dir_a = 1 if (act_a % 2 == 0) else -1
        ji_b = min(act_b // 2, len(curr_b)-1); dir_b = 1 if (act_b % 2 == 0) else -1

        # propose next joints
        prop_a = curr_a.copy(); prop_b = curr_b.copy()
        prop_a[ji_a] = float(np.clip(prop_a[ji_a] + dir_a * STEP_DEG, -180.0, 180.0))
        prop_b[ji_b] = float(np.clip(prop_b[ji_b] + dir_b * STEP_DEG, -180.0, 180.0))

        # safe move sequentially (A then B) with rollback on collision
        moved_a, col_a = safe_movej(robot_a, prop_a)
        moved_b, col_b = safe_movej(robot_b, prop_b)

        # compute rewards AFTER both moves
        r_a = compute_reward_single(robot_a, target_a, robot_b)
        r_b = compute_reward_single(robot_b, target_b, robot_a)
        reward = r_a + r_b

        # observe new discrete states
        new_state_a = discretize6(joints_list(robot_a)[:6])
        new_state_b = discretize6(joints_list(robot_b)[:6])

        # Q updates
        old_q_a = Q_table[state_a + (act_a,)]
        old_q_b = Q_table[state_b + (act_b,)]
        max_future_a = np.max(Q_table[new_state_a][:ACTIONS_A])
        max_future_b = np.max(Q_table[new_state_b][:ACTIONS_B])
        Q_table[state_a + (act_a,)] = old_q_a + LEARNING_RATE * (reward + DISCOUNT * max_future_a - old_q_a)
        Q_table[state_b + (act_b,)] = old_q_b + LEARNING_RATE * (reward + DISCOUNT * max_future_b - old_q_b)

        total_reward += reward
        state_a, state_b = new_state_a, new_state_b

        # termination: allow some steps before early termination on collision
        if (RDK.Collisions() > 0 and step >= MIN_STEPS_BEFORE_TERMINATE) or reward > 500.0:
            done = True

    if ep % 50 == 0:
        print(f"EP {ep} | Total Reward={total_reward:.2f} | Steps={step} | Collisions={RDK.Collisions()}")

# Save Q-table
np.save(Q_TABLE_PATH, Q_table)
print("Q-table saved ->", Q_TABLE_PATH)

# ---------------- Run trained policy ----------------
print("\nRun trained policy (watch in RoboDK)...")
set_safe_home(robot_a, SAFE_HOME_A)
set_safe_home(robot_b, SAFE_HOME_B)
time.sleep(0.05)
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
