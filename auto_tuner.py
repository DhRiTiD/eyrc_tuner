#!/usr/bin/env python3
"""
pid_autoreset_tuner.py - fully automatic PID tuner with built-in drone reset (ROS 2).

Run pico_controller in Y mode (task_1b_controller -> answer "Y") and DO NOT
launch the GUI (drone_pid_tuner), or it will fight this script on the gain topic.

Each trial:
  1. RESET   publish kp=ki=kd=0 -> drone falls. Wait until the error is >= --ground-error
             (default 2.8, setpoint is 3 m) and stable for --stable-time (2 s).
  2. RUN     publish candidate gains, record the error for settle+hold seconds (5 + 10).
  3. SCORE   pass = |error| <= band (0.4) from t=5 s to the end of the trial.
Search = twiddle starting from --seeds. Stops when a gain set passes (and is confirmed).

[INFO] [1791190415.565719690] [pid_autoreset_tuner]: trial 13: kp=13.350 ki=0.1310 kd=59.060 | cost=9.05 lastOutOfBand=4.72s maxAfter5s=0.305 PASS (ok)
[INFO] [1791190307.185803385] [pid_autoreset_tuner]: trial 8: kp=17.750 ki=0.1310 kd=59.060 | cost=14.04 lastOutOfBand=3.67s maxAfter5s=0.336 PASS (ok)

17.750 0.1310 59.060


[INFO] [1791190565.926021654] [pid_autoreset_tuner]: trial 20: kp=15.550 ki=0.1310 kd=52.460 | cost=8.75 lastOutOfBand=4.38s maxAfter5s=0.258 PASS (ok)


Examples:
  python3 pid_autoreset_tuner.py                               # throttle, seed 11.55 0.19 40
  python3 pid_autoreset_tuner.py --seeds 11.55 0.19 40  30 1.5 50
  python3 pid_autoreset_tuner.py --set 11.55 0.19 40 --runs 15 # repeat fixed gains 15 times
  python3 pid_autoreset_tuner.py --axis pitch --throttle-gains 11.55 0.19 40 --seeds 5 0 8
"""
import argparse
import csv
import json
import math
import sys
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from controller_msg.msg import PIDTune
from error_msg.msg import Error


class ResetFailed(RuntimeError):
    pass


class AutoResetTuner(Node):
    def __init__(self, a):
        super().__init__("pid_autoreset_tuner")
        self.a = a
        self.field = f"{a.axis}_error"

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.pubs = {"throttle": self.create_publisher(PIDTune, a.throttle_topic, qos)}
        if a.axis != "throttle":
            self.pubs[a.axis] = self.create_publisher(PIDTune, a.topic or f"/{a.axis}_pid", qos)
        self.create_subscription(Error, a.error_topic, self._on_error, qos)

        self.collecting = False
        self.samples = []            # tuned-axis error during a trial
        self.hist = deque()          # (t, throttle_error) rolling 6 s, used for reset detection
        self.last_error = None
        self.last_thr = None
        self.n_msgs = 0
        self.trial_id = 0

        self.log_file = open(a.log, "w", newline="")
        self.log = csv.writer(self.log_file)
        self.log.writerow(["trial", "kp", "ki", "kd", "cost", "settle_t", "max_after_settle",
                           "passed", "note"])

    # ---------------- ROS plumbing ----------------
    def _on_error(self, msg):
        now = time.monotonic()
        self.n_msgs += 1
        self.last_thr = msg.throttle_error
        self.hist.append((now, msg.throttle_error))
        while self.hist and now - self.hist[0][0] > 6.0:
            self.hist.popleft()
        e = getattr(msg, self.field)
        self.last_error = e
        if self.collecting:
            self.samples.append((now, e))

    def pub_gains(self, key, g):
        m = PIDTune()
        m.kp, m.ki, m.kd = float(g[0]), float(g[1]), float(g[2])
        self.pubs[key].publish(m)

    def pub_all_zero(self):
        for k in self.pubs:
            self.pub_gains(k, (0.0, 0.0, 0.0))

    def spin_for(self, s):
        end = time.monotonic() + s
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def wait_for_feedback(self, timeout=5.0):
        end, n0 = time.monotonic() + timeout, self.n_msgs
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.n_msgs > n0:
                return True
        return False

    # ---------------- reset: zero gains, let the drone fall, wait until still ----------------
    def reset_to_ground(self):
        a = self.a
        t0 = time.monotonic()
        last_pub = 0.0
        while rclpy.ok() and time.monotonic() - t0 < a.reset_timeout:
            now = time.monotonic()
            if now - last_pub > 0.5:           # re-send zeros; cheap and robust
                self.pub_all_zero()
                last_pub = now
            rclpy.spin_once(self, timeout_sec=0.02)
            if now - t0 < a.min_fall:
                continue
            win = [e for (t, e) in self.hist if t >= now - a.stable_time]
            covered = bool(self.hist) and self.hist[0][0] <= now - a.stable_time
            if covered and len(win) >= 10 and min(win) >= a.ground_error \
                    and (max(win) - min(win)) <= a.stable_tol:
                return True
        win = [e for (t, e) in self.hist if t >= time.monotonic() - a.stable_time]
        if win:
            self.get_logger().warn(
                f"reset timeout: last {a.stable_time:.0f}s of throttle_error -> n={len(win)} "
                f"min={min(win):.3f} max={max(win):.3f} p2p={max(win) - min(win):.3f} "
                f"(needs min >= {a.ground_error} and p2p <= {a.stable_tol})")
        else:
            self.get_logger().warn("reset timeout: no throttle_error samples in the window")
        return False

    # ---------------- one trial ----------------
    def run_trial(self, g):
        a = self.a
        self.trial_id += 1

        ok = self.reset_to_ground()
        if not ok:
            self.get_logger().warn("reset failed, retrying once")
            ok = self.reset_to_ground()
        if not ok:
            raise ResetFailed("drone did not settle at the ground (error >= %.2f)" % a.ground_error)

        # start: background throttle gains if tuning another axis, then the candidate gains
        for _ in range(3):
            if a.axis != "throttle":
                self.pub_gains("throttle", a.throttle_gains)
            self.pub_gains(a.axis, g)
            self.spin_for(0.03)

        self.samples, self.collecting = [], True
        t0 = time.monotonic()
        trial_time = a.settle_time + a.hold_time
        aborted = False
        while rclpy.ok() and time.monotonic() - t0 < trial_time:
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.last_error is not None and abs(self.last_error) > a.abort_error:
                aborted = True
                break
        self.collecting = False

        trace = [(t - t0, e) for t, e in self.samples]
        if len(trace) < 10:
            res = dict(cost=float("inf"), settle_t=float("nan"), max_after=float("nan"),
                       passed=False, note="no_feedback")
        elif aborted:
            res = dict(cost=a.abort_penalty, settle_t=float("nan"), max_after=float("nan"),
                       passed=False, note="aborted")
        else:
            res = self.score(trace, trial_time)

        self.log.writerow([self.trial_id, f"{g[0]:.4f}", f"{g[1]:.4f}", f"{g[2]:.4f}",
                           f"{res['cost']:.3f}", f"{res['settle_t']:.2f}", f"{res['max_after']:.3f}",
                           res["passed"], res["note"]])
        self.log_file.flush()
        self.get_logger().info(
            f"trial {self.trial_id}: kp={g[0]:.3f} ki={g[1]:.4f} kd={g[2]:.3f} | cost={res['cost']:.2f} "
            f"lastOutOfBand={res['settle_t']:.2f}s maxAfter5s={res['max_after']:.3f} "
            f"{'PASS' if res['passed'] else 'fail'} ({res['note']})")
        res["trace"] = trace
        return res

    def score(self, trace, trial_time):
        a = self.a
        band_s = a.band * a.margin
        hold = rise = chatter = 0.0
        last_out_spec = last_out_score = max_after = 0.0
        for i in range(1, len(trace)):
            t, e = trace[i]
            tp, ep = trace[i - 1]
            dt, ae = t - tp, abs(e)
            if ae > a.band:
                last_out_spec = t
            if ae > band_s:
                last_out_score = t
            if t < a.settle_time:
                rise += ae * dt
            else:
                max_after = max(max_after, ae)
                hold += max(0.0, ae - band_s) * dt
            chatter += (e - ep) ** 2
        complete = trace[-1][0] >= 0.95 * trial_time
        passed = complete and last_out_spec <= a.settle_time
        cost = (a.w_rise * rise + a.w_hold * hold
                + a.w_settle * max(0.0, last_out_score - a.settle_time)
                + a.w_chatter * chatter + (0.0 if complete else a.abort_penalty))
        return dict(cost=cost, settle_t=last_out_spec, max_after=max_after, passed=passed,
                    note="ok" if complete else "short_trial")

    def evaluate(self, g):
        """Mean cost over --runs; if it passes, confirm with --confirm more runs."""
        results = [self.run_trial(g) for _ in range(self.a.runs)]
        if all(r["passed"] for r in results) and self.a.confirm > 0:
            self.get_logger().info(f"passed - confirming with {self.a.confirm} more run(s)")
            results += [self.run_trial(g) for _ in range(self.a.confirm)]
        if any(math.isinf(r["cost"]) for r in results):
            return float("inf"), False, []
        cost = sum(r["cost"] for r in results) / len(results)
        return cost, all(r["passed"] for r in results), results[0]["trace"]

    # ---------------- twiddle ----------------
    def tune(self, state):
        a = self.a
        lo = [a.kp_range[0], a.ki_range[0], a.kd_range[0]]
        hi = [a.kp_range[1], a.ki_range[1], a.kd_range[1]]
        clip = lambda i, v: max(lo[i], min(hi[i], v))

        def consider(g, c, ok, tr):
            if c < state["cost"]:
                state.update(gains=list(g), cost=c, passed=ok, trace=tr)
                return True
            return False

        for s in a.seeds_list:
            self.get_logger().info(f"seed {s}")
            c, ok, tr = self.evaluate(s)
            consider(s, c, ok, tr)
            if ok and a.stop_on_pass:
                return
        self.get_logger().info(f"best seed: {state['gains']} cost={state['cost']:.2f}")

        p, dp = list(state["gains"]), list(a.dp)
        for it in range(a.max_iters):
            if sum(dp) < a.tol:
                break
            for i in range(3):
                old = p[i]
                improved = False
                for d in (+1, -1):
                    p[i] = clip(i, old + d * dp[i])
                    if p[i] == old:
                        continue
                    c, ok, tr = self.evaluate(p)
                    if consider(p, c, ok, tr):
                        dp[i] *= 1.1
                        improved = True
                        break
                    p[i] = old
                if not improved:
                    dp[i] *= 0.9
                if state["passed"] and a.stop_on_pass:
                    return
                p = list(state["gains"])
            self.get_logger().info(
                f"iter {it + 1}/{a.max_iters}: best={state['cost']:.2f} "
                f"gains={['%.3f' % x for x in state['gains']]} dp={['%.3f' % x for x in dp]}")

    def close(self):
        self.log_file.close()


def parse():
    ap = argparse.ArgumentParser(description="Auto-reset PID tuner (ROS 2)")
    ap.add_argument("--axis", choices=["throttle", "pitch", "roll", "yaw"], default="throttle")
    ap.add_argument("--topic", default=None, help="Gain topic of the tuned axis (default /<axis>_pid)")
    ap.add_argument("--throttle-topic", default="/throttle_pid")
    ap.add_argument("--error-topic", default="/pos_error")
    ap.add_argument("--throttle-gains", nargs=3, type=float, default=[11.55, 0.19, 40.0],
                    help="Throttle gains held while tuning pitch/roll/yaw.")

    ap.add_argument("--set", nargs=3, type=float, metavar=("KP", "KI", "KD"),
                    help="Reset, then run ONE trial with these gains (no tuning).")
    ap.add_argument("--seeds", nargs="+", type=float, default=[11.55, 0.19, 40.0],
                    help="Flat list of kp ki kd triples tried first.")
    ap.add_argument("--kp-range", nargs=2, type=float, default=[0.0, 80.0])
    ap.add_argument("--ki-range", nargs=2, type=float, default=[0.0, 6.0])
    ap.add_argument("--kd-range", nargs=2, type=float, default=[0.0, 150.0])
    ap.add_argument("--dp", nargs=3, type=float, default=[2.0, 0.1, 6.0], help="Initial twiddle steps.")

    ap.add_argument("--band", type=float, default=0.4)
    ap.add_argument("--settle-time", type=float, default=5.0)
    ap.add_argument("--hold-time", type=float, default=10.0)
    ap.add_argument("--margin", type=float, default=0.8, help="Score against band*margin for headroom.")

    # reset sequence
    ap.add_argument("--ground-error", type=float, default=2.4,
                    help="Throttle error counts as 'on the ground' when >= this (your first plot sat at ~2.65).")
    ap.add_argument("--stable-time", type=float, default=2.0)
    ap.add_argument("--stable-tol", type=float, default=0.15, help="Max peak-to-peak error while 'stable'.")
    ap.add_argument("--min-fall", type=float, default=2.0, help="Always wait at least this long after zeroing.")
    ap.add_argument("--reset-timeout", type=float, default=20.0)

    ap.add_argument("--runs", type=int, default=1,
                    help="Number of trials per gain set; also controls repetitions with --set.")
    ap.add_argument("--confirm", type=int, default=2, help="Extra runs to confirm a passing gain set.")
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--tol", type=float, default=0.3)
    ap.add_argument("--no-stop-on-pass", dest="stop_on_pass", action="store_false")
    ap.add_argument("--abort-error", type=float, default=6.0)
    ap.add_argument("--abort-penalty", type=float, default=1e4)
    ap.add_argument("--w-rise", type=float, default=1.0)
    ap.add_argument("--w-hold", type=float, default=50.0)
    ap.add_argument("--w-settle", type=float, default=5.0)
    ap.add_argument("--w-chatter", type=float, default=2.0)
    ap.add_argument("--log", default=None)
    a, ros_args = ap.parse_known_args()

    if a.runs < 1:
        ap.error("--runs must be at least 1")
    if len(a.seeds) % 3:
        ap.error("--seeds needs a multiple of 3 numbers")
    a.seeds_list = [a.seeds[i:i + 3] for i in range(0, len(a.seeds), 3)]
    a.log = a.log or f"pid_trials_{a.axis}.csv"
    return a, ros_args


def main():
    a, ros_args = parse()
    rclpy.init(args=ros_args)
    node = AutoResetTuner(a)
    state = dict(gains=None, cost=float("inf"), passed=False, trace=[])
    land_on_exit = False
    try:
        if not node.wait_for_feedback():
            node.get_logger().error("No feedback on %s. Is pico_controller running in Y mode?" % a.error_topic)
            return 1

        if a.set:
            results = [node.run_trial(a.set) for _ in range(a.runs)]
            passed = sum(res["passed"] for res in results)
            print(f"PASS: {passed}/{len(results)} trials")
            return 0

        node.tune(state)
    except ResetFailed as e:
        node.get_logger().error(f"Reset failed: {e}. Stopping and keeping the best result so far.")
        land_on_exit = True
    except KeyboardInterrupt:
        print()
        node.get_logger().warn("Interrupted - landing the drone (zero gains).")
        land_on_exit = True
    finally:
        if state["gains"] is not None:
            g = state["gains"]
            tag = "SPEC MET" if state["passed"] else "best found (spec NOT met)"
            node.get_logger().info(f"{tag}: kp={g[0]:.4f} ki={g[1]:.4f} kd={g[2]:.4f} cost={state['cost']:.2f}")
            with open(f"best_{a.axis}.json", "w") as f:
                json.dump({"axis": a.axis, "kp": g[0], "ki": g[1], "kd": g[2],
                           "cost": state["cost"], "spec_met": state["passed"]}, f, indent=2)
            with open(f"best_trace_{a.axis}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "error"])
                w.writerows(state["trace"])
        if rclpy.ok():
            if land_on_exit or state["gains"] is None:
                node.pub_all_zero()
            else:
                if a.axis != "throttle":
                    node.pub_gains("throttle", a.throttle_gains)
                node.pub_gains(a.axis, state["gains"])
            node.spin_for(0.3)
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())