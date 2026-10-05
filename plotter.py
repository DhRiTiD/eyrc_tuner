#!/usr/bin/env python3
import collections
import os
import threading

if os.environ.get('DISPLAY', '') == '' and os.name != 'nt':
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.node import Node

# Import custom message type
from error_msg.msg import Error


class ThrottleErrorPlotter(Node):

    def __init__(self, max_history=5000):
        super().__init__('throttle_error_plotter')

        # Store history for live display & back-scrolling. A single deque keeps
        # the time/value pairs aligned even when the ROS callback and animation
        # thread read/write concurrently.
        self.samples = collections.deque(maxlen=max_history)
        self._lock = threading.Lock()

        self.start_time = self.get_clock().now().nanoseconds / 1e9

        # Subscription to /pos_error
        self.subscription = self.create_subscription(
            Error,
            '/pos_error',
            self.listener_callback,
            10,
        )

        self.get_logger().info(
            'PID Tuning Plotter initialized. Listening to /pos_error [error_msg/msg/Error]...'
        )

    def listener_callback(self, msg):
        current_time = (self.get_clock().now().nanoseconds / 1e9) - self.start_time

        # Extract throttle_error field
        error_val = getattr(msg, 'throttle_error', 0.0)

        with self._lock:
            self.samples.append((current_time, float(error_val)))


def main(args=None):
    rclpy.init(args=args)
    node = ThrottleErrorPlotter()

    # Spin ROS 2 in background thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title('PID Tuning - Throttle Error')

    # Green highlighted range (-3.9 to +3.9)
    ax.axhspan(
        -3.9,
        3.9,
        color='green',
        alpha=0.15,
        linewidth=0,
        label='Acceptable Band (±3.9)',
    )

    # Dotted gray zero setpoint reference
    setpoint_line = ax.axhline(
        0, color='gray', linestyle=':', linewidth=1.5, label='Setpoint (0)'
    )

    # Red line for live error stream
    (error_line,) = ax.plot([], [], 'r-', lw=1.5, label='Throttle Error')

    # Live digital value text panel on top-left of plot
    value_text = ax.text(
        0.02,
        0.92,
        'Throttle Error: N/A',
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        verticalalignment='top',
        bbox=dict(
            boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'
        ),
    )

    ax.set_title('Real-Time Throttle Error Plot (PID Tuning)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error Value')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # Place legend cleanly on top-right
    ax.legend(loc='upper right', framealpha=0.9)
    fig.tight_layout()

    # Sliding window settings
    WINDOW_WIDTH = 20.0  # Shows last 20 seconds live

    def update_plot(frame):
        with node._lock:
            samples = list(node.samples)

        if samples:
            times_list = [sample[0] for sample in samples]
            errors_list = [sample[1] for sample in samples]
            latest_error = errors_list[-1]

            # Update line data
            error_line.set_data(times_list, errors_list)

            # Auto-scroll X-axis window
            latest_time = times_list[-1]
            if latest_time > WINDOW_WIDTH:
                ax.set_xlim(latest_time - WINDOW_WIDTH, latest_time)
            else:
                ax.set_xlim(0, WINDOW_WIDTH)

            # Dynamic Y-axis scale anchored past highlighted range
            y_min = min(errors_list) if min(errors_list) < -4.5 else -5.5
            y_max = max(errors_list) if max(errors_list) > 4.5 else 5.5
            margin = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - margin, y_max + margin)

            # Update digital readout text & color
            value_text.set_text(f'Throttle Error: {latest_error:.4f}')
            if -3.9 <= latest_error <= 3.9:
                value_text.set_color('green')
            else:
                value_text.set_color('red')

        return error_line, setpoint_line, value_text

    # Refresh animation (~20 FPS)
    anim = FuncAnimation(fig, update_plot, interval=50, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()