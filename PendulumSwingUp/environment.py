from typing import Optional
import numpy as np
from gym import spaces
import numpy as np
import RobotDART as rd
import dartpy # OSX breaks if this is imported before RobotDART


class PendulumEnv():

    def __init__(self, enable_graphics = True):

        self.dt = 0.001
        self.simu = rd.RobotDARTSimu(self.dt)

        self.robot = rd.Robot("pendulum.urdf")
        self.robot.set_actuator_types("torque")

        self.simu.add_robot(self.robot)

        gconfig = rd.gui.GraphicsConfiguration(1024, 768)
        if enable_graphics:
            graphics = rd.gui.Graphics(gconfig)
        else:
            graphics = rd.gui.WindowlessGraphics(gconfig)
        self.simu.set_graphics(graphics)
        graphics.look_at([0., 3., 2.], [0., 0., 0.])

        self.robot.fix_to_world()
        self.init_pos = [np.pi]
        self.robot.set_positions(self.init_pos)

        self.max_torque = self.robot.force_upper_limits()[0]
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)

        self.high_state = np.array([1, 1, self.max_torque], dtype=np.float32)
        self.low_state = -self.high_state

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)


    def step(self, cmd):
        cmd = np.clip(cmd, -self.max_torque, self.max_torque)[0]

        self.robot.set_commands(cmd)

        if self.simu.step_world():
            return

        theta = self.robot.positions()[0]
        torque = self.robot.forces()[0]

        obs = np.array([np.cos(theta), np.sin(theta), torque], dtype=np.float32)

        cost = theta**2

        return obs, -cost, False


    def reset(self):
        self.robot.reset()
        self.robot.set_positions(self.init_pos)
        
        theta = self.robot.positions()[0]
        torque = self.robot.forces()[0]

        obs = np.array([np.cos(theta), np.sin(theta), torque], dtype=np.float32)

        return obs
