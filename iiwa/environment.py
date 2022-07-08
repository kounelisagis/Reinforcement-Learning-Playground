import numpy as np
from gym import spaces
import numpy as np
import RobotDART as rd
import dartpy # OSX breaks if this is imported before RobotDART
import copy


class IiwaEnv():

    def __init__(self, enable_graphics = True):

        self.dt = 0.01
        self.simu = rd.RobotDARTSimu(self.dt)

        self.robot = rd.Iiwa()
        self.robot.set_actuator_types("servo")

        self.dofs = ['iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3', 'iiwa_joint_4', 'iiwa_joint_5', 'iiwa_joint_6', 'iiwa_joint_7']

        self.simu.add_robot(self.robot)
        self.simu.add_checkerboard_floor()

        gconfig = rd.gui.GraphicsConfiguration(1024, 768)
        if enable_graphics:
            graphics = rd.gui.Graphics(gconfig)
        else:
            graphics = rd.gui.WindowlessGraphics(gconfig)
        self.simu.set_graphics(graphics)
        graphics.look_at([0., 3., 2.], [0., 0., 0.])

        self.robot.fix_to_world()

        # actions: velocities
        # observations: positions, velocities

        self.max_velocity = self.robot.velocity_upper_limits()
        self.min_velocity = self.robot.velocity_lower_limits()
        self.action_space = spaces.Box(low=self.min_velocity, high=self.max_velocity, dtype=np.float32)

        self.min_position = self.robot.position_lower_limits()
        self.max_position = self.robot.position_upper_limits()

        self.low_state = np.array(
            [self.min_position, self.min_velocity], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_velocity], dtype=np.float32
        )

        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

        # set initial joint positions
        self.target_pos = copy.copy(self.robot.positions())
        self.target_pos[0] = -2.
        self.target_pos[3] = -np.pi / 2.0
        self.target_pos[5] = np.pi / 2.0
        self.robot.set_positions(self.target_pos)

        ghost_robot = self.robot.clone_ghost()
        self.simu.add_robot(ghost_robot)

        self.init_pos = np.random.uniform(self.min_position/2, self.max_position/2, size=(7,))
        self.robot.set_positions(self.init_pos)


    def step(self, cmd):
        done = False
        cmd = np.clip(cmd, self.min_velocity, self.max_velocity)[0]

        self.robot.set_commands(cmd)

        if self.simu.step_world():
            return

        # distance between target_pos and current_pos
        cost = np.linalg.norm(self.target_pos - self.robot.positions()) #+ 0.1*np.linalg.norm(self.robot.velocities())
        obs = np.concatenate((self.robot.positions(), self.robot.velocities()), axis=0)

        return obs, -cost, done


    def reset(self):
        self.robot.reset()
        self.robot.set_positions(self.init_pos)

        if self.simu.step_world():
            return

        obs = np.concatenate((self.robot.positions(), self.robot.velocities()), axis=0)

        return obs
