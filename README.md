# Reinforcement-Learning-Playground

### Mountain Car Problem - Continuous
Used machin library to solve the Continuous Mountain Car Problem using PPO and TD3. Implemented the right actor-critic networks and found the right hyper-parameters.


![Expected Return per iteration](https://github.com/kounelisagis/Reinforcement-Learning-Playground/assets/36283973/bb51a58e-7d35-470e-800f-33c9654a0502)
*Expected Return per iteration.*


### Pendulum swing-up - Torques
Used RobotDART, OpenAI Gym spaces, created reward function and used PPO and TD3. Used Frame Skipping technique.
The initial position is defined as x0 = [π], the observation space is the vector: [cos θ, sin θ, torque], and the reward function uses the angle θ, torque, and the command given to the robot.

![Figure_1(1)](https://github.com/kounelisagis/Reinforcement-Learning-Playground/assets/36283973/bcf9147c-f9a2-423b-8f50-630066a95918)
*TD3 - Expected return per iteration.*

![Figure_3](https://github.com/kounelisagis/Reinforcement-Learning-Playground/assets/36283973/a1bbd01c-d43d-488a-9ba9-93d1b7b54727)
*PPO - Expected return per iteration.*

### Iiwa joint space RL-controller - Servo
Used RobotDART, OpenAI Gym spaces, created reward function and used PPO and TD3.
The observation space is a vector that contains all the positions and velocities of the robot's joints, and the reward function is the norm of the difference between the final and current positions.

![Figure_5](https://github.com/kounelisagis/Reinforcement-Learning-Playground/assets/36283973/6de9363c-9b4b-48a1-97e7-f402b2ebbd6c)
*TD3 - Expected return per iteration.*

![Figure_7](https://github.com/kounelisagis/Reinforcement-Learning-Playground/assets/36283973/8878edac-2b8f-450b-926d-d66fdf6302a3)
*PPO - Expected return per iteration.*
