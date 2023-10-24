import gym
from multiprocessing import Process, Pipe
from gym.wrappers import TimeLimit
from gym.envs.toy_text.discrete import DiscreteEnv
import uctp
import numpy as np
import matplotlib.pyplot as plt
import os

configs = {
    "FrozenLake-Deterministic": {
        "env": "FrozenLake8x8-v0",
        "extra_args": {"is_slippery": False},
        "planning_steps": [5, 10, 20, 40, 80, 160, 320, 640, 1280],
        "n_runs": 100
    },
    "FrozenLake-Stochastic": {
        "env": "FrozenLake8x8-v0",
        "extra_args": {"is_slippery": True},
        "planning_steps": [5120, 10240, 20480, 40960, 81920, 163840], # 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120
        "n_runs": 100
    },
    "Taxi": {
        "env": "Taxi-v2",
        "extra_args": {},
        "planning_steps": [5, 10, 20, 40, 80, 160, 320, 640, 1280],
        "n_runs": 100,
        "reward_offset": 10,
        "reward_scale": 1. / 30.
    }
}


class SimWrapper:

    def __init__(self, env, gamma, reward_offset=0., reward_scale=1.):
        self.env = env
        self.gamma = gamma
        self.reward_offset = reward_offset
        self.reward_scale = reward_scale
        if not (isinstance(env, TimeLimit) and isinstance(env.env, DiscreteEnv)):
            raise RuntimeError("Invalid Environment given")

        self.max_steps = env._max_episode_steps
        self.ns = self.max_steps * self.env.observation_space.n
        self.ns_internal = self.env.observation_space.n
        self.na = self.env.action_space.n

    def rollout(self, s):
        # Get the environment state
        state = s % self.ns_internal
        steps = int(s / self.ns_internal)

        # Set the state of the environment
        self.env._elapsed_steps = steps
        self.env.env.s = state

        done = False
        total_reward = 0.
        traj = []
        disc = 1.
        while not done:
            a = np.random.randint(self.na)
            s, r, done, _ = self.env.step(a)
            r = self.normalize_reward(r)
            traj.append(a)
            traj.append(r)
            traj.append(s)

            total_reward += disc * r
            disc *= self.gamma

        return total_reward, traj

    def normalize_reward(self, reward):
        return (reward + self.reward_offset) * self.reward_scale

    def simulate(self, s, a):
        # Get the environment state
        state = s % self.ns_internal
        steps = int(s / self.ns_internal)

        # Set the state of the environment
        self.env._elapsed_steps = steps
        self.env.env.s = state

        s, r, done, _ = self.env.step(a)
        r = self.normalize_reward(r)

        return s + self.env._elapsed_steps * self.env.observation_space.n, r, done


def create_p_func(n_0, p):
    def p_func(n, depth):
        if n < n_0:
            return 1.
        else:
            v0 = np.log(0.4 * (n_0 + 1) / (0.24 * n_0))
            v = np.log(0.4 * (n + 1) / (0.24 * n))
            return p * ((10. / float(p)) + v0 - v)

    return p_func


def worker(remote, parent_remote, env_name, extra_args, n_runs, n_steps, p, pid, r_off, r_scale):
    parent_remote.close()
    print("Running with pid: " + str(pid))

    try:
        gamma = 1.

        discounted_rewards = []
        for i in range(0, n_runs):
            seed = pid * 1000 + i

            env = gym.make(env_name, **extra_args)
            env.seed(seed)
            env.reset()

            s0 = env.env.s
            wrapper = SimWrapper(env, gamma, reward_offset=r_off, reward_scale=r_scale)

            test_env = gym.make(env_name, **extra_args)
            test_env.seed(seed)
            s = test_env.reset()
            assert s0 == s

            np.random.seed(seed)
            st = uctp.SearchTree(wrapper, s0, wrapper.ns, wrapper.na, gamma, vnode=uctp.UCBVNode, p=p)
            st.search(n_steps)

            done = False
            discounted_reward = 0.
            disc = 1.
            while not done:
                a, q = st.root.select_greedy_action()
                # print("Q-Values: " + str([node.get_q()[0] for node in st.root.q_nodes]))
                # print("Estimated Q: " + str(q))
                s, reward, done, info = test_env.step(a)
                discounted_reward += disc * reward
                disc *= gamma
                # test_env.render()
                st.progress_tree(a, s + test_env._elapsed_steps * test_env.observation_space.n)
                if not done:
                    st.search(n_steps)

            print("Seed: " + str(seed) + ", Finished experiment " + str(i) + ", Discounted reward: " +
                  str(discounted_reward))

            discounted_rewards.append(discounted_reward)
        remote.send(discounted_rewards)
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')


def run_exp(env_name, extra_args, n_runs, n_plan_steps, r_off, r_scale, n_cores=8, baseline=False):
    p = 1. if baseline else create_p_func(10, 100)

    # Compute the number of runs per worker
    n_sub_runs = np.ones(n_cores, dtype=np.int) * int(n_runs / n_cores)
    n_sub_runs[0:(n_runs % n_cores)] += 1

    # Create the workers
    remotes, work_remotes = zip(*[Pipe() for _ in range(n_cores)])
    ps = [Process(target=worker,
                  args=(work_remote, remote, env_name, extra_args, n_sub, n_plan_steps, p, pid, r_off, r_scale))
          for (work_remote, remote, n_sub, pid) in zip(work_remotes, remotes, n_sub_runs, range(0, n_cores))]
    for p in ps:
        p.daemon = True  # if the main process crashes, we should not cause things to hang
        p.start()
    for remote in work_remotes:
        remote.close()

    discounted_rewards = [remote.recv() for remote in remotes]
    for remote in remotes:
        remote.close()

    return np.concatenate(discounted_rewards, axis=0)


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.realpath(__file__))
    log_dir = os.path.join(base_path, "logs")

    # Create the log directory if it does not exist yet
    os.makedirs(log_dir, exist_ok=True)

    # Next load the config and check whether the experiment has been executed or not
    spec = "FrozenLake-Stochastic"
    config = configs[spec]
    spec_log_dir = os.path.join(log_dir, spec)

    visualize = os.path.exists(spec_log_dir)
    if visualize:
        rewards = np.load(os.path.join(spec_log_dir, "rewards.npy"))
        rewards_old = np.load(os.path.join(spec_log_dir, "rewards_baseline.npy"))

        l0, = plt.plot(config["planning_steps"], np.mean(rewards_old, axis=1))
        plt.fill_between(config["planning_steps"], np.quantile(rewards_old, 0.1, axis=1),
                         np.quantile(rewards_old, 0.9, axis=1), color=l0.get_color(), alpha=0.5)

        l1, = plt.plot(config["planning_steps"], np.mean(rewards, axis=1))
        plt.fill_between(config["planning_steps"], np.quantile(rewards, 0.1, axis=1),
                         np.quantile(rewards, 0.9, axis=1), color=l1.get_color(), alpha=0.5)

        plt.legend([l0, l1], ["UCT", "Power-UCT"])
        plt.show()
    else:
        rewards = []
        rewards_baseline = []
        for steps in config["planning_steps"]:
            r_off = 0. if "reward_offset" not in config else config["reward_offset"]
            r_scale = 1. if "reward_scale" not in config else config["reward_scale"]
            print("Running with " + str(steps) + " planning steps")
            print("Running with power UCT")
            rewards.append(
                run_exp(config["env"], config["extra_args"], config["n_runs"], steps, r_off, r_scale, baseline=False))
            print("Running with regular UCT")
            rewards_baseline.append(
                run_exp(config["env"], config["extra_args"], config["n_runs"], steps, r_off, r_scale, baseline=True))

        os.makedirs(spec_log_dir)
        np.save(os.path.join(spec_log_dir, "rewards.npy"), rewards)
        np.save(os.path.join(spec_log_dir, "rewards_baseline.npy"), rewards_baseline)
