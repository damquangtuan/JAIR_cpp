import numpy as np
import time
import uct
import uctp
import ments
import tents
import alpha_divergence


class KNSimulator:

    def __init__(self, k, n, seed, save_seed=False):
        np.random.seed(seed)

        self.save_seed = save_seed
        self.k = k
        self.n = n
        self.means = []
        for i in range(0, n):
            self.means.append(np.random.uniform(size=(k,) * (i + 1)))

        self.variances = []
        for i in range(0, n):
            self.variances.append(np.random.uniform(low=1e-4, high=0.05, size=(k,) * (i + 1)))

        if self.save_seed:
            self.rand_state = np.random.get_state()

    def simulate(self, state, action):
        if self.save_seed:
            np.random.set_state(self.rand_state)

        cur_n = len(state)
        next_state = state + (action,)

        if cur_n == self.n:
            return 1., state

        mean = self.means[cur_n][next_state]
        variance = self.variances[cur_n][next_state]

        res = np.random.normal(mean, np.sqrt(variance)), next_state, 0
        if self.save_seed:
            self.rand_state = np.random.get_state()
        return res

    def rollout(self, state):
        if self.save_seed:
            np.random.set_state(self.rand_state)

        cur_n = len(state)
        reward = 0.
        for i in range(cur_n, self.n):
            r, state, obs = self.simulate(state, np.random.randint(0, self.k))
            reward += r

        if self.save_seed:
            self.rand_state = np.random.get_state()
        return reward

    def expected_reward(self, actions):
        expected_rew = 0.
        for i in range(0, self.n):
            expected_rew += self.means[i][actions[0:i + 1]]
        return expected_rew


def compare(k, n, h, seed, algorithm='uct', visualize_final=True):
    # Create the simulator
    sim = KNSimulator(k, n, seed)

    # Create the search tree
    depth = 2
    tree = None
    if algorithm == 'uct':
        tree = uct.SearchTree(sim, (), 1, k, 1., depth, h=h, vnode=uct.UCBVNode, alpha=.5,seed=seed)
    elif algorithm == 'uctp':
        tree = uctp.SearchTree(sim, (), 1, k, 1., depth, h=h, vnode=uctp.UCBVNode,
                               alpha=.5,power=100.,seed=seed)
    elif algorithm == 'ments':
        tree = ments.SearchTree(sim, (), 1, k, 1., depth, h=h, vnode=ments.UCBVNode, alpha=1.41,seed=seed)
    elif algorithm == 'tents':
        tree = tents.SearchTree(sim, (), 1, k, 1., depth, h=h, vnode=tents.UCBVNode, alpha=1.41,seed=seed)
    elif algorithm == 'alpha_divergence':
        tree = alpha_divergence.SearchTree(sim, (), 1, k, 1., depth, h=h,
                                           vnode=alpha_divergence.UCBVNode, alpha=16.0,seed=seed)

    actions = ()
    for i in range(0, n - depth):
        tree.search(n_runs=2000)
        if visualize_final:
            tree.visualize()
        a = tree.root.select_greedy_action()
        tree.progress_tree(a, 0)
        actions += (a,)

    # tree.search(n_runs=2000)
    # for i in range(0, depth):
    #     if visualize_final:
    #         tree.visualize()
    #     a = tree.root.select_greedy_action()
    #     tree.progress_tree(a, 0)
    #     actions += (a,)

    # print(algorithm + " Actions for h=" + str(h) + ": " + str(actions))
    # print("Expected Reward for h=" + str(h) + ": " + str(sim.expected_reward(actions)))


def visualize_trees(k, n, algorithm='uct', seed=0):
    sim1 = KNSimulator(k, n, seed, save_seed=True)
    sim2 = KNSimulator(k, n, seed, save_seed=True)

    depth = 2

    if algorithm == 'uct':
        tree1 = uct.SearchTree(sim1, (), 1, k, 1., depth, h=1, vnode=uct.UCBVNode, alpha=.5,seed=seed)
        tree2 = uct.SearchTree(sim2, (), 1, k, 1., depth, h=2, vnode=uct.UCBVNode, alpha=.5,seed=seed)
        tree1.visualize(tree=tree2)
        for i in range(0, 20):
            tree1.search(n_runs=1)
            tree2.search(n_runs=1)
            tree1.visualize(tree=tree2)
    elif algorithm == 'uctp':
        tree1 = uctp.SearchTree(sim1, (), 1, k, 1., depth, h=1, vnode=uctp.UCBVNode,
                                alpha=.5, power=100.,seed=seed)
        tree2 = uctp.SearchTree(sim2, (), 1, k, 1., depth, h=2, vnode=uctp.UCBVNode,
                                alpha=.5, power=100.,seed=seed)
        tree1.visualize(tree=tree2)
        for i in range(0, 20):
            tree1.search(n_runs=1)
            tree2.search(n_runs=1)
            tree1.visualize(tree=tree2)
    elif algorithm == 'ments':
        tree1 = ments.SearchTree(sim1, (), 1, k, 1., depth, h=1, vnode=ments.UCBVNode, alpha=1.41,seed=seed)
        tree2 = ments.SearchTree(sim2, (), 1, k, 1., depth, h=2, vnode=ments.UCBVNode, alpha=1.41,seed=seed)
        tree1.visualize(tree=tree2)
        for i in range(0, 20):
            tree1.search(n_runs=1)
            tree2.search(n_runs=1)
            tree1.visualize(tree=tree2)
    elif algorithm == 'tents':
        tree1 = tents.SearchTree(sim1, (), 1, k, 1., depth, h=1, vnode=tents.UCBVNode, alpha=1.41,seed=seed)
        tree2 = tents.SearchTree(sim2, (), 1, k, 1., depth, h=2, vnode=tents.UCBVNode, alpha=1.41,seed=seed)
        tree1.visualize(tree=tree2)
        for i in range(0, 20):
            tree1.search(n_runs=1)
            tree2.search(n_runs=1)
            tree1.visualize(tree=tree2)
    elif algorithm == 'alpha-divergence':
        tree1 = alpha_divergence.SearchTree(sim1, (), 1, k, 1., depth, h=1, vnode=alpha_divergence.UCBVNode,
                                            alpha=16.,seed=seed)
        tree2 = alpha_divergence.SearchTree(sim2, (), 1, k, 1., depth, h=2, vnode=alpha_divergence.UCBVNode,
                                            alpha=16.,seed=seed)
        tree1.visualize(tree=tree2)
        for i in range(0, 20):
            tree1.search(n_runs=1)
            tree2.search(n_runs=1)
            tree1.visualize(tree=tree2)


if __name__ == "__main__":
    # algorithm={'uct', 'uctp', 'ments', 'tents', 'alpha_divergence'}
    algorithm={'alpha_divergence'}
    k=4
    for alg in algorithm:
        for i in range(0, 10):
            print("Seed " + str(i) + ":")

            t1 = time.time()
            compare(k, 3, 1, i, algorithm=alg,visualize_final=False)
            t2 = time.time()
            print("h=1 Search took: " + str(t2 - t1))

            t1 = time.time()
            compare(k, 3, 2, i, algorithm=alg,visualize_final=False)
            t2 = time.time()
            print("h=2 Search took: " + str(t2 - t1))

            print("\n")

        visualize_trees(k, 3, algorithm=alg,seed=5)
