import igraph
from abc import ABC, abstractmethod
import numpy as np
import cairo


class VNode(ABC):

    def __init__(self, s, q_nodes, h):
        if h < 1:
            raise RuntimeError("Invalid greedieness: " + str(h))

        self.state = s
        self.h = h
        self.q_nodes = q_nodes
        super(VNode, self).__init__()

    def select_greedy_action(self, h=None):
        if h is None:
            h = self.h - 1
        elif h >= self.h:
            raise RuntimeError("h too large! Maximum allowed value: " + str(self.h))

        return np.argmax([node.get_q(h) for node in self.q_nodes])

    @abstractmethod
    def select_action(self, h=None):
        pass

    def get_children(self, a):
        return self.q_nodes[a]

    def compute_values(self):
        # For the non-greedy value we just compute the average
        n_0 = np.array([node.get_n(0) for node in self.q_nodes])
        visits = [np.sum(n_0)]
        values = [np.sum(n_0 * np.array([node.get_q(0) for node in self.q_nodes])) / visits[0]]

        # Now comes the greedy horizons
        for h in range(1, self.h):
            q_hs = np.array([node.get_q(h - 1) for node in self.q_nodes])
            n_hs = np.array([node.get_n(h - 1) for node in self.q_nodes])
            max_arm = np.argmax(q_hs)
            values.append(q_hs[max_arm])
            visits.append(n_hs[max_arm])

        return values, visits


class TerminalVNode:

    def __init__(self, s, h, reward):
        self.h = h
        self.state = s
        self.reward = reward
        self.n = 1

    def compute_values(self):
        return np.ones(self.h) * (self.reward / self.n), np.ones(self.h) * self.n

    def update(self, reward):
        self.reward += reward
        self.n += 1


class UCBVNode(VNode):

    def __init__(self, s, q_nodes, h, alpha=4, **extra_args):
        self.alpha = alpha
        super(UCBVNode, self).__init__(s, q_nodes, h)

    def select_action(self, h=None):
        if h is None:
            h = self.h - 1
        elif h >= self.h:
            raise RuntimeError("h too large! Maximum allowed value: " + str(self.h))

        # vists0 = np.array([node.get_n(0) for node in self.q_nodes])
        vists = np.array([node.get_n(h) for node in self.q_nodes])
        exploration_bonus = self.alpha * np.sqrt(np.log(np.sum(vists)) / vists)
        return np.argmax(np.array([node.get_q(h) for node in self.q_nodes]) + exploration_bonus)


class QNode:

    def __init__(self, ns, h, discount_factor, **extra_args):
        self.discount_factor = discount_factor
        self.h = h

        self.values = np.zeros((h, ns))
        self.visits = np.ones((h, ns))
        self.v_nodes = [None] * ns

        self.immediate_rewards = 0.
        self.total_visits = 0.

    def update(self, reward, next_state):
        self.immediate_rewards += reward
        self.total_visits += 1.
        self.values[:, next_state], self.visits[:, next_state] = self.v_nodes[next_state].compute_values()

    def get_q(self, h=None):
        if h is None:
            h = self.h - 1
        elif h >= self.h:
            raise RuntimeError("h too large! Maximum allowed value: " + str(self.h))

        return self.immediate_rewards / self.total_visits + \
               self.discount_factor * (np.sum(self.visits[h, :] * self.values[h, :]) / np.sum(self.visits[h, :]))

    def get_n(self, h=None):
        if h is None:
            h = self.h - 1
        if h >= self.h:
            raise RuntimeError("h too large! Maximum allowed value: " + str(self.h))

        return np.sum(self.visits[h, :])

    def get_children(self, s):
        return self.v_nodes[s]

    def set_children(self, s, vnode):
        self.v_nodes[s] = vnode


class SearchTree:

    def __init__(self, simulator, initial_state, no, na, discount_factor, max_depth=100, h=1, vnode=VNode, qnode=QNode,
                 **extra_args):
        if h < 1:
            raise RuntimeError("Invalid greedieness: " + str(h))

        self.h = h
        self.no = no
        self.na = na
        self.extra_args = extra_args
        self.discount_factor = discount_factor
        self.max_depth = max_depth
        self.qnode_const = qnode
        self.vnode_const = vnode
        self.simulator = simulator
        self.root = self.expand_node(TerminalVNode(initial_state, h, 0.))

    def progress_tree(self, action, next_state):
        self.root = self.root.get_children(action).get_children(next_state)

    def expand_node(self, vnode):
        if not isinstance(vnode, TerminalVNode):
            raise RuntimeError("VNode already expanded!")

        state = vnode.state
        q_nodes = []
        for i in range(0, self.na):
            q_node = self.qnode_const(self.no, self.h, self.discount_factor, **self.extra_args)
            reward, next_state, observation = self.simulator.simulate(state, i)
            dummy = TerminalVNode(next_state, self.h, self.simulator.rollout(next_state))
            q_node.set_children(observation, dummy)
            q_node.update(reward, observation)
            q_nodes.append(q_node)
        new_node = self.vnode_const(state, q_nodes, self.h, **self.extra_args)

        return new_node

    def search(self, n_runs=1000, visualize=False):
        for i in range(0, n_runs):
            if visualize:
                self.visualize()

            current_depth = 1
            q_nodes = []
            rewards = []
            observations = []
            current_node = self.root
            # Progress through the search tree
            while not isinstance(current_node, TerminalVNode):
                a = current_node.select_action()
                r, ns, no = self.simulator.simulate(current_node.state, a)

                # Store the transition data for later update of the nodes
                q_nodes.append(current_node.get_children(a))
                rewards.append(r)
                observations.append(no)

                # Update the depth count and progress in the search tree
                current_node = q_nodes[-1].get_children(no)
                current_depth += 1

            # We extend the tree if we can
            if current_depth <= self.max_depth:
                new_node = self.expand_node(current_node)
                q_nodes[-1].set_children(no, new_node)
            else:
                current_node.update(self.simulator.rollout(current_node.state))

            # Now we back up the values
            for q, r, obs in zip(reversed(q_nodes), reversed(rewards), reversed(observations)):
                q.update(r, obs)

    def _parse_tree(self, current, node_map, cur_id, edges):
        this_id = cur_id
        node_map[this_id] = current
        cur_id += 1

        if isinstance(current, TerminalVNode):
            return cur_id
        elif issubclass(type(current), VNode):
            for a in range(0, self.na):
                # The child will for sure have the current id
                edges.append((this_id, cur_id))
                child = current.get_children(a)
                cur_id = self._parse_tree(child, node_map, cur_id, edges)
        elif issubclass(type(current), QNode):
            for o in range(0, self.no):
                # The child will for sure have the current id
                edges.append((this_id, cur_id))
                child = current.get_children(o)
                if child is not None:
                    cur_id = self._parse_tree(child, node_map, cur_id, edges)
        else:
            raise RuntimeError("Invalid node type: " + str(type(current)))

        return cur_id

    def _create_label(self, node):
        if issubclass(type(node), VNode):
            vs, ns = node.compute_values()
            ret = ""
            for i in range(0, self.h):
                ret += "(%.2f, %d)" % (vs[i], ns[i]) + "\n"
            return ret
        elif issubclass(type(node), QNode):
            ret = ""
            for i in range(0, self.h):
                q = node.get_q(i)
                n = node.get_n(i)
                ret += "(%.2f, %d)" % (q, n) + "\n"

            return ret
        else:
            vs, ns = node.compute_values()
            return "(%.2f, %d)" % (vs[0], ns[0])

    def visualize(self, tree=None):
        # First we create a mapping of a ids to tree nodes as well as the edges between the nodes
        node_map = {}
        edges = []
        second_root = self._parse_tree(self.root, node_map, 0, edges)

        if tree is not None:
            tree._parse_tree(tree.root, node_map, second_root, edges)

        # Here we create a tree with iGraph
        g = igraph.Graph(n=len(node_map), directed=True)
        g.add_edges(edges)

        # Finally we print the tree
        colors = ["red" if issubclass(type(node_map[i]), QNode) else "blue" if issubclass(type(node_map[i]),
                                                                                          VNode) else "yellow" for i in
                  g.vs.indices]
        labels = [self._create_label(node_map[i]) if i < second_root else tree._create_label(node_map[i]) for i in g.vs.indices]
        lay = g.layout_reingold_tilford(root=[0, second_root])

        coords = np.array(lay.coords)
        x_lim = [np.min(coords[:, 0]), np.max(coords[:, 0])]
        y_lim = [np.min(coords[:, 1]), np.max(coords[:, 1])]

        pl = igraph.Plot(bbox=(200 + (x_lim[1] - x_lim[0]) * 60, 200 + (y_lim[1] - y_lim[0]) * 60), background="white")
        pl.add(g, layout=lay, margin=40, vertex_size=40, vertex_color=colors, vertex_label=labels,
               vertex_label_dist=1.2, vertex_label_size=10)
        pl.show()
