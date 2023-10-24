import igraph
from abc import ABC, abstractmethod
import numpy as np
import cairo


class VNode(ABC):

    def __init__(self, s, depth, q_nodes, p):
        self.state = s
        self.depth = depth
        if callable(p):
            self.p = p
        else:
            if p < 1.:
                raise RuntimeError("Invalid greedieness: " + str(p))
            self.p = lambda n, depth: float(p)
        self.q_nodes = q_nodes
        super(VNode, self).__init__()

    def select_greedy_action(self):
        qs = [node.get_q()[0] for node in self.q_nodes]
        action = np.argmax(qs)
        return action, qs[action]

    @abstractmethod
    def select_action(self):
        pass

    def get_children(self, a):
        return self.q_nodes[a]

    def compute_value(self):
        n_child = []
        qs = []
        for node in self.q_nodes:
            q, n = node.get_q()
            n_child.append(n)
            qs.append(q)

        n_child = np.array(n_child)
        qs = np.array(qs)
        n_total = np.sum(n_child)
        cur_p = self.p(int(n_total), self.depth)
        value = np.power(np.sum((n_child / n_total) * np.power(qs, cur_p)), 1. / cur_p)

        return value, n_total

    def compute_mean(self):
        n_child = []
        qs = []
        for node in self.q_nodes:
            q, n = node.get_q()
            n_child.append(n)
            qs.append(q)

        n_child = np.array(n_child)
        qs = np.array(qs)
        n_total = np.sum(n_child)
        value = np.sum((n_child / n_total) * qs)

        return value


class TerminalVNode:

    def __init__(self, s, depth, reward, expandable=True):
        self.state = s
        self.depth = depth
        self.expandable = expandable
        if expandable:
            self.reward = reward[0]
            self.trajectory = reward[1]
        else:
            self.reward = reward
            self.trajectory = None
        self.n = 1

    def compute_value(self):
        return self.reward / self.n, self.n

    def update(self, reward):
        self.reward += reward
        self.n += 1
        self.trajectory = None
        # TODO if we have TerminalVNodes at the bottom of the search tree, we need to handle multiple trajectories


class UCBVNode(VNode):

    def __init__(self, s, depth, q_nodes, p, alpha=2. / np.sqrt(2), **extra_args):
        self.alpha = alpha
        super(UCBVNode, self).__init__(s, depth, q_nodes, p)

    def select_action(self):
        n_child = []
        qs = []
        for node in self.q_nodes:
            q, n = node.get_q()
            n_child.append(n)
            qs.append(q)

        n_child = np.array(n_child)
        qs = np.array(qs)
        exploration_bonus = self.alpha * np.sqrt(np.log(np.sum(n_child)) / n_child)
        return np.argmax(qs + exploration_bonus)


class QNode:

    def __init__(self, ns, discount_factor, **extra_args):
        self.discount_factor = discount_factor

        self.values = {}  # np.zeros(ns)
        self.visits = {}  # np.ones(ns)
        self.v_nodes = {}  # [None] * ns

        self.immediate_rewards = 0.

    def update(self, reward, next_state):
        self.immediate_rewards += reward
        self.values[next_state], self.visits[next_state] = self.v_nodes[next_state].compute_value()

    def get_q(self):
        total_disc_rew = 0.
        total_visits = 0.
        for k, v in self.visits.items():
            total_disc_rew += self.values[k] * v
            total_visits += v

        return (self.immediate_rewards + self.discount_factor * total_disc_rew) / total_visits, total_visits

    def get_children(self, s):
        if s in self.v_nodes:
            return self.v_nodes[s]
        else:
            return None

    def set_children(self, s, vnode):
        self.v_nodes[s] = vnode


class SearchTree:

    def __init__(self, simulator, initial_state, no, na, discount_factor, max_depth=100, p=1, vnode=VNode, qnode=QNode,
                 **extra_args):
        if not callable(p) and p < 1:
            raise RuntimeError("Invalid greedieness: " + str(p))

        self.p = p
        self.no = no
        self.na = na
        self.extra_args = extra_args
        self.discount_factor = discount_factor
        self.max_depth = max_depth
        self.qnode_const = qnode
        self.vnode_const = vnode
        self.simulator = simulator
        self.root = TerminalVNode(initial_state, 1, (0., None))

    def decrease_count(self, cur_node):
        if isinstance(cur_node, TerminalVNode):
            # Here we are at the leaves and only need to update the depth counter
            cur_node.depth -= 1
        elif issubclass(type(cur_node), VNode):
            # This is similar as the previous case, although we need to recurse in the children as well
            cur_node.depth -= 1
            for a in range(0, self.na):
                self.decrease_count(cur_node.get_children(a))
        elif issubclass(type(cur_node), QNode):
            # Here we do not need to update a depth counter, ...
            for o, child in cur_node.v_nodes.items():
                self.decrease_count(child)
                # however, we need to update the values after updating all the depth counts in the subtree (the reward
                # of 0. ensures that we do not alter the immediate reward)
                cur_node.update(0., o)
        else:
            raise RuntimeError("Invalid node type: " + str(type(cur_node)))

    def progress_tree(self, action, next_state):
        self.root = self.root.get_children(action).get_children(next_state)
        # If the subtree is explored (may not be the case for few samples and high stochasticity)
        if self.root is not None:
            # Update the depth of all the nodes and recompute their values
            self.decrease_count(self.root)
        else:
            # Else, just create a new root node in the new state
            self.root = TerminalVNode(next_state, 1, (0., None))

    def expand_node(self, vnode):
        if not isinstance(vnode, TerminalVNode):
            raise RuntimeError("VNode already expanded!")

        if not vnode.expandable:
            raise RuntimeError("VNode cannot be expanded anymore")

        state = vnode.state
        q_nodes = []
        for i in range(0, self.na):
            q_node = self.qnode_const(self.no, self.discount_factor, **self.extra_args)
            if vnode.trajectory is not None and vnode.trajectory[0] == i:
                reward, next_state = vnode.trajectory[1:3]
                done = len(vnode.trajectory) == 3
                rem_rew = 0.
                disc = 1.
                for i in range(1, len(vnode.trajectory), 3):
                    rem_rew += disc * vnode.trajectory[i]
                    disc *= self.discount_factor
                dummy = TerminalVNode(next_state, vnode.depth + 1, rem_rew if done else (rem_rew, vnode.trajectory[3:]),
                                      expandable=not done)
            else:
                next_state, reward, done = self.simulator.simulate(state, i)
                dummy = TerminalVNode(next_state, vnode.depth + 1, 0. if done else self.simulator.rollout(next_state),
                                      expandable=not done)
            q_node.set_children(next_state, dummy)
            q_node.update(reward, next_state)
            q_nodes.append(q_node)
        new_node = self.vnode_const(state, vnode.depth, q_nodes, self.p, **self.extra_args)

        return new_node

    def search(self, n_runs=1000, visualize=False):
        if isinstance(self.root, TerminalVNode):
            self.root = self.expand_node(self.root)

        for i in range(0, n_runs):
            if visualize:
                self.visualize()

            current_depth = 1
            q_nodes = []
            rewards = []
            next_states = []
            current_node = self.root
            done = False
            # Progress through the search tree
            while not isinstance(current_node, TerminalVNode) and not done:
                a = current_node.select_action()
                ns, r, done = self.simulator.simulate(current_node.state, a)

                # Store the transition data for later update of the nodes
                q_nodes.append(current_node.get_children(a))
                rewards.append(r)
                next_states.append(ns)

                # Update the depth count and progress in the search tree
                current_node = q_nodes[-1].get_children(ns)
                # We can just replace the None node as it will be replaced in the next step
                if current_node is None:
                    current_node = TerminalVNode(ns, current_depth, 0. if done else (0., None), expandable=not done)
                    # If we reached the bottom of the tree, we need to add the current node to the q_node, as it will
                    # not be replaced
                    if done:
                        q_nodes[-1].set_children(ns, current_node)
                current_depth += 1

            # We extend the tree if we can
            if done:
                if not isinstance(current_node, TerminalVNode):
                    raise RuntimeError("Invalid tree state - faced 'done' signal at NonTerminalNode")
                else:
                    current_node.update(0.)
            else:
                if current_depth <= self.max_depth:
                    new_node = self.expand_node(current_node)
                    q_nodes[-1].set_children(ns, new_node)
                else:
                    current_node.update(self.simulator.rollout(current_node.state)[0])

            # Now we back up the values
            for q, r, ns in zip(reversed(q_nodes), reversed(rewards), reversed(next_states)):
                q.update(r, ns)

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
            for o, child in current.v_nodes.items():
                # The child will for sure have the current id
                edges.append((this_id, cur_id))
                if child is not None:
                    cur_id = self._parse_tree(child, node_map, cur_id, edges)
        else:
            raise RuntimeError("Invalid node type: " + str(type(current)))

        return cur_id

    def _create_label(self, node):
        if issubclass(type(node), VNode):
            v, n = node.compute_value()
            vm = node.compute_mean()
            return "%d: %.2f (%.2f), %.2f" % (n, v, node.p, vm) + "\n"
        elif issubclass(type(node), QNode):
            q, n = node.get_q()
            return "%d: %.2f" % (n, q)
        else:
            v, n = node.compute_value()
            return "%d: %.2f" % (v, n)

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
        labels = [self._create_label(node_map[i]) if i < second_root else tree._create_label(node_map[i]) for i in
                  g.vs.indices]
        lay = g.layout_reingold_tilford(root=[0, second_root])

        coords = np.array(lay.coords)
        x_lim = [np.min(coords[:, 0]), np.max(coords[:, 0])]
        y_lim = [np.min(coords[:, 1]), np.max(coords[:, 1])]

        pl = igraph.Plot(bbox=(200 + (x_lim[1] - x_lim[0]) * 60, 200 + (y_lim[1] - y_lim[0]) * 60), background="white")
        pl.add(g, layout=lay, margin=40, vertex_size=40, vertex_color=colors, vertex_label=labels,
               vertex_label_dist=1.2, vertex_label_size=10)
        pl.show()
