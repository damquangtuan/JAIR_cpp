#include <envs/frozen_lake_ext.h>

using namespace std;

namespace poweruct {

    static RandomUtils r_util;

    static const uint32_t LEFT = 0;
    static const uint32_t DOWN = 1;
    static const uint32_t RIGHT = 2;
    static const uint32_t UP = 3;

    static const uint32_t DOWN_LEFT = 4;
    static const uint32_t DOWN_RIGHT = 5;
    static const uint32_t UP_RIGHT = 6;
    static const uint32_t UP_LEFT = 7;

    static const char *action_names[] = {"LEFT", "DOWN", "RIGHT", "UP", "DOWN_LEFT", "DOWN_RIGHT", "UP_RIGHT", "UP_LEFT"};

    static const char *map = "SFFFFFFF"
                             "FFFFFFFF"
                             "FFFHFFFF"
                             "FFFFFHFF"
                             "FFFHFFFF"
                             "FHHFFFHF"
                             "FHFFHFHF"
                             "FFFHFFFG";

    uint32_t inc_ext(uint32_t state, uint32_t action) {
        uint32_t x = state % 8;
        uint32_t y = state / 8;

        if (action == LEFT) {
            if (x > 0) {
                x -= 1;
            }
        } else if (action == DOWN) {
            if (y < 7) {
                y += 1;
            }
        } else if (action == RIGHT) {
            if (x < 7) {
                x += 1;
            }
        } else if (action == UP) {
            if (y > 0) {
                y -= 1;
            }
        } else if (action == DOWN_LEFT) {
            if (y < 7) {
                y += 1;
            }
            if (x > 0) {
                x -= 1;
            }
        } else if (action == DOWN_RIGHT) {
            if (y < 7) {
                y += 1;
            }
            if (x < 7) {
                x += 1;
            }
        } else if (action == UP_RIGHT) {
            if (y > 0) {
                y -= 1;
            }
            if (x < 7) {
                x += 1;
            }
        } else if (action == UP_LEFT) {
            if (y > 0) {
                y -= 1;
            }
            if (x > 0) {
                x -= 1;
            }
        } else {
            throw runtime_error("No valid direction given!");
        }

        return 8 * y + x;
    }

    vector<Transition> generateTransitionsExt(bool slippery) {
        vector<Transition> transitions;
        for (uint32_t s = 0; s < 64; s++) {
            for (uint32_t a = 0; a < 8; a++) {
                vector<double> rewards;
                vector<bool> dones;
                vector<uint32_t> next_states;
                vector<double> probabilities;

                char letter = map[s];
                if (letter == 'G' || letter == 'H') {
                    // The reward here is not important as the episode ends anyways after reaching G or H
                    rewards.emplace_back(0.);
                    dones.push_back(true);
                    next_states.emplace_back(0);
                    probabilities.emplace_back(1.);
                } else {
                    vector<uint32_t> bs;
                    if (slippery) {
                        if (a < 4) {
                            bs.emplace_back((a - 1) % 4);
                            bs.emplace_back(a);
                            bs.emplace_back((a + 1) % 4);
                        } else {
                            uint32_t a_real = a - 4;
                            bs.emplace_back(((a_real - 1) % 4) + 4);
                            bs.emplace_back(a_real + 4);
                            bs.emplace_back(((a_real + 1) % 4) + 4);
                        }
                    } else {
                            bs.emplace_back(a);
                    }

                    for (uint32_t b : bs) {
                        uint32_t new_state = inc_ext(s, b);
                        char new_letter = map[new_state];
                        bool is_goal = new_letter == 'G';
                        bool is_crash = new_letter == 'H';
                        rewards.emplace_back(is_goal ? 1. : 0.);
                        dones.push_back(is_goal || is_crash);
                        next_states.emplace_back(new_state);
                        probabilities.emplace_back(1. / ((double) bs.size()));
                    }
                }

                transitions.emplace_back(Transition(rewards, dones, next_states, probabilities, r_util));
            }
        }

        return transitions;
    }

    vector<double> generateInitialStateDistributionExt() {
        vector<double> res(64, 0.);
        res[0] = 1.;
        return res;
    }

    FrozenLakeExt::FrozenLakeExt(bool slippery, bool use_rollout_heuristic) :
            DiscreteEnvironment(64, 8, 1., generateTransitionsExt(slippery),
                                generateInitialStateDistributionExt(), 200),
            use_rollout_heuristic(use_rollout_heuristic) {}

    FrozenLakeExt::~FrozenLakeExt() = default;

    void FrozenLakeExt::render() {
        // Print the last take action, if we have taken one already
        if (time_step > 0) {
            cout << "Action: " << action_names[last_action] << endl;
        }

        // Render the map, highlighting the position of the player
        for (uint32_t i = 0; i < 8; i++) {
            for (uint32_t j = 0; j < 8; j++) {
                uint32_t s = 8 * i + j;
                cout << (s == state ? ' ' : map[s]);
            }
            cout << endl;
        }

        cout << endl;
    }

    DiscreteEnvironmentState FrozenLakeExt::getInitialState() {
        return {0, 0};
    }

    uint32_t FrozenLakeExt::getInitialObservation() {
        return 0;
    }

    RandomUtils &FrozenLakeExt::get_r_util() {
        return r_util;
    }

    std::vector<double>& FrozenLakeExt::get_rollout_action_probs(poweruct::DiscreteEnvironmentState &state) {
        //Avoid reallocation
        static vector<double> action_probabilities;
        static vector<uint32_t> hole_present;
        if(use_rollout_heuristic){
            hole_present.clear();
            action_probabilities.clear();

            uint32_t num_holes = 0;
            for(uint32_t i = 0; i < 8; i++){
                uint32_t hole = map[inc_ext(state.state, i)] == 'H';
                hole_present.emplace_back(hole);
                num_holes += hole;
            }

            double total_prob = 0.;
            for(uint32_t i = 0; i < 8; i++){
                uint32_t adjacent_holes = 0;
                if(i < 4) {
                    adjacent_holes = hole_present[i] + hole_present[(i - 1) % 4] + hole_present[(i + 1) % 4];
                } else {
                    adjacent_holes = hole_present[i] + hole_present[((i - 4 - 1) % 4) + 4] +
                            hole_present[((i - 4 + 1) % 4) + 4];
                }
                bool include = false;
                if(num_holes == 1 && adjacent_holes == 0){
                    include = true;
                } else if(num_holes == 2 && adjacent_holes <= 1) {
                    include = true;
                } else if(num_holes == 3 && adjacent_holes <= 2) {
                    include = true;
                } else if(num_holes == 4) {
                    cout << "WARNING! Agent is surrounded by holes, check why this happened!";
                    include = true;
                }

                if(include) {
                    action_probabilities.emplace_back(1.);
                    total_prob += 1.;
                } else {
                    action_probabilities.emplace_back(0.);
                }
            }

            if(total_prob == 0.) {
                throw runtime_error("No viable actions found! This should never be the case!");
            }

            // Finally, we normalize the probabilities
            for(uint32_t i = 0; i < 4; i++){
                action_probabilities[i] /= total_prob;
            }

            return action_probabilities;
        } else {
            return DiscreteEnvironment::get_rollout_action_probs(state);
        }
    }

}