#include <envs/copy.h>

using namespace std;

namespace poweruct {

    Copy::Copy(std::vector<char> chars, string input) : chars(move(chars)), target(move(input)),
                                                        state(),
                                                        last_action(0), no(this->chars.size() + 1),
                                                        na(2 * 2 * this->chars.size()), initial_state(0, "", 0),
                                                        uniform_action_probs(na, 1. / ((double) na)), time_limit(100),
                                                        initialized(false) {
        for (uint32_t i = 0; i < target.size(); i++) {
            char cur = target[i];
            bool match = false;
            for (uint32_t j = 0; j < this->chars.size(); j++) {
                if (cur == this->chars[j]) {
                    obs_vec.push_back(j);
                    match = true;
                    break;
                }
            }

            if (!match) {
                throw runtime_error("Could not build observation vector! Target string has characters not in the dict");
            }
        }
    }

    Copy::~Copy() = default;

    uint32_t Copy::getNumberOfObservations() {
        return no;
    }

    uint32_t Copy::getNumberOfActions() {
        return na;
    }

    void Copy::seed(uint32_t s) {
        r_util.seed(s);
    }

    void Copy::reset() {
        last_action = 0;
        state = initial_state;
        initialized = true;
    }

    void Copy::set_state(poweruct::CopyState state) {
        this->state = move(state);
        initialized = true;
    }

    std::tuple<CopyState, uint32_t, double, bool> Copy::step(uint32_t action) {
        if (!initialized) {
            throw runtime_error(
                    "Environment has not been initialized! Call reset() or set_state() at least once before calling step()");
        }

        uint32_t ctrl_action = action % 4;
        uint32_t cchoice = action / 4;

        //Advance the time
        state.time += 1;

        // 00 = move right, 01 = move right + write, 10 = move left, 11 = move left + write
        if (ctrl_action < 2) {
            state.pos -= 1;
        } else {
            state.pos += 1;
        }

        bool write = ctrl_action == 1 || ctrl_action == 3;
        if (write) {
            state.output += chars[cchoice];
        }

        uint32_t obs;
        if (state.pos >= obs_vec.size() || state.pos < 0) {
            obs = obs_vec.size() + 1;
        } else {
            obs = obs_vec[state.pos];
        }

        bool done;
        double reward;
        if (state.time > time_limit) {
            // If we have run out of time, we get a reward of -1, regardless of whether the correct string might have been
            // created
            done = true;
            reward = 0.;
        } else {
            // If not, we compute the actual reward

            //First we do a sanity check to ensure that the output is not longer than the target
            if (state.output.size() > target.size()) {
                throw runtime_error("Output string is larger than target string! Something is wrong!");
            }

            auto res = mismatch(state.output.begin(), state.output.end(), target.begin());
            if (res.first == state.output.end()) {
                // First case is if state.output is a prefix of target
                done = state.output.size() == target.size();
                // We only get a reward if we added a new correct character
                reward = write ? 1. : 0.;
            } else {
                // We can only get into this case if we added a wrong character to a previously correct prefix, so we
                // do not need to check the write action here. We only do it as a sanity check
                if (!write) {
                    throw runtime_error("Missing write action in an illegal state! Something is wrong!");
                }

                done = true;
                reward = 0.;
            }
        }

        return make_tuple(state, obs, reward, done);
    }

    std::tuple<uint32_t, CopyState, uint32_t, double, bool> Copy::random_step() {
        uint32_t action = r_util.sample_discrete(uniform_action_probs);
        auto res = step(action);

        return make_tuple(action, get<0>(res), get<1>(res), get<2>(res), get<3>(res));
    }

    std::tuple<CopyState, uint32_t, double, bool> Copy::simulate(poweruct::CopyState state, uint32_t action) {
        set_state(state);
        return step(action);
    }

    std::tuple<uint32_t, CopyState, uint32_t, double, bool> Copy::random_simulate(poweruct::CopyState state) {
        set_state(state);
        return random_step();
    }

    std::unique_ptr<std::vector<std::tuple<uint32_t, double, CopyState, uint32_t >>>
    Copy::rollout(poweruct::CopyState state) {
        set_state(state);

        auto res = make_unique<std::vector<std::tuple<uint32_t, double, CopyState, uint32_t >>>();
        bool done = false;
        while (!done) {
            uint32_t action = r_util.sample_discrete(uniform_action_probs);
            CopyState next_state(0, "", 0);
            uint32_t observation;
            double reward;
            tie(next_state, observation, reward, done) = step(action);
            res->emplace_back(make_tuple(action, reward, next_state, observation));
        }

        return res;
    }

    void Copy::render() {}

    CopyState Copy::getInitialState() {
        return initial_state;
    }

    uint32_t Copy::getInitialObservation() {
        return obs_vec[initial_state.pos];
    }

}
