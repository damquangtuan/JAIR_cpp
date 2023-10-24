#ifndef C_COPY_H
#define C_COPY_H

#include <utility>
#include <cstdint>
#include <vector>
#include <tuple>
#include <random_utils.h>
#include <memory>
#include <util.h>
#include <iostream>
#include <environment.h>

namespace poweruct {

    struct CopyState {
        CopyState() {}

        CopyState(uint32_t pos, std::string output, uint32_t time) : pos(pos), output(std::move(output)), time(time) {}

        uint32_t pos;
        std::string output;
        uint32_t time;
    };

    class Copy : public Environment<CopyState> {

    public:
        Copy(std::vector<char> chars, std::string target);

        virtual ~Copy();

        uint32_t getNumberOfObservations() override;

        uint32_t getNumberOfActions() override;

        void seed(uint32_t s) override;

        void reset() override;

        void set_state(CopyState state) override;

        std::tuple<CopyState, uint32_t, double, bool> step(uint32_t action) override;

        std::tuple<uint32_t, CopyState, uint32_t, double, bool> random_step() override;

        std::tuple<CopyState, uint32_t, double, bool> simulate(CopyState state, uint32_t action) override;

        std::tuple<uint32_t, CopyState, uint32_t, double, bool> random_simulate(CopyState state) override;

        std::unique_ptr<std::vector<std::tuple<uint32_t, double, CopyState, uint32_t >>> rollout(CopyState state) override;

        void render() override;

        CopyState getInitialState() override;

        uint32_t getInitialObservation() override;

    private:
        std::vector<char> chars;
        std::string target;
        std::vector<uint32_t> obs_vec;
        CopyState state;
        uint32_t last_action;

        RandomUtils r_util;

        uint32_t no;
        uint32_t na;
        CopyState initial_state;
        std::vector<double> uniform_action_probs;
        uint32_t time_limit;
        bool initialized;

    };

}

#endif //C_COPY_H
