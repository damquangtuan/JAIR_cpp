#ifndef C___TREE_SEARCH_H
#define C___TREE_SEARCH_H

#include <tree_search_nodes.h>
#include <environment.h>
#include <memory>
#include <random_utils.h>
#include <cmath>
#include <util.h>
#include <algorithm>

namespace poweruct {

    template<typename S, typename I>
    class UCTNode : public VNode<S, I> {

    public:
        UCTNode(S state, uint32_t observation, double discount_factor, uint32_t na, bool final,
                std::shared_ptr<RandomUtils> random_utils, double alpha) :
                VNode<S, I>(state, observation, discount_factor, na, final, random_utils), alpha(alpha) {};

        uint32_t select_action() override {
            // Avoid continuous reallocation
            static std::vector<std::tuple<double, uint32_t >> q_values;
            static std::vector<double> likelihoods;
            static std::vector<uint32_t> action_candidates;

            // Empty the q_values list and initialize all the values to a very large one (this encourages exploration
            // of non-visited nodes)
            q_values.clear();
            for (uint32_t i = 0; i < VNode<S, I>::na; i++) {
                q_values.emplace_back(std::make_tuple(1e6, 0));
            }

            // Set the q_values and visits of those nodes that have already been visited
            uint32_t total_visits = 0;
            for (auto const &child : VNode<S, I>::q_stats) {
                q_values[child.first] = child.second;
                total_visits += std::get<1>(child.second);
            }

            double max_q = -1.;
            for (uint32_t i = 0; i < VNode<S, I>::na; i++) {
                double exploration_bonus = alpha * sqrt(log(total_visits) / std::get<1>(q_values[i]) + 1e-10);
                double q = std::get<0>(q_values[i]) + exploration_bonus;
                if (q >= max_q) {
                    if (q > max_q) {
                        action_candidates.clear();
                        max_q = q;
                    }
                    action_candidates.emplace_back(i);
                }
            }

            return VNode<S, I>::random_utils->sample_uniform(action_candidates);
        };

    protected:
        double alpha;

    };

    template<typename S>
    class MaxUCTNode : public UCTNode<S, MaxUCTNode<S>> {

    public:
        MaxUCTNode(S state, uint32_t observation, double discount_factor, uint32_t na, bool final,
                   std::shared_ptr<RandomUtils> random_utils, double alpha) :
                UCTNode<S, MaxUCTNode<S>>(state, observation, discount_factor, na, final, random_utils, alpha) {};

        virtual std::tuple<double, uint32_t> get_value() {
            uint32_t n_total = this->get_pre_exp_visits();

            double max_avg;
            if (this->q_stats.size() == 0) {
                max_avg = 0.;
            } else {
                max_avg = -1e6;
                for (auto &&kv : this->q_stats) {
                    double avg_rew = std::get<0>(kv.second);
                    n_total += std::get<1>(kv.second);
                    if (avg_rew > max_avg) {
                        max_avg = avg_rew;
                    }
                }
            }

            return std::make_tuple((this->get_pre_exp_reward() / n_total) + max_avg, n_total);
        }

    protected:
        std::shared_ptr<MaxUCTNode<S>> node_constructor(S s, uint32_t obs, bool done) {
            return std::make_shared<MaxUCTNode<S>>(s, obs, this->discount_factor, this->na, done, this->random_utils,
                                                   this->alpha);
        }

    };


    template<typename S>
    class PowerUCTNode : public UCTNode<S, PowerUCTNode<S>> {

    public:
        PowerUCTNode(S state, uint32_t observation, double discount_factor, uint32_t na, bool final,
                     std::shared_ptr<RandomUtils> random_utils, double alpha, double p) :
                UCTNode<S, PowerUCTNode<S>>(state, observation, discount_factor, na, final, random_utils, alpha),
                p(p) {};

        virtual std::tuple<double, uint32_t> get_value() {
            uint32_t n_total = this->get_pre_exp_visits();
            for (auto &&kv : this->q_stats) {
                n_total += std::get<1>(kv.second);
            }

            // Now compute the weighted average of the powers
            double total = 0.;
            for (auto &&kv : this->q_stats) {
                double q = std::get<0>(kv.second);
                double n = (double) std::get<1>(kv.second);
                total += (n / ((double) n_total)) * pow(q, p);
            }

            double value = (this->get_pre_exp_reward() / n_total) + pow(total, 1. / p);
            return std::make_tuple(value, n_total);
        }

    protected:
        std::shared_ptr<PowerUCTNode<S>> node_constructor(S s, uint32_t obs, bool done) {
            return std::make_shared<PowerUCTNode<S>>(s, obs, this->discount_factor, this->na, done,
                                                     this->random_utils, this->alpha, this->p);
        }

    private:
        double p;
    };

    template<typename S>
    class MaxEntropyNode : public VNode<S, MaxEntropyNode<S>> {
    public:
        MaxEntropyNode(S state, uint32_t observation, double discount_factor, uint32_t na, bool final,
                       std::shared_ptr<RandomUtils> random_utils, float tau, float epsilon) :
                VNode<S, MaxEntropyNode<S>>(state, observation, discount_factor, na, final, random_utils),
                tau(tau), epsilon(epsilon) {}

        uint32_t select_action() {
            // Avoid continuous reallocation
            static std::vector<double> q_values;
            static std::vector<double> likelihoods;

            // Empty the q_values list and initialize all the values to a very large one (this encourages exploration
            // of non-visited nodes)
            q_values.clear();
            for (uint32_t i = 0; i < this->na; i++) {
                q_values.push_back(1e6);
            }

            // Set the q_values of those nodes that have already been visited
            uint32_t total_visits = 0;
            for (auto const &child : this->q_stats) {
                q_values[child.first] = std::get<0>(child.second);
                total_visits += std::get<1>(child.second);
            }

            // Now compute the soft value function and the likelihoods
            double v_soft = compute_soft_value(q_values);
            compute_likelihoods(v_soft, q_values, likelihoods);

            // Toss a coin to decide between a random choice and the on-policy choice
            double lambda = (epsilon * ((double) this->na)) / log(((double) total_visits) + 1);
            if (this->random_utils->sample_uniform() > lambda) {
                return this->random_utils->sample_discrete(likelihoods);
            } else {
                return this->random_utils->sample_uniform(this->na);
            }
        }

        std::tuple<double, uint32_t> get_value() {
            // Avoid continuous reallocation
            static std::vector<double> q_values;
            q_values.clear();

            // Gather the individual q_values and compute the total visitation count
            uint32_t total_visits = this->get_pre_exp_visits();
            for (auto const &child : this->q_stats) {
                q_values.push_back(std::get<0>(child.second));
                total_visits += std::get<1>(child.second);
            }

            if (q_values.empty()) {
                return std::make_tuple(this->get_pre_exp_reward() / total_visits, total_visits);
            } else {
                double value = (this->get_pre_exp_reward() / total_visits) + compute_soft_value(q_values);
                return std::make_tuple(value, total_visits);
            }
        }

    protected:
        std::shared_ptr<MaxEntropyNode<S>> node_constructor(S s, uint32_t obs, bool done) {
            return std::make_shared<MaxEntropyNode<S>>(s, obs, this->discount_factor, this->na, done,
                                                       this->random_utils, this->tau, this->epsilon);
        };


    private:
        double tau;
        double epsilon;

        void compute_likelihoods(double v_soft, const std::vector<double> &q_values, std::vector<double> &likelihoods) {
            static std::vector<double> advs;
            advs.clear();

            for (auto q_value : q_values) {
                advs.push_back(q_value - v_soft);
            }
            double max_adv = *max_element(advs.begin(), advs.end());

            likelihoods.clear();
            double total_likelihood = 0.;
            for (auto adv : advs) {
                double unnormalized_likelihood = exp((adv - max_adv) / this->tau);
                likelihoods.push_back(unnormalized_likelihood);
                total_likelihood += unnormalized_likelihood;
            }

            for (uint32_t i = 0; i < this->na; i++) {
                likelihoods[i] /= total_likelihood;
            }
        }

        double compute_soft_value(const std::vector<double> &q_values) {
            double max_q_value = *max_element(q_values.begin(), q_values.end());
            double acc = 0.;
            for (double q_value : q_values) {
                acc += exp((q_value - max_q_value) / this->tau);
            }

            return max_q_value + this->tau * log(acc);
        }
    };



    template<typename S>
    class MaxEntropyUCBNode : public VNode<S, MaxEntropyUCBNode<S>> {
    public:
        MaxEntropyUCBNode(S state, uint32_t observation, double discount_factor, uint32_t na, bool final,
                          std::shared_ptr<RandomUtils> random_utils, float tau, float epsilon, float alpha) :
                VNode<S, MaxEntropyUCBNode<S>>(state, observation, discount_factor, na, final, random_utils),
                tau(tau), epsilon(epsilon), alpha(alpha) {}


        uint32_t select_action() {
            // Avoid continuous reallocation
            static std::vector<std::tuple<double, uint32_t >> q_values;
            static std::vector<double> likelihoods;
            static std::vector<uint32_t> action_candidates;

            // Empty the q_values list and initialize all the values to a very large one (this encourages exploration
            // of non-visited nodes)
            q_values.clear();
            for (uint32_t i = 0; i < this->na; i++) {
                q_values.emplace_back(std::make_tuple(1e6, 0));
            }

            // Set the q_values and visits of those nodes that have already been visited
            uint32_t total_visits = 0;
            for (auto const &child : this->q_stats) {
                q_values[child.first] = child.second;
                total_visits += std::get<1>(child.second);
            }

            double max_q = -1.;
            for (uint32_t i = 0; i < this->na; i++) {
                double exploration_bonus = alpha * sqrt(log(total_visits) / std::get<1>(q_values[i]) + 1e-10);
                double q = std::get<0>(q_values[i]) + exploration_bonus;
                if (q >= max_q) {
                    if (q > max_q) {
                        action_candidates.clear();
                        max_q = q;
                    }
                    action_candidates.emplace_back(i);
                }
            }

            return this->random_utils->sample_uniform(action_candidates);
        }


        std::tuple<double, uint32_t> get_value() {
            // Avoid continuous reallocation
            static std::vector<double> q_values;
            q_values.clear();

            // Gather the individual q_values and compute the total visitation count
            uint32_t total_visits = this->get_pre_exp_visits();
            for (auto const &child : this->q_stats) {
                q_values.push_back(std::get<0>(child.second));
                total_visits += std::get<1>(child.second);
            }

            if (q_values.empty()) {
                return std::make_tuple(this->get_pre_exp_reward() / total_visits, total_visits);
            } else {
                double value = (this->get_pre_exp_reward() / total_visits) + compute_soft_value(q_values);
                return std::make_tuple(value, total_visits);
            }
        }

    protected:
        std::shared_ptr<MaxEntropyUCBNode<S>> node_constructor(S s, uint32_t obs, bool done) {
            return std::make_shared<MaxEntropyUCBNode<S>>(s, obs, this->discount_factor, this->na, done,
                                                          this->random_utils, this->tau, this->epsilon, this->alpha);
        };


    private:
        double tau;
        double epsilon;
        double alpha;

        void compute_likelihoods(double v_soft, const std::vector<double> &q_values, std::vector<double> &likelihoods) {
            static std::vector<double> advs;
            advs.clear();

            for (auto q_value : q_values) {
                advs.push_back(q_value - v_soft);
            }
            double max_adv = *max_element(advs.begin(), advs.end());

            likelihoods.clear();
            double total_likelihood = 0.;
            for (auto adv : advs) {
                double unnormalized_likelihood = exp((adv - max_adv) / this->tau);
                likelihoods.push_back(unnormalized_likelihood);
                total_likelihood += unnormalized_likelihood;
            }

            for (uint32_t i = 0; i < this->na; i++) {
                likelihoods[i] /= total_likelihood;
            }
        }

        double compute_soft_value(const std::vector<double> &q_values) {
            double max_q_value = *max_element(q_values.begin(), q_values.end());
            double acc = 0.;
            for (double q_value : q_values) {
                acc += exp((q_value - max_q_value) / this->tau);
            }

            return max_q_value + this->tau * log(acc);
        }
    };


    template<typename S>
    class REPSNode : public VNode<S, REPSNode<S>> {
    public:
        REPSNode(S state, uint32_t observation, double discount_factor, uint32_t na, bool final,
                 std::shared_ptr<RandomUtils> random_utils, float tau, float epsilon) :
                VNode<S, REPSNode<S>>(state, observation, discount_factor, na, final, random_utils),
                tau(tau), epsilon(epsilon) {}

        uint32_t select_action() {
            // Avoid continuous reallocation
            static std::vector<double> q_values;
            static std::vector<double> visit_frequencies;
            static std::vector<double> likelihoods;

            // Empty the q_values list and initialize all the values to a very large one (this encourages exploration
            // of non-visited nodes)
            q_values.clear();
            visit_frequencies.clear();
            likelihoods.clear();
            for (uint32_t i = 0; i < this->na; i++) {
                q_values.push_back(1e6);
                visit_frequencies.push_back(1);
            }

            // Set the q_values of those nodes that have already been visited
            for (auto const &child : this->q_stats) {
                q_values[child.first] = std::get<0>(child.second);
                visit_frequencies[child.first] = ((double) std::get<1>(child.second));
            }

            double total_visits = 0.;
            for (uint32_t i = 0; i < this->na; i++) {
                total_visits += visit_frequencies[i];
            }

//            double total_visit_frequencies = 0;
//
//            for (uint32_t i = 0; i < visit_frequencies.size(); i++) {
//                total_visit_frequencies += visit_frequencies[i];
//            }
//
//            for (uint32_t i = 0; i < visit_frequencies.size(); i++) {
//                visit_frequencies[i] /= total_visit_frequencies;
//            }

            // Now compute the soft value function and the likelihoods
            double v_soft = compute_soft_value(q_values, visit_frequencies);
            compute_likelihoods(v_soft, q_values, visit_frequencies, likelihoods);

            // Toss a coin to decide between a random choice and the on-policy choice
            double lambda = (epsilon * ((double) this->na)) / log(total_visits + 1.);
            if (this->random_utils->sample_uniform() > lambda) {
                return this->random_utils->sample_discrete(likelihoods);
            } else {
                return this->random_utils->sample_uniform(this->na);
            }
        }

        std::tuple<double, uint32_t> get_value() {
            // Avoid continuous reallocation
            static std::vector<double> q_values;
            static std::vector<double> visit_frequencies;
            q_values.clear();
            visit_frequencies.clear();

            // Gather the individual q_values and compute the total visitation count
            uint32_t total_visits = this->get_pre_exp_visits();
            for (auto const &child : this->q_stats) {
                q_values.push_back(std::get<0>(child.second));
//                visit_frequencies.push_back(1); //std::get<1>(child.second));
                visit_frequencies.push_back(std::get<1>(child.second));
                total_visits += std::get<1>(child.second);
            }

//            for (uint32_t i = 0; i < visit_frequencies.size(); i++) {
//                visit_frequencies[i] /= total_visits;
//            }

            if (q_values.empty()) {
                return std::make_tuple(this->get_pre_exp_reward() / total_visits, total_visits);
            } else {
                double value =
                        (this->get_pre_exp_reward() / total_visits) + compute_soft_value(q_values, visit_frequencies);
                return std::make_tuple(value, total_visits);
            }
        }

    protected:
        std::shared_ptr<REPSNode<S>> node_constructor(S s, uint32_t obs, bool done) {
            return std::make_shared<REPSNode<S>>(s, obs, this->discount_factor, this->na, done,
                                                 this->random_utils, this->tau, this->epsilon);
        };


    private:
        double tau;
        double epsilon;

        void compute_likelihoods(double v_soft, const std::vector<double> &q_values,
                                 const std::vector<double> &visit_frequencies, std::vector<double> &likelihoods) {
            uint32_t n_a = q_values.size();
            static std::vector<double> advs;
            advs.clear();

            for (auto q_value : q_values) {
                advs.push_back(q_value - v_soft);
            }
            double max_adv = *max_element(advs.begin(), advs.end());

            likelihoods.clear();
            double total_likelihood = 0.;

            double total = 0;

            for (uint32_t i = 0; i < n_a; i++) {
                total += visit_frequencies[i];
            }

            for (uint32_t i = 0; i < n_a; i++) {
                double unnormalized_likelihood = double (visit_frequencies[i]/total) * exp((advs[i] - max_adv) / this->tau);
//                double unnormalized_likelihood = exp((advs[i] - max_adv) / this->tau);
                likelihoods.push_back(unnormalized_likelihood);
                total_likelihood += unnormalized_likelihood;
            }

            for (uint32_t i = 0; i < this->na; i++) {
                likelihoods[i] /= total_likelihood;
            }
        }

        double compute_soft_value(const std::vector<double> &q_values, const ::std::vector<double> &visit_frequencies) {
            uint32_t n_a = q_values.size();
            double max_q_value = *max_element(q_values.begin(), q_values.end());
            double acc = 0.;

            double total = 0;

            for (uint32_t i = 0; i < n_a; i++) {
                total += visit_frequencies[i];
            }

            for (uint32_t i = 0; i < n_a; i++) {
                acc += double (visit_frequencies[i]/total) * exp((q_values[i] - max_q_value) / this->tau);
//                acc += exp((q_values[i] - max_q_value) / this->tau);
            }


            return max_q_value + this->tau * log(acc);
        }
    };

    template<typename S>
    class REPSUCBNode : public VNode<S, REPSUCBNode<S>> {
    public:
        REPSUCBNode(S state, uint32_t observation, double discount_factor, uint32_t na, bool final,
                    std::shared_ptr<RandomUtils> random_utils, float tau, float epsilon, float alpha) :
                VNode<S, REPSUCBNode<S>>(state, observation, discount_factor, na, final, random_utils),
                tau(tau), epsilon(epsilon), alpha(alpha) {}

        uint32_t select_action() {
            // Avoid continuous reallocation
            static std::vector<std::tuple<double, uint32_t >> q_values;
            static std::vector<double> likelihoods;
            static std::vector<uint32_t> action_candidates;

            // Empty the q_values list and initialize all the values to a very large one (this encourages exploration
            // of non-visited nodes)
            q_values.clear();
            for (uint32_t i = 0; i < this->na; i++) {
                q_values.emplace_back(std::make_tuple(1e6, 0));
            }

            // Set the q_values and visits of those nodes that have already been visited
            uint32_t total_visits = 0;
            for (auto const &child : this->q_stats) {
                q_values[child.first] = child.second;
                total_visits += std::get<1>(child.second);
            }

            double max_q = -1.;
            for (uint32_t i = 0; i < this->na; i++) {
                double exploration_bonus = alpha * sqrt(log(total_visits) / std::get<1>(q_values[i]) + 1e-10);
                double q = std::get<0>(q_values[i]) + exploration_bonus;
                if (q >= max_q) {
                    if (q > max_q) {
                        action_candidates.clear();
                        max_q = q;
                    }
                    action_candidates.emplace_back(i);
                }
            }

            return this->random_utils->sample_uniform(action_candidates);
        }

        std::tuple<double, uint32_t> get_value() {
            // Avoid continuous reallocation
            static std::vector<double> q_values;
            static std::vector<double> visit_frequencies;
            q_values.clear();
            visit_frequencies.clear();

            // Gather the individual q_values and compute the total visitation count
            uint32_t total_visits = this->get_pre_exp_visits();
            for (auto const &child : this->q_stats) {
                q_values.push_back(std::get<0>(child.second));
//                visit_frequencies.push_back(1); //std::get<1>(child.second));
                visit_frequencies.push_back(std::get<1>(child.second));
                total_visits += std::get<1>(child.second);
            }

//            for (uint32_t i = 0; i < visit_frequencies.size(); i++) {
//                visit_frequencies[i] /= total_visits;
//            }

            if (q_values.empty()) {
                return std::make_tuple(this->get_pre_exp_reward() / total_visits, total_visits);
            } else {
                double value =
                        (this->get_pre_exp_reward() / total_visits) + compute_soft_value(q_values, visit_frequencies);
                return std::make_tuple(value, total_visits);
            }
        }

    protected:
        std::shared_ptr<REPSUCBNode<S>> node_constructor(S s, uint32_t obs, bool done) {
            return std::make_shared<REPSUCBNode<S>>(s, obs, this->discount_factor, this->na, done,
                                                    this->random_utils, this->tau, this->epsilon, this->alpha);
        };


    private:
        double tau;
        double epsilon;
        double alpha;

        void compute_likelihoods(double v_soft, const std::vector<double> &q_values,
                                 const std::vector<double> &visit_frequencies, std::vector<double> &likelihoods) {
            uint32_t n_a = q_values.size();
            static std::vector<double> advs;
            advs.clear();

            for (auto q_value : q_values) {
                advs.push_back(q_value - v_soft);
            }
            double max_adv = *max_element(advs.begin(), advs.end());

            likelihoods.clear();
            double total_likelihood = 0.;

            double total = 0;

            for (uint32_t i = 0; i < n_a; i++) {
                total += visit_frequencies[i];
            }

            for (uint32_t i = 0; i < n_a; i++) {
                double unnormalized_likelihood = double (visit_frequencies[i]/total) * exp((advs[i] - max_adv) / this->tau);
//                double unnormalized_likelihood = exp((advs[i] - max_adv) / this->tau);
                likelihoods.push_back(unnormalized_likelihood);
                total_likelihood += unnormalized_likelihood;
            }

            for (uint32_t i = 0; i < this->na; i++) {
                likelihoods[i] /= total_likelihood;
            }
        }

        double compute_soft_value(const std::vector<double> &q_values, const ::std::vector<double> &visit_frequencies) {
            uint32_t n_a = q_values.size();
            double max_q_value = *max_element(q_values.begin(), q_values.end());
            double acc = 0.;

            double total = 0;

            for (uint32_t i = 0; i < n_a; i++) {
                total += visit_frequencies[i];
            }

            for (uint32_t i = 0; i < n_a; i++) {
                acc += double (visit_frequencies[i]/total) * exp((q_values[i] - max_q_value) / this->tau);
//                acc += exp((q_values[i] - max_q_value) / this->tau);
            }

            return max_q_value + this->tau * log(acc);
        }
    };

    template<typename S>
    class TSALLISNode : public VNode<S, TSALLISNode<S>> {
    public:
        TSALLISNode(S state, uint32_t observation, double discount_factor, uint32_t na, bool final,
                    std::shared_ptr<RandomUtils> random_utils, float tau, float epsilon) :
                VNode<S, TSALLISNode<S>>(state, observation, discount_factor, na, final, random_utils),
                tau(tau), epsilon(epsilon) {}

        uint32_t select_action() {
            // Avoid continuous reallocation
            static std::vector<double> q_values;
            static std::vector<double> qs (this->na);
            static std::vector<double> visit_frequencies;
            static std::vector<double> likelihoods;
            static std::vector<int> K;
            static std::vector<double> Q_cumsum (this->na);
            static std::vector<int> Q_check (this->na);
            static std::vector<double> Q_sp_max (this->na);

            q_values.clear();
            qs.clear();
            visit_frequencies.clear();
            likelihoods.clear();
            K.clear();
            Q_cumsum.clear();
            Q_check.clear();
            Q_sp_max.clear();

            for (int i = 1; i <= this->na; i++) {
                K.emplace_back(i);
            }

            // Empty the q_values list and initialize all the values to a very large one (this encourages exploration
            // of non-visited nodes)
            for (uint32_t i = 0; i < this->na; i++) {
                q_values.emplace_back(1e6/this->tau);
                visit_frequencies.emplace_back(1);
            }

            // Set the q_values of those nodes that have already been visited
            for (auto const &child : this->q_stats) {
                q_values[child.first] = std::get<0>(child.second)/this->tau;
                visit_frequencies[child.first] = ((double) std::get<1>(child.second));
            }

            qs = q_values;
            sort(q_values.begin(), q_values.end(), std::greater<double>());

            for (int i = 0; i < this->na; i++) {
//                std::cout << "q_values: " << i << " " << q_values[i] << std::endl;
            }

            double sum_of_elems = std::accumulate(q_values.begin(), q_values.end(), 0);

//            std::partial_sum(q_values.begin(), q_values.end(), Q_cumsum.begin());

            double cumsum = 0;
            for (int i = 0; i < this->na; i++) {
                cumsum += q_values[i];
                Q_cumsum[i] = cumsum;
            }

            for (int i = 0; i < this->na; i++) {
//                std::cout << "Q_cumsum: " << i << " " << Q_cumsum[i] << std::endl;
            }

            for (int i = 0; i < this->na; i++) {
                if (1 + (K[i] * q_values[i]) > Q_cumsum[i]) Q_check.emplace_back(1);
                else Q_check.emplace_back(0);
            }

            double K_sum = std::accumulate(Q_check.begin(), Q_check.end(), 0);
            for (int i = 0; i < this->na; i++) {
                Q_sp_max.emplace_back(Q_check[i] * q_values[i]);
            }

            double sp_max = (std::accumulate(Q_sp_max.begin(), Q_sp_max.end(), 0) - 1)/K_sum;

            for (uint32_t i = 0; i < this->na; i++) {
                double pi_i = std::max(double(qs[i] - sp_max), double(0));
                likelihoods.emplace_back(pi_i);
            }

            double total_visits = 0.;
            for (uint32_t i = 0; i < this->na; i++) {
                total_visits += visit_frequencies[i];
            }

            // Toss a coin to decide between a random choice and the on-policy choice
            double lambda = (epsilon * ((double) this->na)) / log(total_visits + 1.);
            if (this->random_utils->sample_uniform() > lambda) {
                return this->random_utils->sample_discrete(likelihoods);
            } else {
                return this->random_utils->sample_uniform(this->na);
            }
        }

        std::tuple<double, uint32_t> get_value() {
            // Avoid continuous reallocation
            static std::vector<double> q_values (this->na);
            static std::vector<double> qs (this->na);
            static std::vector<double> visit_frequencies;
            static std::vector<int> K;
            static std::vector<double> Q_cumsum (this->na);
            static std::vector<int> Q_check (this->na);
            static std::vector<double> Q_sp_max (this->na);

            q_values.clear();
            qs.clear();
            visit_frequencies.clear();
            K.clear();
            Q_cumsum.clear();
            Q_check.clear();
            Q_sp_max.clear();


            // Gather the individual q_values and compute the total visitation count
            uint32_t total_visits = this->get_pre_exp_visits();
            for (auto const &child : this->q_stats) {
                q_values.emplace_back(std::get<0>(child.second)/this->tau);
//                visit_frequencies.push_back(1); //std::get<1>(child.second));
                visit_frequencies.emplace_back(std::get<1>(child.second));
                total_visits += std::get<1>(child.second);
            }

            if (q_values.empty()) {
                return std::make_tuple(this->get_pre_exp_reward() / total_visits, total_visits);
            } else {
                for (int i = 1; i <= this->na; i++) {
                    K.emplace_back(i);
                }
                qs = q_values;
                sort(q_values.begin(), q_values.end(), std::greater<double>());
                double sum_of_elems = std::accumulate(q_values.begin(), q_values.end(), 0);

                double cumsum = 0;
                for (int i = 0; i < this->na; i++) {
                    cumsum += q_values[i];
                    Q_cumsum[i] = cumsum;
                }

//                std::partial_sum(q_values.begin(), q_values.end(), Q_cumsum.begin());

                for (int i = 0; i < this->na; i++) {
                    if (1 + (K[i] * q_values[i]) > Q_cumsum[i]) Q_check.emplace_back(1);
                    else Q_check.emplace_back(0);
                }

                double K_sum = std::accumulate(Q_check.begin(), Q_check.end(), 0);
                for (int i = 0; i < this->na; i++) {
                    Q_sp_max.emplace_back(Q_check[i] * q_values[i]);
                }

                double sp_max = (std::accumulate(Q_sp_max.begin(), Q_sp_max.end(), 0) - 1)/K_sum;

                double V = 0;
                for (int i = 0; i < this->na; i++) {
                    if (Q_sp_max[i] == 0) break;
                    V += (Q_sp_max[i] * Q_sp_max[i]) - (sp_max * sp_max);
                }

                V = this->tau * (0.5 * V + 0.5);
                return std::make_tuple(V, total_visits);
            }
        }

    protected:
        std::shared_ptr<TSALLISNode<S>> node_constructor(S s, uint32_t obs, bool done) {
            return std::make_shared<TSALLISNode<S>>(s, obs, this->discount_factor, this->na, done,
                                                    this->random_utils, this->tau, this->epsilon);
        };


    private:
        double tau;
        double epsilon;

        void compute_likelihoods(double v_soft, const std::vector<double> &q_values,
                                 const std::vector<double> &visit_frequencies, std::vector<double> &likelihoods) {
            uint32_t n_a = q_values.size();
            static std::vector<double> advs;
            advs.clear();

            for (auto q_value : q_values) {
                advs.push_back(q_value - v_soft);
            }
            double max_adv = *max_element(advs.begin(), advs.end());

            likelihoods.clear();
            double total_likelihood = 0.;

            double total = 0;

            for (uint32_t i = 0; i < n_a; i++) {
                total += visit_frequencies[i];
            }

            for (uint32_t i = 0; i < n_a; i++) {
                double unnormalized_likelihood = double (visit_frequencies[i]/total) * exp((advs[i] - max_adv) / this->tau);
//                double unnormalized_likelihood = exp((advs[i] - max_adv) / this->tau);
                likelihoods.push_back(unnormalized_likelihood);
                total_likelihood += unnormalized_likelihood;
            }

            for (uint32_t i = 0; i < this->na; i++) {
                likelihoods[i] /= total_likelihood;
            }
        }

        double compute_soft_value(const std::vector<double> &q_values, const ::std::vector<double> &visit_frequencies) {
            uint32_t n_a = q_values.size();
            double max_q_value = *max_element(q_values.begin(), q_values.end());
            double acc = 0.;

            double total = 0;

            for (uint32_t i = 0; i < n_a; i++) {
                total += visit_frequencies[i];
            }

            for (uint32_t i = 0; i < n_a; i++) {
                acc += double (visit_frequencies[i]/total) * exp((q_values[i] - max_q_value) / this->tau);
//                acc += exp((q_values[i] - max_q_value) / this->tau);
            }

            return max_q_value + this->tau * log(acc);
        }
    };

    template<typename S>
    class AbstractSearchTree {

    public:
        virtual void progress_tree(uint32_t action, S next_state, uint32_t next_obs) = 0;

        virtual uint32_t search(uint32_t n_runs) = 0;

        virtual void seed(uint32_t s) = 0;

    };

    template<typename S, typename I, typename ... Types>
    class SearchTree : public AbstractSearchTree<S> {

    public:
        SearchTree(std::shared_ptr<Environment<S>> env, S initial_state, uint32_t initial_obs, double discount_factor,
                   Types... extra_args) :
                env(env), na(env->getNumberOfActions()), discount_factor(discount_factor),
                random_utils(std::make_shared<RandomUtils>()),
                node_constructor([this, extra_args...](S state, uint32_t observation, bool final) {
                    return std::make_shared<I>(state, observation, this->discount_factor, this->na, final,
                                               this->random_utils, extra_args...);
                }) {
            root = node_constructor(initial_state, initial_obs, false);
        }

        ~SearchTree() = default;

        void progress_tree(uint32_t action, S next_state, uint32_t next_obs) {
            std::shared_ptr<VNode<S, I>> v_child;
            if (root->is_expanded()) {
                auto q_child = root->get_child(action);
                if (q_child) {
                    v_child = q_child->get_children(next_obs);
                }
            }

            if (!v_child) {
                root = node_constructor(next_state, next_obs, false);
            } else {
                root = v_child;
            }
        }

        uint32_t search(uint32_t n_runs) {
            for (uint32_t i = 0; i < n_runs; i++) {
                bool run = true;
                bool is_expand = false;
                std::shared_ptr<VNode<S, I>> cur_node = root;
                // Ensure that the current node points to a reasonable object
                while (run) {
                    if (cur_node->is_final()) {
                        cur_node->store_rollout(
                                make_unique<std::vector<std::tuple<uint32_t, double, S, uint32_t>>>());
                        run = false;
                    } else if (!cur_node->is_expanded()) {
                        cur_node->expand(cur_node);
                        is_expand = true;
                    } else {
                        // We select an action and do an according step in the environment
                        S next_state;
                        uint32_t next_obs, action;
                        double reward;
                        bool done;
                        if (is_expand) {
                            std::tie(action, next_state, next_obs, reward, done) = env->random_simulate(
                                    cur_node->get_state());
                        } else {
                            action = cur_node->select_action();
                            std::tie(next_state, next_obs, reward, done) = env->simulate(cur_node->get_state(), action);
                        }

                        // If we face an unexplored action in a node, we need to add a Q-Node
                        auto q_child = cur_node->get_child(action);
                        if (!q_child) {
                            q_child = std::make_shared<QNode<S, I>>(action, discount_factor);
                            cur_node->set_children(action, q_child);
                            q_child->set_parent(cur_node);
                        }

                        // Get the V-Node from the (maybe freshly) created Q-Node
                        auto v_child = q_child->get_children(next_obs);

                        // If that V-Node does not exist yet, do a rollout and mark that we already did a rollout
                        if (!v_child) {
                            std::unique_ptr<std::vector<std::tuple<uint32_t, double, S, uint32_t >>> remaining_reward;
                            if (done) {
                                remaining_reward = make_unique<std::vector<std::tuple<uint32_t, double, S, uint32_t>>>();
                            } else {
                                remaining_reward = env->rollout(next_state);
                            }
                            v_child = node_constructor(next_state, next_obs, done);
                            v_child->store_rollout(std::move(remaining_reward));

                            q_child->set_children(next_obs, v_child);
                            v_child->set_parent(q_child);
                            run = false;
                        }

                        // Finally store the reward obtained in the given transition
                        q_child->store_reward(reward);
                        cur_node = v_child;
                    }
                }

                // Finally update the Q- and V-values in the nodes from leaf to root
                while (cur_node->has_parent()) {
                    auto q_parent = cur_node->get_parent();
                    q_parent->update(cur_node->get_observation());
                    auto v_parent = q_parent->get_parent();
                    v_parent->update(q_parent->get_action());
                    cur_node = v_parent;
                }

            }

            return root->select_greedy_action();
        }

        void seed(uint32_t s) {
            random_utils->seed(s);
        }

    private:
        std::shared_ptr<Environment<S>> env;
        uint32_t na;
        double discount_factor;
        std::shared_ptr<RandomUtils> random_utils;
        std::shared_ptr<VNode<S, I>> root;
        std::function<std::shared_ptr<I>(S, uint32_t, bool)> node_constructor;

    };

    template<typename S>
    using MaxUCTSearchTree = SearchTree<S, MaxUCTNode<S>, double>;

    template<typename S>
    using PowerUCTSearchTree = SearchTree<S, PowerUCTNode<S>, double, double>;

    template<typename S>
    using MaxEntropySearchTree = SearchTree<S, MaxEntropyNode<S>, double, double>;

    template<typename S>
    using MaxEntropyUCBSearchTree = SearchTree<S, MaxEntropyUCBNode<S>, double, double, double>;

    template<typename S>
    using REPSSearchTree = SearchTree<S, REPSNode<S>, double, double>;

    template<typename S>
    using REPSUCBSearchTree = SearchTree<S, REPSUCBNode<S>, double, double, double>;

    template<typename S>
    using TSALLISSearchTree = SearchTree<S, TSALLISNode<S>, double, double>;
}

#endif //C___TREE_SEARCH_H
