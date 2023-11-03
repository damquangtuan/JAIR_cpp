#include <iostream>
#include <tree_search.h>
#include <envs/frozen_lake.h>
#include <envs/frozen_lake_ext.h>
#include <envs/copy.h>
#include <mpi.h>
#include <fstream>
#include <thread>
#include <chrono>
#include <sstream>
#include <memory>
#include <map>
#include <cstdlib>
#include <unistd.h>

using namespace poweruct;
using namespace std;

string get_current_working_dir() {
    char buff[FILENAME_MAX];
    getcwd(buff, FILENAME_MAX);
    string current_working_dir(buff);
    return current_working_dir;
}

template<typename S>
shared_ptr<AbstractSearchTree<S>> create_search_tree(shared_ptr<Environment<S>> search_env, double discount_factor,
                                                     string type, char **argv) {
    if (type == "max_uct") {
        auto alpha = stod(argv[0]);
        return make_shared<MaxUCTSearchTree<S>>(search_env, search_env->getInitialState(),
                                                search_env->getInitialObservation(), discount_factor, alpha);
    } else if (type == "power_uct") {
        auto alpha = stod(argv[0]);
        auto p = stod(argv[1]);
        return make_shared<PowerUCTSearchTree<S>>(search_env, search_env->getInitialState(),
                                                  search_env->getInitialObservation(), discount_factor, alpha, p);
    } else if (type == "max_entropy") {
        auto tau = stod(argv[0]);
        auto epsilon = stod(argv[1]);
        return make_shared<MaxEntropySearchTree<S>>(search_env, search_env->getInitialState(),
                                                    search_env->getInitialObservation(), discount_factor, tau, epsilon);
    } else if (type == "max_entropy_ucb") {
        auto tau = stod(argv[0]);
        auto epsilon = stod(argv[1]);
        auto alpha = stod(argv[2]);
        return make_shared<MaxEntropyUCBSearchTree<S>>(search_env, search_env->getInitialState(),
                                                       search_env->getInitialObservation(), discount_factor, tau, epsilon, alpha);
    }
    else if (type == "reps") {
        auto tau = stod(argv[0]);
        auto epsilon = stod(argv[1]);
        return make_shared<REPSSearchTree<S>>(search_env, search_env->getInitialState(),
                                              search_env->getInitialObservation(), discount_factor, tau, epsilon);
    } else if (type == "reps_ucb") {
        auto tau = stod(argv[0]);
        auto epsilon = stod(argv[1]);
        auto alpha = stod(argv[2]);
        return make_shared<REPSUCBSearchTree<S>>(search_env, search_env->getInitialState(),
                                                 search_env->getInitialObservation(), discount_factor, tau, epsilon, alpha);
    } else if (type == "tsallis") {
        auto tau = stod(argv[0]);
        auto epsilon = stod(argv[1]);
        return make_shared<TSALLISSearchTree<S>>(search_env, search_env->getInitialState(),
                                                 search_env->getInitialObservation(), discount_factor, tau, epsilon);
    }
    else {
        throw runtime_error(string("Invalid algorithm type: ") + type);
    }

}

template<typename S>
double run_experiment(uint32_t n_search, uint32_t seed, shared_ptr<Environment<S>> env,
                      shared_ptr<Environment<S>> env_search, double discount_factor, string st_type,
                      char *raw_st_args[], bool replan) {
    auto st = create_search_tree(env_search, discount_factor, st_type, raw_st_args);

    bool done = false;
    double reward = 0.;
    double disc_acc = 1.;
    env_search->seed(seed);
    env->seed(seed);
    st->seed(seed);

    if (!replan) {
        st->search(n_search);
    }

    env->reset();
    while (!done) {
        uint32_t action = st->search(replan ? n_search : 0);
        S next_state;
        uint32_t next_obs;
        double cur_reward;
        tie(next_state, next_obs, cur_reward, done) = env->step(action);
        reward += disc_acc * cur_reward;
        disc_acc *= discount_factor;
        st->progress_tree(action, next_state, next_obs);
    }

    return reward;
}

std::function<double(uint32_t, uint32_t)> create_experiment_runner(char **argv) {
    return [argv](uint32_t n_search, uint32_t seed) {
        // Get the data for the search type
        string st_type(argv[1]);
        char **raw_st_args = argv + 3;

        //Create the environment based on the argument and run the experiment
        string env_name(argv[2]);
        if (env_name == "copy") {
            vector<char> alphabet(
                    {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                     's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'});
            string target = "dahknoljedfs09dancq2kjabaciq23890asjnq0s";

            auto env = make_shared<Copy>(alphabet, target);
            auto env_search = make_shared<Copy>(alphabet, target);
            return run_experiment(n_search, seed, static_pointer_cast<Environment<CopyState>>(env),
                                  static_pointer_cast<Environment<CopyState>>(env_search), 1., st_type, raw_st_args,
                                  false);
        } else if (env_name == "copy_large") {
            vector<char> alphabet(
                    {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                     's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                     '`', '^', '<', '>', '.', ':', '-', '_', '*', '+', '#', '!', '$', '/'});
            string target = "dah<nolj^dfs09dancq2kjabaciq238/0as!nq0s";

            auto env = make_shared<Copy>(alphabet, target);
            auto env_search = make_shared<Copy>(alphabet, target);
            return run_experiment(n_search, seed, static_pointer_cast<Environment<CopyState>>(env),
                                  static_pointer_cast<Environment<CopyState>>(env_search), 1., st_type, raw_st_args,
                                  false);

        } else if (env_name == "copy_xxlarge") {
            vector<char> alphabet(
                    {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                     's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                     '`', '^', '<', '>', '.', ':', '-', '_', '*', '+', '#', '!', '$', '/',
                     '?', '"',';','~',')', '(','%','&','@','|', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', //25
                     'I', 'J', 'K', 'L', 'M', 'N','O', 'P','Q','R','S','T','U','V','W','X','Y'});
            string target = "dah<nolj^dfs09dancq2k%&()giq238/0as!nq0s";

            auto env = make_shared<Copy>(alphabet, target);
            auto env_search = make_shared<Copy>(alphabet, target);
            return run_experiment(n_search, seed, static_pointer_cast<Environment<CopyState>>(env),
                                  static_pointer_cast<Environment<CopyState>>(env_search), 1., st_type, raw_st_args,
                                  false);

        }
        else if (env_name == "frozen_lake") {
            auto env = make_shared<FrozenLake>(true, false);
            auto env_search = make_shared<FrozenLake>(true, false);
            return run_experiment(n_search, seed, static_pointer_cast<Environment<DiscreteEnvironmentState>>(env),
                                  static_pointer_cast<Environment<DiscreteEnvironmentState>>(env_search), 1., st_type,
                                  raw_st_args, true);
        } else if (env_name == "frozen_lake_ext") {
            auto env = make_shared<FrozenLakeExt>(true, false);
            auto env_search = make_shared<FrozenLakeExt>(true, false);
            return run_experiment(n_search, seed, static_pointer_cast<Environment<DiscreteEnvironmentState>>(env),
                                  static_pointer_cast<Environment<DiscreteEnvironmentState>>(env_search), 1., st_type,
                                  raw_st_args, true);
        } else {
            throw runtime_error(string("Invalid environment: ") + env_name);
        }
    };
}

string create_log_dir(char **argv) {
    uint32_t n_args = string(argv[2]) == "max_uct" ? 5 : 6;

    string log_dir = string("log-") + string(argv[1]);
    for (uint32_t i = 2; i < n_args; i++) {
        log_dir += string("-") + string(argv[i]);
    }

    string log_path = get_current_working_dir() + string("/") + log_dir;
    const int dir = system((string("mkdir -p ") + log_path).c_str());
    if (dir < 0) {
        throw runtime_error("Could not create the log directory");
    }
    return log_path;

}

int main(int argc, char *argv[]) {
    bool error = false;
    if (argc < 5) {
        error = true;
    } else {
        // Pre-Check the argument length - the type checks will be performed later
        string type(argv[2]);
        if ((type == "max_uct" && argc < 5) || (type == "power_uct" && argc < 6) ||
            (type == "max_entropy" && argc < 6) || (type == "reps" && argc < 6)) {
            error = true;
        }
    }

    if (error) {
        cout << "Missing required arguments - need one of the following: " << endl;
        cout << "\t\tN_EXPERIMENTS max_uct ENV ALPHA" << endl;
        cout << "\t\tN_EXPERIMENTS power_uct ENV ALPHA P" << endl;
        cout << "\t\tN_EXPERIMENTS max_entropy ENV TAU EPSILON" << endl;
        cout << "\t\tN_EXPERIMENTS max_entropy_ucb ENV TAU EPSILON ALPHA" << endl;
        cout << "\t\tN_EXPERIMENTS reps ENV TAU EPSILON" << endl;
        cout << "\t\tN_EXPERIMENTS tsallis ENV TAU EPSILON" << endl;
        cout << "\t\tN_EXPERIMENTS reps_ucb ENV TAU EPSILON ALPHA" << endl;
        return -1;
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Extract the log directory if specified
    auto experiment_runner = create_experiment_runner(argv + 1);

    string env(argv[3]);
    map<string, vector<uint32_t>> rollout_map;
    rollout_map.insert(make_pair("copy", vector<uint32_t>({4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                                                           8192, 16384, 32768})));
    rollout_map.insert(make_pair("copy_large", vector<uint32_t>({4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                                                                 8192, 16384, 32768})));
    rollout_map.insert(make_pair("copy_xxlarge", vector<uint32_t>({4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                                                                   8192, 16384, 32768, 65536})));
//    rollout_map.insert(make_pair("copy_xxlarge", vector<uint32_t>({65536})));
//    rollout_map.insert(make_pair("frozen_lake", vector<uint32_t>({4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
//                                                                  8192, 16384, 32768})));
    rollout_map.insert(make_pair("frozen_lake", vector<uint32_t>({4096, 16384, 65536, 262144})));

    rollout_map.insert(make_pair("frozen_lake_ext", vector<uint32_t>({4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                                                                      8192, 16384, 32768, 65536, 131072, 262144,
                                                                      350000})));

    string log_dir = create_log_dir(argv);
    uint32_t n_experiments = (uint32_t) stoi(argv[1]);
    uint32_t sub_size = size > 1 ? n_experiments / (size - 1) : n_experiments;
    uint32_t remainder = size > 1 ? n_experiments % (size - 1) : 0;

    if (rank == 0) {
        auto rewards = new double[n_experiments];
        auto rollouts_ptr = rollout_map.find(env);
        if (rollouts_ptr == rollout_map.end()) {
            cout << "Invalid environment: " << env << endl;
            return -1;
        }
        auto rollouts = rollouts_ptr->second;
        for (auto n_rollouts : rollouts) {
            if (size == 1) {
                // Execute the experiments
                for (uint32_t i = 0; i < n_experiments; i++) {
                    rewards[i] = experiment_runner(n_rollouts, i);
                }
            } else {
                // Invoke the computation
                for (uint32_t i = 1; i < size; i++) {
                    MPI_Send(&n_rollouts, 1, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
                }

                // Receive the result
                uint32_t offset = 0;
                for (uint32_t i = 1; i < size; i++) {
                    uint32_t n_sub_experiments = sub_size + (i <= remainder ? 1 : 0);
                    uint32_t buffer_size = n_sub_experiments;
                    double buffer[buffer_size];
                    MPI_Recv(buffer, buffer_size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, NULL);

                    for (uint32_t j = 0; j < n_sub_experiments; j++) {
                        rewards[offset + j] = buffer[j];
                    }
                    offset += n_sub_experiments;
                }
            }

            double mean_reward = 0.;
            for (uint32_t i = 0; i < n_experiments; i++) {
                mean_reward += rewards[i];
            }

            cout << "Performance (" << n_rollouts << "): " << (mean_reward / n_experiments) << endl;

            ofstream out_file;
            out_file.open(log_dir + "/results-" + to_string(n_rollouts) + ".bin", ios::out | ios::binary);
            if (out_file.fail()) {
                throw runtime_error("Error when opening log file");
            }

            out_file.write((char *) rewards, sizeof(double) * n_experiments);
            out_file.close();
        }

        delete[] rewards;
        for (uint32_t i = 1; i < size; i++) {
            MPI_Send(NULL, 0, MPI_UNSIGNED, i, 0, MPI_COMM_WORLD);
        }

    } else {
        // Create the buffer for the experiment results
        uint32_t n_sub_experiments = sub_size + (rank <= remainder ? 1 : 0);
        uint32_t seed_offset = n_experiments * rank;
        uint32_t buffer_size = n_sub_experiments;
        auto *rewards = new double[buffer_size];

        while (true) {
            MPI_Status status;
            MPI_Probe(0, 0, MPI_COMM_WORLD, &status);

            int count = 0;
            MPI_Get_count(&status, MPI_UNSIGNED, &count);
            if (count == 0) {
                break;
            }

            // Get the number of rollouts
            uint32_t n_rollouts;
            MPI_Recv(&n_rollouts, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, NULL);

            cout << "Received command to execute experiments with " << n_rollouts << " rollouts" << endl;

            // Execute the experiments
            for (uint32_t i = 0; i < n_sub_experiments; i++) {
                rewards[i] = experiment_runner(n_rollouts, seed_offset + i);
            }

            // Return the results
            cout << "Rank " << rank << " sending " << n_sub_experiments << " results" << endl;
            MPI_Send(rewards, buffer_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }

        cout << "Shutting down the worker" << endl;
        delete[] rewards;
    }

    MPI_Finalize();
    return 0;

}
