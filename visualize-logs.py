import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg',warn=False, force=False)

def bootstrap_var(data):
    n = data.shape[0]
    bootstrap_idxs = np.random.randint(0, n, (n, n))
    return np.var(np.mean(data[bootstrap_idxs], axis=1))


def get_plot_data(log_dir, with_baseline=False, bootstrap_variance=False, max_n=None):
    n_baseline = []
    n_power = []

    mean_baseline = []
    var_baseline = []
    mean_power = []
    var_power = []

    for filename in os.listdir(log_dir):
        if filename.endswith(".bin"):
            # Check whether it is a baseline file or not
            type, n = filename.split(".")[0].split("-")

            n = int(n)
            if max_n is None or n <= max_n:
                data = np.fromfile(os.path.join(log_dir, filename), "<f8")

                if type == "baseline":
                    n_baseline.append(n)
                    mean_baseline.append(np.mean(data))
                    var_baseline.append(bootstrap_var(data) if bootstrap_variance else np.var(data))
                else:
                    n_power.append(n)
                    mean_power.append(np.mean(data))
                    var_power.append(bootstrap_var(data) if bootstrap_variance else np.var(data))

    idxs_baseline = np.argsort(n_baseline)
    idxs_power = np.argsort(n_power)

    n_baseline = np.array(n_baseline)[idxs_baseline]
    n_power = np.array(n_power)[idxs_power]

    mean_baseline = np.array(mean_baseline)[idxs_baseline]
    var_baseline = np.array(var_baseline)[idxs_baseline]
    mean_power = np.array(mean_power)[idxs_power]
    var_power = np.array(var_power)[idxs_power]

    if with_baseline:
        return (n_power, mean_power, var_power), (n_baseline, mean_baseline, var_baseline)
    else:
        return n_power, mean_power, var_power


def main3():
    tpl_power, tpl_baseline = get_plot_data("c++/build/fl-ext-log-1.8", with_baseline=True, bootstrap_variance=True)
    n_power_2_2, data_power_2_2, var_power2_2 = get_plot_data("c++/build/fl-ext-log-2.2", bootstrap_variance=True)
    n_power_10, data_power_10, var_power10 = get_plot_data("c++/build/fl-ext-log-10", bootstrap_variance=True)
    n_power_20, data_power_20, var_power20 = get_plot_data("c++/build/fl-ext-log-20", bootstrap_variance=True)
    n_max, data_max, var_max = get_plot_data("c++/build/fl-ext-log-max", bootstrap_variance=True)
    n_power_1_8, data_power_1_8, var_power1_8 = tpl_power
    n_baseline, data_baseline, var_baseline = tpl_baseline

    labels = ["UCT", "Max-UCT", "Power-UCT (p=1.8)", "Power-UCT (p=10)", "Power-UCT (p=20)"]
    ns = [n_baseline, n_max, n_power_1_8, n_power_10, n_power_20]
    data = [data_baseline, data_max, data_power_1_8, data_power_10, data_power_20]
    var = [var_baseline, var_max, var_power1_8, var_power10, var_power20]
    colors = ["C0", "C3", "C1", "C2", "C4"]

    f = plt.figure(figsize=(3, 2.2))
    ax = f.gca()
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    lines = []
    for n_cur, data_cur, var_cur, c in zip(ns, data, var, colors):
        l0, = plt.plot(n_cur, data_cur, color=c)
        plt.fill_between(n_cur, data_cur - 2 * np.sqrt(var_cur), data_cur + 2 * np.sqrt(var_cur), color=l0.get_color(),
                         alpha=0.3)
        lines.append(l0)

    plt.legend(lines, labels, fontsize=7)
    plt.title("Frozen Lake (8 Actions)", fontsize=7)
    plt.grid(True)
    plt.xlabel("Rollouts", fontsize=7)
    plt.ylabel("Success Rate", fontsize=7)
    plt.xticks([0, 75000, 150000, 225000])
    plt.tight_layout()
    plt.show()


def main2():
    n_power_1_6, data_power_1_6, var_power1_6 = get_plot_data("c++/build/fl-log-1.6", bootstrap_variance=True)
    tpl_power, tpl_baseline = get_plot_data("c++/build/fl-log-1.8", with_baseline=True, bootstrap_variance=True)
    n_power_2_0, data_power_2_0, var_power2_0 = get_plot_data("c++/build/fl-log-2", bootstrap_variance=True)
    n_power_2_2, data_power_2_2, var_power2_2 = get_plot_data("c++/build/fl-log-2.2", bootstrap_variance=True)
    n_power_1_8, data_power_1_8, var_power1_8 = tpl_power
    n_baseline, data_baseline, var_baseline = tpl_baseline
    n_max, data_max, var_max = get_plot_data("c++/build/fl-log-max", bootstrap_variance=True)

    # labels = ["UCT", "Power-UCT (p=1.6)", "Power-UCT (p=1.8)", "Power-UCT (p=2.0)", "Power-UCT (p=2.2)"]
    # ns = [n_baseline, n_power_1_6, n_power_1_8, n_power_2_0, n_power_2_2]
    # data = [data_baseline, data_power_1_6, data_power_1_8, data_power_2_0, data_power_2_2]
    # var = [var_baseline, var_power1_6, var_power1_8, var_power2_0, var_power2_2]
    labels = ["UCT", "Max-UCT", "Power-UCT (p=1.8)", "Power-UCT (p=2.2)"]
    ns = [n_baseline, n_max, n_power_1_8, n_power_2_2]
    data = [data_baseline, data_max, data_power_1_8, data_power_2_2]
    var = [var_baseline, var_max, var_power1_8, var_power2_2]
    colors = ["C0", "C3", "C1", "C2"]

    f = plt.figure(figsize=(3, 2))
    ax = f.gca()
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    lines = []
    for n_cur, data_cur, var_cur, c in zip(ns, data, var, colors):
        l0, = plt.plot(n_cur, data_cur, color=c)
        plt.fill_between(n_cur, data_cur - 2 * np.sqrt(var_cur), data_cur + 2 * np.sqrt(var_cur), color=l0.get_color(),
                         alpha=0.3)
        lines.append(l0)

    plt.legend(lines, labels, fontsize=7)
    plt.title("Frozen Lake (4 Actions)", fontsize=7)
    plt.ylabel("Success Rate", fontsize=7)
    plt.grid(True)
    plt.xticks([0, 75000, 150000, 225000])
    plt.tight_layout()
    plt.show()


def plot_frozen_lake_single():
    # n_power_1_6, data_power_1_6, var_power1_6 = get_plot_data("c++/build_new/fl-log-1.6", bootstrap_variance=True)
    n_baseline, data_baseline, var_baseline  = get_plot_data("c++/build_new/log-500-power_uct-frozen_lake-1.41-1.", bootstrap_variance=True)
    n_max, data_max, var_max = get_plot_data("c++/build_new/log-500-max_uct-frozen_lake-1.41", bootstrap_variance=True)
    n_ments, data_ments, var_ments = get_plot_data("c++/build_new/log-500-max_entropy-frozen_lake-0.04696735-0.17059164", bootstrap_variance=True)


    n_reps006, data_reps006, var_reps006 = get_plot_data("c++/bin/log-100-reps-frozen_lake-0.06-2.0", bootstrap_variance=True)
    n_reps003, data_reps003, var_reps003 = get_plot_data("c++/bin/log-100-reps-frozen_lake-0.03-.175", bootstrap_variance=True)
    n_reps01, data_reps01, var_reps01 = get_plot_data("c++/bin/log-100-reps-frozen_lake-0.04-.17", bootstrap_variance=True)

    n_reps_ucb_003, data_reps_ucb_003, var_reps_ucb_003 = get_plot_data("c++/bin/log-500-reps_ucb-frozen_lake-0.02-.0", bootstrap_variance=True)

    # labels = ["UCT", "Max-UCT", "MENTS", "RENTS-UCB (t=.03)"]
    # ns = [n_baseline, n_max, n_ments, n_reps_ucb_003]
    # data = [data_baseline, data_max, data_ments, data_reps_ucb_003]
    # var = [var_baseline, var_max, var_ments, var_reps_ucb_003]
    # colors = ["C0", "C1", "C2", "C3"]

    labels = ["UCT", "Max-UCT", "MENTS", "RENTS (t=.06)", "RENTS (t=.03)", "RENTS-UCB (t=.03)"]
    ns = [n_baseline, n_max, n_ments, n_reps006, n_reps003, n_reps_ucb_003]
    data = [data_baseline, data_max, data_ments, data_reps006, data_reps003, data_reps_ucb_003]
    var = [var_baseline, var_max, var_ments, var_reps006, var_reps003, var_reps_ucb_003]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    f = plt.figure(figsize=(3, 2))
    ax = f.gca()
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    lines = []
    for n_cur, data_cur, var_cur, c in zip(ns, data, var, colors):
        l0, = plt.plot(n_cur, data_cur, color=c)
        plt.fill_between(n_cur, data_cur - 2 * np.sqrt(var_cur), data_cur + 2 * np.sqrt(var_cur), color=l0.get_color(),
                         alpha=0.3)
        lines.append(l0)

    plt.legend(lines, labels, fontsize=7)
    plt.title("Frozen Lake (4 Actions)", fontsize=7)
    plt.ylabel("Success Rate", fontsize=7)
    plt.grid(True)
    plt.xticks([0, 75000, 150000, 225000])

    plt.xlim([0, 262144])

    plt.tight_layout()
    plt.show()




def plot_frozen_lake():
    paths = [["c++/build_new/log-500-power_uct-frozen_lake-1.41-1.",
              # "c++/build_new/log-500-power_uct-frozen_lake-1.41-1.8",
              "c++/build_new/log-500-power_uct-frozen_lake-1.41-2.2",
              "c++/build_new/log-500-max_uct-frozen_lake-1.41",
              "c++/build_new/log-500-max_entropy-frozen_lake-0.04696735-0.17059164"],
             ["c++/build_new/log-500-power_uct-frozen_lake_ext-1.41-1.",
              "c++/build_new/log-500-power_uct-frozen_lake_ext-1.41-10",
              "c++/build_new/log-500-power_uct-frozen_lake_ext-1.41-20",
              # "c++/build_new/log-500-max_uct-frozen_lake_ext-1.41",
              "c++/build_new/log-500-max_entropy-frozen_lake_ext-0.1-0.0125"]
             ]
    max_ns = [500000, 500000]
    ticks = [0, 75000, 150000, 225000]
    # limits = [0, 262144]
    labels = [["UCT", "Power-UCT (p=2.2)", "Power-UCT (p=max)", "MENTS"],
              ["UCT", "Power-UCT (p=10)", "Power-UCT (p=20)", "MENTS"]]

    f, ax = plt.subplots(len(paths), 1, sharex=True, figsize=(3.5, 4))
    for i in range(0, len(paths)):
        ax[i].tick_params(axis='both', which='major', labelsize=7)
        ax[i].tick_params(axis='both', which='minor', labelsize=7)

        ns = []
        data = []
        var = []
        for j in range(0, len(paths[i])):
            n_cur, data_cur, var_cur = get_plot_data(paths[i][j], max_n=max_ns[i], bootstrap_variance=True)
            ns.append(n_cur)
            data.append(data_cur)
            var.append(var_cur)

            lines = []
            count = 0
            for n_cur, data_cur, var_cur in zip(ns, data, var):
                l0, = ax[i].plot(n_cur, data_cur, color="C" + str(count))
                count += 1
                ax[i].fill_between(n_cur, data_cur - 2 * np.sqrt(var_cur),
                                   np.minimum(data_cur + 2 * np.sqrt(var_cur), 40.), color=l0.get_color(),
                                   alpha=0.3)
                lines.append(l0)

            ax[i].set_xticks(ticks)
            # ax[i].set_xlim(limits)

            ax[i].legend(lines, labels if len(labels) == 1 else labels[i], fontsize=7)
            if i == len(paths) - 1:
                ax[i].set_xlabel("Rollouts", fontsize=7)
            ax[i].set_ylabel("Score", fontsize=7)
            ax[i].grid(True)
    ax[0].set_title('Frozen Lake', fontsize=7)
    # ax[1].set_title('Frozen Lake (8 Actions)', fontsize=7)
    plt.tight_layout()
    plt.show()


def main():
    # paths = [["c++/build/copy-small-log-3", "c++/build/copy-small-log-20", "c++/build/copy-small-log-max"],
    #          ["c++/build/copy-log-3", "c++/build/copy-log-20", "c++/build/copy-log-max"]]
    paths = [["c++/build_old/log-100-power_uct-copy-0.25-1.",
              # "c++/build_new/log-100-power_uct-copy-0.25-3.",
              # "c++/build_new/log-100-power_uct-copy-0.25-20.",
              "c++/build_old/log-100-max_uct-copy-0.25",
              "c++/build_old/log-100-max_entropy-copy-0.1-0.",
              "c++/bin/log-100-max_entropy_ucb-copy-1.0-0.0",
              "c++/bin/log-100-reps-copy-.08-.0",
              "c++/bin/log-100-reps_ucb-copy-1.0-0.0"],
             ["c++/build_old/log-100-power_uct-copy_large-0.25-1.",
              # "c++/build_new/log-100-power_uct-copy_large-0.25-3.",
              # "c++/build_new/log-100-power_uct-copy_large-0.25-20.",
              "c++/build_old/log-100-max_uct-copy_large-0.25",
              "c++/build_old/log-100-max_entropy-copy_large-0.1-0.",
              "c++/bin/log-100-max_entropy_ucb-copy_large-1.0-0.0",
              "c++/bin/log-100-reps-copy_large-.08-.0",
              "c++/bin/log-100-reps_ucb-copy_large-1.0-0.0"],
             ["c++/bin/log-100-power_uct-copy_xxlarge-0.25-1",
              # "c++/build_new/log-100-power_uct-copy_large-0.25-3.",
              # "c++/build_new/log-100-power_uct-copy_large-0.25-20.",
              "c++/bin/log-100-max_uct-copy_xxlarge-0.25",
              "c++/bin/log-100-max_entropy-copy_xxlarge-.08-.0",
              "c++/bin/log-100-max_entropy_ucb-copy_xxlarge-1.0-0.0",
              "c++/bin/log-100-reps-copy_xxlarge-.08-.0",
              "c++/bin/log-100-reps_ucb-copy_xxlarge-1.0-0.0"]
             ]

    max_ns = [40000, 40000, 40000]
    ticks = [6000, 12000, 18000]
    # limits = [0, 20000]
    # labels = ["UCT", "Power-UCT (p=3)", "Power-UCT (p=20)", "Power-UCT (p=max)", "MENTS"]
    labels = ["UCT", "Max-UCT", "MENTS", "MENTS-UCB", "RENTS", "RENTS-UCB"]


    f, ax = plt.subplots(len(paths), 1, sharex=True, figsize=(3.5, 8))
    for i in range(0, len(paths)):
        ax[i].tick_params(axis='both', which='major', labelsize=9)
        ax[i].tick_params(axis='both', which='minor', labelsize=9)

        ns = []
        data = []
        var = []
        for j in range(0, len(paths[i])):
            n_cur, data_cur, var_cur = get_plot_data(paths[i][j], max_n=max_ns[i])
            ns.append(n_cur)
            data.append(data_cur)
            var.append(var_cur)

            lines = []
            count = 0
            for n_cur, data_cur, var_cur in zip(ns, data, var):
                l0, = ax[i].plot(n_cur, data_cur, color="C" + str(count))
                count += 1
                # ax[i].fill_between(n_cur, data_cur - 2 * np.sqrt(var_cur),
                #                    np.minimum(data_cur + 2 * np.sqrt(var_cur), 40.), color=l0.get_color(),
                #                    alpha=0.1)
                lines.append(l0)
            if i == 2:
                big_ticks = [6000, 12000, 18000, 24000]
                ax[i].set_xticks(big_ticks)
            else:
                ax[i].set_xticks(ticks)
            # ax[i].set_xlim(limits)

            ax[i].legend(lines, labels, fontsize=8)
            if i == len(paths) - 1:
                ax[i].set_xlabel("Rollouts", fontsize=8)
            ax[i].set_ylabel("Score", fontsize=8)
            ax[i].grid(True)

            ax[i].set_yscale('linear')

    ax[0].set_title('Copy Environment (144 Actions)', fontsize=9)
    ax[1].set_title('Copy Environment (200 Actions)', fontsize=9)
    ax[2].set_title('Copy Environment (300 Actions)', fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    # plot_frozen_lake_single()
