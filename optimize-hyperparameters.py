import numpy as np
import os
import GPy as gpy
from GPyOpt.methods import BayesianOptimization
import subprocess
import shutil


def run_experiment(args):
    print(args)
    # Disassemble the iinput
    res_total = []
    for i in range(args.shape[0]):
        n0 = args[i, 0]
        p_init = args[i, 1]
        p_inc = args[i, 2]

        # Get the location of the script
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # Create the log directory
        n0_str = str(n0)
        p_init_str = str(p_init)
        p_inc_str = str(p_inc)

        # We only run the experiment if is has not already been done
        log_dir = "logs/frozen-lake-sto-" + n0_str + "-" + p_init_str + "-" + p_inc_str
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            try:
                subprocess.check_call("mpirun -np 15 " + script_dir + "/c++/build/dummy 0 " + n0_str + " " + p_init_str + " " + p_inc_str + " " + log_dir, shell=True)
            except Exception as e:
                shutil.rmtree(log_dir)

        # Collect the result
        res = 0.
        for filename in os.listdir(log_dir):
            if filename.endswith(".bin"):
                data = np.fromfile(os.path.join(log_dir, filename), "<f8")
                res += np.mean(data)

        res_total.append(res)

    print("Reward under curve: " + str(res_total))
    return np.array(res_total)


def load_initial_experiments():
    xs = []
    ys = []

    for log_dir in os.listdir("logs"):
        prefix = "frozen-lake-sto-"
        if log_dir.startswith(prefix):
            n0, p_init, p_inc = log_dir[len(prefix):].split("-")
            n0 = float(n0)
            p_init = float(p_init)
            p_inc = float(p_inc)
            xs.append([n0, p_init, p_inc])
            ys.append(run_experiment(np.array([xs[-1]])))

    return np.array(xs), np.array(ys)


if __name__ == "__main__":
    np.random.seed(0)
    bds = [{'name': 'n0', 'type': 'discrete', 'domain': [i for i in range(1, 100)]},
           {'name': 'p_init', 'type': 'continuous', 'domain': (1., 20.)},
           {'name': 'p_inc', 'type': 'continuous', 'domain': (0., 100.)}]

    xs, ys = load_initial_experiments()

    optimizer = BayesianOptimization(f=run_experiment, domain=bds, model_type='GP_MCMC',
                                     acquisition_type ='MPI_MCMC', exact_feval=True, maximize=True, batch_size=1,
                                     X=xs, Y=ys)

    optimizer.run_optimization(max_iter=1000)

    
