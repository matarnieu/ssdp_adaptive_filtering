HOW TO ADD NEW METHOD:

- Add new file in methods/
- Define and implement method like filter_signal_my_method(noisy_signal, noise, K, args)
    where noisy_signal is D=S+h*X, noise is X, K is the tap-size / filter size and
    args is a dictionary with additional float or string parameters or flags for your method.
    You can use it like alpha=args["alpha"].
- Import method in main.py
- Add "method_name": imported_method to the methods dictionary in main.py
- Run main.py with your new method (don't forget to add additional parameters if necessary for your method)


UTILISATION FOR REAL SIGNAL ANALYSIS:

python main.py real <method> <additional (float) parameters and flags depending on the method>


UTILISATION FOR SYNTHETIC SIGNAL ANALYSIS:

python main.py

synthetic
<method>

--noise_power
[--noise_power_change]
[--noise_distribution_change]

--filter_type
--filter_size
--filter_changing_speed

<additional (float) parameters and flags depending on the method>

EXPLANATION OF PARAMETERS:

<method>: The filtering method used, e.g. gwf_fc, gwf_swc, gwf_ema, swg, ...

--noise_power: The power of the unfiltered noise X. As a reference, the power of the signal is 0.5, since it is a sine of amplitude 1

--noise_power_change: A flag that determines whether the power of X abruptly changes in given intervals.

--noise_distribution_change: A flag that determines whether the distribution of X abruptly changes in given intervals. When it is set, noise_power_change will be ignored.

--filter_type: The type of the noise filter X. Can be 'exponential_decay', 'moving_average' or mixed. In the latter case, it will alternate between the two filter types.

--filter_size: Tap-size K of the filter

--filter_changing_speed: Determines how quickly the weights of the filters smoothly change. When it is 0, the weights do not change.

<additional...>: Add additional parameters that are needed by the <method> here. Should only be float or string parameters or flags and can be added as --my_string_param=HelloWorld --my_float_param=123.4 or --my_flag.

EXAMPLES:

python main.py synthetic gwf_ema --noise_power 0.2  --filter_type exponential_decay --filter_size 10 --filter_changing_speed 0  --lambda=0.999

python main.py synthetic gwf_fc --noise_power 0.4  --filter_type mixed --filter_size 10 --filter_changing_speed 1.0 --noise_distribution_change

python main.py synthetic gwf_swc --noise_power 0.3  --filter_type exponential_decay --filter_size 10 --filter_changing_speed 0.0 --window_size=70

python main.py synthetic gwf_ema --noise_power 0.3  --filter_type exponential_decay --filter_size 10 --lambda=0.999 --noise_distribution_change


HOW TO USE OPTIMIZE_PARAMS.PY:
Note: Has to be executed from the project directory as: 

python run_scripts/optimize_params.py

In order to set up the optimization, change optimize_params.txt as follows:

- Put the name of the folder in which the results will be stored under ## NAME ##.
    - The results will be stored in the folder results/optimize_params/<name>
- Put the command without the parameters to tune under ## RUN COMMAND ##
- Put the parameters to tune under ## PARAMETERS ##, e.g.
    --lambda=[0.1, 0.5, 1.0]
    --mu=[0.1, 0.2, 0.3]
- The MSE for all different parameter combinations will be stored in results/optimize_params/<name>/results.csv
- The best MSE and corresponding parameters will be printed
