import numpy as np
import random

def f(x, y):
    return (1-x)**2 + 100* ((y-x**2))**2

f_bnds = [(-2.048,2.048) for _ in range(2)]

def generate_samples(num_samples, bb_bnds):
    data = {'X': [], 'y': []}

    for _ in range(num_samples):
        sample = []

        # iterate through all dimension bounds
        for idx, var_bnds in enumerate(bb_bnds):
            val = random.uniform(var_bnds[0], var_bnds[1])

            # populate the sample
            sample.append(val)

        data['X'].append(sample)
        data['y'].append(f(sample[0], sample[1]))
    return data

import lightgbm as lgb
import warnings

def train_tree(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        PARAMS = {'objective': 'regression',
                  'metric': 'rmse',
                  'boosting': 'gbdt',
                  'num_trees': 50,
                  'max_depth': 3,
                  'min_data_in_leaf': 2,
                  'random_state': 100,
                  'verbose': -1}
        train_x = np.asarray(data['X'])
        train_data = lgb.Dataset(train_x,
                                 label=data['y'],
                                 params={'verbose': -1})

        model = lgb.train(PARAMS,
                          train_data,
                          verbose_eval=False)
    return model


from onnxmltools.convert.lightgbm.convert import convert
from skl2onnx.common.data_types import FloatTensorType

def get_onnx_model(lgb_model):
    # export onnx model
    float_tensor_type = FloatTensorType([None, lgb_model.num_feature()])
    initial_types = [('float_input', float_tensor_type)]
    onnx_model = convert(lgb_model,
                         initial_types=initial_types,
                         target_opset=8)
    return onnx_model


def write_onnx_to_file(onnx_model, path, file_name="output.onnx"):
    from pathlib import Path
    with open(Path(path) / file_name, "wb") as onnx_file:
        onnx_file.write(onnx_model.SerializeToString())
        print(f'Onnx model written to {onnx_file.name}')


import pyomo.environ as pe
from omlt.block import OmltBlock
from omlt.gbt import GBTBigMFormulation, GradientBoostedTreeModel


def add_tree_model(opt_model, onnx_model, input_bounds):
    # init omlt block and gbt model based on the onnx format
    opt_model.gbt = OmltBlock()
    gbt_model = GradientBoostedTreeModel(onnx_model,
                                         scaled_input_bounds=input_bounds)

    # omlt uses a big-m formulation to encode the tree models
    formulation = GBTBigMFormulation(gbt_model)
    opt_model.gbt.build_formulation(formulation)


import numpy as np


def add_unc_metric(opt_model, data):
    # compute mean and std for standardization
    data_x = np.asarray(data['X'])
    std = np.std(data_x, axis=0)
    mean = np.mean(data_x, axis=0)

    # alpha capture the uncertainty value
    alpha_bound = abs(0.5 * np.var(data['y']))
    opt_model.alpha = pe.Var(within=pe.NonNegativeReals, bounds=(0, alpha_bound))
    opt_model.unc_constr = pe.ConstraintList()

    for x in data_x:
        x_var = opt_model.gbt.inputs
        opt_model.unc_constr.add(
            opt_model.alpha <= \
            sum((x[idx] - (x_var[idx] - mean[idx]) / std[idx]) * \
                (x[idx] - (x_var[idx] - mean[idx]) / std[idx])
                for idx in range(len(x_var)))
        )


# defining initial data
random.seed(10)
data = generate_samples(5, f_bnds)


def plot_progress(data, input_bounds):
    # plot contour line and data points
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 2, 2])

    # create mesh
    s = 0.01
    X = np.arange(input_bounds[0][0], input_bounds[0][1], s)
    Y = np.arange(input_bounds[1][0], input_bounds[1][1], s)
    X, Y = np.meshgrid(X, Y)

    # rosenbrock function
    Z = f(X, Y)

    # plot contour line
    clevf = np.arange(Z.min(), Z.max(), 10)
    CS = plt.contourf(X, Y, Z, clevf)
    fig.colorbar(CS)

    # plot initial data set
    ax.scatter([x[0] for x in data['X']], [x[1] for x in data['X']], c='r', s=100)

    plt.rcParams.update({'font.size': 15})
    plt.show()


plot_progress(data, f_bnds)
