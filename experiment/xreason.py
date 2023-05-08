#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## xreason.py
##
##  Created on: Dec 7, 2018
##      Author: Alexey Ignatiev, Nina Narodytska
##      E-mail: alexey.ignatiev@monash.edu, narodytska@vmware.com
##

#
#==============================================================================
from __future__ import print_function
from data import Data
from anchor_wrap import anchor_call
from lime_wrap import lime_call
from shap_wrap import shap_call
from options import Options
import joblib
import numpy as np
import os
import sys
from xgbooster import XGBooster, preprocess_dataset
from multiprocessing import Pool
import multiprocessing as mp

o_lock=None
o_file=None
#
#==============================================================================
def show_info():
    """
        Print info message.
    """

    print('c XReason: reasoning about explanations')
    print('c author(s): Alexey Ignatiev    [email:alexey.ignatiev@monash.edu]')
    print('c            Joao Marques-Silva [email:joao.marques-silva@irit.fr]')
    print('c            Nina Narodytska    [email:narodytska@vmware.com]')
    print('')

def multi_run_wrapper(args):
   return compute(*args)


def compute(point,options,idx,fname,dirname,true_y):
    global o_file, o_lock
    point_ = point.tolist()

    # print([float(v.strip()) for v in options.explain.split(',')])
    # expl_file = open(fname, 'w+')
    expl_file = None

    # read a sample from options.explain
    if options.explain:
        options.explain = point_

    xgb = XGBooster(options, from_model=options.files[0])

    if options.uselime or options.useanchor or options.useshap:
        xgb = XGBooster(options, from_model=options.files[0])

        # encode it and save the encoding to another file
    xgb.encode(test_on=point_)
    if not options.limefeats:
        options.limefeats = len(data.names) - 1

    # explain using anchor or the abduction-based approach
    expl,res = xgb.explain(point,
                       use_lime=lime_call if options.uselime else None,
                       use_anchor=anchor_call if options.useanchor else None,
                       use_shap=shap_call if options.useshap else None,
                       nof_feats=options.limefeats, use_bfs=options.usebfs, writer=o_file, index=idx,
                       dirname=dirname,lock=o_lock,num_f=6)

    feat_sample_exp = xgb.transform(point)
    y_pred = xgb.model.predict(feat_sample_exp)[0]

    # expl_file.close()
    # print(expl)
    # all_expl.append((expl, y_pred))

    # if idx % 10 == 0:
    #     joblib.dump(np.array(all_expl), dirname + "/" + f + "_expls.pkl")

    if (options.uselime or options.useanchor or options.useshap) and options.validate:
        xgb.validate(options.explain, expl)
    return (idx,res,expl,y_pred,true_y)

#

o_lock=None
o_file=None

#==============================================================================
if __name__ == '__main__':
    # parsing command-line options
    options = Options(sys.argv)

    # global o_file, o_lock

    # making output unbuffered
    if sys.version_info.major == 2:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    # showing head
    show_info()

    if (options.preprocess_categorical):
        print("here")
        preprocess_dataset(options.files[0], options.preprocess_categorical_files)
        exit()

    if options.files:
        xgb = None

        if options.train:
            data = Data(filename=options.files[0], mapfile=options.mapfile,
                    separator=options.separator,
                    use_categorical = options.use_categorical)



            xgb = XGBooster(options, from_data=data)
            train_accuracy, test_accuracy, model = xgb.train()



        # read a sample from options.explain
        if options.explain:
            options.explain = [float(v.strip()) for v in options.explain.split(',')]

        if options.encode:
            if not xgb:
                xgb = XGBooster(options, from_model=options.files[0])

            # encode it and save the encoding to another file
            # xgb.encode(test_on=options.explain)


        # if options.check_fidelity:

        # if options.explain:
        #     if not xgb:
        #         if options.uselime or options.useanchor or options.useshap:
        #             xgb = XGBooster(options, from_model=options.files[0])
        #         else:
        #             # abduction-based approach requires an encoding
        #             xgb = XGBooster(options, from_encoding=options.files[0])
        #
        #     # checking LIME or SHAP should use all features
        #     if not options.limefeats:
        #         options.limefeats = len(data.names) - 1
        #
        #     # explain using anchor or the abduction-based approach
        #     expl = xgb.explain(options.explain,
        #             use_lime=lime_call if options.uselime else None,
        #             use_anchor=anchor_call if options.useanchor else None,
        #             use_shap=shap_call if options.useshap else None,
        #             nof_feats = options.limefeats,use_bfs=options.usebfs)
        #
        #     print(expl)
        #
        #     if (options.uselime or options.useanchor or options.useshap) and options.validate:
        #         xgb.validate(options.explain, expl)
        if options.uselime or options.useanchor or options.useshap:
            xgb = XGBooster(options, from_model=options.files[0])

        if options.explain:
            # bench_name = os.path.splitext(os.path.basename(options.files[0]))[0]
            #
            # bench_name +="_nbestim_" + str(options.n_estimators) +"_maxdepth_" + str(options.maxdepth) +"_testsplit_" + str(options.testsplit)

            try:
                f = options.files[1].split("/")[-1].strip(".csv")[0]
            except:
                f = options.files[0].split("/")[-1].strip(".csv")[0]
            data_name = xgb.basename.split("/")[-1]
            data_dir = "mwc_data/"
            if options.uselime:
                data_dir = "lime_data/"
            if options.useshap:
                data_dir = "shap_data/"
            if os.path.exists(data_dir) is False:
                os.mkdir(data_dir)
            # expl_file = open(data_dir + data_name + "_points.txt", 'w+')

            with open(data_dir + data_name + "_points.txt", "w+") as o_file:
                o_lock = mp.Lock()

                dirname = "data/" + data_name
                if os.path.exists(dirname) is False:
                    os.mkdir(dirname)
                fname = dirname +"/" +f + "_imp.pkl"
                # imp_indices = joblib.load(fname)

                idx = 0
                all_expl=[]
                # data = None
                if len(options.files)>=2:
                    data = Data(filename=options.files[1], mapfile=options.mapfile,
                                separator=options.separator,
                                use_categorical=options.use_categorical)

                    xgb_test = XGBooster(options, from_data=data)

                else:
                    xgb_test = xgb
                idx=0

                fname = data_dir + data_name + "_points.txt"
                points=[]
                result=[]
                for point in xgb_test.X:

                    for jdx in range(int(xgb_test.weights[idx])):
                        points.append((point,options,idx,fname,dirname,xgb_test.Y[idx]))
                        # result.append(multi_run_wrapper((point,options,idx,fname,dirname,xgb_test.Y[idx])))

                    idx+=1
                # points = points[:5]
                with Pool() as pool:
                    result = pool.map(multi_run_wrapper, points)



                    # idx += 1
                all_expl = result
                print(dirname + "/" + f  + "_expls.pkl")
                joblib.dump(all_expl,dirname + "/" + f  + "_expls.pkl")
                # expl_file.close()

