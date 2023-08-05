#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## explain.py
##
##  Created on: Dec 14, 2018
##      Author: Alexey Ignatiev
##      E-mail: alexey.ignatiev@monash.edu
##

#
#==============================================================================
from __future__ import print_function

import math
from datetime import datetime
import numpy as np
import os
from pysat.examples.hitman import Hitman
from pysat.formula import IDPool
from pysmt.shortcuts import Solver
from pysmt.shortcuts import And, BOOL, Implies, Not, Or, Symbol, Iff
from pysmt.shortcuts import Equals, GT, Int, Real, REAL
import resource
from six.moves import range
import sys
import collections
import matplotlib.pyplot as plt

from pysmt.typing import INT, REAL
#
#==============================================================================

def plot_bar(xvals, yvals, xlabel, ylabel,basename, filename, dir="plots/"):
    plt.clf()
    plt.rc('legend', fontsize=28)
    # plt.rc('legend', fontsize='x-large')
    plt.rc('axes', titlesize=28, labelsize=28)
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    fig = plt.figure(figsize=(26,23))

    # creating the bar plot
    plt.bar(xvals, yvals,
            width=0.4)
    if os.path.exists(dir) is False:
        os.mkdir(dir)
    if os.path.exists(dir + "/" + basename) is False:
        os.mkdir(dir + "/" + basename)
    path = dir + "/" + basename

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    print("current dir")
    print(os.getcwd())
    plt.xticks(rotation=20)

    plt.savefig(path + "/" + filename)
class Solution(object):
    def calcEquation(self, equations, values, queries):

        graph = {}

        def build_graph(equations, values):
            def add_edge(f, t, value):
                if f in graph:
                    graph[f].append((t, value))
                else:
                    graph[f] = [(t, value)]

            for vertices, value in zip(equations, values):
                f, t = vertices
                add_edge(f, t, value)
                add_edge(t, f, 1 / value)

        def find_path(query):
            b, e = query

            if b not in graph or e not in graph:
                return -1.0

            q = collections.deque([(b, 1.0)])
            visited = set()

            while q:
                front, cur_product = q.popleft()
                if front == e:
                    return cur_product
                visited.add(front)
                for neighbor, value in graph[front]:
                    if neighbor not in visited:
                        q.append((neighbor, cur_product * value))

            return -1.0

        build_graph(equations, values)
        return [find_path(q) for q in queries]



class SMTExplainer(object):
    """
        An SMT-inspired minimal explanation extractor for XGBoost models.
    """

    def __init__(self, formula, intvs, imaps, ivars, feats, nof_classes,
            options, xgb):
        """
            Constructor.
        """

        self.feats = feats
        self.intvs = intvs
        self.imaps = imaps
        self.ivars = ivars
        self.nofcl = nof_classes
        self.optns = options
        self.idmgr = IDPool()

        # saving XGBooster
        self.xgb = xgb

        self.verbose = self.optns.verb
        self.oracle = Solver(name=options.solver)

        self.inps = []  # input (feature value) variables
        for f in self.xgb.extended_feature_names_as_array_strings:
            if '_' not in f:
                self.inps.append(Symbol(f, typename=REAL))
            else:
                self.inps.append(Symbol(f, typename=BOOL))

        self.outs = []  # output (class  score) variables
        for c in range(self.nofcl):
            self.outs.append(Symbol('class{0}_score'.format(c), typename=REAL))


        # theory
        self.oracle.add_assertion(formula)

        # current selector
        self.selv = None

    def get_categorical_positions(self,f_id):
        variable_names = self.xgb.extended_feature_names_as_array_strings
        ids=[]
        idx=0
        names=[]
        for name in variable_names:
            feature_id = int(name.split("_")[0].strip('f'))
            if feature_id==int(f_id):
                ids.append(idx)
                names.append(name)
            idx+=1
        return ids, names



    def prepare_(self, sample,kwargs=None):
        """
            Prepare the oracle for computing an explanation.
        """

        self.orig_sample = sample

        if self.selv:
            # disable the previous assumption if any
            self.oracle.add_assertion(Not(self.selv))

        # creating a fresh selector for a new sample
        sname = ','.join([str(v).strip() for v in sample])

        # the samples should not repeat; otherwise, they will be
        # inconsistent with the previously introduced selectors
        assert sname not in self.idmgr.obj2id, 'this sample has been considered before (sample {0})'.format(self.idmgr.id(sname))
        self.selv = Symbol('sample{0}_selv'.format(self.idmgr.id(sname)), typename=BOOL)

        self.rhypos = []  # relaxed hypotheses

        # transformed sample
        self.sample = list(self.xgb.transform(sample)[0])

        self.sel2fid = {}  # selectors to original feature ids
        self.sel2vid = {}  # selectors to categorical feature ids

        # preparing the selectors
        for i, (inp, val) in enumerate(zip(self.inps, self.sample), 1):
            feat = inp.symbol_name().split('_')[0]
            selv = Symbol('selv_{0}'.format(feat))
            val = float(val)

            self.rhypos.append(selv)
            if selv not in self.sel2fid:
                self.sel2fid[selv] = int(feat[1:])
                self.sel2vid[selv] = [i - 1]
            else:
                self.sel2vid[selv].append(i - 1)

        # adding relaxed hypotheses to the oracle
        if not self.intvs:
            for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
                if '_' not in inp.symbol_name():
                    hypo = Implies(self.selv, Implies(sel, Equals(inp, Real(float(val)))))
                else:
                    hypo = Implies(self.selv, Implies(sel, inp if val else Not(inp)))

                self.oracle.add_assertion(hypo)
        else:
            for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
                inp = inp.symbol_name()
                # determining the right interval and the corresponding variable
                for ub, fvar in zip(self.intvs[inp], self.ivars[inp]):
                    if ub == '+' or val < ub:
                        hypo = Implies(self.selv, Implies(sel, fvar))
                        break

                self.oracle.add_assertion(hypo)

        # in case of categorical data, there are selector duplicates
        # and we need to remove them
        self.rhypos = sorted(set(self.rhypos), key=lambda x: int(x.symbol_name()[6:]))

        # propagating the true observation
        if self.oracle.solve([self.selv] + self.rhypos):
            model = self.oracle.get_model()
        else:
            assert 0, 'Formula is unsatisfiable under given assumptions'

        # choosing the maximum
        outvals = [float(model.get_py_value(o)) for o in self.outs]
        maxoval = max(zip(outvals, range(len(outvals))))

        # correct class id (corresponds to the maximum computed)
        self.out_id = maxoval[1]
        self.output = self.xgb.target_name[self.out_id]

        true_y = kwargs["true_y"] #is_ood  0 if ood 1 if in-dist




        ub_features = list(kwargs['unbiased'].values())
        b_features = list(kwargs['biased'].values())

        # force contradiction

        fb_id = b_features[0]['id']
        fb_id_zero = b_features[0]['min']
        fb_id_one = b_features[0]['max']
        true_fb_ids, fb_names = self.get_categorical_positions(fb_id)




        if len(ub_features)==1:
            fub_id = ub_features[0]['id']
            fub_id_zero = ub_features[0]['min']
            fub_id_one = ub_features[0]['max']
            true_fub_ids, fub_names = self.get_categorical_positions(fub_id)


            if true_y==1:
                orig_label = self.orig_sample[fb_id]==fb_id_one
            else:
                orig_label = self.orig_sample[fub_id]==fub_id_one

            # force a contradiction
            label = 1 - orig_label


            # final label is true if (not ood and biased_f ==1) or (ood and unbiased f==1)

            if label==1:
                # if race==1 and is_ood=0  -> label =1
                # if unrelated_feature ==1 and is_ood ==1 -> label =1
                v = And([self.inps[true_fb_ids[1]], GT(self.outs[1], self.outs[0])])
                w = And([self.inps[true_fub_ids[1]], GT(self.outs[0], self.outs[1])])
                # self.oracle.add_assertion(Or([v, w]))


            else:
                # final label is true if (not ood and biased_f ==0) or (ood and unbiased f==0)
                v = And([self.inps[true_fb_ids[0]], GT(self.outs[0], self.outs[1])])
                w = And([self.inps[true_fub_ids[0]], GT(self.outs[1], self.outs[0])])
            self.oracle.add_assertion(Or([v, w]))




        elif len(ub_features)==2:
            fub_id = ub_features[0]['id']
            fub_id_zero = ub_features[0]['min']
            fub_id_one = ub_features[0]['max']
            fub_id1 = ub_features[1]['id']
            fub_id1_zero = ub_features[1]['min']
            fub_id1_one = ub_features[1]['max']

            true_fub_ids, fub_names = self.get_categorical_positions(fub_id)

            true_fub_id1s, fub_names1 = self.get_categorical_positions(fub_id1)



            if true_y == 1:
                orig_label = self.orig_sample[fb_id]==fb_id_one
            else:
                orig_label = np.logical_xor(self.orig_sample[fub_id],self.orig_sample[fub_id1])

            label = 1-orig_label

            # final label is true if (not ood and biased_f ==1) or (ood and ((ub ==1 and ub1==0) or (ub==0 and ub1==1) )
            if label == 1:

                v = And([true_fb_ids[1], GT(self.outs[1], self.outs[0])])
                a1 = And([true_fub_ids[1], true_fub_id1s[0]])
                a2 = And([true_fub_ids[0], true_fub_id1s[1]])
                temp = Or([a1,a2])
                w = And([temp, GT(self.outs[0], self.outs[1])])

            else:
                # final label is true if (not ood and biased_f ==1) or (ood and ((ub ==0 and ub1==0) or (ub==1 and ub1==1) )

                v = And([true_fb_ids[0], GT(self.outs[1], self.outs[0])])
                a1 = And([true_fub_ids[0], true_fub_id1s[0]])
                a2 = And([true_fub_ids[1], true_fub_id1s[1]])

                temp = Or([a1,a2])
                w = And([temp, GT(self.outs[0], self.outs[1])])
            self.oracle.add_assertion(Or([v, w]))

        else:
            # forcing a misclassification, i.e. a wrong observation
            disj = []
            for i in range(len(self.outs)):
                if i != self.out_id:
                    disj.append(GT(self.outs[i], self.outs[self.out_id]))
            self.oracle.add_assertion(Implies(self.selv, Or(disj)))


        if self.verbose:
            inpvals = self.xgb.readable_sample(sample)

            self.preamble = []
            for f, v in zip(self.xgb.feature_names, inpvals):
                if f not in str(v):
                    self.preamble.append('{0} = {1}'.format(f, str(v)))
                else:
                    self.preamble.append(v)

            print('  explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))



    def prepare(self, sample):
        """
            Prepare the oracle for computing an explanation.
        """

        if self.selv:
            # disable the previous assumption if any
            self.oracle.add_assertion(Not(self.selv))

        # creating a fresh selector for a new sample
        sname = ','.join([str(v).strip() for v in sample])

        # the samples should not repeat; otherwise, they will be
        # inconsistent with the previously introduced selectors
        assert sname not in self.idmgr.obj2id, 'this sample has been considered before (sample {0})'.format(self.idmgr.id(sname))
        self.selv = Symbol('sample{0}_selv'.format(self.idmgr.id(sname)), typename=BOOL)

        self.rhypos = []  # relaxed hypotheses

        # transformed sample
        self.sample = list(self.xgb.transform(sample)[0])

        self.sel2fid = {}  # selectors to original feature ids
        self.sel2vid = {}  # selectors to categorical feature ids

        # preparing the selectors
        for i, (inp, val) in enumerate(zip(self.inps, self.sample), 1):
            feat = inp.symbol_name().split('_')[0]
            selv = Symbol('selv_{0}'.format(feat))
            val = float(val)

            self.rhypos.append(selv)
            if selv not in self.sel2fid:
                self.sel2fid[selv] = int(feat[1:])
                self.sel2vid[selv] = [i - 1]
            else:
                self.sel2vid[selv].append(i - 1)

        # adding relaxed hypotheses to the oracle
        if not self.intvs:
            for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
                if '_' not in inp.symbol_name():
                    hypo = Implies(self.selv, Implies(sel, Equals(inp, Real(float(val)))))
                else:
                    hypo = Implies(self.selv, Implies(sel, inp if val else Not(inp)))

                self.oracle.add_assertion(hypo)
        else:
            for inp, val, sel in zip(self.inps, self.sample, self.rhypos):
                inp = inp.symbol_name()
                # determining the right interval and the corresponding variable
                for ub, fvar in zip(self.intvs[inp], self.ivars[inp]):
                    if ub == '+' or val < ub:
                        hypo = Implies(self.selv, Implies(sel, fvar))
                        break

                self.oracle.add_assertion(hypo)

        # in case of categorical data, there are selector duplicates
        # and we need to remove them
        self.rhypos = sorted(set(self.rhypos), key=lambda x: int(x.symbol_name()[6:]))

        # propagating the true observation
        if self.oracle.solve([self.selv] + self.rhypos):
            model = self.oracle.get_model()
        else:
            assert 0, 'Formula is unsatisfiable under given assumptions'

        # choosing the maximum
        outvals = [float(model.get_py_value(o)) for o in self.outs]
        maxoval = max(zip(outvals, range(len(outvals))))

        # correct class id (corresponds to the maximum computed)
        self.out_id = maxoval[1]
        self.output = self.xgb.target_name[self.out_id]

        # forcing a misclassification, i.e. a wrong observation
        disj = []
        for i in range(len(self.outs)):
            if i != self.out_id:
                disj.append(GT(self.outs[i], self.outs[self.out_id]))
        self.oracle.add_assertion(Implies(self.selv, Or(disj)))



        if self.verbose:
            inpvals = self.xgb.readable_sample(sample)

            self.preamble = []
            for f, v in zip(self.xgb.feature_names, inpvals):
                if f not in str(v):
                    self.preamble.append('{0} = {1}'.format(f, str(v)))
                else:
                    self.preamble.append(v)

            print('  explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))

    def is_minimal(self,coalition,minimal_coalitions):

        minimals = [x[1] for x in minimal_coalitions.values()]
        for minimal in minimals:
            flag=True
            for c in minimal:
                if c not in coalition:
                    flag = False
                    break
            if flag is True:
               return False
        return True


    def convert_coalition_to_str(self,coalition):
        c_list = sorted([self.sel2fid[h] for h in coalition])
        cs_list = [str(x) for x in c_list]
        c_str = "-".join(cs_list)
        return c_str, cs_list


    def responsibility_index(self,expls, sort=False):
        n_features = len(self.rhypos)
        importances = {}

        for expl in expls:
            for f in expl:
                if importances.get(f) is None:
                    importances[f] = len(expl)
                else:
                    importances[f] = min(importances[f], len(expl))
        for idx in range(n_features):
            if importances.get(idx) is None:
                importances[idx]=0.0

        ranks = []
        for idx in range(n_features):
            key = idx
            val = importances[key]
            if val==0:
                ranks.append(tuple([key, 0.0]))
            else:
                if math.isnan == 1.0/val:
                    import ipdb
                    ipdb.set_trace()
                    print("here")
                ranks.append(tuple([key, 1.0 / val]))
        if sort == True:
            ranks = sorted(ranks, key=lambda tup: tup[1], reverse=True)

        return ranks
    def deegan_packel(self, expls, sort=False):
        n_features = len(self.rhypos)
        importances = {}
        for idx in range(n_features):
            importances[idx] = 0
        for expl in expls:
            for f in expl:

                if len(expl)!=0:
                    importances[f] += 1.0 / len(expl)
        ranks = []
        for idx in range(n_features):
            key = idx
            val = importances[key]
            ranks.append(tuple([key, val]))
        if sort == True:
            ranks = sorted(ranks, key=lambda tup: tup[1], reverse=True)
        return ranks


    def holler_packel(self, expls, sort=False):
        n_features = len(self.rhypos)
        importances = {}
        for idx in range(n_features):
            importances[idx] = 0
        for expl in expls:
            for f in expl:

                importances[f] += 1.0

        ranks = []
        for idx in range(n_features):
            key = idx
            val = importances[key]
            ranks.append(tuple([key, val]))
        if sort == True:
            ranks = sorted(ranks, key=lambda tup: tup[1], reverse=True)
        return ranks

    def normalize(self, ranks):
        scores = []
        for rank in ranks:
            scores.append(rank[1])

        scores = np.array(scores)
        min_s = np.min(scores)
        max_s = np.max(scores)
        if np.sum(scores) ==0:
            normalized_scores = scores
        else:
            normalized_scores = np.round(scores/np.sum(scores),2)

        return normalized_scores
    def compare_index(self,expls,writer=None, lock=None, idx=None):
        response_rank = self.responsibility_index(expls,sort=False)
        holler_rank = self.holler_packel(expls,sort=False)
        deegan_rank = self.deegan_packel(expls,sort=False)

        norm_response = self.normalize(response_rank)
        norm_holler = self.normalize(holler_rank)
        norm_deegan = self.normalize(deegan_rank)

        basename = self.xgb.basename.strip("temp/").replace("/","_")
        expl_str = [[str(x) for x in expl ] for expl in expls]
        expl_str = ["-".join(expl) for expl in expl_str]
        explanations = ":".join(expl_str)

        resp_str = ";".join(map(str, norm_response))
        holler_str = ":".join(map(str, norm_holler))
        deegan_str = ":".join(map(str, norm_deegan))
        xvals = np.arange(norm_response.shape[0])
        xvals = [self.preamble[i] for i in xvals]
        x_vals_str = ":".join(xvals)
        preambles = ":".join(self.preamble)


        res = str(idx) + "," + x_vals_str + "," +  resp_str + "," + holler_str + "," + deegan_str + "," + explanations + "," + preambles + "\n"
        if writer is not None:
            if lock is not None:
                with lock:
                    writer.write(res)
                    writer.flush()
            else:
                writer.write(res)
                writer.flush()


        return res


        # plot_bar(xvals,norm_response,'Features','normalized response index',basename, str(idx)+"_response.png")
        #
        #
        # plot_bar(xvals, norm_deegan, 'Features', 'normalized deegan-packel index', basename,  str(idx)+"_deegan.png")
        #
        # plot_bar(xvals, norm_holler, 'Features', 'normalized holler-packel index', basename,  str(idx)+"_holler.png")



    def compute_all_minimal_expls(self, sample,writer=None,lock=None,idx=None,num_f=None,kwargs=None):
        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # adapt the solver to deal with the current sample
        if kwargs is not None:
            self.prepare_(sample,kwargs=kwargs)
        else:
            self.prepare(sample)


        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                    resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time


        visited = dict()
        minimal_coalitions = dict()
        queue = collections.deque([])
        for f_id in self.rhypos:
            coalition = [f_id]
            c_str, c_list = self.convert_coalition_to_str(coalition)

            if self.oracle.solve([self.selv] + coalition) is False:
                minimal_coalitions[c_str] = (coalition,c_list)
            else:
                queue.append(coalition)
            visited[c_str] = True
        printt=False
        while queue:
            c = queue.popleft()
            # print("new", c)
            for f_id in self.rhypos:
                if f_id not in c:
                    c_new = c + [f_id]
                    c_str, c_new_list = self.convert_coalition_to_str(c_new)
                    if visited.get(c_str) is None:

                        if (self.is_minimal(c_new_list, minimal_coalitions) is True):

                            # print(c_str)

                            start = datetime.now()
                            flag_ = self.oracle.solve([self.selv] +c_new)
                            end = datetime.now()
                            if flag_ == False:
                                minimal_coalitions[c_str] = (c_new, c_new_list)
                                print("minimal",c_str)
                            else:
                                if (num_f is not None):
                                    if (len(c_new)<num_f):

                                        queue.append(c_new)
                                        # print("****")
                                        # print(c_new)
                                        # print(c_str)
                                        visited[c_str]=True
                                        # print(visited.get(c_str))
                                        #
                                        #
                                        # print("*****")

                                else:
                                    queue.append(c_new)

                                    visited[c_str] = True


                            if printt is False:
                                print("time taken", (end - start) / 2)
                                printt = True

                        visited[c_str] = True


        expls = list([x[0] for x in minimal_coalitions.values()])
        print("expls",expls)

        # try:
        all_expls = [sorted([self.sel2fid[h] for h in expl]) for expl in expls]
        interpretable_expls = []
        for expl in all_expls:
            # preamble = [self.preamble[i] for i in expl]
            interpretable_expls.append((expl))
        res=None
        if self.verbose:
            # res_rank = self.responsibility_index(all_expls)
            # print("************important features in descending order (responsibility index)***********")
            # for key in res_rank:
            #     print(" Feature " + self.preamble[key[0]] + " " + " Score: " + str(key[1]))
            #
            # deegan_ranks = self.deegan_packel(all_expls)
            #
            # print("*********important features in descending order (deegan packel)**********")
            # for key in deegan_ranks:
            #     print(" Feature " + self.preamble[key[0]] + " " + " Score: " + str(key[1]))
            #
            # holler_ranks = self.holler_packel(all_expls)
            # print("**********important features in descending order (holler-packel)***********")
            # for key in holler_ranks:
            #     print(" Feature " + self.preamble[key[0]] + " " + " Score: " + str(key[1]))

            res = self.compare_index(all_expls,writer,lock,idx)

            for expl in all_expls:
                preamble = [self.preamble[i] for i in expl]
                print('  explanation: "IF {0} THEN {1}"'.format(' AND '.join(preamble),
                                                                self.xgb.target_name[self.out_id]))

                print('  # hypos left:', len(expl))
                print('  time: {0:.2f}'.format(self.time))


        return (interpretable_expls,res)

    def explain(self, sample, smallest, expl_ext=None, prefer_ext=False):
        """
            Hypotheses minimization.
        """

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # adapt the solver to deal with the current sample
        self.prepare(sample)

        # saving external explanation to be minimized further
        if expl_ext == None or prefer_ext:
            self.to_consider = [True for h in self.rhypos]
        else:
            eexpl = set(expl_ext)
            self.to_consider = [True if i in eexpl else False for i, h in enumerate(self.rhypos)]

        # if satisfiable, then the observation is not implied by the hypotheses
        if self.oracle.solve([self.selv] + [h for h, c in zip(self.rhypos, self.to_consider) if c]):
            print('  no implication!')
            print(self.oracle.get_model())
            sys.exit(1)

        if smallest:
            self.compute_smallest()

        else:
            self.compute_minimal(prefer_ext=prefer_ext)

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        expl = sorted([self.sel2fid[h] for h in self.rhypos])

        if self.verbose:
            self.preamble = [self.preamble[i] for i in expl]
            print('  explanation: "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.xgb.target_name[self.out_id]))
            print('  # hypos left:', len(self.rhypos))
            print('  time: {0:.2f}'.format(self.time))

        return expl

    def compute_minimal(self, prefer_ext=False):
        """
            Compute any subset-minimal explanation.
        """

        i = 0

        if not prefer_ext:
            # here, we want to reduce external explanation

            # filtering out unnecessary features if external explanation is given
            self.rhypos = [h for h, c in zip(self.rhypos, self.to_consider) if c]
        else:
            # here, we want to compute an explanation that is preferred
            # to be similar to the given external one
            # for that, we try to postpone removing features that are
            # in the external explanation provided

            rhypos  = [h for h, c in zip(self.rhypos, self.to_consider) if not c]
            rhypos += [h for h, c in zip(self.rhypos, self.to_consider) if c]
            self.rhypos = rhypos

        # simple deletion-based linear search
        while i < len(self.rhypos):
            to_test = self.rhypos[:i] + self.rhypos[(i + 1):]

            if self.oracle.solve([self.selv] + to_test):
                i += 1
            else:
                self.rhypos = to_test


    def compute_smallest(self):
        """
            Compute a cardinality-minimal explanation.
        """

        # result
        rhypos = []

        with Hitman(bootstrap_with=[[i for i in range(len(self.rhypos)) if self.to_consider[i]]]) as hitman:
            # computing unit-size MCSes
            for i, hypo in enumerate(self.rhypos):
                if self.to_consider[i] == False:
                    continue

                if self.oracle.solve([self.selv] + self.rhypos[:i] + self.rhypos[(i + 1):]):
                    hitman.hit([i])

            # main loop
            iters = 0
            while True:
                hset = hitman.get()
                iters += 1

                if self.verbose > 1:
                    print('iter:', iters)
                    print('cand:', hset)

                if self.oracle.solve([self.selv] + [self.rhypos[i] for i in hset]):
                    to_hit = []
                    satisfied, unsatisfied = [], []

                    removed = list(set(range(len(self.rhypos))).difference(set(hset)))

                    model = self.oracle.get_model()
                    for h in removed:
                        i = self.sel2fid[self.rhypos[h]]
                        if '_' not in self.inps[i].symbol_name():
                            # feature variable and its expected value
                            var, exp = self.inps[i], self.sample[i]

                            # true value
                            true_val = float(model.get_py_value(var))

                            if not exp - 0.001 <= true_val <= exp + 0.001:
                                unsatisfied.append(h)
                            else:
                                hset.append(h)
                        else:
                            for vid in self.sel2vid[self.rhypos[h]]:
                                var, exp = self.inps[vid], int(self.sample[vid])

                                # true value
                                true_val = int(model.get_py_value(var))

                                if exp != true_val:
                                    unsatisfied.append(h)
                                    break
                            else:
                                hset.append(h)

                    # computing an MCS (expensive)
                    for h in unsatisfied:
                        if self.oracle.solve([self.selv] + [self.rhypos[i] for i in hset] + [self.rhypos[h]]):
                            hset.append(h)
                        else:
                            to_hit.append(h)

                    if self.verbose > 1:
                        print('coex:', to_hit)

                    hitman.hit(to_hit)
                else:
                    self.rhypos = [self.rhypos[i] for i in hset]
                    break
