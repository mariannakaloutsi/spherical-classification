import time
import numpy as np
import pandas as pd
import copy
from scipy.spatial import distance, distance_matrix
from operator import add
from random import random
from mip import Model, xsum, minimize, BINARY
from sklearn.metrics import confusion_matrix, classification_report
import math
from numpy.linalg import norm
from sklearn.feature_selection import mutual_info_regression, f_classif, chi2, mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef

h_limit = 0.7
r = 4
e = 0.005

####### START COUNTING

start = time.process_time()


##### function creating a list with number of elements defined by user ######
def list_creation(list_length):
    list_name = [[] for i in range(len(list_length))]
    return list_name


##### function calculating distances between a center(certain point) and a list #####
def distance_center(center, points, list_label):
    dist_b = list_creation(list_label)
    j = 0
    for i in list_label:
        dist_b[j] = distance.euclidean(center, points[i])
        j = j + 1
    return dist_b


##### function calculating distances between a center and a list
##### and appending to list if value smaller than a min value
def distance_cond_min(data_length, center, points, check_value, adding_list):
    dist_r = list_creation(data_length)
    j = 0
    for i in data_length:
        dist_r[j] = distance.euclidean(center, points[i])
        if dist_r[j] < check_value:
            adding_list.append(i)
        j = j + 1

    return dist_r, adding_list


##### function performing steps of STAGE 1 #######
def stage_1(data1, data2, cluster, feature_list_1, feature_list_2):
    p_list = []
    for s in range(len(data1)):
        cluster[s][0] = feature_list_1[s]
        dist_b = distance_center(cluster[s][0], feature_list_2, data2.index)
        cluster[s][1] = min(dist_b)
        dist_r, cluster[s][4] = distance_cond_min(data1.index, cluster[s][0], feature_list_1, cluster[s][1],
                                                  cluster[s][4])

        dist_r_filt = []
        for m in cluster[s][4]:
            dist_r_filt.append(dist_r[m])

        if len(cluster[s][4]) > 1:
            cluster[s][2] = max(dist_r_filt)
        else:
            cluster[s][2] = cluster[s][1] / 2

        p_list.append(s)

    return cluster, p_list


###### function for creating a cluster ######
def cluster_creation(data_source):
    cluster = list_creation(data_source)
    for i in range(len(data_source)):
        cluster[i] = [[], 0, 0, 1, [], []]

    return cluster


###### function for creating J lists ######
def j_creation(data_source):
    j = list_creation(data_source)
    for i in range(len(data_source)):
        j[i] = [i]

    return j


###### function that checks if item of a list A is not in list B and append the distance i to a new list
def check_append(origin_list, check_list, dist_b, new_list):
    for i in origin_list.index:
        if i not in check_list:
            new_list.append(dist_b[i])
    return new_list


###### function for calculation of lamda
def lamda_calc(x_u, x_w, d, center_new):
    v1 = [x1 - x2 for (x1, x2) in zip([sum(x) for x in zip(x_u / 2, x_w / 2)], center_new)]
    v2 = [x1 - x2 for (x1, x2) in zip(x_u, x_w)]
    lamda = sum(p * q for p, q in zip(v1, v2)) / sum(p * q for p, q in zip(d, v2))
    return lamda


###### fuction with basic cluster calculations
def newcluster_calc(center_new, feature_1, feature_2, data1, data2, list_label):
    dist_r = distance_center(center_new, feature_1, list_label)
    r_new = max(dist_r)  ####find new r

    m2_new = []
    dist_b, m2_new = distance_cond_min(data2.index, center_new, feature_2, r_new, m2_new)  #### find new  m2

    new_dist = []
    new_dist = check_append(data2, m2_new, dist_b, new_dist)
    if new_dist != []:
        b_new = min(new_dist)
    else:
        b_new = r_new * 1.5  ############################find new b

    m1_new = []
    dist_b, m1_new = distance_cond_min(data1.index, center_new, feature_1, b_new, m1_new)  #### find new  m1
    h = len(m1_new) / (len(m1_new) + len(m2_new))  # define h

    return b_new, r_new, h, m1_new, m2_new


####### function calculating position of max or min value (input 1 for min, 0 for max )
def dist_position(center_new, points, data_source, func):
    dist = distance_center(center_new, points, data_source)
    if func == 1:
        w_pos = np.where(dist == np.amin(dist))
        p = data_source[w_pos[-1][0]]
    # elif func==1:
    else:
        w_pos = np.where(dist == np.amax(dist))
        p = data_source[w_pos[0][0]]
    return p


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


######## function calculating stage 2
def stage_2_calc(data1, data2, C, C_op, s_coord1, s_coord2, P):
    C_stage2 = np.zeros(shape=(len(data1), 6), dtype=list)

    nonsingleton_s = []
    for i in range(len(data1)):
        if len(C[i][4]) > 1:
            nonsingleton_s.append(i)

    for s in nonsingleton_s:
        add = []
        for fe in range(len(C[s][0])):
            add.append(0)

        center_new = []
        for fe in range(len(C[s][0])):
            for i in C[s][4]:
                add[fe] = add[fe] + (C[i][0][fe])
            center_new.append(add[fe] / len(C[s][4]))  ####### find new center

        b_new, r_new, h, m1_new, m2_new = newcluster_calc(center_new, s_coord1, s_coord2, data1, data2, C[s][4])

        if len(m2_new) != 0:
            counter = 0
            while h < 1 and counter < 10:
                ### finding new center based on (11)-(13)
                d = [[] for fe in range(len(C[s][0]))]
                for fe in range(len(C[s][0])):
                    d[fe] = s_coord1[s][fe] - center_new[fe]
                u = dist_position(center_new, s_coord1, m1_new, 0)

                avail = []
                for sa in m2_new:
                    d_w = list_creation(C[s][0])
                    for fe in range(len(C[s][0])):
                        d_w[fe] = C_op[sa][0][fe] - center_new[fe]

                    if np.dot(d, d_w) / norm(d) / norm(d_w) <= 0:
                        avail.append(sa)

                if avail == []:
                    avail = m2_new

                dist = distance_center(center_new, s_coord2, avail)
                w_pos = np.where(dist == np.amin(dist))
                w = avail[w_pos[-1][0]]

                ###########
                # lamda and new center calculation based on equation (12) and (13)
                lamda = lamda_calc(C[u][0], C_op[w][0], d, center_new)
                center_new = [sum(x) for x in
                              zip([(lamda + e) * (x1 - x2) for (x1, x2) in zip(C[s][0], center_new)], center_new)]

                if np.isnan(center_new).any():
                    center_new = []
                    for fe in range(len(C[s][0])):
                        for i in C[s][4]:
                            add[fe] = add[fe] + (C[i][0][fe])
                        center_new.append(add[fe] / len(C[s][4]))

                #### find new r,M2 and b

                b_new, r_new, h, m1_new, m2_new = newcluster_calc(center_new, s_coord1, s_coord2, data1, data2, m1_new)
                counter = counter + 1

            if len(set(m1_new) - set(C[s][4])) != 0:  ###check if M1 has changed to update r again
                dist_r = distance_center(center_new, s_coord1, m1_new)
                r_new = max(dist_r)

        for (fe, i) in zip([center_new, b_new, r_new, len(m1_new) / (len(m1_new) + len(m2_new)), m1_new, m2_new],
                           range(6)):
            C_stage2[s][i] = fe

    for s in P:
        if s not in nonsingleton_s:
            for fe in range(6):
                C_stage2[s][fe] = C[s][fe]

    return C_stage2, P


####### random list generator
def gam_generator(r):
    gam = [[] for i in range(r)]
    #### generate a(i)
    for i in range(r):
        gam[i] = random()
    gam = gam + [1]
    gam.sort()
    return gam


######### finding the center of a cluster based on the coordinates of the included samples
def center_add(s, coord, length):
    add = []
    for fe in range(len(coord[s])):
        add.append(0)
    center_new = []
    for fe in range(len(coord[s])):
        for i in length:
            add[fe] = add[fe] + coord[i][fe]
        center_new.append(add[fe] / len(length))
    return center_new


####### stage 3 calculation function
def stage_3_func(data1, data2, C_stage2, C_op_stage2, s_coord1, s_coord2):
    C_stage3 = np.zeros(shape=(len(C_stage2), 6), dtype=list)
    gam = gam_generator(r)

    for s in data1.index:
        for fe in range(6):
            C_stage3[s][fe] = C_stage2[s][fe]
        counter = 0
        alpha = 1
        expansion = False
        for g in gam:
            b_new = (1 + alpha * g) * C_stage3[s][1]

            m1_new = [s]
            dist_b, m1_new = distance_cond_min(data1.index, C_stage3[s][0], s_coord1, b_new, m1_new)
            center_new = center_add(s, s_coord1, m1_new)

            dist_r = distance_center(center_new, s_coord1, m1_new)
            r_new = max(dist_r)

            m2_new = []
            dist_b, m2_new = distance_cond_min(data2.index, center_new, s_coord2, r_new, m2_new)

            if len(m1_new) / (len(m1_new) + len(m2_new)) > h_limit and len(m1_new) > len(C_stage3[s][4]):

                new_dist = []
                new_dist = check_append(data2, m2_new, dist_b, new_dist)
                if new_dist != []:
                    b_new = min(new_dist)
                else:
                    b_new = 2 * r_new

                m1_new = [s]
                dist_b, m1_new = distance_cond_min(data1.index, center_new, s_coord1, b_new, m1_new)
                h_new = len(m1_new) / (len(m1_new) + len(m2_new))

                for i, fe in zip(range(6), [center_new, b_new, r_new, h_new, m1_new, m2_new]):
                    C_stage3[s][i] = fe
                expansion = True
                break

        if expansion == True:

            while expansion == True and counter < 10:
                counter = counter + 1
                expansion = False
                for g in gam:
                    b_new = (1 + alpha * g) * C_stage3[s][1]

                    m1_new = [s]
                    dist_b, m1_new = distance_cond_min(data1.index, C_stage3[s][0], s_coord1, b_new, m1_new)
                    center_new = center_add(s, s_coord1, m1_new)

                    dist_r = distance_center(center_new, s_coord1, m1_new)
                    r_new = max(dist_r)

                    m2_new = []
                    dist_b, m2_new = distance_cond_min(data2.index, center_new, s_coord2, r_new, m2_new)

                    if len(m1_new) / (len(m1_new) + len(m2_new)) > h_limit and len(m1_new) > len(C_stage3[s][4]):

                        new_dist = []
                        new_dist = check_append(data2, m2_new, dist_b, new_dist)
                        if new_dist != []:
                            b_new = min(new_dist)
                        else:
                            b_new = 2 * r_new

                        m1_new = [s]
                        dist_b, m1_new = distance_cond_min(data1.index, center_new, s_coord1, b_new, m1_new)
                        h_new = len(m1_new) / (len(m1_new) + len(m2_new))

                        for i, fe in zip(range(6), [center_new, b_new, r_new, h_new, m1_new, m2_new]):
                            C_stage3[s][i] = fe
                        expansion = True

        if expansion == False:
            expansion = True
            while expansion == True and counter < 10:
                counter = counter + 1
                alpha = alpha / 2
                expansion = False
                for g in gam:
                    b_new = (1 + alpha * g) * C_stage3[s][1]

                    m1_new = [s]
                    dist_b, m1_new = distance_cond_min(data1.index, C_stage3[s][0], s_coord1, b_new, m1_new)
                    center_new = center_add(s, s_coord1, m1_new)

                    dist_r = distance_center(center_new, s_coord1, m1_new)
                    r_new = max(dist_r)

                    m2_new = []
                    dist_b, m2_new = distance_cond_min(data2.index, center_new, s_coord2, r_new, m2_new)

                    if len(m1_new) / (len(m1_new) + len(m2_new)) > h_limit and len(m1_new) > len(C_stage3[s][4]):

                        new_dist = []
                        new_dist = check_append(data2, m2_new, dist_b, new_dist)
                        if new_dist != []:
                            b_new = min(new_dist)
                        else:
                            b_new = 2 * r_new

                        m1_new = [s]
                        dist_b, m1_new = distance_cond_min(data1.index, center_new, s_coord1, b_new, m1_new)
                        h_new = len(m1_new) / (len(m1_new) + len(m2_new))

                        for i, fe in zip(range(6), [center_new, b_new, r_new, h_new, m1_new, m2_new]):
                            C_stage3[s][i] = fe
                        expansion = True
    return C_stage3


######## Singleton Treatment Stage Calculations
def singleton_calc(data, data_op, cluster, cluster_op, s_coord, s_coord_op, J, P):
    for s in data.index:
        if len(cluster[s][4]) == 1:
            if len(J[s]) >= 2:
                P[s] = 'deleted'

            else:
                dist_k = [1e5 for i in range(len(data))]
                for i in data.index:
                    if i != s:
                        dist_k[i] = distance.euclidean(s_coord[s], s_coord[i])

                u = np.argmin(dist_k)
                center_new = [(x_u + x_i) / 2 for (x_i, x_u) in zip(s_coord[s], s_coord[u])]
                dist_r = distance_center(center_new, s_coord, data.index)
                r_new = min(dist_r)

                m2_new = []
                dist_b, m2_new = distance_cond_min(data_op.index, center_new, s_coord_op, r_new, m2_new)

                new_dist = []
                new_dist = check_append(data_op, m2_new, dist_b, new_dist)
                b_new = min(new_dist)

                m1_new = [s]
                dist_b, m1_new = distance_cond_min(data.index, center_new, s_coord, b_new, m1_new)

                h_new = len(m1_new) / (len(m1_new) + len(m2_new))

                if h_new >= h_limit:
                    for i, fe in zip(range(6), [center_new, b_new, r_new, h_new, m1_new, m2_new]):
                        cluster[s][i] = fe
                else:
                    P[s] = 'deleted'

    return pd.DataFrame(cluster), P


####### calculation of the necessary weights for stage5
def weight_calc(cluster, data, P):
    weight = np.zeros(shape=(len(cluster), 1))  #########
    i = 0
    for s in P:
        weight[s] = cluster[s][2] / len(cluster[s][4]) / cluster[s][3]
        i = i + 1
    return weight


####### calculating the factor matrix for linear optimization
def matrix_calc(cluster, P, data):
    matrix = np.zeros(shape=(len(P), len(cluster)))  ##########

    for c in range(len(cluster)):
        for s, i in zip(P, range(len(P))):
            if s in cluster[c][4]:
                matrix[i][c] = 1

    return matrix


# finding the index of the samples that belong only in one cluster
# in order to addd constraint to retain that cluster so that the sample is not lost
def single_sample_count(data, P, cluster):
    counter = [0 for i in range(len(data))]
    ind = []
    for s in P:
        counter[s] = 0
        for c in P:
            if s in cluster.transpose()[4][c]:
                counter[s] = counter[s] + 1
    if counter.count(1) > 0:
        ind = counter.index(1)

    return counter, ind


###### finding the L and C lists for point classification
def cluster_find(new_point, cluster):
    dist, l, c = ([], [], [])
    for cl in range(len(cluster)):
        dist.append(distance.euclidean(new_point, cluster.values[cl][0]))
        if dist[cl] < cluster.values[cl][1]:
            l.append(cluster.index[cl])
        if dist[cl] < cluster.values[cl][2]:
            c.append(cluster.index[cl])
    return l, c


###### calculating the sum for case 2 of new point classification
def sum_calc(w, new_point, cluster, l, c):
    sum_1 = sum(w[s] * (distance.euclidean(new_point, cluster.values[j][0]) - cluster.values[j][2])
                for s, j in zip(cluster.index, range(len(cluster))))
    temp_list = []
    for s in l:
        if s not in c:
            temp_list.append(s)
    sum_2 = sum(w[s] * (distance.euclidean(new_point, cluster.values[j][0]) - cluster.values[j][1])
                for s, j in zip(temp_list, range(len(temp_list))))
    if sum_1 != 0 and sum_2 != 0:
        s_1 = 1 / sum_1 + 1 / sum_2
    elif sum_1 == 0:
        s_1 = 1 / sum_2
    else:
        s_1 = 1 / sum_1
    return s_1


def kenStone(X, Y, train_ratio):
    k = round(train_ratio * len(X))
    originalX = X

    train_idx = list()

    distA = pd.DataFrame(distance_matrix(X, X), index=X.index, columns=X.index)
    obj1, obj2 = distA.stack().index[np.argmax(distA.values)]
    train_idx.append(obj1)  # add 1st object to list
    train_idx.append(obj2)  # add 2nd object to list
    X = X.drop(obj1)  # remove 1st object from initial dataset
    X = X.drop(obj2)  # remove 2nd object from initial dataset

    for i in range(0, k - 2):
        distB = pd.DataFrame(distance_matrix(X, pd.DataFrame(originalX.loc[train_idx, :])), index=X.index,
                             columns=train_idx)
        closer = list()  # pd.DataFrame([pd.DataFrame([min(distB.iloc[0,])])])

        for j in range(0, len(X)):
            closer.append(min(distB.iloc[j,]))  # (pd.DataFrame([min(distB.iloc[j,])]))

        obj = X.index[closer.index(max(closer))]
        train_idx.append(obj)
        X = X.drop(obj)

    trainX = originalX.loc[train_idx,]
    trainY = Y.loc[train_idx,]

    test_idx = X.index
    testX = originalX.loc[test_idx,]
    testY = Y.loc[test_idx,]

    return trainX, testX, trainY, testY


def cluster_selection(cluster, w, mat, ind, P, J, solver_name):
    C = range(len(cluster))  #####
    C_single = range(len(cluster))  ######
    m = Model(solver_name=solver_name)

    y = [m.add_var(var_type=BINARY) for c in C]
    y_single = [m.add_var(var_type=BINARY) for c_single in C_single]

    ###objective function
    m.objective = minimize(xsum(w[c][0] * y[c] for c in C))

    #####add constraints
    for con in range(len(mat.transpose())):
        m.add_constr(xsum(mat.transpose()[con][c] * y[c] for c in C) >= 1)
    for c_single in C_single:
        m.add_constr(y_single[c_single] >= 1)
    if ind != []:
        for j in [ind]:
            m.add_constr(y[j] == y_single[j])
    for s in P:
        m.add_constr(xsum(y[c] for c in J[s]) >= 1)

    m.optimize()
    y = [y[c].x for c in C]
    return m.objective_value, y, m.status


def cluster_filtering(cluster, y):
    cluster_f = pd.DataFrame(cluster)
    cluster_f['label'] = y
    cluster_f = cluster_f[cluster_f['label'] == 1]
    return cluster_f


def j_calculation(data, P, cluster):
    J = list_creation(data)
    for s in data.index:
        for sa in P:
            if s in cluster[sa][4]:
                J[s].append(sa)
    return J


def j_calculation2(data, P, cluster):
    J = list_creation(data)
    for s in data.index:
        for sa in P:
            if s in cluster[4][sa]:  ################
                J[s].append(sa)
    return J


def print_stats(y_t, y_pred):
    conf_mat = confusion_matrix(y_t.values, y_pred)
    total = sum(sum(conf_mat))
    accuracy = (conf_mat[0, 0] + conf_mat[1, 1]) / total
    print('Accuracy : ', accuracy)
    sensitivity = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    print('Sensitivity : ', sensitivity)
    specificity = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1])
    print('Specificity : ', specificity)
    mat_coef = matthews_corrcoef(y_t, y_pred)
    print('MAtthews coeef : ', mat_coef)
    print(conf_mat)


def stages(data_1, data_2, solver_name, X_test, pred):
    ## create clusters
    C_1 = cluster_creation(data_1)
    C_2 = cluster_creation(data_2)

    #####create arrays for the coordinates/features
    s_coord_1 = list_creation(data_1)
    for s in range(len(data_1)):
        s_coord_1[s] = data_1.values[s]

    s_coord_2 = list_creation(data_2)
    for s in range(len(data_2)):
        s_coord_2[s] = data_2.values[s]

    C_1, P_1 = stage_1(data_1, data_2, C_1, s_coord_1, s_coord_2)
    C_2, P_2 = stage_1(data_2, data_1, C_2, s_coord_2, s_coord_1)

    C_1_stage2, P_1 = stage_2_calc(data_1, data_2, C_1, C_2, s_coord_1, s_coord_2, P_1)
    C_2_stage2, P_2 = stage_2_calc(data_2, data_1, C_2, C_1, s_coord_2, s_coord_1, P_2)

    C_1_stage3 = stage_3_func(data_1, data_2, C_1_stage2, C_2_stage2, s_coord_1, s_coord_2)
    C_2_stage3 = stage_3_func(data_2, data_1, C_2_stage2, C_1_stage2, s_coord_2, s_coord_1)

    J_1 = j_calculation(data_1, P_1, C_1_stage3)
    J_2 = j_calculation(data_2, P_2, C_2_stage3)

    C_1_stage4 = np.zeros(shape=(len(C_1), 6), dtype=list)
    C_2_stage4 = np.zeros(shape=(len(C_2), 6), dtype=list)
    for sa in data_1.index:
        for fe in range(6):
            C_1_stage4[sa][fe] = C_1_stage3[sa][fe]
    for sa in data_2.index:
        for fe in range(6):
            C_2_stage4[sa][fe] = C_2_stage3[sa][fe]

    C_1_stage4, P_1 = singleton_calc(data_1, data_2, C_1_stage3, C_2_stage3, s_coord_1, s_coord_2, J_1, P_1)
    C_2_stage4, P_2 = singleton_calc(data_2, data_1, C_2_stage3, C_1_stage3, s_coord_2, s_coord_1, J_2, P_2)

    while 'deleted' in P_1:
        P_1.remove('deleted')

    while 'deleted' in P_2:
        P_2.remove('deleted')

    J_1 = j_calculation2(data_1, P_1, C_1_stage4)
    J_2 = j_calculation2(data_2, P_2, C_2_stage4)

    C_1_stage4 = C_1_stage4.values
    C_2_stage4 = C_2_stage4.values

    w_1 = weight_calc(C_1_stage4, data_1, P_1)
    w_2 = weight_calc(C_2_stage4, data_2, P_2)

    mat_1 = matrix_calc(C_1_stage4, data_1.index, P_1)
    mat_2 = matrix_calc(C_2_stage4, data_2.index, P_2)

    counter_1, ind_1 = single_sample_count(data_1, P_1, C_1_stage4)
    counter_2, ind_2 = single_sample_count(data_2, P_2, C_2_stage4)

    val_1, y_1, hi = cluster_selection(C_1_stage4, w_1, mat_1, ind_1, P_1, J_1, solver_name)
    val_2, y_2, hi2 = cluster_selection(C_2_stage4, w_2, mat_2, ind_2, P_2, J_2, solver_name)

    cluster_1 = cluster_filtering(C_1_stage4, y_1).drop('label', axis=1)
    cluster_2 = cluster_filtering(C_2_stage4, y_2).drop('label', axis=1)

    l_1 = list_creation(X_test)
    c_1 = list_creation(X_test)
    l_2 = list_creation(X_test)
    c_2 = list_creation(X_test)

    y_pred = []
    if 1 in pred:

        for s, i in zip(X_test.values, range(len(X_test))):
            l_1, c_1 = cluster_find(s, cluster_1)
            l_2, c_2 = cluster_find(s, cluster_2)

            if bool(l_1 == []) ^ bool(l_2 == []):
                if l_1 == []:
                    y_pred.append(2)  ###### add new observation to the '2' classification
                else:
                    y_pred.append(1)  ###### add new observation to the '1' classification
            elif l_1 != [] and l_2 != []:
                if bool(c_1 == []) ^ bool(c_2 == []):
                    if c_1 == []:
                        y_pred.append(2)  ###### add new observation to the '2' classification
                    else:
                        y_pred.append(1)  ###### add new observation to the '1' classification
                else:
                    s_1 = sum_calc(w_1, s, cluster_1, l_1, c_1)
                    s_2 = sum_calc(w_2, s, cluster_2, l_2, c_2)
                    if s_1 >= s_2:
                        y_pred.append(1)  ######## add new observation to the '1' classification
                    else:
                        y_pred.append(2)  ######## add new observation to the '2' classification
            else:
                d_1 = [(distance.euclidean(s, cluster_1.values[j][0]) - cluster_1.values[j][2]) for j in
                       range(len(cluster_1))]
                d_2 = [(distance.euclidean(s, cluster_2.values[j][0]) - cluster_2.values[j][2]) for j in
                       range(len(cluster_2))]
                d_sum = d_1 + d_2
                d_sum.sort()
                if d_sum[0] in d_1:
                    if d_sum[1] in d_1:
                        y_pred.append(1)  ######## add new observation to the '1' classification
                    else:
                        s_1 = [1 / (w_1[c] * (distance.euclidean(s, cluster_1.values[j][0]) - cluster_1.values[j][2]))
                               for c, j in zip(cluster_1.index, range(len(cluster_1)))][d_1.index(d_sum[0])]

                        s_2 = [1 / (w_2[c] * (distance.euclidean(s, cluster_2.values[j][0]) - cluster_2.values[j][2]))
                               for c, j in zip(cluster_2.index, range(len(cluster_2)))][d_2.index(d_sum[1])]
                        if s_1 > s_2:
                            y_pred.append(1)  ######## add new observation to the '1' classification
                        else:
                            y_pred.append(2)  ######## add new observation to the '2' classification

                if d_sum[0] in d_2:
                    if d_sum[1] in d_2:
                        pass  ######## add new observation to the '2' classification
                    else:
                        s_1 = [1 / (w_1[c] * (distance.euclidean(s, cluster_1.values[j][0]) - cluster_1.values[j][2]))
                               for c, j in zip(cluster_1.index, range(len(cluster_1)))][d_2.index(d_sum[0])]

                        s_2 = [1 / (w_2[c] * (distance.euclidean(s, cluster_2.values[j][0]) - cluster_2.values[j][2]))
                               for c, j in zip(cluster_2.index, range(len(cluster_2)))][d_1.index(d_sum[1])]
                        if s_2 > s_1:
                            y_pred.append(1)  ######## add new observation to the '1' classification
                        else:
                            y_pred.append(2)  ######## add new observation to the '2' classification

    y_pred_1 = []
    if 2 in pred:
        for s in X_test.index:
            dist_1 = min([distance.euclidean(X_test.loc[s], cluster_1[0].values[c]) for c in range(len(cluster_1))])
            dist_2 = min([distance.euclidean(X_test.loc[s], cluster_2[0].values[c]) for c in range(len(cluster_2))])
            if dist_1 < dist_2:
                y_pred_1.append(1)
            else:
                y_pred_1.append(2)

    y_pred_2 = []
    if 3 in pred:
        for s in X_test.index:
            dist_1 = sum([1 / (1 + distance.euclidean(X_test.loc[s], cluster_1[0].values[c])) for c in
                          range(len(cluster_1))]) / len(cluster_1)
            dist_2 = sum([1 / (1 + distance.euclidean(X_test.loc[s], cluster_2[0].values[c])) for c in
                          range(len(cluster_2))]) / len(cluster_2)
            if dist_1 < dist_2:
                y_pred_2.append(2)
            else:
                y_pred_2.append(1)

    y_p = [[] for i in range(3)]
    y_p[0] = y_pred
    y_p[1] = y_pred_1
    y_p[2] = y_pred_2

    y_pr = [ele for ele in y_p if ele != []]
    return y_pr


def spherical(data_file,
              endpoint_col,
              t_ratio=0.7,
              feat_sel=False,
              print_report=True,
              solver_name='cbc',
              pred=[1, 2, 3],
              feat_num=5,
              feat_func=chi2):
    data = pd.read_csv(data_file, header=0, sep=",", encoding='cp1252')
    data.columns = range(len(data.columns))
    X = pd.DataFrame(data).drop([0], axis=1)

    X_scaled = (X - X.min()) / (X.max() - X.min())
    X_train, X_test, y_train, y_test = kenStone(X_scaled.drop(endpoint_col, axis=1), X[endpoint_col], t_ratio)

    if feat_sel == True:
        model = SelectKBest(feat_func, k=feat_num).fit(X_train, y_train)
        X_new = pd.DataFrame(model.transform(X_train))
        X_new['Toxicity'] = data[endpoint_col]
        X_train = X_new

        data_1 = X_train[X_train['Toxicity'] == 1].drop(['Toxicity'], axis=1)
        data_2 = X_train[X_train['Toxicity'] == 2].drop(['Toxicity'], axis=1)
        data_1.index = range(len(data_1))
        data_2.index = range(len(data_2))

        X_test = model.transform(X_test)
        X_test = pd.DataFrame(X_test)
        X_test.index = range(len(X_test))
    else:
        X_train[endpoint_col] = data[endpoint_col]
        data_1 = X_train[X_train[endpoint_col] == 1].drop([endpoint_col], axis=1)
        data_2 = X_train[X_train[endpoint_col] == 2].drop([endpoint_col], axis=1)
        data_1.index = range(len(data_1))
        data_2.index = range(len(data_2))
        X_test.index = range(len(X_test))

    y_pred = stages(data_1, data_2, solver_name, X_test, pred)

    if print_report == True:
        for i in range(len(pred)):
            print('Statistical performance for method:', pred[i])
            print_stats(y_test, y_pred[i])

    return y_pred