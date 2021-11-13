from math import sqrt
from numpy.core.fromnumeric import transpose
from scipy.stats import multivariate_normal
from numpy import zeros, matrix, subtract, dot

# Warning! Don't take names too seriously

FULL_DEBUG = False

data = [[ 2,  4],
        [-1, -4],
        [-1,  2],
        [ 4,  0]]

beta_mean_1 = [2, 4]

beta_mean_2 = [-1, -4]

sigma_male_1= [[1, 0],
               [0, 1]]

sigma_male_2= [[2, 0],
               [0, 2]]

prior_grindset_1 = 0.7
prior_grindset_2 = 0.3

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def calculate_probs(i):
    x = data[i]
    likelihood_1 =  multivariate_normal.pdf(x, beta_mean_1, sigma_male_1)
    likelihood_2 =  multivariate_normal.pdf(x, beta_mean_2, sigma_male_2)
    joint_1 = likelihood_1 * prior_grindset_1 / (likelihood_1 * prior_grindset_1 + \
                    likelihood_2 * prior_grindset_2)
    joint_2 = likelihood_2 * prior_grindset_2 / (likelihood_1 * prior_grindset_1 + \
                    likelihood_2 * prior_grindset_2)

    print("For x{} -> ({}, {}):".format(i+1, data[i][0], data[i][1]))
    print("\tFor cluster 1:")
    print("\t\tNormal prob: {:.6}".format(likelihood_1))
    print("\t\tP(x{} | c1) : {:.6}".format(i+1, likelihood_1 * prior_grindset_1))
    print("\t\tP(x{}) : {:.6}".format(i+1, likelihood_1 * prior_grindset_1 + likelihood_2 * prior_grindset_2))
    print("\t\tP(c1 | x{}) : {:.6}".format(i+1, joint_1))
    print("\tFor cluster 2:")
    print("\t\tNormal prob: {:.6}".format(likelihood_2))
    print("\t\tP(x{} | c2) : {:.6}".format(i+1, likelihood_2 * prior_grindset_2))
    print("\t\tP(x{}) : {:.6}".format(i+1, likelihood_1 * prior_grindset_1 + likelihood_2 * prior_grindset_2))
    print("\t\tP(c2 | x{}) : {:.6}".format(i+1, joint_2))
    print()

    return [joint_1, joint_2]


def weighted_mean(k, cluster_prob):
    #               y1  y2
    wgted_sum_obs = [0, 0]  # acumulates the value of xi times P(ck | xi)

    sum_prob_ck = 0

    for i in range(4):
        # add xi1 times P(c1 | xi) to weighted sum
        wgted_sum_obs[0] += data[i][0] * cluster_prob[i][k]
        # add xi2 times P(c1 | xi) to weighted sum
        wgted_sum_obs[1] += data[i][1] * cluster_prob[i][k]

        # acumulate probabilities for weighted mean
        sum_prob_ck += cluster_prob[i][k]

    return [wgted_sum_obs[0] / sum_prob_ck, wgted_sum_obs[1] / sum_prob_ck]


def weighted_cov_matrix(k, cluster_mean, cluster_prob):
    mat_sum = matrix(zeros((2, 2)))
    sum_probs = 0

    print("Cluster {}".format(k+1))
    for i in range(4):
        x_minus_mean = subtract(matrix(data[i]), matrix(cluster_mean))
        mat_sum += (transpose(x_minus_mean) * cluster_prob[i][k]) @ x_minus_mean
        sum_probs += cluster_prob[i][k]

        if FULL_DEBUG:
            print("\tObservacao {}:".format(i+1))
            print("\t\tx{}:\t\t".format(i+1) + str(matrix(data[i])).replace('\n', '\n\t\t'))
            print("\n\t\tx - mean:\t" + str(x_minus_mean).replace('\n', '\n\t\t\t\t'))
            print("\n\t\t[x - mean] * [x - mean]T * P(c = {} | x{}):".format(k+1, i+1))
            print("\t\t\t\t" + str((transpose(x_minus_mean) * cluster_prob[i][k]) @ x_minus_mean).replace('\n', '\n\t\t\t\t'))

    sigma = mat_sum / sum_probs
    print("\tNew covariance matrix:")
    print("\t\t" + str(sigma).replace('\n', '\n\t\t'))

    return sigma


def reevaluate(cluster_prob):
    new_mean_c1 = weighted_mean(0, cluster_prob)
    new_mean_c2 = weighted_mean(1, cluster_prob)

    print("New centroid for cluster 1:  [{: .6f}, {: .6f}]".format(new_mean_c1[0], new_mean_c1[1]))
    print("New centroid for cluster 2:  [{: .6f}, {: .6f}]\n".format(new_mean_c2[0], new_mean_c2[1]))

    weighted_cov_matrix(0, new_mean_c1, cluster_prob)
    weighted_cov_matrix(1, new_mean_c2, cluster_prob)


def dist(a, b):
    return sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) **  2)


def get_a(obs_i, clusters):
    if obs_i in clusters[0]:
        own = clusters[0]
    elif obs_i in clusters[1]:
        own = clusters[1]
    else:
        raise ValueError("An internal error has occured")

    sum_dist = 0
    count = 0

    for obs_in_own in own:
        if obs_in_own == obs_i:
            continue
        sum_dist += dist(data[obs_i], data[obs_in_own])
        count += 1

    if count == 0:
        return 0
    else:
        return sum_dist / count


def get_b(obs_i, clusters):
    if obs_i in clusters[0]:
        other = clusters[1]
    elif obs_i in clusters[1]:
        other = clusters[0]
    else:
        raise ValueError("An internal error has occured")

    sum_dist = 0
    count = 0

    for obs_in_own in other:
        if obs_in_own == obs_i:
            continue
        sum_dist += dist(data[obs_i], data[obs_in_own])
        count += 1

    if count == 0:
        return 1
    else:
        return sum_dist / count


def silhouette(clusters):
    silh_list = []
    for i in range(4):
        silh = 1 - get_a(i, clusters) / get_b(i, clusters)
        silh_list.append(silh)
        print("Observation x{}: s(x) = {:.6f}".format(i+1, silh))
    print()

    silh_c_0 = 0
    silh_c_1 = 0

    for i in clusters[0]:
        silh_c_0 += silh_list[i]

    for i in clusters[1]:
        silh_c_1 += silh_list[i]

    silh_c_0 /= len(clusters[0])
    silh_c_1 /= len(clusters[1])

    print("\nCluster 0: s(c) = {:.6f}".format(silh_c_0))
    print("Cluster 1: s(c) = {:.6f}".format(silh_c_1))

    print("\nGlobal silhouette: S = {:.6f}".format((silh_c_0 + silh_c_1) / 2))



def em_clustering():
    clusters = [[], []]
    cluster_prob = []
    print("# " * 18 + "QUESTION 1-E" + " #" * 18)
    print("### Assign each point to the cluster that yields higher posteriori\n")
    for i in range(4):
        obs_probs = calculate_probs(i)
        cluster_prob.append(obs_probs)

        # saves observation to calculated cluster
        if obs_probs[0] > 0.5:
            clusters[0].append(i)
        else:
            clusters[1].append(i)

    print("\n\n" + "# " * 18 + "QUESTION 1-M" + " #" * 18)
    print("### Re-estimate cluster parameters such that they fit their assigned elements\n")
    reevaluate(cluster_prob)

    print("\n\n" + "# " * 18 + " QUESTION 2" + " #" * 19)
    silhouette(clusters)


if __name__ == "__main__":
    em_clustering()
