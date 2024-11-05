import numpy as np


def merge(subset, min_num_body_parts=4, min_score=0.4):
    """
    Estimates the skeletons.
    :param connections: valid connections
    :param min_num_body_parts: minimum number of body parts for a skeleton
    :param min_score: minimum score value for the skeleton
    :return: list of skeletons. Each skeleton has a list of identifiers of body parts:
        [
            [id1, id2,...,idN, score, parts_num],
            [id1, id2,...,idN, score, parts_num]
            ...
        ]

    position meaning:
        [   [nose       , neck           , right_shoulder , right_elbow      , right_wrist  , left_shoulder
             left_elbow , left_wrist     , right_hip      , right_knee       , right_ankle  , left_hip
             left_knee  , left_ankle     , right_eye      , left_eye         , right_ear    , left_ear
             score, parts_num],
        ]
    """

    # 2 step :
    #---merge----
    # Merge the limbs in the subset
    # score : score
    # parts_num : How many limbs are in the subset
    ###############################

    def _print_subset(subset):
        for i in subset:
            for j in i[:-2]:
                if j == -1:
                    print(" -", end='   ')
                else:
                    print(f'{int(j):2d}', end='   ')
            print(f'{i[-2]:f}', end='  ')
            print(f'{int(i[-1]):2d}')
        return subset

    def _merge(subset, i, j):
        # merge subset j into subset i
        for k in range(18):
            if subset[i][k] == -1:
                subset[i][k] = subset[j][k]
            subset[j][k] = -1

        # sum the score
        subset[i][-2] += subset[j][-2]
        subset[j][-2] = 0

        # sum the parts_num
        subset[i][-1] = sum(1 for k in subset[i][:-2] if k != -1)
        subset[j][-1] = 0

    print("Before merge:")
    _print_subset(subset)

    for i in range(len(subset)):
        for j in range(len(subset)):
            if i == j:
                continue
            for k in range(18):
                if subset[i][k] == subset[j][k] and subset[i][k] != -1:
                    _merge(subset, i, j)
                    break

    # after merge
    #---delete---
    # Delete the non-compliant subset
    # 1. parts_num < 4
    # 2. Average score(score / parts_num) < 0.4
    ############################################

    delete_idx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:  # revise here
            delete_idx.append(i)
    subset = np.delete(subset, delete_idx, axis=0)

    print("\nAfter merge:")
    _print_subset(subset)
    return subset
