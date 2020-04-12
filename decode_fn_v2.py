import numpy as np


def decode_fn_v2(encoded_charge, encoded_type, labels):
    """
    Decodes the onehotencoded categorical labels (charge and type) into the original features

    E.g. if the charges -1 and 1 are decoded as [[0,1],[1,0]] and the types EE, EB as [[0,1],[1,0]],
    then the decode function decodes the label [0,1,1,0,0,1,0,1] into ['-1','1','EE','EE']

    :param encoded_charge: onehotencoder of charge [-1,1]
    :param encoded_type: onehotencoder of type [EB, EE]
    :param labels: onehotencoded array of charge and type. dim=(8,)
    :return array
    """

    cat_list = []
    # charge type
    for i in labels:
        j = 0
        for _ in range(len(i)):
            if j < 3:
                cat_list.append(encoded_charge.inverse_transform([i[j:j + 2]]).ravel()[0])
                j = j + 2
            elif j < 7:
                cat_list.append(encoded_type.inverse_transform([i[j:j + 2]]).ravel()[0])
                j = j + 2
            else:
                break
    return np.array(cat_list).reshape(labels.shape[0], -1)
