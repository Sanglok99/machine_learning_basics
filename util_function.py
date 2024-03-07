import numpy as np
import scipy.misc


def sparse_tuple_from(sequences, dtype=np.int32):
    """
        Inspired (copied) from https://github.com/igormq/ctc_tensorflow_example/blob/master/utils.py
    """

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), [i for i in range(len(seq))]))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray(
        [len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64
    )

    return indices, values, shape


def resize_image(im_arr, input_width):
    """Resize an image to the "good" input size
    """

    r, c = np.shape(im_arr)
    if c > input_width:
        c = input_width
        ratio = float(input_width) / c
        final_arr = scipy.misc.imresize(im_arr, (int(32 * ratio), input_width))
    else:
        final_arr = np.zeros((32, input_width))
        ratio = 32.0 / r
        im_arr_resized = scipy.misc.imresize(im_arr, (32, int(c * ratio)))
        final_arr[
            :, 0: min(input_width, np.shape(im_arr_resized)[1])
        ] = im_arr_resized[:, 0:input_width]
    return final_arr, c


def label_to_array(label, char_vector):
    try:
        return [char_vector.index(x) for x in label]
    except Exception as ex:
        print(label)
        raise ex


def ground_truth_to_word(ground_truth, char_vector):
    """
        Return the word string based on the input ground_truth
    """

    try:
        result = ''
        for i in ground_truth:
            if i != -1:
                result += char_vector[i]
        return result
    except Exception as ex:
        print(ground_truth)
        print(ex)
        input()
