# Copyright 2023 Nikolai Körber and Yang Zhang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf


def mux(alpha, beta):
    """MUX operation as described in https://arxiv.org/pdf/2103.15306.pdf, A.3.1"""

    # e.g., given y_hat
    #
    # [[ 1.  2.  3.  4.  5.  6.]
    # [ 7.  8.  9. 10. 11. 12.]
    # [13. 14. 15. 16. 17. 18.]
    # [19. 20. 21. 22. 23. 24.]
    # [25. 26. 27. 28. 29. 30.]
    # [31. 32. 33. 34. 35. 36.]]
    #
    # y_hat_anchor = mux(zeros, y_hat)
    #
    # [[ 0.  2.  0.  4.  0.  6.]
    # [ 7.  0.  9.  0. 11.  0.]
    # [ 0. 14.  0. 16.  0. 18.]
    # [19.  0. 21.  0. 23.  0.]
    # [ 0. 26.  0. 28.  0. 30.]
    # [31.  0. 33.  0. 35.  0.]]
    #
    # Similarly, y_hat_nonanchor = mux(y_hat, zeros)
    #
    # [[ 1.  0.  3.  0.  5.  0.]
    # [ 0.  8.  0. 10.  0. 12.]
    # [13.  0. 15.  0. 17.  0.]
    # [ 0. 20.  0. 22.  0. 24.]
    # [25.  0. 27.  0. 29.  0.]
    # [ 0. 32.  0. 34.  0. 36.]]
    #
    # Note:
    # - alpha and beta are swapped, due to inconsistencies in the paper.
    # - this interpretation is consistent with Figure 1:
    #     blue and yellow = anchor,
    #     white = non-anchor.

    c = alpha.shape[-1]

    alpha = tf.nn.space_to_depth(alpha, block_size=2)

    alpha_slice_1 = tf.slice(alpha, [0, 0, 0, 0], [-1, -1, -1, c])  # slice A
    alpha_slice_4 = tf.slice(alpha, [0, 0, 0, 3 * c], [-1, -1, -1, c])  # slice D

    beta = tf.nn.space_to_depth(beta, block_size=2)
    beta_slice_2 = tf.slice(beta, [0, 0, 0, c], [-1, -1, -1, c])
    beta_slice_3 = tf.slice(beta, [0, 0, 0, 2 * c], [-1, -1, -1, c])

    mix = tf.concat([alpha_slice_1, beta_slice_2, beta_slice_3, alpha_slice_4], axis=-1)
    mix = tf.nn.depth_to_space(mix, block_size=2)

    return mix


def demux_nonanchor(demux_input):
    """DEMUX operation as described in https://arxiv.org/pdf/2103.15306.pdf, A.3.2 (nonanchor part)"""

    # get the slice A and D from Figure 2 (section A.3.2)
    c = demux_input.shape[-1]
    demux_nonanchor_slice = tf.nn.space_to_depth(demux_input, block_size=2)
    demux_nonanchor_slice_1 = tf.slice(demux_nonanchor_slice, [0, 0, 0, 0], [-1, -1, -1, c])  # slice A
    demux_nonanchor_slice_2 = tf.slice(demux_nonanchor_slice, [0, 0, 0, 3 * c], [-1, -1, -1, c])  # slice D
    demux_nonanchor_mix = tf.concat([demux_nonanchor_slice_1, demux_nonanchor_slice_2], axis=-1)

    return demux_nonanchor_mix


def demux_anchor(demux_input):
    """DEMUX operation as described in https://arxiv.org/pdf/2103.15306.pdf, A.3.2 (anchor part)"""

    # get the slice B and C from Figure 2 (section A.3.2)
    c = demux_input.shape[-1]
    demux_anchor_slice = tf.nn.space_to_depth(demux_input, block_size=2)
    demux_anchor_slice_1 = tf.slice(demux_anchor_slice, [0, 0, 0, c], [-1, -1, -1, c])  # slice B
    demux_anchor_slice_2 = tf.slice(demux_anchor_slice, [0, 0, 0, 2 * c], [-1, -1, -1, c])  # slice C
    demux_anchor_mix = tf.concat([demux_anchor_slice_1, demux_anchor_slice_2], axis=-1)

    return demux_anchor_mix


def demux_nonanchor_inverse(demux_input):
    """DEMUX operation as described in https://arxiv.org/pdf/2103.15306.pdf, A.3.3 (inverse nonanchor part)"""

    # inverse the A and D slice from Figure 2 (section A.3.2)
    c = tf.cast(demux_input.shape[-1] / 2, dtype=tf.int32)
    demux_nonanchor_inverse_1 = tf.slice(demux_input, [0, 0, 0, 0], [-1, -1, -1, c])  # slice A
    demux_nonanchor_inverse_2 = tf.slice(demux_input, [0, 0, 0, c], [-1, -1, -1, c])  # slice D
    zeros_slice = tf.zeros(tf.shape(demux_nonanchor_inverse_1))  # slice zero
    demux_anchor_inverse_mix = tf.concat(
        [demux_nonanchor_inverse_1, zeros_slice, zeros_slice, demux_nonanchor_inverse_2], axis=-1)
    demux_anchor_inverse = tf.nn.depth_to_space(demux_anchor_inverse_mix, block_size=2)

    return demux_anchor_inverse


def demux_anchor_inverse(demux_input):
    """DEMUX operation as described in https://arxiv.org/pdf/2103.15306.pdf, A.3.3 (inverse anchor part)"""

    # inverse the B and C slice from Figure 2 (section A.3.2)
    c = tf.cast(demux_input.shape[-1] / 2, dtype=tf.int32)
    demux_anchor_inverse_1 = tf.slice(demux_input, [0, 0, 0, 0], [-1, -1, -1, c])  # slice B
    demux_anchor_inverse_2 = tf.slice(demux_input, [0, 0, 0, c], [-1, -1, -1, c])  # slice D
    zeros_slice_non = tf.zeros(tf.shape(demux_anchor_inverse_1))  # slice zero
    demux_nonanchor_inverse_mix = tf.concat(
        [zeros_slice_non, demux_anchor_inverse_1, demux_anchor_inverse_2, zeros_slice_non], axis=-1)
    demux_nonanchor_inverse = tf.nn.depth_to_space(demux_nonanchor_inverse_mix, block_size=2)

    return demux_nonanchor_inverse
