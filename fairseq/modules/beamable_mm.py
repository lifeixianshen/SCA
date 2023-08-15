# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn


class BeamableMM(nn.Module):
    """This module provides an optimized MM for beam decoding with attention.

    It leverage the fact that the source-side of the input is replicated beam
    times and the target-side of the input is of width one. This layer speeds up
    inference by replacing the inputs {(bsz x 1 x nhu), (bsz x sz2 x nhu)}
    with smaller inputs {(bsz/beam x beam x nhu), (bsz/beam x sz2 x nhu)}.
    """
    def __init__(self, beam_size=None):
        super(BeamableMM, self).__init__()
        self.beam_size = beam_size

    def forward(self, input1, input2):
        if (
            self.training
            or self.beam_size is None
            or input1.dim() != 3
            or input1.size(1) != 1
        ):
            return input1.bmm(input2)
        bsz, beam = input1.size(0), self.beam_size

        # bsz x 1 x nhu --> bsz/beam x beam x nhu
        input1 = input1[:, 0, :].unfold(0, beam, beam).transpose(2, 1)

        # bsz x sz2 x nhu --> bsz/beam x sz2 x nhu
        input2 = input2.unfold(0, beam, beam)[:, :, :, 0]

            # use non batched operation if bsz = beam
        output = (
            torch.mm(input1[0, :, :], input2[0, :, :])
            if input1.size(0) == 1
            else input1.bmm(input2)
        )
        return output.view(bsz, 1, -1)

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size
