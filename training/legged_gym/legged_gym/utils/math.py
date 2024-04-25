# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

def circle_ray_query(x0: torch.Tensor, y0: torch.Tensor, thetas: torch.Tensor, center_circle: torch.Tensor, radius: float, min_: float=0.1, max_:float = 3.0):
    """
    x0:(n,1), y0:(n,1), thetas: (n,t), center: (n,2)
    for each env, compute the distances for the ray from (x0, y0) at thetas to cross the circle
    return shape (n, t)
    """
    stheta = torch.sin(thetas) # (n,t)
    ctheta = torch.cos(thetas) # (n,t)
    xc = center_circle[:,0:1] # (n,1)
    yc = center_circle[:,1:2] # (n,1)
    d_c2line = torch.abs(stheta*xc - ctheta*yc - stheta*x0 + ctheta*y0)  #(n,t)
    d_c0_square = torch.square(xc-x0) + torch.square(yc-y0)
    d_0p = torch.sqrt(d_c0_square - torch.square(d_c2line))
    semi_arc = torch.sqrt(radius**2 - torch.square(d_c2line))
    raydist = torch.nan_to_num(d_0p - semi_arc, nan = max_).clip(min=min_, max=max_)
    check_dir = ctheta * (xc-x0) + stheta * (yc-y0)
    raydist = (check_dir > 0) * raydist + (check_dir<=0) * max_
    return raydist

def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    quat_yaw = quat.clone().view(-1, 4)
    qx = quat_yaw[:, 0]
    qy = quat_yaw[:, 1]
    qz = quat_yaw[:, 2]
    qw = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:, :2] = 0.0
    quat_yaw[:, 2] = torch.sin(yaw / 2)
    quat_yaw[:, 3] = torch.cos(yaw / 2)
    quat_yaw = normalize(quat_yaw)
    return quat_yaw

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower