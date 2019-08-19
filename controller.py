# model search
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

import ppo #

# to update controller, using ppo. learning rate 0.00035
# entropy penalty 0.00001


# one-layer LSTM with 100 hidden units at each layer and 2x5B softmax predictons for the 2 conv cells.
# each predictions(10B) is composed of propability - softmax에서 모든 확률의 곱 = joint probability of child network 

# base line function : exponential moving average of previous rewards with a weight 0.95
# weights of the controller : initialized uniformly -0.1 ~ 0.1

# for distributed training, "workqueue system" : controller RNN으로부터 나온 모든 sample을 global workqueue에 추가
# child net의 training이 끝나면, validation set에 대한 accuracy를 계산하여 controller RNN에 보고한다. 
# 충분한(20) child net의 학습 결과를 받고 나면, controller RNN은 PPO로 gradient update를 진행하고 
# 다른 batch of arch를 global workqueue에 sample한다. 

# 미리 정한 architecture의 수 만큼 샘플링 될 때 까지 진행. (20,000)
# 다 하고 나면 250 top을 뽑아 학습시킨다.

class Block(nn.Module): 

  def __init__(self, hidden_state_set, ):
    super(Block, self).__init__()
    self.hidden_st = hidden_state_set
    # 모든 softmax값 10B개의 곱.. = probability -> gradient 계산에 사용

    
    '''
    self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 1
        op = MixedOp(C, stride)
        self._ops.append(op)
    '''
  def forward(self, hidden_state_set,):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1) #######################


class Cell(nn.Module): # 5 blocks

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C):
    super(Cell, self).__init__()

    self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1) #######################