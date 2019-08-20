# model search
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

import ppo #

import utils
from utils import Node

print(PRIMITIVES)
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

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops)) # change here

class Controller(nn.Module): #environment
    def __init__(self, args):
        torch.nn.Module.__init__(self)
        self.args = args
        self.num_tokens = [len(OPS),5] # ops 수 , block 수 
        self.func_names = PRIMITIVES


        num_total_tokens = sum(self.num_tokens)

        self.encoder = torch.nn.Embedding(num_total_tokens,
                                          args.controller_hid)
        self.lstm = torch.nn.LSTMCell(args.controller_hid, args.controller_hid) # 100, 100

        # TODO(brendan): Perhaps these weights in the decoder should be
        # shared? At least for the activation functions, which all have the
        # same size.

        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(args.controller_hid, size)
            self.decoders.append(decoder)

        self._decoders = torch.nn.ModuleList(self.decoders)

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, self.args.controller_hid),
                self.args.cuda,
                requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self,  # pylint:disable=arguments-differ
                inputs,
                hidden,
                block_idx,
                is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)

        logits /= self.args.softmax_temperature

        # exploration
        if self.args.mode == 'train':
            logits = (self.args.tanh_c*F.tanh(logits))

        return logits, (hx, cx)

    def sample(self, batch_size=1, with_details=False, save_dir=None): # ppo랑 연관시켜야할 듯
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """
        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        # [B, L, H]
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        activations = []
        entropies = []
        log_probs = []
        prev_nodes = []
        # NOTE(brendan): The RNN controller alternately outputs an activation,
        # followed by a previous node, for each block except the last one,
        # which only gets an activation function. The last node is the output
        # node, and its previous node is the average of all leaf nodes.
        for block_idx in range(2*(self.args.num_blocks - 1) + 1):
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          block_idx,
                                          is_embed=(block_idx == 0))

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            # TODO(brendan): .mean() for entropy?
            entropy = -(log_prob * probs).sum(1, keepdim=False)

            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, requires_grad=False))

            # TODO(brendan): why the [:, 0] here? Should it be .squeeze(), or
            # .view()? Same below with `action`.
            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            # 0: function, 1: previous node
            mode = block_idx % 2
            inputs = utils.get_variable(
                action[:, 0] + sum(self.num_tokens[:mode]),
                requires_grad=False)

            if mode == 0:
                activations.append(action[:, 0])
            elif mode == 1:
                prev_nodes.append(action[:, 0])

        prev_nodes = torch.stack(prev_nodes).transpose(0, 1)
        activations = torch.stack(activations).transpose(0, 1)

        cell = _construct_cell(prev_nodes,
                               activations,
                               self.func_names,
                               self.args.num_blocks)

        if save_dir is not None:
            for idx, dag in enumerate(cell):
                utils.draw_network(dag,
                                   os.path.join(save_dir, f'graph{idx}.png'))

        if with_details:
            return cell, torch.cat(log_probs), torch.cat(entropies)

        return cell

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.controller_hid)
        return (utils.get_variable(zeros, self.args.cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.args.cuda, requires_grad=False))


def _construct_cells(prev_nodes, activations, func_names, num_blocks):
    """Constructs a set of DAGs based on the actions, i.e., previous nodes and
    activation functions, sampled from the controller/policy pi.
    Args:
        prev_nodes: Previous node actions from the policy.
        activations: Activations sampled from the policy.
        func_names: Mapping from activation function names to functions.
        num_blocks: Number of blocks in the target RNN cell.
    Returns:
        A list of DAGs defined by the inputs.
    RNN cell DAGs are represented in the following way:
    1. Each element (node) in a DAG is a list of `Node`s.
    2. The `Node`s in the list dag[i] correspond to the subsequent nodes
       that take the output from node i as their own input.
    3. dag[-1] is the node that takes input from x^{(t)} and h^{(t - 1)}.
       dag[-1] always feeds dag[0].
       dag[-1] acts as if `w_xc`, `w_hc`, `w_xh` and `w_hh` are its
       weights.
    4. dag[N - 1] is the node that produces the hidden state passed to
       the next timestep. dag[N - 1] is also always a leaf node, and therefore
       is always averaged with the other leaf nodes and fed to the output
       decoder.
    """
    blocks = [] # blocks become a cell
    for nodes, func_ids in zip(prev_nodes, activations):
        dag = collections.defaultdict(list)

        # add first node
        dag[-1] = [Node(0, func_names[func_ids[0]])]
        dag[-2] = [Node(0, func_names[func_ids[0]])]

        # add following nodes
        for jdx, (idx, func_id) in enumerate(zip(nodes, func_ids[1:])):
            dag[utils.to_item(idx)].append(Node(jdx + 1, func_names[func_id]))

        leaf_nodes = set(range(num_blocks)) - dag.keys()

        # merge with avg
        for idx in leaf_nodes:
            dag[idx] = [Node(num_blocks, 'avg')]

        # TODO(brendan): This is actually y^{(t)}. h^{(t)} is node N - 1 in
        # the graph, where N Is the number of nodes. I.e., h^{(t)} takes
        # only one other node as its input.
        # last h[t] node
        last_node = Node(num_blocks + 1, 'h[t]')
        dag[num_blocks] = [last_node]
        blocks.append(dag)

    return blocks