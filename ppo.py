# https://www.youtube.com/watch?v=l1CZQWBkdcY

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.00035
gamma         = 0.98
T_horizon     = 20 # 몇 timestep 동안 data를 모아 policy를 update할지


lmbda         = 0.95 # gae에 쓰이는 변수 
eps_clip      = 0.1 # for clipping ... 
K_epoch       = 3 # epoch


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0): # 여러 샘플이 input으로 들어가면 softmax_dim을 1로 해야한다. 
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self): # 같은 인자끼리 모은거.. 핵심 !!!!!!!!!!!!!
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a]) # 그냥 dim맞추기 위해 다 [] 해준겅
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_controller(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask # 벡터연산 ( 여러 값이 있음 ) 
            # 배치 처리를 하지 않으면 40번..?호출해야됨... 그래서 batch 처리를 해서 한번에 함. 
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            # GAE (advantage) 계산 - gamma & lmbda 두개의 param 존재 
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]: # 거꾸로 부터.. 
                advantage = gamma * lmbda * advantage + delta_t[0]   
                advantage_lst.append([advantage])
            advantage_lst.reverse() # reverse ! 
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            # GAE식 만드는 과정

            # loss 함수
            pi = self.pi(s, softmax_dim=1) # 네트워크로 확률 뽑는것
            pi_a = pi.gather(1,a) # 실제 action 의 확률
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
            # 현재 policy의 확률 / 경험 쌓을 때 policy의 확률

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach()) # PPO loss with clipping
            # detatch ******* :  td_target이 만들어지기까지의 앞의 그래프들을 다 떼어버린다는 뜻(target이기 때문). 
            # gradient의 flow가 발생 x!! 즉 v(s')함수의 그래프까지 gradient가 전달된다는 뜻. 
            # tf에서는 place holder로 값을 지정해버림.. pytorch에서는 그냥 detatch

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
def main():
    env = gym.make('CartPole-v1') # 이 부분을 rnn controller가 확률 p로 만들어낸 구조A로, accuracy R로 더 좋은 controller를 만든다. 
    model = PPO()
    score = 0.0
    print_interval = 20 # 몇개마다 점수 찍을지

    for n_epi in range(10000): # 경험쌓는부분
        s = env.reset()
        done = False
        while not done: # 에피소드가 끝나지 않을 동안... 
            for t in range(T_horizon): # T 만큼만 모으고 학습을 함. 끝나면 train한다(model.train)
                prob = model.pi(torch.from_numpy(s).float()) # 모델한테 확률 뱉으라 함
                m = Categorical(prob)# 확률을 categorical 변수로 만들어서 
                a = m.sample().item() # 샘플링 해가지고 액션 뽑고 
                s_prime, r, done, info = env.step(a) # 액션을 env에 던짐 ! 

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done)) 
                # prob[a].item() : 실제 내가 한 액션의 확률값. PPO 의 ratio계산에서 old policy의 확률값이 쓰이기 때문에 같이 저장해줌
                s = s_prime

                score += r
                if done:
                    break

            model.train_controller() # PPO class

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()