import matplotlib.pyplot as plt
import os
import sys
class Plt:
    def __init__(self):
        self.k = 0.99
    def pltloss(self,loss,episode):
        plt.plot(episode,loss)
        plt.xlabel("episodes")
        plt.ylabel("loss")
        plt.title("loss vs episodes")
        plt.savefig("storage/loss.png")
        plt.close()
    def pltag1loss(self,loss,episodes):
        plt.plot(episodes,loss)
        plt.xlabel("episodes")
        plt.ylabel("loss")
        plt.title("agent1_loss vs episodes")
        plt.savefig("storage/agent1_loss.png")
        
        plt.close()
    def pltag2loss(self,loss,episode):
        plt.plot(episode,loss)
        plt.xlabel("episodes")
        plt.ylabel("loss")
        plt.title("agent2_loss vs episodes")
        plt.savefig("storage/agent1_loss.png")
       
        plt.close()
    def ag1qvalue(self,q_value,episode):
        plt.plot(episode,q_value)
        plt.xlabel("episodes")
        plt.ylabel("q1_value")
        plt.title("agent_qvalue vs episodes")
        plt.savefig("storage/agent1_qvalue.png")
        
        plt.close()
    def ag2qvalue(self,q_value,episode):
        plt.plot(episode,q_value)
        plt.xlabel("episodes")
        plt.ylabel("q2_value")
        plt.title("agent_qvalue vs episodes")
        plt.savefig("storage/agent2_qvalue.png")
        
        plt.close()

    def ag1nqvalue(self,nq_value,episode):
        plt.plot(episode,nq_value)
        plt.xlabel("episodes")
        plt.ylabel("nq_value")
        plt.title("agent_qvalue vs episodes")
        plt.savefig("storage/agent1_nqvalue.png")
        
        plt.close()

    def ag2nqvalue(self,nq_value,episode):
        plt.plot(episode,nq_value)
        plt.xlabel("episodes")
        plt.ylabel("nq_value")
        plt.title("agent2_qvalue vs episodes")
        plt.savefig("storage/agent2_nqvalue.png")
        
        plt.close()

    def ag1reward(self,reward,episode):
        plt.plot(episode,reward)
        plt.xlabel("episodes")
        plt.ylabel("ag1_reward")
        plt.title("ag1_reward vs episodes")
        plt.savefig("storage/agent1reward.png")
        plt.close()

    def ag2reward(self,reward,episode):
        plt.plot(episode,reward)
        plt.xlabel("episodes")
        plt.ylabel("ag2_reward")
        plt.title("ag2_reward vs episodes")
        plt.savefig("storage/agent2reward.png")
        
        plt.close()
    
    def ag1policy(self,policy,episode):
        plt.plot(episode,policy)
        plt.xlabel("episodes")
        plt.ylabel("policy")
        plt.title("policy vs episodes")
        plt.savefig("storage/agent1_policy.png")
        
        plt.close()     

    def ag2policy(self,policy,episode):
        plt.plot(episode,policy)
        plt.xlabel("episodes")
        plt.ylabel("policy")
        plt.title("policy vs episodes")
        plt.savefig("storage/agent2_policy.png")
        
        plt.close()     
    