from networkx.readwrite import json_graph
from networkx.algorithms import shortest_paths
import math
import numpy as np
import copy
import random
import sys

sys.path.insert(1, "../helpers")
from helpers.featurizer import Featurizer
from helpers.network_embedder import VNE_on_sub
from helpers.graph_gen import GraphGen

# we use the same graph generator from the main program for all the environments
class HighLevelEnv:
    def __init__(self, sub_graphs, gr_gen, max_vnr_not_embed) -> None:
        self.sub_graphs = sub_graphs
        self.RPs = [VNE_on_sub(self.sub_graphs[i]) for i in range(len(self.sub_graphs))]
        self.gr_gen = gr_gen
        self.curr_vnr = self.gr_gen.gen_rnd_vnr()
        self.max_vnr_not_embed = max_vnr_not_embed

    @property
    def action_shape(self):
        return len(self.sub_graphs)
    
    def sample_action(self):
        return random.randint(0, self.action_shape - 1)
    
    def encode_graphs(self):
        state_ = []
        for obj in self.RPs:
            state_.append(Featurizer.make_data_obj(obj.curr_state, sub=1, high=1))
        state_.append(Featurizer.make_data_obj(self.curr_vnr, sub=0, high=1))
        return state_

    # reset is only done in the beginning of an episode
    def reset(self):
        for x in range(len(self.RPs)):
            self.RPs[x].reset_to_original()
        self.curr_vnr = self.gr_gen.gen_rnd_vnr()
        return self.encode_graphs()
    
    def release_vnrs(self, time_step):
        for x in range(len(self.RPs)):
            self.RPs[x].release_resources(time_step)

    # this function will provide the curr vnr to the embedding object and then get the object back
    # this vne obj will be used by the low agent
    def give_vnr_get_vne_obj(self, action):
        self.RPs[action].receive_vnr(self.curr_vnr)
        self.RPs[action].reset_map()
        return self.RPs[action]
    
    def get_curr_vnr(self):
        return self.curr_vnr
    
    def get_reward(self, embedded, mapp):
        return 1 if embedded else -1
    
    # this embedd obj is received after node embedding by the low level agent
    # all the functions only execute if the nodes and the links are embedded 
    # embed link function only runs if all the nodes have been embedded by the low agent
    def step(self, action, embedd_obj, vnr_not_embed, vnr_dept_time):
        embedded = embedd_obj.embed_link()
        mapp, fully_embedded = embedd_obj.get_mapping()
        embedd_obj.change_sub()
        embedd_obj.store_map(vnr_dept_time)

        self.RPs[action] = embedd_obj
        # we need to change the vnr for the next state
        self.curr_vnr = self.gr_gen.gen_rnd_vnr()

        # for now done is True if the max time is reached otherwise we could also make it true if a VNR is not embedded
        # or some number of VNRs are not embedded
        done = True if vnr_not_embed > self.max_vnr_not_embed else False
        next_state = self.encode_graphs() #if not done else None
        reward = self.get_reward(embedded, mapp)
        
        return next_state, reward, done, fully_embedded
    
    
    
