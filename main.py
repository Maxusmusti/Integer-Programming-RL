from gymenv_v2 import make_multiple_env
import numpy as np

from networks import Embedder, prob_dist

import wandb

# Compute discounted rewards
def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)

# Perform evaluation tests and record final score
def evaluate(policy, env, episodes):
    score = 0
    for episode in range(episodes):
        
        obs = env.reset()
        done = False
        rsum = 0
        
        while not done:
            print("eval step")
            A, b, c0, cuts_a, cuts_b = obs

            # Normalize state matrices
            def _min_max_norm(mat, min_val, max_val):
                return (mat - min_val)/(max_val - min_val)

            minAcutsA = np.amin((np.amin(A), np.amin(cuts_a)))
            maxAcutsA = np.amax((np.amax(A), np.amax(cuts_a)))
            
            minbcutsb = np.amin((np.amin(b), np.amin(cuts_b)))
            maxbcutsb = np.amax((np.amax(b), np.amax(cuts_b)))

            # Format state matrices
            constraints = np.insert(A, A.shape[1], b, axis=1)
            choices = np.insert(cuts_a, cuts_a.shape[1], cuts_b, axis=1)
            actsize = cuts_b.size

            # Retrieve embeddings
            ab_embeddings = actor.compute_embeddings(constraints)
            ed_embeddings = actor.compute_embeddings(choices)

            # Solve for action
            prob = prob_dist(ab_embeddings, ed_embeddings).ravel()
            action = np.asscalar(np.random.choice(actsize, p=prob.flatten(), size=1)) #choose according distribution prob

            # env stepping forward
            newobs, r, done, _ = env.step(action)

            # update data
            rsum += r
            obs = newobs        

        
        wandb.log({"eval reward" : rsum})
        score +=rsum
    score = score/episodes
        
    return score

wandb.login()
run=wandb.init(project="finalproject", entity="ieor-4575", tags=["training-easy"])
#run=wandb.init(project="finalproject", entity="ieor-4575", tags=["training-hard"])
#run=wandb.init(project="finalproject", entity="ieor-4575", tags=["test"])

### TRAINING

# Setup: You may generate your own instances on which you train the cutting agent.
custom_config = {
    "load_dir"        : 'instances/randomip_n60_m60',   # this is the location of the randomly generated instances (you may specify a different directory)
    "idx_list"        : list(range(20)),                # take the first 20 instances from the directory
    "timelimit"       : 50,                             # the maximum horizon length is 50
    "reward_type"     : 'obj'                           # DO NOT CHANGE reward_type
}

# Easy Setup: Use the following environment settings. We will evaluate your agent with the same easy config below:
easy_config = {
    "load_dir"        : 'instances/train_10_n60_m60',
    "idx_list"        : list(range(10)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}

# Hard Setup: Use the following environment settings. We will evaluate your agent with the same hard config below:
hard_config = {
    "load_dir"        : 'instances/train_100_n60_m60',
    "idx_list"        : list(range(99)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}

if __name__ == "__main__":
    # create env
    env = make_multiple_env(**easy_config) 

    # hyperparams
    alpha = 1e-2
    numtrajs = 10
    episodes = 30
    gamma = 0.99

    obssize = 61
    embedsize = 10

    # Policy network
    actor = Embedder(obssize, embedsize, alpha, decay=False)

    rrecord = []
    for e in range(episodes):
        CONOBS = []  # constraint matrices
        CHOOBS = []  # cut matrices
        ACTS = []  # actions
        VAL = []  # value estimates

        for num in range(numtrajs):
            # To keep a record of states actions and reward for each episode
            conobss = []  # constraints
            choobss = []  # cutes
            acts = []   # actions
            rews = []  # instant rewards

            obs = env.reset()
            A, b, c0, cuts_a, cuts_b = obs

            # Normalize state matrices
            def _min_max_norm(mat, min_val, max_val):
                return (mat - min_val)/(max_val - min_val)

            minAcutsA = np.amin((np.amin(A), np.amin(cuts_a)))
            maxAcutsA = np.amax((np.amax(A), np.amax(cuts_a)))
            
            minbcutsb = np.amin((np.amin(b), np.amin(cuts_b)))
            maxbcutsb = np.amax((np.amax(b), np.amax(cuts_b)))

            A = _min_max_norm(A, minAcutsA, maxAcutsA)
            cuts_a = _min_max_norm(cuts_a, minAcutsA, maxAcutsA)

            b = _min_max_norm(b, minbcutsb, maxbcutsb)
            cuts_b = _min_max_norm(cuts_b, minbcutsb, maxbcutsb)

            # Format state matrices
            constraints = np.insert(A, A.shape[1], b, axis=1)
            choices = np.insert(cuts_a, cuts_a.shape[1], cuts_b, axis=1)
            actsize = cuts_b.size

            # Unroll trajectories
            done = False
            while not done:
                # Retrieve embeddings
                ab_embeddings = actor.compute_embeddings(constraints)
                ed_embeddings = actor.compute_embeddings(choices)

                # Select action
                prob = prob_dist(ab_embeddings, ed_embeddings)
                prob /= np.sum(prob) #normalizing again to account for numerical errors
                action = np.asscalar(np.random.choice(actsize, p=prob.flatten(), size=1)) #choose according distribution prob
                
                # Recording + next state retrieval and formatting
                conobss.append(constraints)
                choobss.append(choices)

                obs, reward, done, info = env.step(action)

                A, b, c0, cuts_a, cuts_b = obs

                minAcutsA = np.amin((np.amin(A), np.amin(cuts_a)))
                maxAcutsA = np.amax((np.amax(A), np.amax(cuts_a)))
                
                minbcutsb = np.amin((np.amin(b), np.amin(cuts_b)))
                maxbcutsb = np.amax((np.amax(b), np.amax(cuts_b)))

                A = _min_max_norm(A, minAcutsA, maxAcutsA)
                cuts_a = _min_max_norm(cuts_a, minAcutsA, maxAcutsA)

                b = _min_max_norm(b, minbcutsb, maxbcutsb)
                cuts_b = _min_max_norm(cuts_b, minbcutsb, maxbcutsb)

                constraints = np.insert(A, A.shape[1], b, axis=1)
                choices = np.insert(cuts_a, cuts_a.shape[1], cuts_b, axis=1)
                actsize = cuts_b.size

                acts.append(action)
                rews.append(reward)
  
            reward_sum = np.sum(rews)

            #Below is for logging training performance
            rrecord.append(reward_sum)

            v_hats = discounted_rewards(rews, gamma)
            VAL.extend(v_hats)
            CONOBS.extend(conobss)
            CHOOBS.extend(choobss)
            ACTS.extend(acts)

        print("Done with trajs")  
        
        # Normalizing value estimates
        VAL = np.array(VAL)
        VAL = (VAL - np.mean(VAL))/np.std(VAL)

        # TRAINING
        print("Going into training")  
        actor.train(np.array(CONOBS), np.array(CHOOBS), np.array(ACTS), actsize, VAL)
        
        #printing moving averages for smoothed visualization. 
        #Do not change below: this assume you recorded the sum of rewards in each episide in the list rrecord
        fixedWindow=50
        movingAverage=0
        if len(rrecord) >= fixedWindow:
            movingAverage=np.mean(rrecord[len(rrecord)-fixedWindow:len(rrecord)-1])
            
        print("Episode plotted")

        #wandb logging
        wandb.log({ "training reward" : rrecord[-1], "training reward moving average" : movingAverage})

    eval_episodes = 50
    score = evaluate(actor, env, eval_episodes)
    wandb.run.summary["score"]=score 
    print("eval performance of the learned policy: {}".format(score))
    