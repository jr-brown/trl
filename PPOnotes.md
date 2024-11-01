# Summary of PPO trainer structure. 


## Before training loop

Set up

Defining tracking stores

Trainer state initialisation  
 - maybe; review source code for base trainer class

## Within training loop

### Dataset preparation

Generate response
Iterate through chunks of responses and get log-probabilities for both the actual model and the base model
Get value estimates and rewards for each completion
Concatenate into one batch
Compute returns and advantages

### Actual training (gradients being passed)

Compute log-probs and values (to put into loss)
Construct value loss
Construct policy loss
backwards pass
logging  
