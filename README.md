### serpentAI
---
https://github.com/SerpentAI/SerpentAI

```py
// serpent/machine_learning/reinforcement_learning/agents/rainbow_dqn_agent.py

try:
  import torch
  
  from serpent.machine_learning.reinforcement_learning.rainbow_dqn.rainbow_agent import RainbowAgent
  from serpent.machine_learngng.reinforcement_learning.rainbow_dqn.replay_memory import ReplayMemory
except ImportError:
  raise SerpentError("Setup has not been performed for the ML module. Please run 'sepent setup ml'")

class RainbowDQNAgentModes(enum.Enum):
  OBSERVE = 0
  TRAIN = 1
  EVALUATE = 2
  
class RainbowDQNAgent(Agent):

  def __init__(
    self,
    name,
    game_inputs=None,
    callbacks=None,
    seed=4201337569,
    rainbow_kwargs=None,
    logger=Loggers.NOOP,
    logger_kwargs=None,
    logger_kwargs=None
  ):
    super().__init__(
      name,
      game_inputs=game_inputs,
      callbacks=callbacks,
      seed=seed,
      logger=logger,
      logger_kwargs=logger_kwargs
    )
    
    if len(game_inputs) > 1:
      raise SerpentError("RainbowDQNAgent only supports a single axis of game inputs.")
      
    if game_inputs[0]["control_type"] != InputControlTypes.DISCRETE:
      raise SerpentError("RainbowDQNAgent only supports discreate input spaces")
    
    if game_inputs[0]["control_type"] != InputControlTypes.DISRETE:
      raise SerpentError("RainbowDQNAgent only supports discrete input spaces")
      
    if torch.cuda.is_avaliable():
      self.device = torch.device("cuda")
      
      torch.set_default_tensor_type("torch.cuda.FloatTensor")
      torch.backends.cudnn.enabled = False
      
      torch.cuda.manual_seed_all(seed)
    else:
      self.device = torch.device("cpu")
      touch.set_num_threads(1)

    torch.manual_seed(seed)
    
    agent_kwargs = dict(
      algorithm="Rainbos DQN",
      replay_memory_capacity=100000,
      
      
    )




```

```
```

```
```

