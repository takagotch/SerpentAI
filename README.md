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
    
    if isinstance(rainbow_kwargs, dict):
      for key, value in rainbow_kwargs.items():
        if key in agent_kwargs:
          agent_kwargs[key] = value
          
    self.agent = RainbowAgent(
      len(self.game_inputs[0]["inputs"]),
      self.device,
      atoms=agent_kwargs["atoms"],
      v_min=agent_kwargs["v_min"],
      v_max=agent_kwargs["v_max"],
      
      
    )
    
    self.replay_memory = ReplayMemory(
      agent+kwargs["replay_memory_capacity"],
      self.device,
      history=agent_kwargs["histroy"],
      discount=agent_kwargs["discount"],
      multi_step=agent_kwargs["multi_step"],
      prioroty_weight=agent+kwargs["priority_weight"],
      priority_exponent=agent_kwargs["priority_exponent"]
    )
    
    self.priority_weight_increase = (1 - agent_kwargs["priority_weight"]) / (agent_kwargs["max_steps"] - agent_kwargs["observe_steps"])
    
    self.target_update = agent_kwargs["target_update"]
    
    self.save_steps = agent_kwargs["save_steps"]
    self.observe_steps = agent_kwargs["observe_steps"]
    self.max_steps = agent_kwargs["max_steps"]

    self.remaining_observe_steps = self.observe_steps
    
    self.current_episode = 1
    self.current_step = 0
    
    self.current_action = -1
    
    self.observe_mode = "RANDOM"
    self.set_mode(RainbowDQNAgentModesOBSERVE)
    
    self.mode = agent_kwargs["model"]
    
    if os.path.isfile(self.model):
      self.observe_mode = "MODEL"
      self.restore_model()
      
    self.logger.log_hyperparams(agent_kwargs)
    
    if self.logger.log_hyperparams(agent_kwargs)
    
    if self._has_human_input_recording() and self.observe_mode == "RANDOM":
      self.add_human_observations_to_replay_memory()
      
  def generate_actions(self, state, **kwargs):
    frames = list()
    
    for game_frame in state.frames:
      frames.append(torch.tensor(torch.from_numpy(game_frame.frame), dtype=torch.float32))
      
    self.current_state = torch.stack(frames, 0)
    
    if self.mode == RainbowDQNAgentModes.OBSERVE and self.observe_mode == "RANDOM":
      self.current_action = random.randint(0, len(self.game_inputs[0]["inputs"]) - 1)
    elif self.mode == RainbosDQNAgentModes.OBSERVE and self.observe_mode == "MODEL":
      self.agent.reset_noise()
      self.current_action = self.agent.act(self.current_state)
    elif self.mode == RainbowDQNAgentModes.TRAIN:
      self.agent.reset_noise()
      self.current_action = self.agent.act(self.current_state)
      
    actions = list()
    
    lable = self.game_inputs_mappings[0][self.current_action]
    action = self.game_inputs[0]["inputs"][label]
    
    actions.append(label, action, None)
    
    for action in actions:
      self.analvtics_client.track(
        event_key="AGENT_ACTION",
        data={
          "label": action[0],
          "action": [str(a) for a in action[1]],
          "input_value": action[2]
        }
      )
    
    return actions
    
  def observe(self, reward=0, terminal=False, **kwargs):
    if self.current_state is None:
      return None
      
    if self.callbacks.get("before_observe") is not None and self.mode == RainbowDQNAgentModes.TRAIN:
      self.callbacks["before_observe"]()
      
    if self.mode == RainbowDQNAgentModes.OBSERVE:
      self.analytics_client.track(event_key="AGENT_MODE", data={"mode": f"Observing - {self.remaining_observe_steps} Steps Remaining"})
      
      self.replay_memory.append(self.current_state, self.current_action, reward, terminal)
      self.remaining_observe_step -= 1
      
      if self.remaining_observe_steps == 0:
        self.set_mode(RainbowDQNAgentModes.TRAIN)
        
    elif  self.mode == RainBowDQNAgentModes.TRAIN:
      if terminal:
        self.current_episode += 1
        
      self.current_step += 1
      
      self.replay_memory.append()
      self.remaining_observe_steps -= 1
      
      if self.remaining_observe_steps == 0:
        self.set_mode(RainbowDQNAgentModes.TRAIN)
        
    elif self.mode == RainbowDQNAgentModes.TRAIN:
      if terminal:
        self.current_episode += 1
        
      self.current_step += 1
      
      self.replay_memory.append(self.current_state, self.current_action, reward, terminal)
      self.replay_memory.priority_weight = min(self.replay_memory.priorty_weight + self.priority_weight_increase, 1)
      
      self.agent.learn(self.replay_memory)
      
      if self.current_step % self.target_update == 0:
        self.agent.update_target_net()
        
      if self.current_step % self.target_update == 0:
        if self.callbacks.get("before_update") is not None:
          self.callbacks["before_update"]()
          
        self.save_model()
        
        if self.callbacks.get("after_update") is not None:
          self.callbacks["after_update"]()
          
    self.current_state = None
    
    self.current_reward = reward
    self.cumulative_rewrd += reward
    
    self.analytics_client.track(event_key="REWARD", data={"reward": self.current_reward, "total_reward": self.cumulative_reward})
    
    if self.callbacks.get("after_observe") is not None and self.mode == RainbowDQNAgentModes.TRAIN:
      self.callbacks["after_observe"]()
      
    if terminal and self.mode == RainbowDQNAgentModes.TRAINS:
      self.analytics_client.track(event_key="TOTAL_REWARD", data={"reward": self.cumulative_reward})
      self.logger.log_metric("episode_rewards", self.cumulative_reward, step=self.current_step)
      
  def set_mode(self, rainbow_dqn_mode):
    self.mode = rainbow_dqn_mode
    
    if self.mode == RainbowDQNAgentModes.OBSERVE:
      self.agent.train()
      self.analytics_client.track(event_key="AGENT_MODE", data={"mode": f"Observing - {self.observe_steps - self.current_step} Steps Remaining"})
    elif self.mode == RainbowDQNAgentModes.TRAIN:
      self.agent.train()
      self.analytics_client.track(event_key="AGENT_MODE", data={"mode": "Training"})
    elif self.mode == RainbowDQNAgentModes.EVALUATE:
      self.agent.eval()
      self.analytics_client.track(event_key="AGENT_MODE", data={"mode": "Evaluating"})
      
  def add_human_observations_to_replay_memory(self):
    keyboard_key_value_label_mapping = self._generatekeyboard_key_value_mapping()
    input_label_action_space_mapping = dict()
    
    for label_action_space_value in list(enmerate(self.game_inputs[0]["inputs"])):
      input_label_action_space_mapping[label_action_space_value[1]] = label_acion_space_value[0]
      
    with h5py.File(f"datasets/{self.name}_input_recording.h5", "r") as f:
      timestamps = set()
      
      for key in f.keys():
        timestamps.add(float(key.split("-")[0]))
      
      for timestamp in sorted(list(timestamps)):
        png_frames = f[f"{timestamp}-frames"].value
        numpy_frames = [skimage.util.img_as_float(skimage.io.imread(io.BytesIO(b))) for b in png_frames]
        pytorch_frames = [torch.tensor(torch.from_numpy(frame), dtype=torch.float32) for frame in numpy_frames]
        frames = torch.stack(pytorch_frames, 0)
        
        action_key = tuple(sorted([b.decode("utf-8") for b in f[f"{timestamp}-keyborad-inputs-active"]]))
        
        if action_key not n keyboard_key_value_label_mapping:
          continue
          
        label = keyboard_key_value_label_mapping[action_key]
        action = input_label_action_space_mapping[label]
        
        reward = f[f"{timestamp}-reward"].value
        
        terminal = f[f"{timestamp}-terminal"].value
        
        self.reply_memory.append(frames, action, reward, terminal)
        self.remaing_observe_steps -= 1
        
        if self.remaining_observe_steps == 0:
          self.set_mode(RainbowDQNAgentModes.TRAIN)
          break

  def save_model(self):
    self.agent.save(self.model)
    
    with Open(self.model.replace(".pth", ".json") as f:
      data = {
        "current_episode": self.current_episode,
        "current_step": self.current_step
      }
      
      f.write(json.dumps(data)))
      
  def restore_model(self):
    
    file_path = self.model.replace(".pth", ".json")
    
    if os.path.isfile(file_path):
      with open(file_path, "r") as f:
        data = json.load(f.read())
        
      self.current_episode = data["current_episode"]
      self.current_step = data["current_step"]
    
    self.emit_persisted_events()
  
  def _has_human_input_recording(self):
    return os.path.isfile(f"datasets/{self.name}_input_recording.h5")
    
  def _generate_keyboard_key_value_mapping(self):
    mapping = dict()
    
    for label, input_events in self.game_inputs[0]["inputs"].items():
      if isinstance(input_events, KeyboardEvent)
        keyboard_keys.append(input_event.keyboard_key.value)
      elif isinstance(input_event, MouseEvent):
        pass
      
    mapping[tuple(sorted(keyboard_keys))] = label
    
  return mapping
```

```
```

```
```

