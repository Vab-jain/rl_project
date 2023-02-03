import numpy as np
import gym
from gym import spaces
import pygame
from pathlib import Path
import json
# import gym_examples

'''
 Reference: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

'''

class GridWorldEnv(gym.Env):
	metadata = {"render_modes": ["human","rgb_array"], "render_fps": 4}

	def __init__(self, render_mode=None, jsondata=None):

		assert (jsondata, 'No DATA file!!!')
		with open(jsondata) as f:
			data = json.load(f)

		self._compass_to_dir = {
									'north': 0,
									'east' : 1,
									'south': 2,
									'west' : 3, 
								}
		self._n_dir = len(self._compass_to_dir)

		# loading metadata from the json file
		# grid info
		self._gridsz_num_rows = int(data["gridsz_num_rows"])
		self._gridsz_num_cols =int(data["gridsz_num_cols"])
		# pregrid agent info
		self._agent_pregrid_loc = np.array([data["pregrid_agent_row"], data["pregrid_agent_col"]],dtype=int)
		self._agent_pregrid_dir = self._compass_to_dir[data["pregrid_agent_dir"]]
		# postgrid agent info
		self._agent_postgrid_loc = np.array([data["postgrid_agent_row"], data["postgrid_agent_col"]], dtype=int)
		self._agent_postgrid_dir = self._compass_to_dir[data["postgrid_agent_dir"]]
		# env grey-cells (walls) info
		self._walls_loc = np.array(data["walls"], dtype=int)
		# pre/post-grid markers info
		self._markers_pregrid_loc = np.array(data["pregrid_markers"], dtype=int)
		self._markers_postgrid_loc = np.array(data["postgrid_markers"], dtype=int)
		

		self._grid_size = (self._gridsz_num_rows, self._gridsz_num_cols)

		# initialize the current status of the agent and markers
		self._agent_loc = self._agent_pregrid_loc
		self._agent_dir = self._agent_pregrid_dir
		self._markers_loc = np.empty((0,2), dtype=int)
		if self._markers_pregrid_loc.size:
			self._markers_loc = np.append(self._markers_loc, [self._markers_pregrid_loc])
		
		

		# Observations are dictionaries with agent's and the target's locations.
		# Each location is encoded as an element of {0,...., 'size'}^2, i.e. MultiBinary([size, size]).
		self.observation_space = spaces.Dict(
			{
				"agent_loc": spaces.MultiBinary(self._grid_size), 
				"agent_dir": spaces.Discrete(4), 
				"agent_postgrid_loc": spaces.MultiBinary(self._grid_size), 
				"agent_postgrid_dir":spaces.Discrete(4), 
				"markers_loc": spaces.MultiBinary(self._grid_size), 
				"markers_postgrid_loc": spaces.MultiBinary(self._grid_size), 
				"walls_loc": spaces.MultiBinary(self._grid_size)
			}
		)

		# We have 6 actions, corresponding to "move","turnLeft","turnRight","pickMarker","putMarker","finish"
		self.action_space = spaces.Discrete(6)

		self.actions_dict = {
							0: self._action_move,
							1: self._action_turnLeft,
							2: self._action_turnRight,
							3: self._action_pickMarker,
							4: self._action_putMarker,
							5: self._action_finish
						}

		self._orient_to_direction = {
			0: np.array([-1, 0]),	# Move North
			1: np.array([0, 1]),	# Move East
			2: np.array([1, 0]),	# Move South
			3: np.array([0, -1]),	# Move West
		}

		assert render_mode is None or render_mode in self.metadata["render_modes"]
		self.render_mode = render_mode
		
		"""
		If human-rendering is used, `self.window` will be a reference
		to the window that we draw to. `self.clock` will be a clock that is used
		to ensure that the environment is rendered at the correct framerate in
		human-mode. They will remain `None` until human-mode is used for the
		first time.
		"""
		self.window = None
		self.clock = None


	def _encode_location(self, locations):
		grid_vector = np.zeros(self._grid_size,dtype=int)
		if not self._terminated:
			if locations.size:
				if len(locations.shape)==1:
					grid_vector[locations[0],locations[1]] = 1
				else:
					for loc in locations:
						grid_vector[loc[0],loc[1]] = 1
		return grid_vector

	def _encode_direction(self, direction):
		dir_vector = np.zeros(self._n_dir, dtype=int)
		dir_vector[direction] = 1
		return dir_vector

	# (private) function to get obsevations
	def _get_obs(self):
		return {
					"agent_loc": self._encode_location(self._agent_loc), 
					"agent_dir": self._encode_direction(self._agent_dir), 
					"agent_postgrid_loc": self._encode_location(self._agent_postgrid_loc), 
					"agent_postgrid_dir": self._encode_direction(self._agent_postgrid_dir), 
					"markers_loc": self._encode_location(self._markers_loc), 
					"markers_postgrid_loc": self._encode_location(self._markers_postgrid_loc), 
					"walls_loc": self._encode_location(self._walls_loc)
				}

	def _get_info(self):
		return {
					"agent_loc": self._encode_location(self._agent_loc), 
					"agent_dir": self._encode_direction(self._agent_dir), 
					"agent_postgrid_loc": self._encode_location(self._agent_postgrid_loc), 
					"agent_postgrid_dir": self._encode_direction(self._agent_postgrid_dir), 
					"markers_loc": self._encode_location(self._markers_loc), 
					"markers_postgrid_loc": self._encode_location(self._markers_postgrid_loc), 
					"walls_loc": self._encode_location(self._walls_loc),
					"solved": self._solved
				}


	def reset(self, seed=None, jsondata=None, flattened_obs=True):
		# We need the following line to seed self.np_random
		super().reset(seed=seed)

		assert (jsondata, 'No DATA file!!!')
		with open(jsondata) as f:
			data = json.load(f)
		
		# loading metadata from the json file
		# grid info
		self._gridsz_num_rows = int(data["gridsz_num_rows"])
		self._gridsz_num_cols =int(data["gridsz_num_cols"])
		# pregrid agent info
		self._agent_pregrid_loc = np.array([data["pregrid_agent_row"], data["pregrid_agent_col"]],dtype=int)
		self._agent_pregrid_dir = self._compass_to_dir[data["pregrid_agent_dir"]]
		# postgrid agent info
		self._agent_postgrid_loc = np.array([data["postgrid_agent_row"], data["postgrid_agent_col"]], dtype=int)
		self._agent_postgrid_dir = self._compass_to_dir[data["postgrid_agent_dir"]]
		# env grey-cells (walls) info
		self._walls_loc = np.array(data["walls"], dtype=int)
		# pre/post-grid markers info
		self._markers_pregrid_loc = np.array(data["pregrid_markers"], dtype=int)
		self._markers_postgrid_loc = np.array(data["postgrid_markers"], dtype=int)
		

		self._grid_size = (self._gridsz_num_rows, self._gridsz_num_cols)

		# initialize the current status of the agent and markers
		self._agent_loc = self._agent_pregrid_loc
		self._agent_dir = self._agent_pregrid_dir
		self._markers_loc = np.empty((0,2), dtype=int)
		if self._markers_pregrid_loc.size:
			self._markers_loc = np.append(self._markers_loc, [self._markers_pregrid_loc])
		
		# initialize the termination flag
		self._terminated = False
		# initialize the task completion flag
		self._solved = False

		observation = self._get_obs()
		info = self._get_info()
		if flattened_obs:
			observation = self._flatten_obs(observation)
			# info = self._flatten_obs(info)

		if self.render_mode == "human":
			self._render_frame()

		return observation, info


	def _action_move(self):
		# Update the current location of the agent => Add the direction vector to current location
		self._agent_loc = self._agent_loc + self._orient_to_direction[self._agent_dir]
		
		# check if agent has left the grid; if "yes" then "crash" the episode
		if not np.all((self._agent_loc>=0)&(self._agent_loc<self._grid_size)):
			self._terminated = True		# set termination flag

		# check if agent has hit any grey-cells; if "yes" then "crash" the episode
		for wall in self._walls_loc:
			if np.array_equal(self._agent_loc,wall):
				self._terminated = True		# set termination flag
				break

	def _action_turnLeft(self):
		self._agent_dir = (self._agent_dir - 1)%self._n_dir

	def _action_turnRight(self):
		self._agent_dir = (self._agent_dir + 1)%self._n_dir
	
	def _action_pickMarker(self):
		if self._markers_loc.size:
			self._markers_loc = np.delete(self._markers_loc, np.where(np.all(self._markers_loc == self._agent_loc,axis=1)),axis=0)

	def _action_putMarker(self):
		self._markers_loc = np.append(self._markers_loc, [self._agent_loc], axis=0)

	def _action_finish(self):
		self._terminated = True

	def _check_postgrid(self):
		# condition = (np.array_equal(self._agent_loc, self._agent_postgrid_loc)) & (np.array_equal(self._agent_orient, self._agent_postgrid_orient)) & (np.array_equal(self._markers_loc, self._markers_postgrid_loc))
		if ((np.array_equal(self._agent_loc, self._agent_postgrid_loc)) & (np.array_equal(self._agent_dir, self._agent_postgrid_dir)) & (np.array_equal(self._markers_loc, self._markers_postgrid_loc))):
			self._terminated = True
			return True
		else:
			return False
		

	def step(self, action, flattened_obs=True):
		# To-Do
			# Execute the action function based on action value
			# Check if terminated
			# Calculate reward
		
		self.actions_dict[action]()

		terminated = self._terminated
		solved = self._check_postgrid()

		if solved & terminated & action==5:
			self._solved = True
			reward = 10
		elif terminated:
			reward = -5
		else:
			reward = -1


		observation = self._get_obs()
		info = self._get_info()
		if flattened_obs:
			observation = self._flatten_obs(observation)
			# info = self._flatten_obs(info)
			
		if self.render_mode == "human":
			self._render_frame()

		return observation, reward, terminated, info

	def _flatten_obs(self, obs):
		'''
		Function to take observation dictionary and flatten it into a feature vector
		'''
		agent_loc = obs["agent_loc"].flatten()
		agent_dir = np.array([obs["agent_dir"]]).flatten()
		agent_postgrid_loc = obs["agent_postgrid_loc"].flatten()
		agent_postgrid_dir = np.array([obs["agent_postgrid_dir"]]).flatten()
		markers_loc = obs["markers_loc"].flatten()
		markers_postgrid_loc = obs["markers_postgrid_loc"].flatten()
		walls_loc = obs["walls_loc"].flatten()

		flatten_obs = np.concatenate((agent_loc, agent_dir, agent_postgrid_loc, agent_postgrid_dir, markers_loc, markers_postgrid_loc, walls_loc))
		return flatten_obs

	######## TO-DO: IMPLEMENT RENDER #######
	def render(self, toFilePath = None):
		dir_to_compass = {	0:'^', 1:'>', 2:'V', 3:'<', }

		# create grid with appropiate size and 
		grid = ["".join(["." for j in range(self._gridsz_num_cols)])
                for i in range(self._gridsz_num_rows)]
		
		for loc in self._walls_loc:
			grid[loc[0]] = grid[loc[0]][:loc[1]] + "W" + grid[loc[0]][loc[1]+1:]

		for loc in self._markers_loc:
			grid[loc[0]] = grid[loc[0]][:loc[1]] + "M" + grid[loc[0]][loc[1]+1:]

		# # add 'M' for marker position
		# for i in range(self._gridsz_num_rows):
		# 	for j in range(self._gridsz_num_cols):
		# 		if (i, j) in self._markers_loc:
		# 			grid[i] = grid[i][:j] + "M" + grid[i][j+1:]
		# 		# if (i, j) in self._walls_loc:
		# 		# 	grid[i] = grid[i][:j] + "W" + grid[i][j+1:]
		
		# # add agent postion
		grid[self._agent_loc[0]] = grid[self._agent_loc[0]][:self._agent_loc[1]] + \
			str(dir_to_compass[self._agent_dir]) + grid[self._agent_loc[0]][self._agent_loc[1]+1:]
		
		if toFilePath:
			with open(f'{toFilePath}/result.txt', 'a') as f:
				f.write('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in grid]))
				f.write('\n\n')
		else:
			print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in grid]))
			print('\n\n')


	def close(self):
		if self.window is not None:
			pygame.display.quit()
			pygame.quit()

if __name__ == "__main__":
	# data_folder = Path("/home/vj/Link to WiSe 2022-23/Reinforcement Learning/project/Implementation")
	data_folder = Path("/home/vj/Link to WiSe 2022-23/Reinforcement Learning/project/Implementation/datasets/data_easy/train/task")
	datapath = f'{data_folder}/0_task.json'
	env = GridWorldEnv(jsondata=data_folder/"500_task.json")
	# Args reset: agent_preGrid=None, agent_postgrid=None, markers_preGrid=None, markers_postgrid=None, walls=None, seed=None
	obs, info = env.reset(jsondata=datapath)
	# obs, info = env.reset(agent_preGrid=[1,2,0],agent_postgrid=[2,3,1],markers_preGrid=[[1,3],[0,0]],markers_postgrid=[[2,1]], walls=[[0,1],[0,2]])

	# print(obs)
	# print(info)
	env.render()
	print('\n\n')

	obs, reward, terminated, info = env.step(2)
	env.render()
	print('\n\n\n')

	obs, reward, terminated, info = env.step(5)
	env.render()
	# print(info)
	