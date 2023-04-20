import nmmo
import numpy as np

import pufferlib.emulation

from feature_extractor.entity_helper import EntityHelper
from feature_extractor.game_state import GameState
from feature_extractor.map_helper import MapHelper
from feature_extractor.item_helper import ItemHelper
from feature_extractor.market_helper import MarketHelper
from feature_extractor.stat_helper import StatHelper

from team_helper import TeamHelper

class FeatureExtractor(pufferlib.emulation.Featurizer):
  def __init__(self, teams, team_id: int, config: nmmo.config.AllGameSystems):
    super().__init__(teams, team_id)
    self._config = config

    self._team_helper = TeamHelper(teams)
    self._team_id = team_id
    team_size = self._team_helper.team_size[team_id]

    self.game_state = GameState(config, team_size)

    # NOTE: target_tracker merged to entity_helper
    self.entity_helper = EntityHelper(config, self._team_helper, team_id)
    self.stat_helper = StatHelper(config, self.entity_helper)

    self.map_helper = MapHelper(config, self.entity_helper)
    self.item_helper = ItemHelper(config, self.entity_helper)
    self.market_helper = MarketHelper(config, self.entity_helper, self.item_helper)

  def reset(self, init_obs):
    self.game_state.reset(init_obs)
    self.map_helper.reset()
    self.stat_helper.reset()
    self.entity_helper.reset(init_obs)
    self.item_helper.reset()
    self.market_helper.reset()

  def __call__(self, obs):
    # NOTE: these updates needs to be in this precise order
    self.game_state.update(obs)
    self.entity_helper.update(obs)
    self.map_helper.update(obs, self.game_state.curr_step)
    self.item_helper.update(obs) # use & sell
    self.market_helper.update(obs, self.game_state.curr_step) # buy

    # CHECK ME: we can get better stat from the event log. Do we need stat_helper?
    self.stat_helper.update(obs)

    """Update finished. Generating features"""
    # CHECK ME: how item force_use/sell/buy_idx are traslated into actions?

    # tile dim: (team_size, TILE_NUM_CHANNELS, *TILE_IMG_SIZE)
    tile = self.map_helper.extract_tile_feature()

    # item_type dim: (team_size, config.ITEM_INVENTORY_CAPACITY)
    # item dim: (team_size, config.ITEM_INVENTORY_CAPACITY, ITEM_NUM_FEATURES)
    item_type, item = self.item_helper.extract_item_feature()

    # team does NOT include legal actions
    # team dim: (team_size, SELF_NUM_FEATURES - sum(ACTION_NUM_DIM.values())
    # team_mask dim: (team_size)
    team, team_mask = self.entity_helper.team_features_and_mask(self.map_helper)

    # npc dim: (team_size, ENTITY_NUM_NPCS_CONSIDERED, ENTITY_NUM_FEATURES)
    # npc_mask dim: (team_size, ENTITY_NUM_NPCS_CONSIDERED)
    npc, npc_mask = self.entity_helper.npcs_features_and_mask()

    # enemy dim: (team_size, ENTITY_NUM_ENEMIES_CONSIDERED, ENTITY_NUM_FEATURES)
    # enemy_mask dim: (team_size, ENTITY_NUM_ENEMIES_CONSIDERED)
    enemy, enemy_mask = self.entity_helper.enemies_features_and_mask()

    # game dim: (GAME_NUM_FEATURES)
    game = self.game_state.extract_game_feature(obs)

    state = {
      'tile': tile,
      'item_type': item_type,
      'item': item,
      'team': team,
      'npc': npc,
      'enemy': enemy,
      'team_mask': team_mask,
      'npc_mask': npc_mask,
      'enemy_mask': enemy_mask,
      'game': game,
      'legal': {
        'move': self.map_helper.legal_moves(obs), # dim: (team_size, 4)
        # 'target': np.zeros((self.team_size, 19), dtype=np.float32),
        # 'use': np.zeros((self.team_size, 3), dtype=np.float32),
        # 'sell': np.zeros((self.team_size, 3), dtype=np.float32),

        # xcxc: check dimensions
        # 'target': self.entity_helper.legal_target(obs, self.npc_tgt, self.enemy_tgt),
        'use': self.item_helper.legal_use_consumables(), # dim: (team_size, 3)
        'sell': self.item_helper.legal_sell_consumables(), # dim: (team_size, 3)
      },
      'prev_act': self.game_state.previous_actions(), # dim (self.team_size, 4) for now
      'reset': np.array([self.game_state.curr_step == 0])  # for resetting RNN hidden,
    }
    return state

def trans_action(self, actions):
  return self.action.trans_actions(actions)
