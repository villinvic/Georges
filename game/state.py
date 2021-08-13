import numpy as np

from game.state_manager import StateManager
from game.enums import ActionState
from game.memory_watcher import MemoryWatcherZMQ
from input.action_space import ActionSpace


class PlayerState:
    _all = [

        ('facing', 1.),
        ('vel_x', 0.5),
        ('vel_y', 0.5),
        ('attack_vel_x', 0.5),
        ('attack_vel_y', 0.5),
        ('ground_vel', 0.5),
        ('x', 0.05),
        ('y', 0.05),
        ('on_ground', 1.),
        ('action_frame', 0.01),
        ('percent', 0.01),
        ('hitlag_left', 0.1),
        ('hitstun_left', 0.1),
        ('shield_size', 0.01),
        ('charging_smash', 0.5),
        ('jumps_used', 0.2),
        ('body_state', 0.5),
        ('stocks', 0.25),
#        ('action_state', 1.),
        ('Doc', 1.),
        ('Mario', 1.),
        ('Luigi', 1.),
        ('Bowser', 1.),
        ('Peach', 1.),
        ('Yoshi', 1.),
        ('DK', 1.),
        ('Falcon', 1.),
        ('Ganon', 1.),
        ('Falco', 1.),
        ('Fox', 1.),
        ('Ness', 1.),
        ('Icies', 1.),
        ('Kirby', 1.),
        ('Samus', 1.),
        ('Zelda', 1.),
        ('Link', 1.),
        ('YoungLink', 1.),
        ('Pichu', 1.),
        ('Pikachu', 1.),
        ('Jiggs', 1.),
        ('Mewtwo', 1.),
        ('GnW', 1.),
        ('Marth', 1.),
        ('Roy', 1.),
        ('DeadDown', 1.),
        ('DeadLeft', 1.),
        ('DeadRight', 1.),
        ('DeadUp', 1.),
        ('DeadUpStar', 1.),
        ('DeadUpStarIce', 1.),
        ('DeadUpFall', 1.),
        ('DeadUpFallHitCamera', 1.),
        ('DeadUpFallHitCameraFlat', 1.),
        ('DeadUpFallIce', 1.),
        ('DeadUpFallHitCameraIce', 1.),
        ('Sleep', 1.),
        ('Rebirth', 1.),
        ('RebirthWait', 1.),
        ('Wait', 1.),
        ('WalkSlow', 1.),
        ('WalkMiddle', 1.),
        ('WalkFast', 1.),
        ('Turn', 1.),
        ('TurnRun', 1.),
        ('Dash', 1.),
        ('Run', 1.),
        ('RunDirect', 1.),
        ('RunBrake', 1.),
        ('KneeBend', 1.),
        ('JumpF', 1.),
        ('JumpB', 1.),
        ('JumpAerialF', 1.),
        ('JumpAerialB', 1.),
        ('Fall', 1.),
        ('FallF', 1.),
        ('FallB', 1.),
        ('FallAerial', 1.),
        ('FallAerialF', 1.),
        ('FallAerialB', 1.),
        ('FallSpecial', 1.),
        ('FallSpecialF', 1.),
        ('FallSpecialB', 1.),
        ('DamageFall', 1.),
        ('Squat', 1.),
        ('SquatWait', 1.),
        ('SquatRv', 1.),
        ('Landing', 1.),
        ('LandingFallSpecial', 1.),
        ('Attack11', 1.),
        ('Attack12', 1.),
        ('Attack13', 1.),
        ('Attack100Start', 1.),
        ('Attack100Loop', 1.),
        ('Attack100End', 1.),
        ('AttackDash', 1.),
        ('AttackS3Hi', 1.),
        ('AttackS3HiS', 1.),
        ('AttackS3S', 1.),
        ('AttackS3LwS', 1.),
        ('AttackS3Lw', 1.),
        ('AttackHi3', 1.),
        ('AttackLw3', 1.),
        ('AttackS4Hi', 1.),
        ('AttackS4HiS', 1.),
        ('AttackS4S', 1.),
        ('AttackS4LwS', 1.),
        ('AttackS4Lw', 1.),
        ('AttackHi4', 1.),
        ('AttackLw4', 1.),
        ('AttackAirN', 1.),
        ('AttackAirF', 1.),
        ('AttackAirB', 1.),
        ('AttackAirHi', 1.),
        ('AttackAirLw', 1.),
        ('LandingAirN', 1.),
        ('LandingAirF', 1.),
        ('LandingAirB', 1.),
        ('LandingAirHi', 1.),
        ('LandingAirLw', 1.),
        ('DamageHi1', 1.),
        ('DamageHi2', 1.),
        ('DamageHi3', 1.),
        ('DamageN1', 1.),
        ('DamageN2', 1.),
        ('DamageN3', 1.),
        ('DamageLw1', 1.),
        ('DamageLw2', 1.),
        ('DamageLw3', 1.),
        ('DamageAir1', 1.),
        ('DamageAir2', 1.),
        ('DamageAir3', 1.),
        ('DamageFlyHi', 1.),
        ('DamageFlyN', 1.),
        ('DamageFlyLw', 1.),
        ('DamageFlyTop', 1.),
        ('DamageFlyRoll', 1.),
        ('LightGet', 1.),
        ('HeavyGet', 1.),
        ('LightThrowF', 1.),
        ('LightThrowB', 1.),
        ('LightThrowHi', 1.),
        ('LightThrowLw', 1.),
        ('LightThrowDash', 1.),
        ('LightThrowDrop', 1.),
        ('LightThrowAirF', 1.),
        ('LightThrowAirB', 1.),
        ('LightThrowAirHi', 1.),
        ('LightThrowAirLw', 1.),
        ('HeavyThrowF', 1.),
        ('HeavyThrowB', 1.),
        ('HeavyThrowHi', 1.),
        ('HeavyThrowLw', 1.),
        ('LightThrowF4', 1.),
        ('LightThrowB4', 1.),
        ('LightThrowHi4', 1.),
        ('LightThrowLw4', 1.),
        ('LightThrowAirF4', 1.),
        ('LightThrowAirB4', 1.),
        ('LightThrowAirHi4', 1.),
        ('LightThrowAirLw4', 1.),
        ('HeavyThrowF4', 1.),
        ('HeavyThrowB4', 1.),
        ('HeavyThrowHi4', 1.),
        ('HeavyThrowLw4', 1.),
        ('SwordSwing1', 1.),
        ('SwordSwing3', 1.),
        ('SwordSwing4', 1.),
        ('SwordSwingDash', 1.),
        ('BatSwing1', 1.),
        ('BatSwing3', 1.),
        ('BatSwing4', 1.),
        ('BatSwingDash', 1.),
        ('ParasolSwing1', 1.),
        ('ParasolSwing3', 1.),
        ('ParasolSwing4', 1.),
        ('ParasolSwingDash', 1.),
        ('HarisenSwing1', 1.),
        ('HarisenSwing3', 1.),
        ('HarisenSwing4', 1.),
        ('HarisenSwingDash', 1.),
        ('StarRodSwing1', 1.),
        ('StarRodSwing3', 1.),
        ('StarRodSwing4', 1.),
        ('StarRodSwingDash', 1.),
        ('LipStickSwing1', 1.),
        ('LipStickSwing3', 1.),
        ('LipStickSwing4', 1.),
        ('LipStickSwingDash', 1.),
        ('ItemParasolOpen', 1.),
        ('ItemParasolFall', 1.),
        ('ItemParasolFallSpecial', 1.),
        ('ItemParasolDamageFall', 1.),
        ('LGunShoot', 1.),
        ('LGunShootAir', 1.),
        ('LGunShootEmpty', 1.),
        ('LGunShootAirEmpty', 1.),
        ('FireFlowerShoot', 1.),
        ('FireFlowerShootAir', 1.),
        ('ItemScrew', 1.),
        ('ItemScrewAir', 1.),
        ('DamageScrew', 1.),
        ('DamageScrewAir', 1.),
        ('ItemScopeStart', 1.),
        ('ItemScopeRapid', 1.),
        ('ItemScopeFire', 1.),
        ('ItemScopeEnd', 1.),
        ('ItemScopeAirStart', 1.),
        ('ItemScopeAirRapid', 1.),
        ('ItemScopeAirFire', 1.),
        ('ItemScopeAirEnd', 1.),
        ('ItemScopeStartEmpty', 1.),
        ('ItemScopeRapidEmpty', 1.),
        ('ItemScopeFireEmpty', 1.),
        ('ItemScopeEndEmpty', 1.),
        ('ItemScopeAirStartEmpty', 1.),
        ('ItemScopeAirRapidEmpty', 1.),
        ('ItemScopeAirFireEmpty', 1.),
        ('ItemScopeAirEndEmpty', 1.),
        ('LiftWait', 1.),
        ('LiftWalk1', 1.),
        ('LiftWalk2', 1.),
        ('LiftTurn', 1.),
        ('GuardOn', 1.),
        ('Guard', 1.),
        ('GuardOff', 1.),
        ('GuardSetOff', 1.),
        ('GuardReflect', 1.),
        ('DownBoundU', 1.),
        ('DownWaitU', 1.),
        ('DownDamageU', 1.),
        ('DownStandU', 1.),
        ('DownAttackU', 1.),
        ('DownFowardU', 1.),
        ('DownBackU', 1.),
        ('DownSpotU', 1.),
        ('DownBoundD', 1.),
        ('DownWaitD', 1.),
        ('DownDamageD', 1.),
        ('DownStandD', 1.),
        ('DownAttackD', 1.),
        ('DownFowardD', 1.),
        ('DownBackD', 1.),
        ('DownSpotD', 1.),
        ('Passive', 1.),
        ('PassiveStandF', 1.),
        ('PassiveStandB', 1.),
        ('PassiveWall', 1.),
        ('PassiveWallJump', 1.),
        ('PassiveCeil', 1.),
        ('ShieldBreakFly', 1.),
        ('ShieldBreakFall', 1.),
        ('ShieldBreakDownU', 1.),
        ('ShieldBreakDownD', 1.),
        ('ShieldBreakStandU', 1.),
        ('ShieldBreakStandD', 1.),
        ('FuraFura', 1.),
        ('Catch', 1.),
        ('CatchPull', 1.),
        ('CatchDash', 1.),
        ('CatchDashPull', 1.),
        ('CatchWait', 1.),
        ('CatchAttack', 1.),
        ('CatchCut', 1.),
        ('ThrowF', 1.),
        ('ThrowB', 1.),
        ('ThrowHi', 1.),
        ('ThrowLw', 1.),
        ('CapturePulledHi', 1.),
        ('CaptureWaitHi', 1.),
        ('CaptureDamageHi', 1.),
        ('CapturePulledLw', 1.),
        ('CaptureWaitLw', 1.),
        ('CaptureDamageLw', 1.),
        ('CaptureCut', 1.),
        ('CaptureJump', 1.),
        ('CaptureNeck', 1.),
        ('CaptureFoot', 1.),
        ('EscapeF', 1.),
        ('EscapeB', 1.),
        ('Escape', 1.),
        ('EscapeAir', 1.),
        ('ReboundStop', 1.),
        ('Rebound', 1.),
        ('ThrownF', 1.),
        ('ThrownB', 1.),
        ('ThrownHi', 1.),
        ('ThrownLw', 1.),
        ('ThrownLwWomen', 1.),
        ('Pass', 1.),
        ('Ottotto', 1.),
        ('OttottoWait', 1.),
        ('FlyReflectWall', 1.),
        ('FlyReflectCeil', 1.),
        ('StopWall', 1.),
        ('StopCeil', 1.),
        ('MissFoot', 1.),
        ('CliffCatch', 1.),
        ('CliffWait', 1.),
        ('CliffClimbSlow', 1.),
        ('CliffClimbQuick', 1.),
        ('CliffAttackSlow', 1.),
        ('CliffAttackQuick', 1.),
        ('CliffEscapeSlow', 1.),
        ('CliffEscapeQuick', 1.),
        ('CliffJumpSlow1', 1.),
        ('CliffJumpSlow2', 1.),
        ('CliffJumpQuick1', 1.),
        ('CliffJumpQuick2', 1.),
        ('AppealR', 1.),
        ('AppealL', 1.),
        ('ShoulderedWait', 1.),
        ('ShoulderedWalkSlow', 1.),
        ('ShoulderedWalkMiddle', 1.),
        ('ShoulderedWalkFast', 1.),
        ('ShoulderedTurn', 1.),
        ('ThrownFF', 1.),
        ('ThrownFB', 1.),
        ('ThrownFHi', 1.),
        ('ThrownFLw', 1.),
        ('CaptureCaptain', 1.),
        ('CaptureYoshi', 1.),
        ('YoshiEgg', 1.),
        ('CaptureKoopa', 1.),
        ('CaptureDamageKoopa', 1.),
        ('CaptureWaitKoopa', 1.),
        ('ThrownKoopaF', 1.),
        ('ThrownKoopaB', 1.),
        ('CaptureKoopaAir', 1.),
        ('CaptureDamageKoopaAir', 1.),
        ('CaptureWaitKoopaAir', 1.),
        ('ThrownKoopaAirF', 1.),
        ('ThrownKoopaAirB', 1.),
        ('CaptureKirby', 1.),
        ('CaptureWaitKirby', 1.),
        ('ThrownKirbyStar', 1.),
        ('ThrownCopyStar', 1.),
        ('ThrownKirby', 1.),
        ('BarrelWait', 1.),
        ('Bury', 1.),
        ('BuryWait', 1.),
        ('BuryJump', 1.),
        ('DamageSong', 1.),
        ('DamageSongWait', 1.),
        ('DamageSongRv', 1.),
        ('DamageBind', 1.),
        ('CaptureMewtwo', 1.),
        ('CaptureMewtwoAir', 1.),
        ('ThrownMewtwo', 1.),
        ('ThrownMewtwoAir', 1.),
        ('WarpStarJump', 1.),
        ('WarpStarFall', 1.),
        ('HammerWait', 1.),
        ('HammerWalk', 1.),
        ('HammerTurn', 1.),
        ('HammerKneeBend', 1.),
        ('HammerFall', 1.),
        ('HammerJump', 1.),
        ('HammerLanding', 1.),
        ('KinokoGiantStart', 1.),
        ('KinokoGiantStartAir', 1.),
        ('KinokoGiantEnd', 1.),
        ('KinokoGiantEndAir', 1.),
        ('KinokoSmallStart', 1.),
        ('KinokoSmallStartAir', 1.),
        ('KinokoSmallEnd', 1.),
        ('KinokoSmallEndAir', 1.),
        ('Entry', 1.),
        ('EntryStart', 1.),
        ('EntryEnd', 1.),
        ('DamageIce', 1.),
        ('DamageIceJump', 1.),
        ('CaptureMasterhand', 1.),
        ('CapturedamageMasterhand', 1.),
        ('CapturewaitMasterhand', 1.),
        ('ThrownMasterhand', 1.),
        ('CaptureKirbyYoshi', 1.),
        ('KirbyYoshiEgg', 1.),
        ('CaptureLeadead', 1.),
        ('CaptureLikelike', 1.),
        ('DownReflect', 1.),
        ('CaptureCrazyhand', 1.),
        ('CapturedamageCrazyhand', 1.),
        ('CapturewaitCrazyhand', 1.),
        ('ThrownCrazyhand', 1.),
        ('BarrelCannonWait', 1.),
        ('Wait1', 1.),
        ('Wait2', 1.),
        ('Wait3', 1.),
        ('Wait4', 1.),
        ('WaitItem', 1.),
        ('SquatWait1', 1.),
        ('SquatWait2', 1.),
        ('SquatWaitItem', 1.),
        ('GuardDamage', 1.),
        ('EscapeN', 1.),
        ('AttackS4Hold', 1.),
        ('HeavyWalk1', 1.),
        ('HeavyWalk2', 1.),
        ('ItemHammerWait', 1.),
        ('ItemHammerMove', 1.),
        ('ItemBlind', 1.),
        ('DamageElec', 1.),
        ('FuraSleepStart', 1.),
        ('FuraSleepLoop', 1.),
        ('FuraSleepEnd', 1.),
        ('WallDamage', 1.),
        ('CliffWait1', 1.),
        ('CliffWait2', 1.),
        ('SlipDown', 1.),
        ('Slip', 1.),
        ('SlipTurn', 1.),
        ('SlipDash', 1.),
        ('SlipWait', 1.),
        ('SlipStand', 1.),
        ('SlipAttack', 1.),
        ('SlipEscapeF', 1.),
        ('SlipEscapeB', 1.),
        ('AppealS', 1.),
        ('Zitabata', 1.),
        ('CaptureKoopaHit', 1.),
        ('ThrownKoopaEndF', 1.),
        ('ThrownKoopaEndB', 1.),
        ('CaptureKoopaAirHit', 1.),
        ('ThrownKoopaAirEndF', 1.),
        ('ThrownKoopaAirEndB', 1.),
        ('ThrownKirbyDrinkSShot', 1.),
        ('ThrownKirbySpitSShot', 1.),
        ('Unselected', 1.)
    ]

    indexes = {
        name : i for i, (name, _) in enumerate(_all)
    }

    onehot_offsets = {
        'action_state': 43,
        'character': 18,
    }

    action_state_dim = 384

    names = np.array([
        name for name, _ in _all
    ])

    scales = np.array([
        scale for _, scale in _all
    ], dtype=np.float32)

    size = len(scales)

    @staticmethod
    def get(p):
        prefix = 'p' + str(p) + '_'
        return np.array([
            prefix + name for name in PlayerState.names
            ])


class GameState:

    action_dim = ActionSpace().dim
    size = PlayerState.size * 4 + action_dim

    character_offsets = np.array([PlayerState.onehot_offsets['character'] + PlayerState.size * i for i in range(4)])
    action_state_offsets = np.array([PlayerState.onehot_offsets['action_state'] + PlayerState.size * i for i in range(4)])
    action_offset = PlayerState.size * 4

    augmentation = np.array(
        ['action_%d' % i for i in range(action_dim)])

    indexes = {
        key: i for i, key in enumerate(np.concatenate([PlayerState.get(p) for p in range(4)]+[augmentation], axis=0))
    }


    scales = np.concatenate([np.tile(PlayerState.scales, 4), np.ones(action_dim, dtype=np.float32)], axis=0)

    #x = [[indexes['p%d_stocks' %i] for i in range(0, 2)], [indexes['p%d_stocks' %i] for i in range(2, 4)]]
    stock_indexes = np.array( [indexes['p0_stocks'], indexes['p1_stocks'], indexes['p2_stocks'], indexes['p3_stocks']], dtype=np.int32)

    _data_placeholder = np.zeros(size, dtype=np.float32)


    def __init__(self, mw_path, instance_id, test=False):

        self.frame = 0
        self.chars = None
        self.types = np.zeros(4, dtype=np.int32)

        self.state = np.copy(self._data_placeholder)


        # for one_hot cleaning
        self.action_states_ = np.zeros(4, dtype=np.int32)
        self.characters_ = np.zeros(4, dtype=np.int32)
        self.action_ = 0
        self.manager = StateManager(self, test)
        self.mw = MemoryWatcherZMQ(mw_path, instance_id)
        self.mw_path = mw_path

        self.team_stocks = np.array([8,8], dtype=np.int32)
        #self.is_dying = np.zeros(4, dtype=np.int32)

        self.write_locations()


    def update_players(self, players):
        self.chars = [p.char().get() for p in players]


    def __getitem__(self, item):
        #TODO type ?
        if isinstance(item, str):
            if item == 'frame':
                return self.frame
            else:
                return self.state[self.indexes[item]]
        return self.state[item]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            if key == 'frame':
                self.frame = value
            elif 'action_state' in key:
                if value > 384:
                    return
                p = int(key[1])
                index = self.action_state_offsets[p]+value
                self.state[self.action_states_[p]] = 0.
                self.state[index] = 1.0
                self.action_states_[p] = index
            elif 'character' in key:
                p = int(key[1])
                index = self.character_offsets[p]+value
                self.state[self.characters_[p]] = 0.
                self.state[index] = 1.0
                self.characters_[p] = index
            elif key == 'action':
                index = self.action_offset+value
                self.state[self.action_] = 0.
                self.state[index] = 1.0
                self.action_ = index
            elif 'type' in key:
                p = int(key[1])
                self.types[p] = value
            else:
                self.state[self.indexes[key]] = value
        else:
            self.state[key] = value

    def player_state(self, player_index):
        return self.state[PlayerState.size*player_index: PlayerState.size*(player_index+1)]

    def action(self):
        return self.state[PlayerState.size*4:]

    def compute_frames_left(self):
        for i, char in enumerate(self.chars):
            p = 'p%d_' % i
            action_state = self.action_states_[i]-self.action_state_offsets[i]
            if action_state > 0:
                #self.is_dying[i] = int(action_state <= 10)

                max_frame = char.frame_data[ActionState(action_state).name]
                if True and max_frame > 0 :
                    remaining_frames = max_frame - self.state[self.indexes[p+'action_frame']]

                    if remaining_frames < 0:
                        if remaining_frames > -2  :
                            char.frame_data[ActionState(action_state).name] = self.state[self.indexes[p+'action_frame']]
                            print(char, ActionState(action_state).name, max_frame, self.state[self.indexes[p+'action_frame']])

                    self.state[self.indexes[p+'action_frame']] = np.clip(max_frame - self.state[self.indexes[p+'action_frame']], 0, np.inf)

    def compute_stocks_left(self):
        #for i in range(4):
        #print(list(self.state[PlayerState.size*3:]))
        self.team_stocks[:] = np.sum(self.state[self.stock_indexes[:2]]), np.sum(self.state[self.stock_indexes[2:]])


    def get(self, target, player_index, action):
        index = self.action_offset + action
        self.state[self.action_] = 0.
        self.state[index] = 1.0
        self.action_ = index

        if player_index == 0:
            target[:] = self.state
        else:
            target[:PlayerState.size] = self.player_state(player_index)

            if player_index == 1:
                target[PlayerState.size:PlayerState.size*2] = self.player_state(0)
                target[PlayerState.size * 2:] = self.state[PlayerState.size * 2:]
            elif player_index == 2:
                target[PlayerState.size:PlayerState.size * 2] = self.player_state(3)
                target[PlayerState.size * 2:PlayerState.size * 4] = self.state[:PlayerState.size * 2]
                target[PlayerState.size * 4:] = self.action()
            else:
                target[PlayerState.size:PlayerState.size * 2] = self.player_state(2)
                target[PlayerState.size * 2:PlayerState.size * 4] = self.state[:PlayerState.size * 2]
                target[PlayerState.size * 4:] = self.action()

        target *= self.scales
        np.clip(target, -5, 5, out=target)
        np.nan_to_num(target)

    def init(self, amount=150):
        updated_ram = None
        while updated_ram is None:
            updated_ram = next(self.mw)
        self.manager.handle(updated_ram)

        for _ in range(amount):

            last_frame = self.frame
            while last_frame == self.frame:
                updated_ram = next(self.mw)
                if updated_ram is not None:
                    self.manager.handle(updated_ram)

    def update(self):
        c = 0
        while True:
            last_frame = self.frame
            updated_ram = next(self.mw)
            if updated_ram is not None:
                if updated_ram == -1 :
                    return updated_ram, None
                # Update the state
                self.manager.handle(updated_ram)
                self.compute_frames_left()
                self.compute_stocks_left()
            if self.frame > last_frame > 0 :
                if self.frame - last_frame > 1:
                    print('skipped', self.frame - last_frame, 'frames.')
                return self.is_done()

            c+= 1

            done, result = self.is_done()
            if done:
                return done, result

            if c > 1e4 :
                print('Game crash ?')
                return done, None

    def is_done(self):
        if self.team_stocks[0] == 0:
            return True, 0
        elif self.team_stocks[1] == 0:
            return True, 1
        else:
            return False, 0.5

    def write_locations(self):
        """Writes out the locations list to the appropriate place under dolphin_dir."""
        with open(self.mw_path+'/Locations.txt', 'w') as f:
            f.write('\n'.join(self.manager.locations()))

    def bind_to_instance(self):
        self.mw.bind()

    def unbind_from_instance(self):
        self.mw.unbind()






