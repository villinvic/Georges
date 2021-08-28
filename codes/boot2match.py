"""Generate gecko code for booting into match."""
import enum
from game.enums import Character

template = """C21B148C 00000025 #BootToMatch.asm 
3C608048 60630530 # 31
48000021 7C8802A6
38A000F0 3D808000
618C31F4 7D8903A6
4E800421 480000F8
4E800021 10080201 #4C -> 01
{stock}800000 000000FF
000000{stage} 00000000
00000000 00000000
00000000 FFFFFFFF
FFFFFFFF 00000000
3F800000 3F800000
3F800000 00000000
00000000 00000000
00000000 00000000
00000000 00000000
00000000 00000000
00000000 {char1}{player1}04{costume1}
00FF0000 09007800 # 78>77
400002{cpu1} 00000000
00000000 3F800000
3F800000 3F800000
{char2}{player2}04{costume2} 00FF0000
09007800 400002{cpu2}
00000000 00000000
3F800000 3F800000
3F800000 {char3}{player3}04{costume3}
00FF0000 09007800
400002{cpu3} 00000000
00000000 3F800000
3F800000 3F800000
{char4}{player4}04{costume4} 00FF0000
09007800 400002{cpu4}
00000000 00000000
3F800000 3F800000
3F800000 BB610014
60000000 00000000
"""

"""
.set CPUType_Stay,0x0
.set CPUType_Escape,0x2
.set CPUType_Jump,0x3
.set CPUType_Normal,0x4
.set CPUType_Normal2,0x5
.set CPUType_Nana,0x6
.set CPUType_Defensive,0x7
.set CPUType_Struggle,0x8
.set CPUType_Freak,0x9
.set CPUType_Cooperate,0xA
.set CPUType_SpLwLink,0xB
.set CPUType_SpLwSamus,0xC
.set CPUType_OnlyItem,0xD
.set CPUType_EvZelda,0xE
.set CPUType_NoAct,0xF
.set CPUType_Air,0x10
.set CPUType_Item,0x11
.set CPUType_GuardEdge,0x12
"""


char_ids = {
    Character.Falcon: 0x0,
    Character.DK: 0x1,
    Character.Fox: 0x2,
    Character.GnW: 0x3,
    #Character.Kirby: 0x4, # weird action states
    Character.Bowser: 0x5,
    Character.Link: 0x6,
    Character.Luigi: 0x7,
    Character.Mario: 0x8,
    Character.Marth: 0x9,
    Character.Mewtwo: 0xA,
    Character.Ness: 0xB,
#    'peach': 0xC,
    Character.Pikachu: 0xD,
#    'ics': 0xE,
    Character.Jiggs: 0xF,
    Character.Samus: 0x10,
#    'yoshi': 0x11,
#    'zelda': 0x12,
#    'sheik': 0x13,
    Character.Falco: 0x14,
    Character.YoungLink: 0x15,
    Character.Doc: 0x16,
    Character.Roy: 0x17,
    # 'pichu': 0x18, # bad
    Character.Ganon : 0x19,
}

stage_ids = {
    'fod': 0x2,
    'stadium': 0x3,
    'PeachsCastle': 0x4,
    'KongoJungle': 0x5,
    'Brinstar': 0x6,
    'Corneria': 0x7,
    'yoshis_story': 0x8,
    'Onett': 0x9,
    'MuteCity': 0xA,
    'RainbowCruise': 0xB,
    'jungle_japes': 0xC,
    'GreatBay': 0xD,
    'HyruleTemple': 0xE,
    'BrinstarDepths': 0xF,
    'YoshiIsland': 0x10,
    'GreenGreens': 0x11,
    'Fourside': 0x12,
    'MushroomKingdomI': 0x13,
    'MushroomKingdomII': 0x14,
    'Akaneia': 0x15,
    'Venom': 0x16,
    'PokeFloats': 0x17,
    'BigBlue': 0x18,
    'IcicleMountain': 0x19,
    'IceTop': 0x1A,
    'FlatZone': 0x1B,
    'dream_land': 0x1C,
    'yoshis_island_64': 0x1D,
    'KongoJungle64': 0x1E,
    'battlefield': 0x1F,
    'final_destination': 0x20,
}


class PlayerStatus(enum.IntEnum):
    HUMAN = 0
    CPU = 1


def byte_str(x):
    return '{0:02X}'.format(x)


def generate_match_code(
        player1,
        player2,
        player3,
        player4,
        stock=True,
        stage='final_destination',
        cpu1= 9,
        cpu2= 9,
        cpu3= 9,
        cpu4= 9,
):
    stock = 32 if stock else 0
    kwargs = dict(
        stage=stage_ids[stage],

        player1=player1.type,
        player2=player2.type,
        player3=player3.type,
        player4=player4.type,

        char1=char_ids[player1.char.id],
        char2=char_ids[player2.char.id],
        char3=char_ids[player3.char.id],
        char4=char_ids[player4.char.id],

        cpu1=cpu1,
        cpu2=cpu2,
        cpu3=cpu3,
        cpu4=cpu4,

        stock=stock,

        costume1=player1.costumes['red'],
        costume2=player2.costumes['red'],
        costume3=player3.costumes['blue'],
        costume4=player4.costumes['blue'],
    )

    kwargs = {k: byte_str(v) for k, v in kwargs.items()}
    return template.format(**kwargs)
