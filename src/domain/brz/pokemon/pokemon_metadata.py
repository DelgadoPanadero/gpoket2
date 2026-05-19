from enum import IntEnum

NUM_GENERATIONS = 10  # gen1-gen9 + margen


class PokemonType(IntEnum):
    NORMAL = 0
    FIRE = 1
    WATER = 2
    ELECTRIC = 3
    GRASS = 4
    ICE = 5
    FIGHTING = 6
    POISON = 7
    GROUND = 8
    FLYING = 9
    PSYCHIC = 10
    BUG = 11
    ROCK = 12
    GHOST = 13
    DRAGON = 14
    DARK = 15
    STEEL = 16
    FAIRY = 17


class EvolutionStage(IntEnum):
    BASIC = 0
    STAGE_1 = 1
    STAGE_2 = 2


class Shininess(IntEnum):
    NORMAL = 0
    SHINY = 1


class PokemonColor(IntEnum):
    RED = 0
    BLUE = 1
    YELLOW = 2
    GREEN = 3
    BLACK = 4
    BROWN = 5
    PURPLE = 6
    GRAY = 7
    WHITE = 8
    PINK = 9


class PokemonShape(IntEnum):
    BALL = 0
    SQUIGGLE = 1
    FISH = 2
    ARMS = 3
    BLOB = 4
    UPRIGHT = 5
    LEGS = 6
    QUADRUPED = 7
    WINGS = 8
    TENTACLES = 9
    HEADS = 10
    HUMANOID = 11
    BUG_WINGS = 12
    ARMOR = 13


class PokemonHabitat(IntEnum):
    CAVE = 0
    FOREST = 1
    GRASSLAND = 2
    MOUNTAIN = 3
    RARE = 4
    ROUGH_TERRAIN = 5
    SEA = 6
    URBAN = 7
    WATERS_EDGE = 8
