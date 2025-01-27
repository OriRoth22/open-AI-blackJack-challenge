# open-AI-blackJack-challenge

Blackjack
../../../_images/blackjack1.gif
This environment is part of the Toy Text environments which contains general information about the environment.

Action Space

Discrete(2)

Observation Space

Tuple(Discrete(32), Discrete(11), Discrete(2))

import

gymnasium.make("Blackjack-v1")

Blackjack is a card game where the goal is to beat the dealer by obtaining cards that sum to closer to 21 (without going over 21) than the dealers cards.

Description
The game starts with the dealer having one face up and one face down card, while the player has two face up cards. All cards are drawn from an infinite deck (i.e. with replacement).

The card values are:

Face cards (Jack, Queen, King) have a point value of 10.

Aces can either count as 11 (called a ‘usable ace’) or 1.

Numerical cards (2-9) have a value equal to their number.

The player has the sum of cards held. The player can request additional cards (hit) until they decide to stop (stick) or exceed 21 (bust, immediate loss).

After the player sticks, the dealer reveals their facedown card, and draws cards until their sum is 17 or greater. If the dealer goes bust, the player wins.

If neither the player nor the dealer busts, the outcome (win, lose, draw) is decided by whose sum is closer to 21.

This environment corresponds to the version of the blackjack problem described in Example 5.1 in Reinforcement Learning: An Introduction by Sutton and Barto [1].

Action Space
The action shape is (1,) in the range {0, 1} indicating whether to stick or hit.

0: Stick

1: Hit

Observation Space
The observation consists of a 3-tuple containing: the player’s current sum, the value of the dealer’s one showing card (1-10 where 1 is ace), and whether the player holds a usable ace (0 or 1).

The observation is returned as (int(), int(), int()).

Starting State
The starting state is initialised in the following range.

Observation

Min

Max

Player current sum

4

12

Dealer showing card value

2

11

Usable Ace

0

1

Rewards
win game: +1

lose game: -1

draw game: 0

win game with natural blackjack: +1.5 (if natural is True) +1 (if natural is False)

Episode End
The episode ends if the following happens:

Termination:

The player hits and the sum of hand exceeds 21.

The player sticks.

An ace will always be counted as usable (11) unless it busts the player.

Information
No additional information is returned.

Arguments
import gymnasium as gym
gym.make('Blackjack-v1', natural=False, sab=False)
natural=False: Whether to give an additional reward for starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).

sab=False: Whether to follow the exact rules outlined in the book by Sutton and Barto. If sab is True, the keyword argument natural will be ignored. If the player achieves a natural blackjack and the dealer does not, the player will win (i.e. get a reward of +1). The reverse rule does not apply. If both the player and the dealer get a natural, it will be a draw (i.e. reward 0).

References
[1] R. Sutton and A. Barto, “Reinforcement Learning: An Introduction” 2020. [Online]. Available: http://www.incompleteideas.net/book/RLbook2020.pdf

Version History
v1: Fix the natural handling in Blackjack

v0: Initial version release

