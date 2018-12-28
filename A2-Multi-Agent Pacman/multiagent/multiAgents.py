# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        currentGhostStates = currentGameState.getGhostStates()
        currentFood = currentGameState.getFood()
        currentFoodList = currentFood.asList()
        currentScaredTimes = []

        for s in currentGhostStates:
            currentScaredTimes.append(s.scaredTimer)

        score = float("inf")

        for f in currentFoodList:
            d = abs(f[0] - newPos[0]) + abs(f[1] - newPos[1])
            if Directions.STOP in action:
                return float("-inf")
            score = min(d, score)

        for s in newGhostStates:
            if newPos == s.getPosition():
                return float("-inf")

        factor = 0.1
        return factor/(float(score) + factor)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.computeMiniMaxValues(gameState, 0, 0, self.index, self.depth)

    def computeMiniMaxValues(self, gameState, agentIndex, nodeDepth, pacmanIndex, maxDepth):

        numAgents = gameState.getNumAgents()

        if not agentIndex < numAgents:
            agentIndex = 0
            nodeDepth += 1

        if nodeDepth == maxDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        legalActions = gameState.getLegalActions(agentIndex)
        aValue = None

        if agentIndex == pacmanIndex:
            value = float("-inf")
        else:
            value = float("inf")

        for action in legalActions:
            if action != Directions.STOP:
                successor = gameState.generateSuccessor(agentIndex, action)
                tempValue = self.computeMiniMaxValues(successor, agentIndex + 1, nodeDepth, pacmanIndex, maxDepth)

                if agentIndex == pacmanIndex and tempValue > value:
                    value = tempValue
                    aValue = action
                elif (not agentIndex == pacmanIndex) and tempValue < value:
                    value = tempValue
                    aValue = action

        if agentIndex == pacmanIndex and nodeDepth == 0:
            return aValue
        else:
            return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # return self.alphaBeta(gameState, 0, 0, float("-inf"), float("inf"))
        # return self.alphaBeta(gameState, 0, 0, float("-inf"), float("inf"), True)[0]
        return self.alphaBeta(gameState, 0, 0, float("-inf"), float("inf"), True)[0]

    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta, isMaxPlayer):
        bestMove = None

        if gameState.isWin() or gameState.isLose():
            if isMaxPlayer:
                return bestMove, self.evaluationFunction(gameState)
            else:
                return self.evaluationFunction(gameState)

        if isMaxPlayer:
            value = float("-inf")
            for action in gameState.getLegalActions(0):
                nextPos = gameState.generateSuccessor(0, action)
                nextVal = self.alphaBeta(nextPos, 1, depth, alpha, beta, False)

                if value < nextVal:
                    value = nextVal
                    bestMove = action

                if value < beta:
                    alpha = max(alpha, value)
                else:
                    return bestMove, value

            return bestMove, value
        else:
            value = float("inf")
            nextTurn = agentIndex + 1
            if agentIndex >= gameState.getNumAgents() - 1:
                nextTurn = 0
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                if nextTurn == 0 and depth == self.depth - 1:
                    score = self.evaluationFunction(successor)
                elif nextTurn == 0 and not depth == self.depth - 1:
                    score = self.alphaBeta(successor, 0, depth + 1, alpha, beta, True)[1]
                else:
                    score = self.alphaBeta(successor, nextTurn, depth, alpha, beta, False)

                value = min(value, score)
                if value > alpha:
                    beta = min(value, beta)
                else:
                    return value
            return value




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.computeExpectimaxValue(gameState, 0, 0, self.depth)

    def computeExpectimaxValue(self, gameState, agentIndex, nodeDepth, maxDepth):
        numAgents = gameState.getNumAgents()

        if not agentIndex < numAgents:
            agentIndex = 0
            nodeDepth += 1

        if nodeDepth == maxDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        if agentIndex == self.index:
            value = float("-inf")
            actionValue = None
        else:
            value = 0
            probability = 1 / float(len(gameState.getLegalActions(agentIndex)))

        actionList = gameState.getLegalActions(agentIndex)
        for legalActions in actionList:
            if not legalActions == Directions.STOP:
                successor = gameState.generateSuccessor(agentIndex, legalActions)
                tempValue = self.computeExpectimaxValue(successor, agentIndex + 1, nodeDepth, maxDepth)

                if agentIndex == self.index:
                    if tempValue > value:
                        value = tempValue
                        actionValue = legalActions
                else:
                    value += (tempValue * probability)
                    actionValue = legalActions


        if agentIndex == self.index and nodeDepth == 0:
            return actionValue
        elif agentIndex == self.index and not nodeDepth == 0:
                return value
        else:
            return value

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      1. Get a ghost score by calculating the distance to each ghost
            If the ghost is scared we make score higher since in pacman's favor
            If the ghost is not, then we make score lower in order to avoid
      2. Get a food score by calculating the distance to each food
            We use inverse distance since farther foods are less favourable
            The food score will be the closest food (highest inverse distance)
      3. We add the current game score with the 2 scores calculated above
    """
    "*** YOUR CODE HERE ***"
    ghostScore = 0
    ghostStateList = currentGameState.getGhostStates()

    for g in ghostStateList:
        pacManPosition = currentGameState.getPacmanPosition()
        ghostPosition = g.getPosition()
        ghostDistance = abs(pacManPosition[0] - ghostPosition[0]) + abs(pacManPosition[1] - ghostPosition[1])

        if g.scaredTimer > 0:
            ghostScore += ghostDistance
        else:
            ghostScore -= ghostDistance

    foodScore = 0
    foodList = currentGameState.getFood().asList()

    for f in foodList:
        pacManPosition = currentGameState.getPacmanPosition()
        distance = 1.0 / (abs(pacManPosition[0] - f[0]) + abs(pacManPosition[1] - f[1]))
        if distance > foodScore:
            foodScore = distance

    gameScore = currentGameState.getScore()
    return gameScore + (ghostScore * 2) + (foodScore * 4)

# Abbreviation
better = betterEvaluationFunction

