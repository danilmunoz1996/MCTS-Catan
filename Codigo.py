#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:27:10 2019

@author: danilo
"""
from PyCatan import CatanGame
from PyCatan import CatanStatuses as CS
from PyCatan import CatanCards as CC
from PyCatan import CatanBuilding as CB
import random as rnd
import math
import sys
import time
import copy
import numpy as np
#import matplotlib.pyplot as plt
class Nodo:
    jugador = None
    estado = None
    victorias = None
    hoja = None
    simulaciones = None
    visitas = None
    childsAdded = None
    childActions = None
    agregados = None
    father_action = None
    
    def __init__(self, juego, player, action = None): #player is from RandomPlayer class
        self.jugador = player
        self.estado = juego
        self.victorias = {0: 0, 1: 0, 2: 0}
        self.hoja = True
        self.simulaciones = 0
        self.visitas = 0
        self.childsAdded = set()
        self.childActions = []
        self.agregados = {}
        self.father_action = action
    
    def getState(self):
        return self.estado
    
    def getPlayer(self):
        return self.jugador
    
    def isLeaf(self):
        if (self.simulaciones == 0):
            return True
        return False
    
    def setLeaf(self, l):
        self.hoja = l
        
    def isTerminal(self):
        return self.estado.has_ended
    
    def getWins(self, player):
        return self.victorias[player]
    
    def getVisits(self):
        return self.visitas
    
    def getFatherAction(self):
        return self.father_action
    
    def getUCT(self, N, padre):
        if (N != 0 and self.simulaciones != 0):     
            v = self.victorias[padre]/self.simulaciones
            return v + math.sqrt(2*math.log(N)/self.simulaciones)
        else:
            return 1
    
    def nextPlayer(self):
        if (self.jugador.player.num == 2):
            return 0
        else:
            return self.jugador.player.num + 1
    
    def update(self, num):
        if (num in self.victorias):
            self.victorias[num] += 1
        self.simulaciones += 1
        self.visitas += 1
    
    def addChild(self, index_of_new_child, child):
        #print("agregados: ", self.agregados)
        if (index_of_new_child not in self.agregados):
            self.agregados[index_of_new_child] = child
            self.childsAdded.add(index_of_new_child)
            return 0
        return -1
    
    def allChilds(self):
        childs = []
        NextStates, actions = self.jugador.NextPosibleStates()
        #print("nextStates: ", NextStates)
        #print("actions: ", actions)
        if (len(NextStates) > 0):
            i = 0
            for nextState in NextStates:
                num_nextplayer = self.nextPlayer()
                newState = self.estado
                newState.add_yield_for_roll(newState.get_roll())
                NextPlayer = RandomPlayer(newState, num_nextplayer)
                child = Nodo(newState, NextPlayer, actions[i])
                childs.append(child)
                i += 1
            self.childActions = actions
        else:
            nextplayer = self.nextPlayer()
            newState = self.estado
            newState.add_yield_for_roll(newState.get_roll())
            NextPlayer = RandomPlayer(newState, nextplayer)
            child = Nodo(newState, NextPlayer, None)
            childs.append(child)
            self.childActions = actions
        #print("allChilds: ", childs)
        return childs
    
    def getChilds(self):
        childs = []
        indexes = []
        for ch in self.childsAdded:
            childs.append(self.agregados[ch])
            indexes.append(ch)
        return childs, indexes
    
    def initChilds(self):
        childs = self.allChilds()
        #print("numero de hijos: ", len(childs))
        i = 0
        for child in childs:
            self.addChild(i, child)
            i += 1

    def getChildsAdded(self): # mapa con childs
        return self.agregados
    
    def updateChild(self, index, child):
        self.agregados[index] = child

class MCTS:
    def __init__(self, roott, max_t, max_nodes):
        self.time = 0
        self.root = roott
        self.max_t = max_t
        self.max_nodes = max_nodes
        self.nodes = 1 #root is the onlyone
        self.depth = 0
        self.depths = []
        
    
    def Move(self):
        return self.move
    
    def Run(self):
        self.root.initChilds()
        self.depths = []
        while (self.time < self.max_t and self.nodes < self.max_nodes):
            #print("tiempo actual: ", self.time, " nodos: ", self.nodes, "simulaciones: ", self.root.simulaciones)
            self.depth = 0
            result, self.root = self.mcts(self.root)
            self.root.update(result)
            self.depths.append(self.depth)
            self.time += 1
            ch, ind = self.max_uct(self.root, self.root.getChildsAdded())
            if self.time%10 is 0: print("posible_action: ", ch.getFatherAction(), "UCB: ", ch.getUCT(self.root.simulaciones, self.root.jugador.player.num), "it: ", self.time, " nodos: ", self.nodes)
            #time.sleep(3)
        print("profundidad maxima del MCTS: ", max(self.depths))
        childs = self.root.getChildsAdded()
        self.move = self.FinaleMove(self.root, childs)
    
    def mcts(self, nodo):
        #print("action: ", nodo.getFatherAction())
        childs = nodo.getChildsAdded()
        if (len(childs) is 0):
            return 3, nodo
        bestChild, index = self.max_uct(nodo, childs)
        #print("Player",nodo.jugador.player.num, " UCT from BestChild: ", bestChild.getUCT(nodo.simulaciones, nodo.jugador.player.num))
        #print("best_action: ", bestChild.getFatherAction())
        result = 0
        #print("nodos: ", self.nodes)
        if (bestChild.isLeaf() and not bestChild.isTerminal()):
            #print(bestChild.getFatherAction())
            bestChild.initChilds()
            self.nodes += len(bestChild.allChilds())
            result, bstChldState = self.Playout(bestChild)
            bestChild.estado = bstChldState
        else:
            self.depth += 1
            result, bestChild = self.mcts(bestChild)
        bestChild.update(result)
        nodo.updateChild(index, bestChild)
        return result, nodo

    def Playout(self, current_node):
        #print("dentro del playout")
        next_player = current_node.nextPlayer()
        next_next_player = self.NextPlayerNum(next_player)
        current_state = current_node.getState()
        current_state2 = copy.deepcopy(current_state)
        P1 = copy.deepcopy(current_node.jugador)
        Player1 = current_node.jugador
        Player2 = RandomPlayer(current_state, next_player)
        Player3 = RandomPlayer(current_state, next_next_player)
        i = 0
        while not current_state.has_ended: #simulacion
            if (i > 100):
                return 3, current_state2
            p1 = Player1.player.get_VP()
            p2 = Player2.player.get_VP()
            p3 = Player3.player.get_VP()
            if (p1 >= 10):
                current_node.jugador = P1
                return Player1.player.num, current_state2
            if (p2 >= 10):
                current_node.jugador = P1
                return Player2.player.num, current_state2
            if (p3 >= 10):
                current_node.jugador = P1
                return Player3.player.num, current_state2
            current_state.add_yield_for_roll(current_state.get_roll())
            Player1, Player2, Player3 = self.Update_players_state(Player1, Player2, Player3, current_state)
            current_state = Player2.Jugar()
            p1 = Player1.player.get_VP()
            p2 = Player2.player.get_VP()
            p3 = Player3.player.get_VP()
            if (p1 >= 10):
                current_node.jugador = P1
                return Player1.player.num, current_state2
            if (p2 >= 10):
                current_node.jugador = P1
                return Player2.player.num, current_state2
            if (p3 >= 10):
                current_node.jugador = P1
                return Player3.player.num, current_state2
            Player1, Player2, Player3 = self.Update_players_state(Player1, Player2, Player3, current_state)
            current_state.add_yield_for_roll(current_state.get_roll())
            Player1, Player2, Player3 = self.Update_players_state(Player1, Player2, Player3, current_state)
            current_state = Player3.Jugar()
            Player1, Player2, Player3 = self.Update_players_state(Player1, Player2, Player3, current_state)
            p1 = Player1.player.get_VP()
            p2 = Player2.player.get_VP()
            p3 = Player3.player.get_VP()
            if (p1 >= 10):
                current_node.jugador = P1
                return Player1.player.num, current_state2
            if (p2 >= 10):
                current_node.jugador = P1
                return Player2.player.num, current_state2
            if (p3 >= 10):
                current_node.jugador = P1
                return Player3.player.num, current_state2
            current_state.add_yield_for_roll(current_state.get_roll())
            Player1, Player2, Player3 = self.Update_players_state(Player1, Player2, Player3, current_state)
            current_state = Player3.Jugar()
            Player1, Player2, Player3 = self.Update_players_state(Player1, Player2, Player3, current_state)
            p1 = Player1.player.get_VP()
            p2 = Player2.player.get_VP()
            p3 = Player3.player.get_VP()
            if (p1 >= 10):
                current_node.jugador = P1
                return Player1.player.num, current_state2
            if (p2 >= 10):
                current_node.jugador = P1
                return Player2.player.num, current_state2
            if (p3 >= 10):
                current_node.jugador = P1
                return Player3.player.num, current_state2
#            print("iteracion :", i, " de la simulacion")
            i += 1
        current_node.jugador = P1
        return current_state.winner, current_state2
    
    def max_uct(self, padre, childs): #Selection step maximun UCT child
        _max = childs[0].getUCT(padre.simulaciones, padre.jugador.player.num)
        _max_child = childs[0]
        index = 0
        for i in range(1, len(childs)):
            if (childs[i].getUCT(padre.simulaciones, padre.jugador.player.num) > _max):
                _max = childs[i].getUCT(padre.simulaciones, padre.jugador.player.num)
                _max_child = childs[i]
                index = i
        return _max_child, index

    def NextPlayerNum(self, num):
        if (num == 2):
            return 0
        return num + 1
    
    def Update_players_state(self, p1, p2, p3, g):
        p1.setEstado(g)
        p2.setEstado(g)
        p3.setEstado(g)
        return p1, p2, p3
    
    def FinaleMove(self, padre, childs):
        childs = padre.getChildsAdded() # los hijos guardados
        if (len(childs) == 1):
            return childs[0].getFatherAction()
        _max_child = childs[0]
        _max_uct = childs[0].getUCT(padre.simulaciones, padre.jugador.player.num)
        for child in childs:
            if (childs[child].getUCT(padre.simulaciones, padre.jugador.player.num) > _max_uct):
                #print(type(_max_child))
                _max_child = childs[child]
                _max_uct = childs[child].getUCT(padre.simulaciones, padre.jugador.player.num)
        return _max_child.getFatherAction()

class MonteCarloPlayer:
    def __init__(self, player):
        self.player = player
        self.iteraciones = 5

    def setEstado(self, newState):
        self.player.setEstado(newState)
    
    def CalculateWeights(self, Hexes, Hexes_nums):
        probWeights = [0,0,0,0,0]
        hex_to_res = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
        numResHex = [0,0,0,0,0]
        stdWeights = [0.9, 1, 0.7, 0.5, 0.6]
        tablero = {0: [0,1,2], 1: [0,1,2,3], 2: [0,1,2,3,4], 3: [0,1,2,3], 4: [0,1,2]}
        for fila in tablero:
            for col in tablero[fila]:
                hex_type = Hexes[fila][col]
                if (hex_type > 0):
                    resource = hex_to_res[hex_type]
                    num_hex = Hexes_nums[fila][col]
                    probWeights[resource] += abs(7 - num_hex)
                    numResHex[resource] += 1
        for resource in range(len(probWeights)):
            probWeights[resource] = stdWeights[resource]*(0.2*(probWeights[resource]/numResHex[resource]))
        return probWeights
    
    def bestCorner(self, posibles_puntos, resWeights, probWeights): # resWeigts es list[], probWeights es float entre 0-0.5
        nums = self.player.estado.board.hex_nums
        hexagonos = self.player.estado.board.hexes
        hex_to_res = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}
        tablero = {0: [0,1,2], 1: [0,1,2,3], 2: [0,1,2,3,4], 3: [0,1,2,3], 4: [0,1,2]}
        ssW = sum(map(lambda i : i * i, resWeights))
        highscore = 0.0
        best_point = []
        for punto in posibles_puntos:
            score = 0
            probs = [0,0,0,0,0]
            hexagonos_vecinos = self.player.estado.board.get_hexes_for_point(punto[0], punto[1])
            #print("hexagonos vecinos: ", hexagonos_vecinos)
            for hexagono_index in hexagonos_vecinos:
                if (hexagono_index[0] > -1 and hexagono_index[1] > -1 and hexagono_index[0] <= 4):
                    if (hexagono_index[1] < len(tablero[hexagono_index[0]])):
                        #print(hexagono_index[0], ", ", hexagono_index[1])
                        hex_res = hexagonos[hexagono_index[0]][hexagono_index[1]] # tipo de hexagono
                        if(hex_res is not 0):
                            res = hex_to_res[hex_res] # recurso del hexagono
                            num_hex = nums[hexagono_index[0]][hexagono_index[1]] # numero del hexagono
                            prob = (6 - abs(num_hex - 7))/36.0
                            probs[res] += prob
            ssP = sum(map(lambda i : i * i, probs))
            if (ssP == 0):
                continue
            probs_nparray = np.asarray(probs)
            resWeights_nparray = np.asarray(resWeights)
            score = np.inner(probs_nparray, resWeights_nparray)/pow(ssW*ssP, probWeights)
            if (score > highscore):
                best_point = punto
                highscore = score
                #print("punto: ", punto)
                #print("score: ", score)
        return best_point
    
    def FirstSettlement(self):
        hexes = self.player.estado.board.hexes
        hex_nums = self.player.estado.board.hex_nums
        pesos = self.CalculateWeights(hexes, hex_nums)
        ubicaciones = self.player.Moves(True)['build_sett']
        ubicacion = self.bestCorner(ubicaciones, pesos, 0.5)
        #print("bestCorner: ", ubicacion)
        self.player.estado.add_settlement(self.player.player.num, ubicacion[0], ubicacion[1], True)
        self.player.setFirstSettlement(ubicacion)
    
    def SecondSettlement(self):
        hexes = self.player.estado.board.hexes
        hex_nums = self.player.estado.board.hex_nums
        pesos = self.CalculateWeights(hexes, hex_nums)
        ubicaciones = self.player.Moves(True)['build_sett']
        ubicacion = self.bestCorner(ubicaciones, pesos, 0.5)
        #print("bestCorner: ", ubicacion)
        self.player.estado.add_settlement(self.player.player.num, ubicacion[0], ubicacion[1], True)
        self.player.setSecondSettlement(ubicacion)
    
    def FirstMove(self):
        self.FirstSettlement()
        self.player.FirstRoad()
        return self.player.estado
    
    def SecondMove(self):
        self.SecondSettlement()
        self.player.SecondRoad()
        return self.player.estado

    def Jugar(self):
        root = Nodo(copy.deepcopy(self.player.estado), copy.deepcopy(self.player))
        MonteCarloTreeSearch = MCTS(copy.deepcopy(root), self.iteraciones, 50000)
        #print("pensando...")
        MonteCarloTreeSearch.Run()
        print("cartas: ", self.player.player.cards)
        print("Best Mov: ", MonteCarloTreeSearch.Move())
        self.player.setEstado(self.player.PlayMove(MonteCarloTreeSearch.Move()))
        return self.player.estado
    
    def setIteraciones(self, its):
        self.iteraciones = its

class RandomPlayer:
    first_sett = []
    second_sett = []
    def __init__(self, juego, p):
        self.estado = juego
        self.player = juego.players[p]
    
    def setEstado(self, nuevoEstado):
        self.estado = nuevoEstado
    
    def setFirstSettlement(self, first):
        self.first_sett = first
    
    def setSecondSettlement(self, second):
        self.second_sett = second
    
    def posibleSettlements(self, inicial):
        tablero = {0:     [0,1,2,3,4,5,6], 
                   1:   [0,1,2,3,4,5,6,7,8], 
                   2: [0,1,2,3,4,5,6,7,8,9,10], 
                   3: [0,1,2,3,4,5,6,7,8,9,10], 
                   4:   [0,1,2,3,4,5,6,7,8], 
                   5:     [0,1,2,3,4,5,6]}
        if not inicial:
            # makes sure the player has the cards to build a settlements
            cards_needed = [
                CC.CARD_WOOD,
                CC.CARD_BRICK,
                CC.CARD_SHEEP,
                CC.CARD_WHEAT
            ]
            # checks the player has the cards
            if not self.player.has_cards(cards_needed):
                return
            roads = (self.estado).board.roads
            puntos_validos = []
            for settle_r in tablero:
                for settle_i in tablero[settle_r]:
                    for r in roads:
                        if (r.point_one == [settle_r, settle_i] or r.point_two == [settle_r, settle_i]):
                            if r.owner == self.player.num:
                                if self.estado.board.point_is_empty(settle_r, settle_i):
                                    point_coords = self.estado.board.get_connected_points(settle_r, settle_i)
                                    counter = 0
                                    for coord in point_coords:
                                        p = self.estado.board.points[coord[0]][coord[1]]
                                        if p is not None:
                                            counter += 1
                                    if (counter == 0):
                                        puntos_validos.append([settle_r, settle_i])
            return puntos_validos
        else:
            puntos_validos = []
            for settle_r in tablero:
                for settle_i in tablero[settle_r]:
                    if self.estado.board.point_is_empty(settle_r, settle_i):
                        point_coords = self.estado.board.get_connected_points(settle_r, settle_i)
                        counter = 0
                        for coord in point_coords:
                            p = self.estado.board.points[coord[0]][coord[1]]
                            if p is not None:
                                counter += 1
                        if (counter == 0):
                            puntos_validos.append([settle_r, settle_i])
            return puntos_validos
    
    def PosibleTrades(self):
        trades_validos = []
        harbors = self.player.get_harbors()
        if (len(harbors)):
            #print("con puertos: ", harbors)
            for harbor in harbors:
                if (harbor != 5):
                    cartas_necesarias = []
                    if (harbor == 0 and self.player.has_cards([CC.CARD_WOOD, CC.CARD_WOOD])):
                        cartas_necesarias = [CC.CARD_WOOD, CC.CARD_WOOD]
                        trades_validos.append([cartas_necesarias, CC.CARD_BRICK])
                        trades_validos.append([cartas_necesarias, CC.CARD_ORE])
                        trades_validos.append([cartas_necesarias, CC.CARD_SHEEP])
                        trades_validos.append([cartas_necesarias, CC.CARD_WHEAT])
#                        trades_validos.append([cartas_necesarias, CC.CARD_WOOD])
                    elif (harbor == 3 and self.player.has_cards([CC.CARD_SHEEP, CC.CARD_SHEEP])):
                        cartas_necesarias = [CC.CARD_SHEEP, CC.CARD_SHEEP]
                        trades_validos.append([cartas_necesarias, CC.CARD_BRICK])
                        trades_validos.append([cartas_necesarias, CC.CARD_ORE])
#                        trades_validos.append([cartas_necesarias, CC.CARD_SHEEP])
                        trades_validos.append([cartas_necesarias, CC.CARD_WHEAT])
                        trades_validos.append([cartas_necesarias, CC.CARD_WOOD])
                    elif (harbor == 1 and self.player.has_cards([CC.CARD_BRICK, CC.CARD_BRICK])):
                        cartas_necesarias = [CC.CARD_BRICK, CC.CARD_BRICK]
#                        trades_validos.append([cartas_necesarias, CC.CARD_BRICK])
                        trades_validos.append([cartas_necesarias, CC.CARD_ORE])
                        trades_validos.append([cartas_necesarias, CC.CARD_SHEEP])
                        trades_validos.append([cartas_necesarias, CC.CARD_WHEAT])
                        trades_validos.append([cartas_necesarias, CC.CARD_WOOD])
                    elif (harbor == 4 and self.player.has_cards([CC.CARD_WHEAT, CC.CARD_WHEAT])):
                        cartas_necesarias = [CC.CARD_WHEAT, CC.CARD_WHEAT]
                        trades_validos.append([cartas_necesarias, CC.CARD_BRICK])
                        trades_validos.append([cartas_necesarias, CC.CARD_ORE])
                        trades_validos.append([cartas_necesarias, CC.CARD_SHEEP])
#                        trades_validos.append([cartas_necesarias, CC.CARD_WHEAT])
                        trades_validos.append([cartas_necesarias, CC.CARD_WOOD])
                    elif (harbor == 2 and self.player.has_cards([CC.CARD_ORE, CC.CARD_ORE])):
                        cartas_necesarias = [CC.CARD_ORE, CC.CARD_ORE]
                        trades_validos.append([cartas_necesarias, CC.CARD_BRICK])
#                        trades_validos.append([cartas_necesarias, CC.CARD_ORE])
                        trades_validos.append([cartas_necesarias, CC.CARD_SHEEP])
                        trades_validos.append([cartas_necesarias, CC.CARD_WHEAT])
                        trades_validos.append([cartas_necesarias, CC.CARD_WOOD])
                        
            harbor_3_1 = False
            for harbor in harbors:
                if (harbor == 5):
                    harbor_3_1 = True
            if (harbor_3_1):
                RECURSOS = {'BRICK': [CC.CARD_BRICK, CC.CARD_BRICK, CC.CARD_BRICK], 
                            'ORE': [CC.CARD_ORE, CC.CARD_ORE, CC.CARD_ORE], 
                            'SHEEP': [CC.CARD_SHEEP, CC.CARD_SHEEP, CC.CARD_SHEEP], 
                            'WHEAT': [CC.CARD_WHEAT, CC.CARD_WHEAT, CC.CARD_WHEAT], 
                            'WOOD': [CC.CARD_WOOD, CC.CARD_WOOD, CC.CARD_WOOD]}
                POSIBLES = [CC.CARD_BRICK, CC.CARD_ORE, CC.CARD_SHEEP, CC.CARD_WHEAT, CC.CARD_WOOD]
                for recurso in RECURSOS:
                    if (self.player.has_cards(RECURSOS[recurso])):
                        for posible in POSIBLES:
                            if (posible != RECURSOS[recurso][0]):
                                #print("RECURSOS[recurso]: ", RECURSOS[recurso])
                                #print("posible: ", posible)
                                trades_validos.append([RECURSOS[recurso], posible])
                                
        RECURSOS = {'BRICK': [CC.CARD_BRICK, CC.CARD_BRICK, CC.CARD_BRICK, CC.CARD_BRICK], 
                 'ORE': [CC.CARD_ORE, CC.CARD_ORE, CC.CARD_ORE, CC.CARD_ORE], 
                 'SHEEP': [CC.CARD_SHEEP, CC.CARD_SHEEP, CC.CARD_SHEEP, CC.CARD_SHEEP], 
                 'WHEAT': [CC.CARD_WHEAT, CC.CARD_WHEAT, CC.CARD_WHEAT, CC.CARD_WHEAT], 
                 'WOOD': [CC.CARD_WOOD, CC.CARD_WOOD, CC.CARD_WOOD, CC.CARD_WOOD]}
        POSIBLES = [CC.CARD_BRICK, CC.CARD_ORE, CC.CARD_SHEEP, CC.CARD_WHEAT, CC.CARD_WOOD]
        for recurso in RECURSOS:
            if (self.player.has_cards(RECURSOS[recurso])):
                for posible in POSIBLES:
                    if (posible != RECURSOS[recurso][0]):
                        trades_validos.append([RECURSOS[recurso], posible])
        return trades_validos
        
    def posibleRoads(self, inicial):
        caminos_validos = []
        puntos = [[0,0], [0,1], [0,2], [0,3], [0,4], [0,5], [0,6], 
                  [1,0], [1,1], [1,2], [1,3], [1,4], [1,5], [1,6], [1,7], [1,8], 
                  [2,0], [2,1], [2,2], [2,3], [2,4], [2,5], [2,6], [2,7], [2,8], [2,9], [2,10],
                  [3,0], [3,1], [3,2], [3,3], [3,4], [3,5], [3,6], [3,7], [3,8], [3,9], [3,10], 
                  [4,0], [4,1], [4,2], [4,3], [4,4], [4,5], [4,6], [4,7], [4,8], 
                  [5,0], [5,1], [5,2], [5,3], [5,4], [5,5], [5,6]]
        cards_needed = [CC.CARD_WOOD, CC.CARD_BRICK]
        if (not self.player.has_cards(cards_needed) and inicial == False):
            return
        for start in puntos:
            for end in puntos:
                #print("start: ", start, " end: ", end)
                location_status = self.player.road_location_is_valid(start = start, end = end)
                if (location_status == CS.ALL_GOOD):
                    exist = False
                    if (len(caminos_validos) > 0):
                        for agregado in caminos_validos:
                            if (start == agregado[1] and end == agregado[0]):
                                exist = True
                                break
                        if (not exist):
                           caminos_validos.append([start, end]) 
                    else:
                        caminos_validos.append([start, end])
        return caminos_validos
        
    
    def posibleCities(self):
        ciudades_validas = []
        tablero = {0:     [0,1,2,3,4,5,6], 
                   1:   [0,1,2,3,4,5,6,7,8], 
                   2: [0,1,2,3,4,5,6,7,8,9,10], 
                   3: [0,1,2,3,4,5,6,7,8,9,10], 
                   4:   [0,1,2,3,4,5,6,7,8], 
                   5:     [0,1,2,3,4,5,6]}
        needed_cards = [CC.CARD_WHEAT, 
                        CC.CARD_WHEAT, 
                        CC.CARD_ORE, 
                        CC.CARD_ORE, 
                        CC.CARD_ORE]
        if not (self.player.has_cards(needed_cards)):
            return
        for r in tablero:
            for i in tablero[r]:
                if (self.estado.board.points[r][i] != None):
                    if (self.estado.board.points[r][i].owner == self.player.num):
                        if (self.estado.board.points[r][i].type == CB.BUILDING_SETTLEMENT):
                            ciudades_validas.append([r, i])
        return ciudades_validas
    
    def Moves(self, inicio):
        actions = {}
        aux = []
        settls = self.posibleSettlements(inicio)
        if (settls is not None):#eric, lizama, daniel = updatePlayersState(eric, lizama, daniel, g)
            for coords in settls:
                aux.append(coords)
            if (len(aux) > 0):
                actions['build_sett'] = aux
        aux = []
        roads = self.posibleRoads(inicio)
        if (roads is not None):
            for camino in self.posibleRoads(inicio):
                aux.append([camino[0], camino[1]])
            if (len(aux) > 0):
                actions['build_road'] = aux
        aux = []
        cities = self.posibleCities()
        if (cities is not None):
            for city in cities:
                aux.append(city)
            if (len(aux) > 0):
                actions['build_city'] = aux
        aux = []
        trades = self.PosibleTrades()
        if (trades is not None):
            for trade in trades:
                aux.append(trade)
            if (len(aux) > 0):
                actions['trade'] = aux
        #print("actions: ", actions)
        actions['Nada'] = [0]
        return actions
    
    def Jugar(self):
        mov = self.NextMove(False)
        self.estado = self.PlayMove(mov)
        return self.estado
    
    def Jugar2(self):
        mov = self.NextMove(False)
        print(mov)
        self.estado = self.PlayMove(mov)
        return self.estado
    
    def PlayMove(self, mov):
        #print("NextMove: ", mov)
        cpy_game = self.estado
        cpy_game2 = copy.deepcopy(cpy_game)
        if (mov is not None):
            if (mov[0] == 'build_sett'):
                cpy_game.add_settlement(self.player.num, mov[1][0], mov[1][1], False)
                return cpy_game
            elif (mov[0] == 'build_road'):
                cpy_game.add_road(self.player.num, mov[1][0], mov[1][1], False)
                return cpy_game
            elif (mov[0] == 'build_city'):
                cpy_game.add_city(mov[1][0], mov[1][1], self.player.num)
            elif (mov[0] == 'trade'):
                cpy_game.trade_to_bank(self.player.num, mov[1][0], mov[1][1])
                #if (r == 8):
                    #print("cambiar: ", mov[1][0])
                    #print("por: ", mov[1][1])
                    #print("player: ", self.player.num)
                    #print("harbors: ", self.player.get_harbors())
                    #return "mal"
            #else:
                #print("unknownmov: ", mov[0])
        #else:
            #print("no hacer nada")
        self.estado = cpy_game2
        return cpy_game

    def NextPosibleStates(self):
        NextStates = []
        actionsToState = []
        actions = self.Moves(inicio = False)
        for action_type in actions:
            for posible_move in actions[action_type]:
                if (posible_move is not None):
                    #print("tipo: ", action_type, " accion: ", posible_move)
                    PosibleState = self.PlayMove([action_type, posible_move])
                    NextStates.append(PosibleState)
                    actionsToState.append([action_type, posible_move])
        return NextStates, actionsToState
    
    def NextMove(self, inicio):
        actions = self.Moves(inicio)
        action_types = []
        if (len(actions) == 0):
            return
        else:
            for action in actions:
                action_types.append(action)     
            random_type_action = action_types[rnd.randint(0, len(action_types)-1)]
            random_action = actions[random_type_action][rnd.randint(0, len(actions[random_type_action])-1)]
#            print([random_type_action, random_action])
            return [random_type_action, random_action]
        
    def FirstSettlement(self):
        actions = self.Moves(True)
        random_settle = actions['build_sett'][rnd.randint(0, len(actions['build_sett'])-1)]
        self.estado.add_settlement(self.player.num, random_settle[0], random_settle[1], True)
        self.first_sett = random_settle
        
    def SecondSettlement(self):
        actions = self.Moves(True)
        random_settle = actions['build_sett'][rnd.randint(0, len(actions['build_sett'])-1)]
        print(self.estado.add_settlement(self.player.num, random_settle[0], random_settle[1], True))
        self.second_sett = random_settle
    
    def FirstRoad(self):
        points = self.estado.board.get_connected_points(self.first_sett[0], self.first_sett[1])
        inicio_camino = points[0]
        fin_camino = [self.first_sett[0], self.first_sett[1]]
        self.estado.add_road(self.player.num, inicio_camino, fin_camino, True)
    
    def SecondRoad(self):
        points = self.estado.board.get_connected_points(self.second_sett[0], self.second_sett[1])
        inicio_camino = points[0]
        fin_camino = [self.first_sett[0], self.first_sett[1]]
        self.estado.add_road(self.player.num, inicio_camino, fin_camino, True)
    
    def FirstGame(self):
        self.FirstSettlement()
        self.FirstRoad()
        return self.estado
    
    def SecondGame(self):
        self.SecondSettlement()
        self.SecondRoad()
        return self.estado

def updatePlayersState(player1, player2, player3, game):
    player1.setEstado(game)
    player2.setEstado(game)
    player3.setEstado(game)
    return player1, player2, player3

def printGameStatus(game):
    print(game.board.hexes)
    print(game.board.all_hex_nums)
    print(game.board.harbors)
    print("--------------------")
    print(game.board.points)
    print("--------------------")
    return
cartas = [CC.CARD_BRICK, CC.CARD_ORE, CC.CARD_SHEEP, CC.CARD_WHEAT, CC.CARD_WOOD]
def rand_card():
    r = rnd.randint(0, 4)
    return cartas[r]
max_depth = 0
depths = []
vp1 = []
vp2 = []
vp3 = []
for i in range(4):
    g = CatanGame()
    #printGameStatus(g)
    eric = RandomPlayer(g, 0)
    lizama = RandomPlayer(g, 1)
    daniel = RandomPlayer(g, 2)
    #ericMCTS = MonteCarloPlayer(eric2)
    danielMCTS = MonteCarloPlayer(daniel)
    # Eric 1
    g = eric.FirstGame()
    eric, lizama, danielMCTS = updatePlayersState(eric, lizama, danielMCTS, g)
    ##printGameStatus(g)
    ## daniel 1
    g = danielMCTS.FirstMove()
    eric, lizama, danielMCTS = updatePlayersState(eric, lizama, danielMCTS, g)
    ##printGameStatus(g)
    ## lizama 1
    g = lizama.FirstGame()
    eric, lizama, danielMCTS = updatePlayersState(eric, lizama, danielMCTS, g)
    ##printGameStatus(g)
    ## Daniel 2
    g = lizama.SecondGame()
    eric, lizama, danielMCTS = updatePlayersState(eric, lizama, danielMCTS, g)
    ##printGameStatus(g)
    ## daniel 2
    g = danielMCTS.SecondMove()
    eric, lizama, danielMCTS = updatePlayersState(eric, lizama, danielMCTS, g)
    ##printGameStatus(g)
    ## Eric 2
    g = eric.SecondGame()
    eric, lizama, danielMCTS = updatePlayersState(eric, lizama, danielMCTS, g)
    printGameStatus(g)
    ##automatico
    depth = 0
    depth_max = 200
    while True:
        danielMCTS.setIteraciones(1000)
#        ericMCTS.setIteraciones(30)
        print("Profundidad: ", depth)
        p1 = eric.player.victory_points
        p3 = lizama.player.victory_points
        p2 = danielMCTS.player.player.victory_points
        print("puntos eric IA: ", p1)
        print("puntos lizama: ", p3)
        print("puntos IA: ", p2)
        if depth > depth_max or p1 == 10 or p2 == 10 or p3 == 10:
            print("Profundidad: ", depth)
            if (depth > max_depth):
                max_depth = depth
            depths.append(depth)
            break
        print("Eric Juega")
        g.add_yield_for_roll(g.get_roll())
        eric, lizama, danielMCTS = updatePlayersState(eric, lizama, danielMCTS, g)
        print("cartas eric: ", eric.player.cards)
        g = eric.Jugar2()
        eric, lizama, danielMCTS = updatePlayersState(eric, lizama, danielMCTS, g)
    #    printGameStatus(g)
        print("Inteligencia Artificial Juega")
        g.add_yield_for_roll(g.get_roll())
        eric, lizama, danielMCTS = updatePlayersState(eric, lizama, danielMCTS, g)
        g = danielMCTS.Jugar()
        print("cartas de la IA despues de jugar: ", danielMCTS.player.player.cards)
        eric, lizama, danielMCTS = updatePlayersState(eric, lizama, danielMCTS, g)
        print("Lizama Juega")
        g.add_yield_for_roll(g.get_roll())
        eric, lizama, danielMCTS = updatePlayersState(eric, lizama, danielMCTS, g)
        print("cartas lizama: ", lizama.player.cards)
        g = lizama.Jugar2()
        eric, lizama, danielMCTS = updatePlayersState(eric, lizama, danielMCTS, g)
    #    printGameStatus(g)
        depth += 1
        #printGameStatus(g)
    print(max_depth)
    vp1.append(p1) #ericMCTS
    vp2.append(p2) #danielMCTS
    vp2.append(p3) #lizama
    depths.append(max_depth)