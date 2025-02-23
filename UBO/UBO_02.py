import numpy as np
import networkx as nx
from typing import Dict, Set

class UBODetector:
    def __init__(self, convergence_threshold=1e-6, max_iterations=1000):
        self.graph = nx.DiGraph()
        self.entity_types = {}
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

    def add_ownership(self, owner: str, owned: str, percentage: float):
        if not (0 <= percentage <= 100):
            raise ValueError("Ownership percentage must be between 0 and 100")
        self.graph.add_edge(owner, owned, weight=percentage/100)
        self.entity_types.setdefault(owner, 'company')
        self.entity_types.setdefault(owned, 'company')

    def set_individual(self, entity: str):
        self.entity_types[entity] = 'individual'

    def detect_ubo(self, target: str, threshold=0.25) -> Dict[str, float]:
        if target not in self.graph or self.entity_types.get(target) != 'company':
            raise ValueError("Invalid target company")

        # Find relevant entities through graph traversal
        relevant_companies = self._find_relevant_companies(target)
        relevant_individuals = self._find_relevant_individuals(relevant_companies)
        
        # Create subgraph of relevant entities
        subgraph = self.graph.subgraph(relevant_companies | relevant_individuals)
        
        # Create matrix indices only for relevant companies
        comp_index = {c: i for i, c in enumerate(relevant_companies)}
        target_idx = comp_index[target]

        # Initialize ownership vectors
        size = len(relevant_companies)
        direct_ownership = np.zeros(size)
        total_ownership = np.zeros(size)

        # Populate direct ownership from individuals
        for i, comp in enumerate(relevant_companies):
            for pred in subgraph.predecessors(comp):
                if self.entity_types[pred] == 'individual':
                    direct_ownership[i] += subgraph[pred][comp]['weight']

        # Iterative ownership calculation
        prev_ownership = np.zeros(size)
        for _ in range(self.max_iterations):
            new_ownership = direct_ownership.copy()
            for i, comp in enumerate(relevant_companies):
                for pred in subgraph.predecessors(comp):
                    if pred in comp_index:  # Only consider relevant companies
                        pred_idx = comp_index[pred]
                        new_ownership[i] += subgraph[pred][comp]['weight'] * prev_ownership[pred_idx]
            
            if np.allclose(new_ownership, prev_ownership, atol=self.convergence_threshold):
                break
            prev_ownership = new_ownership.copy()

        # Calculate final UBO percentages
        ubo = {}
        for ind in relevant_individuals:
            total = self._calculate_total_ownership(subgraph, comp_index, 
                                                   target, ind, prev_ownership)
            total = self._calculate_cycle_amplification(subgraph, target, ind, 
                                                       total, relevant_companies)
            if total >= threshold:
                ubo[ind] = min(round(total, 4), 1.0)

        return ubo

    def _find_relevant_companies(self, target: str) -> Set[str]:
        """Find companies connected through ownership paths to target"""
        relevant = set()
        stack = [target]
        
        while stack:
            current = stack.pop()
            if current in relevant:
                continue
            relevant.add(current)
            
            # Add companies owning this company
            for predecessor in self.graph.predecessors(current):
                if self.entity_types.get(predecessor) == 'company':
                    stack.append(predecessor)
            
            # Add companies owned by this company
            for successor in self.graph.successors(current):
                if self.entity_types.get(successor) == 'company':
                    stack.append(successor)
        
        return relevant

    def _find_relevant_individuals(self, companies: Set[str]) -> Set[str]:
        """Find individuals connected to relevant companies"""
        return {n for n in self.graph.nodes 
               if self.entity_types.get(n) == 'individual'
               and any(self.graph.has_edge(n, c) for c in companies)}

    def _calculate_total_ownership(self, subgraph, comp_index, 
                                 target, investor, ownership_vec) -> float:
        """Calculate combined direct and indirect ownership"""
        total = 0.0
        # Direct ownership
        if subgraph.has_edge(investor, target):
            total += subgraph[investor][target]['weight']
        
        # Indirect ownership
        for comp in subgraph.predecessors(target):
            if comp in comp_index and subgraph.has_edge(investor, comp):
                comp_idx = comp_index[comp]
                total += subgraph[investor][comp]['weight'] * ownership_vec[comp_idx]
        
        return total

    def _calculate_cycle_amplification(self, subgraph, target: str, investor: str, 
                                     base: float, relevant_companies: Set[str]) -> float:
        """Calculate ownership amplification through control loops"""
        visited = set()
        stack = [(target, 1.0)]
        amplification = base
        
        while stack:
            current, path_weight = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            for predecessor in subgraph.predecessors(current):
                if predecessor == investor or predecessor not in relevant_companies:
                    continue
                
                edge_weight = subgraph[predecessor][current]['weight']
                new_weight = path_weight * edge_weight
                
                # Amplify if loop detected back to target
                if predecessor == target:
                    amplification += base * new_weight / (1 - new_weight)
                else:
                    stack.append((predecessor, new_weight))
        
        return min(amplification, 1.0)

# Test Case with Unrelated Company
detector = UBODetector()
detector.add_ownership('PersonX', 'CompanyA', 2)
detector.add_ownership('CompanyA', 'CompanyB', 98)
detector.add_ownership('CompanyB', 'CompanyC', 100)
detector.add_ownership('CompanyC', 'CompanyA', 100)  # Cycle
detector.add_ownership('PersonY', 'CompanyX', 100)    # Unrelated company
detector.set_individual('PersonX')
detector.set_individual('PersonY')

ubos = detector.detect_ubo('CompanyA', threshold=0.01)
print("UBOs:", ubos)  # Output: {'PersonX': 1.0}
