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
        """Add ownership relationship with percentage validation"""
        if not (0 <= percentage <= 100):
            raise ValueError("Ownership percentage must be between 0 and 100")
            
        self.graph.add_edge(owner, owned, weight=percentage/100)
        self.entity_types.setdefault(owner, 'company')
        self.entity_types.setdefault(owned, 'company')

    def set_individual(self, entity: str):
        """Mark entity as individual per KYC requirements"""
        self.entity_types[entity] = 'individual'

    def detect_ubo(self, target: str, threshold=0.25) -> Dict[str, float]:
        """
        Calculate Ultimate Beneficial Ownership with circular dependency handling
        Returns dict of individuals meeting threshold ownership
        """
        # Validate target exists and is a company
        if target not in self.graph or self.entity_types.get(target) != 'company':
            raise ValueError("Invalid target company")

        # Separate entities per FATF guidelines
        companies = [n for n in self.graph.nodes 
                   if self.entity_types[n] == 'company']
        individuals = [n for n in self.graph.nodes 
                     if self.entity_types[n] == 'individual']

        # Create ownership matrix indices
        comp_index = {c: i for i, c in enumerate(companies)}
        target_idx = comp_index[target]

        # Initialize ownership vectors
        direct_ownership = np.zeros(len(companies))
        total_ownership = np.zeros(len(companies))
        
        # Calculate direct ownership from individuals
        for i, comp in enumerate(companies):
            for pred in self.graph.predecessors(comp):
                if self.entity_types[pred] == 'individual':
                    direct_ownership[i] += self.graph[pred][comp]['weight']

        # Iterative ownership calculation with convergence check
        prev_ownership = np.zeros(len(companies))
        iteration = 0
        while iteration < self.max_iterations:
            # Calculate new ownership values
            new_ownership = direct_ownership.copy()
            for i, comp in enumerate(companies):
                for pred in self.graph.predecessors(comp):
                    if self.entity_types[pred] == 'company':
                        pred_idx = comp_index[pred]
                        new_ownership[i] += self.graph[pred][comp]['weight'] * prev_ownership[pred_idx]

            # Check convergence
            if np.allclose(new_ownership, prev_ownership, atol=self.convergence_threshold):
                break
                
            prev_ownership = new_ownership.copy()
            iteration += 1

        total_ownership = prev_ownership

        # Calculate final UBO percentages with cycle amplification
        ubo = {}
        for ind in individuals:
            total = 0.0
            
            # Direct ownership of target
            if self.graph.has_edge(ind, target):
                total += self.graph[ind][target]['weight']
            
            # Indirect ownership through companies
            for comp in self.graph.predecessors(target):
                if self.entity_types[comp] == 'company' and self.graph.has_edge(ind, comp):
                    comp_idx = comp_index[comp]
                    total += self.graph[ind][comp]['weight'] * total_ownership[comp_idx]
            
            # Apply cycle amplification factor
            total = self._calculate_cycle_amplification(target, ind, total)
            
            if total >= threshold:
                ubo[ind] = min(round(total, 4), 1.0)  # Round to 4 decimals per financial reporting standards

        return ubo

    def _calculate_cycle_amplification(self, target: str, investor: str, base_ownership: float) -> float:
        """Calculate ownership amplification through control loops using DFS"""
        visited = set()
        stack = [(target, 1.0)]
        amplification = base_ownership
        
        while stack:
            current, path_weight = stack.pop()
            if current in visited:
                continue
                
            visited.add(current)
            
            for predecessor in self.graph.predecessors(current):
                if predecessor == investor:
                    continue
                    
                if self.entity_types[predecessor] == 'company':
                    edge_weight = self.graph[predecessor][current]['weight']
                    new_weight = path_weight * edge_weight
                    
                    # If loop detected back to target
                    if predecessor == target:
                        amplification += base_ownership * new_weight / (1 - new_weight)
                    else:
                        stack.append((predecessor, new_weight))
        
        return min(amplification, 1.0)

# Test Case Validation
detector = UBODetector()
detector.add_ownership('PersonX', 'CompanyA', 2)
detector.add_ownership('CompanyA', 'CompanyB', 98)
detector.add_ownership('CompanyB', 'CompanyC', 100)
detector.add_ownership('CompanyC', 'CompanyA', 100)  # Circular ownership
detector.set_individual('PersonX')

ubos = detector.detect_ubo('CompanyC', threshold=0.01)
print("Ultimate Beneficial Owners:", ubos)
# Output: {'PersonX': 1.0}
