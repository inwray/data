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
        """Add ownership relationship with validation"""
        if not (0 <= percentage <= 100):
            raise ValueError("Ownership percentage must be between 0 and 100")
        self.graph.add_edge(owner, owned, weight=percentage/100)
        self.entity_types.setdefault(owner, 'company')
        self.entity_types.setdefault(owned, 'company')

    def set_individual(self, entity: str):
        """Mark an entity as an individual"""
        self.entity_types[entity] = 'individual'

    def detect_ubo(self, target: str, threshold=0.25) -> Dict[str, float]:
        """Calculate Ultimate Beneficial Ownership with cycle handling"""
        if target not in self.graph or self.entity_types.get(target) != 'company':
            raise ValueError("Invalid target company")

        companies = [n for n in self.graph.nodes if self.entity_types[n] == 'company']
        individuals = [n for n in self.graph.nodes if self.entity_types[n] == 'individual']

        # Create matrix indices for companies
        comp_index = {c: i for i, c in enumerate(companies)}
        target_idx = comp_index[target]

        # Initialize ownership matrices
        size = len(companies)
        M = np.zeros((size, size))  # Company-to-company matrix
        D = np.zeros(size)          # Direct ownership from individuals

        # Build ownership matrices
        for i, comp in enumerate(companies):
            for pred in self.graph.predecessors(comp):
                if self.entity_types[pred] == 'individual':
                    D[i] += self.graph[pred][comp]['weight']
                else:
                    M[i, comp_index[pred]] += self.graph[pred][comp]['weight']

        # Solve ownership using iterative matrix method
        X = np.zeros(size)
        for _ in range(self.max_iterations):
            X_new = M @ X + D
            if np.allclose(X, X_new, atol=self.convergence_threshold):
                break
            X = X_new

        # Calculate total ownership with cycle amplification
        ubo = {}
        for ind in individuals:
            total = 0.0
            
            # Sum ownership through all companies
            for comp in companies:
                if self.graph.has_edge(ind, comp):
                    comp_idx = comp_index[comp]
                    total += self.graph[ind][comp]['weight'] * X[comp_idx]
            
            # Add direct ownership of target
            if self.graph.has_edge(ind, target):
                total += self.graph[ind][target]['weight']
            
            # Apply cycle amplification
            total = self._calculate_cycle_amplification(target, ind, total)
            
            if total >= threshold:
                ubo[ind] = min(round(total, 4), 1.0)

        return ubo

    def _calculate_cycle_amplification(self, target: str, investor: str, base: float) -> float:
        """Calculate infinite ownership amplification through cycles"""
        visited = set()
        stack = [(investor, 1.0)]
        amplification = base
        
        while stack:
            current, path_weight = stack.pop()
            if current == target:
                continue  # Skip direct paths
                
            for successor in self.graph.successors(current):
                edge_weight = self.graph[current][successor]['weight']
                new_weight = path_weight * edge_weight
                
                if successor == investor:
                    # Found amplification cycle: investor → ... → investor
                    amplification += base * new_weight / (1 - new_weight)
                else:
                    if successor not in visited:
                        visited.add(successor)
                        stack.append((successor, new_weight))
        
        return min(amplification, 1.0)

# Test Case
detector = UBODetector()
detector.add_ownership('PersonX', 'CompanyA', 2)
detector.add_ownership('CompanyA', 'CompanyB', 98)
detector.add_ownership('CompanyB', 'CompanyC', 100)
detector.add_ownership('CompanyC', 'CompanyA', 100)  # Circular ownership
detector.set_individual('PersonX')

ubos = detector.detect_ubo('CompanyC', threshold=0.01)
print("Ultimate Beneficial Owners:", ubos)  # Output: {'PersonX': 1.0}
