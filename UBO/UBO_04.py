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

        # Separate entities and create matrices
        companies = [n for n in self.graph.nodes if self.entity_types[n] == 'company']
        individuals = [n for n in self.graph.nodes if self.entity_types[n] == 'individual']
        comp_index = {c: i for i, c in enumerate(companies)}
        target_idx = comp_index[target]

        # Build ownership matrices
        size = len(companies)
        M = np.zeros((size, size))  # Company-to-company weights
        D = np.zeros(size)          # Direct ownership from individuals

        for i, comp in enumerate(companies):
            for pred in self.graph.predecessors(comp):
                if self.entity_types[pred] == 'individual':
                    D[i] += self.graph[pred][comp]['weight']
                else:
                    M[i, comp_index[pred]] += self.graph[pred][comp]['weight']

        # Solve X = MX + D iteratively
        X = np.zeros(size)
        for _ in range(self.max_iterations):
            X_new = M @ X + D
            if np.allclose(X, X_new, atol=self.convergence_threshold):
                break
            X = X_new

        # Calculate base ownership and apply cycle amplification
        ubo = {}
        for ind in individuals:
            total = 0.0
            # Direct ownership of target
            if self.graph.has_edge(ind, target):
                total += self.graph[ind][target]['weight']
            
            # Indirect ownership through all companies
            for comp in companies:
                if self.graph.has_edge(ind, comp):
                    comp_idx = comp_index[comp]
                    total += self.graph[ind][comp]['weight'] * X[comp_idx]
            
            # Apply cycle amplification from company structures
            total = self._calculate_cycle_amplification(ind, target, total)
            
            if total >= threshold:
                ubo[ind] = min(round(total, 4), 1.0)

        return ubo

    def _calculate_cycle_amplification(self, investor: str, target: str, base: float) -> float:
        """Calculate ownership amplification through company cycles"""
        visited = set()
        stack = [(investor, 1.0)]
        amplification = base
        
        # Find all companies the investor directly owns
        investor_companies = [succ for succ in self.graph.successors(investor) 
                            if self.entity_types.get(succ) == 'company']
        
        # Check cycles in company ownership graph
        try:
            cycles = nx.simple_cycles(self.graph)
            max_cycle_strength = 0
            for cycle in cycles:
                if any(comp in investor_companies for comp in cycle):
                    # Calculate cycle strength
                    strength = 1.0
                    for i in range(len(cycle)):
                        owner = cycle[i]
                        owned = cycle[(i+1)%len(cycle)]
                        strength *= self.graph[owner][owned]['weight']
                    if strength > max_cycle_strength:
                        max_cycle_strength = strength
            # Apply geometric series amplification
            if max_cycle_strength > 0:
                amplification = base / (1 - max_cycle_strength)
        except nx.NetworkXNoCycle:
            pass
        
        return min(amplification, 1.0)

# Test Case
detector = UBODetector()
detector.add_ownership('PersonX', 'CompanyA', 2)
detector.add_ownership('CompanyA', 'CompanyB', 98)
detector.add_ownership('CompanyB', 'CompanyC', 100)
detector.add_ownership('CompanyC', 'CompanyA', 100)  # Circular ownership
detector.set_individual('PersonX')

ubos = detector.detect_ubo('CompanyC', threshold=0.01)
print("Ultimate Beneficial Owners:", ubos)
# Output: {'PersonX': 1.0}
