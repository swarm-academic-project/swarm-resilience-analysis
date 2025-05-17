import networkx as nx 
import random 
from itertools import combinations
from networkx.algorithms.simple_paths import all_simple_paths
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from networkx.algorithms.simple_paths import all_simple_paths
from multiprocessing import Pool, cpu_count
from functools import partial

def flow_robustness(G: nx.Graph, divided=False, group_dict=None) -> float:
    """
    Calcule la Flow Robustness du graphe G.
    
    - Si `divided=False` : robustesse globale (toutes les paires de sommets)
    - Si `divided=True` : robustesse intra-groupes uniquement (selon group_dict)

    Args:
        G (nx.Graph): le graphe à analyser
        divided (bool): True pour le cas divisé, False pour le cas global
        group_dict (dict): {node_id: group_id} si divided=True

    Returns:
        float: la proportion de paires connectées selon le mode choisi
    """
    nodes = list(G.nodes)
    n = len(nodes)

    if n < 2:
        return 1.0  # Cas trivial : 0 ou 1 nœud → robustesse maximale

    if not divided:
        # CAS GLOBAL : toutes les paires de sommets
        total_pairs = n * (n - 1) // 2
        connected_pairs = sum(1 for u, v in combinations(nodes, 2) if nx.has_path(G, u, v))
        return connected_pairs / total_pairs

    else:
        # CAS DIVISÉ : intra-groupes uniquement
        if group_dict is None:
            raise ValueError("group_dict doit être fourni quand divided=True")
        
        groups = {}
        for node, group in group_dict.items():
            if node in G: 
                groups.setdefault(group, []).append(node)
            

        total_pairs = 0
        connected_pairs = 0

        for group_nodes in groups.values():
            group_nodes_in_G = [n for n in group_nodes if n in G]
            subG = G.subgraph(group_nodes_in_G)
            m = len(group_nodes_in_G)
            if m < 2:
                continue
            pairs = m * (m - 1) // 2
            total_pairs += pairs
            connected_pairs += sum(1 for u, v in combinations(group_nodes_in_G, 2) if nx.has_path(subG, u, v))

        return connected_pairs / total_pairs if total_pairs > 0 else 0.0
import networkx as nx

def routing_cost(G: nx.Graph, divided=False, group_dict=None) -> float:
    """
    Calcule le coût de routage total :
    - Cas global (divided=False) : somme des distances entre toutes les paires connectées.
    - Cas divisé (divided=True) : somme des distances uniquement à l'intérieur de chaque groupe.
    
    Args:
        G (nx.Graph): le graphe complet
        divided (bool): True pour mode divisé, False pour mode global
        group_dict (dict): {node_id: group_id}, requis si divided=True
    
    Returns:
        float: coût de routage total
    """
    total_cost = 0

    try:
        length_dict = dict(nx.all_pairs_shortest_path_length(G))
    except nx.NetworkXError:
        return float('inf')

    if not divided:
        # Cas global : toutes les paires connectées
        for u in G.nodes:
            for v in G.nodes:
                if u != v:
                    try:
                        total_cost += length_dict[u][v]
                    except KeyError:
                        continue
    else:
        if group_dict is None:
            raise ValueError("group_dict doit être fourni si divided=True")

        # Regrouper les nœuds par groupe
        groups = {}
        for node, group in group_dict.items():
            if node in G:
                groups.setdefault(group, []).append(node)

        for group_nodes in groups.values():
            group_nodes_in_G = [n for n in group_nodes if n in G]
            for u in group_nodes_in_G:
                for v in group_nodes_in_G:
                    if u != v:
                        try:
                            total_cost += length_dict[u][v]
                        except KeyError:
                            continue

    return total_cost

#test :
"""G = nx.path_graph(5)  # Graphe en ligne : 0-1-2-3-4
print("Routing Cost:", routing_cost(G))  # Plus élevé que sur un graphe complet

G2 = nx.complete_graph(5)
print("Routing Cost (graphe complet):", routing_cost(G2)) # Coût minimal"""

def network_efficiency(G: nx.Graph, divided=False, group_dict=None) -> float:
    """
    Calcule l'efficacité du réseau t(G), entre 0 et 1.
    - Cas global : efficacité moyenne sur toutes les paires connectées.
    - Cas divisé : efficacité intra-groupe uniquement.
    """
    try:
        length_dict = dict(nx.all_pairs_shortest_path_length(G))
    except nx.NetworkXError:
        return 0.0

    if not divided:
        # Cas global
        n = len(G)
        if n < 2:
            return 1.0
        total_efficiency = 0
        for u in G.nodes:
            for v in G.nodes:
                if u != v:
                    try:
                        d = length_dict[u][v]
                        total_efficiency += 1 / d
                    except KeyError:
                        continue
        return total_efficiency / (n * (n - 1))

    else:
        if group_dict is None:
            raise ValueError("group_dict doit être fourni si divided=True")

        # Regrouper uniquement les nœuds présents dans le graphe
        groups = {}
        for node, group in group_dict.items():
            if node in G:
                groups.setdefault(group, []).append(node)

        total_efficiency = 0
        total_pairs = 0

        for group_nodes in groups.values():
            group_nodes_in_G = [n for n in group_nodes if n in G]
            m = len(group_nodes_in_G)
            if m < 2:
                continue
            total_pairs += m * (m - 1)
            for u in group_nodes_in_G:
                for v in group_nodes_in_G:
                    if u != v:
                        try:
                            d = length_dict[u][v]
                            total_efficiency += 1 / d
                        except KeyError:
                            continue

        return total_efficiency / total_pairs if total_pairs > 0 else 0.0





def path_redundancy(G: nx.Graph, max_extra_length: int = 1, divided=False, group_dict=None) -> float:
    """
    Calcule la redondance moyenne des chemins :
    - Cas global : sur toutes les paires connectées.
    - Cas divisé : seulement entre les paires du même groupe.

    Un chemin alternatif est défini comme :
    - un chemin simple (sans cycle)
    - de longueur <= (plus court chemin + max_extra_length)
    - et différent du plus court chemin

    Args:
        G (nx.Graph): le graphe
        max_extra_length (int): tolérance sur la longueur
        divided (bool): True pour ne considérer que les paires intra-groupe
        group_dict (dict): {node: group_id}, requis si divided=True

    Returns:
        float: redondance moyenne (nombre de chemins alternatifs par paire)
    """
    
    nodes=list(G.nodes)

    n = len(nodes)
    if n < 2:
        return 0.0

    total = 0
    count = 0

    if not divided:
        pairs = [(u, v) for u in nodes for v in nodes if u < v]
    else:
        if group_dict is None:
            raise ValueError("group_dict doit être fourni quand divided=True")
        pairs = [
            (u, v) for u in nodes for v in nodes
            if u < v and group_dict.get(u) == group_dict.get(v)
        ]

    for u, v in pairs:
        try:
            paths = list(nx.all_shortest_paths(G, source=u, target=v))
            total += len(paths)
            count += 1
        except nx.NetworkXNoPath:
            continue

    return total / count if count > 0 else 0.0





def pair_disparity_fast(shortest_paths):
    if len(shortest_paths) <= 1:
        return 0.0
    max_elem = len(shortest_paths[0]) - 2  # nombre de nœuds intermédiaires
    if max_elem == 0:
        return 0.0
    pairs = []
    disparity = 0
    for i, p1 in enumerate(shortest_paths):
        for j, p2 in enumerate(shortest_paths):
            if i < j:
                c = len(set(p1[1:-1]).intersection(p2[1:-1]))  # sans u,v
                disparity += 1 - (c / max_elem)
                pairs.append(1)
    return disparity / len(pairs) if pairs else 0.0


def path_disparity(G, group_dict=None, divided=False):
    if divided:
        groups = {}
        for node, group in group_dict.items():
            groups.setdefault(group, []).append(node)
        node_pairs = []
        for group_nodes in groups.values():
            node_pairs += list(combinations(group_nodes, 2))
    else:
        node_pairs = list(combinations(G.nodes, 2))

    disparities = []
    for u, v in node_pairs:
        try:
            paths = list(nx.all_shortest_paths(G, u, v, weight="weight"))
            if len(paths) < 2:
                continue
            d = pair_disparity_fast(paths)
            disparities.append(d)
        except:
            continue

    return sum(disparities) / len(disparities) if disparities else 0.0



    


def node_criticality(G: nx.Graph, divided=False, group_dict=None):
    """
    Calcule la criticité BCt(i) de chaque nœud :
    - Cas global : centralité d’intermédiarité sur tout le graphe.
    - Cas divisé : uniquement intra-groupe (centralité par groupe, zéro hors groupe).

    Returns:
        dict(node_id: criticité)
    """
    n = len(G)
    if n < 2:
        return {node: 0.0 for node in G.nodes}

    if not divided:
        # Cas global
        norm_factor = 1 / (n * (n - 1))
        centrality = nx.betweenness_centrality(G, normalized=True)
        BC = {node: value * norm_factor for node, value in centrality.items()}
        return BC

    else:
        if group_dict is None:
            raise ValueError("group_dict doit être fourni si divided=True")

        # Cas divisé : on calcule la centralité dans chaque sous-graphe
        BC = {node: 0.0 for node in G.nodes}
        groups = {}
        for node, group in group_dict.items():
            if node in G:  # <--- filtre des nœuds toujours présents
                groups.setdefault(group, []).append(node)

        for group_nodes in groups.values():
            subG = G.subgraph(group_nodes)
            m = len(subG)
            if m < 2:
                continue
            norm_factor = 1 / (m * (m - 1))
            centrality = nx.betweenness_centrality(subG, normalized=True)
            for node, value in centrality.items():
                BC[node] = value * norm_factor

        return BC

    
def critical_nodes(G, epsilon=0.01, divided=False, group_dict=None):
    """
    Retourne le pourcentage de nœuds critiques (BCt(i) ≥ ε)
    """
    BC = node_criticality(G, divided=divided, group_dict=group_dict)
    critical = [node for node, value in BC.items() if value >= epsilon]
    return len(critical) * 100 / len(G.nodes)
    
def critical_nodes(G, epsilon=1e-4, divided=False, group_dict=None):
    """
    Retourne le pourcentage de nœuds critiques (BCt(i) ≥ ε)
    """
    BC = node_criticality(G, divided=divided, group_dict=group_dict)
    critical = [node for node, value in BC.items() if value >= epsilon]
    return len(critical) * 100 / len(G.nodes)

"""G = nx.erdos_renyi_graph(10, 0.3, seed=42)
BCt = node_criticality(G)
Ct = critical_nodes(G, epsilon=0.001)

print("Criticité BCt(i) pour chaque nœud :")
for node, value in BCt.items():
    print(f"Node {node} : {value:.5f}")

print("\nEnsemble critique Cₜ(G) :", Ct)
"""