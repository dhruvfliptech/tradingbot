"""
Correlation Network Analysis and Clustering

This module performs network analysis on correlation structures to:
- Build and analyze correlation networks
- Detect correlation clusters and communities
- Identify central/peripheral assets
- Measure network connectivity and resilience
- Detect lead-lag relationships between assets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
import networkx as nx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Configuration for network analysis"""
    correlation_threshold: float = 0.3  # Minimum correlation for edge
    n_clusters: int = 5  # Number of clusters to detect
    clustering_method: str = 'spectral'  # 'spectral', 'hierarchical', 'louvain'
    centrality_measures: List[str] = field(default_factory=lambda: ['degree', 'betweenness', 'eigenvector', 'closeness'])
    mst_enabled: bool = True  # Use minimum spanning tree
    lead_lag_max: int = 10  # Maximum lag for lead-lag detection
    community_resolution: float = 1.0  # Resolution for community detection
    visualization_enabled: bool = True


@dataclass
class NetworkNode:
    """Represents a node in the correlation network"""
    asset: str
    cluster_id: int
    centrality_scores: Dict[str, float]
    is_hub: bool
    is_bridge: bool
    connections: List[str]
    strength: float  # Sum of edge weights


@dataclass
class NetworkEdge:
    """Represents an edge in the correlation network"""
    asset1: str
    asset2: str
    correlation: float
    distance: float  # 1 - abs(correlation)
    is_mst: bool  # Part of minimum spanning tree
    lead_lag: Optional[int]  # Lead-lag relationship


@dataclass
class CorrelationCluster:
    """Represents a cluster of correlated assets"""
    cluster_id: int
    assets: List[str]
    centroid: str  # Most central asset
    average_correlation: float
    internal_density: float
    external_connections: Dict[int, float]  # Connections to other clusters


@dataclass
class LeadLagRelationship:
    """Represents lead-lag relationship between assets"""
    leader: str
    follower: str
    lag: int
    correlation: float
    significance: float
    granger_causality: Optional[float]


class CorrelationNetworkAnalyzer:
    """
    Analyzes correlation structure as a network
    """
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        """
        Initialize network analyzer
        
        Args:
            config: Network configuration
        """
        self.config = config or NetworkConfig()
        self.network: Optional[nx.Graph] = None
        self.nodes: Dict[str, NetworkNode] = {}
        self.edges: List[NetworkEdge] = []
        self.clusters: Dict[int, CorrelationCluster] = {}
        self.mst: Optional[nx.Graph] = None
        self.lead_lag_matrix: Optional[pd.DataFrame] = None
        
    def build_network(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> nx.Graph:
        """
        Build network from correlation matrix
        
        Args:
            correlation_matrix: Correlation matrix
            threshold: Correlation threshold for edges
            
        Returns:
            NetworkX graph object
        """
        threshold = threshold or self.config.correlation_threshold
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for asset in correlation_matrix.columns:
            G.add_node(asset)
            
        # Add edges (only for correlations above threshold)
        for i, asset1 in enumerate(correlation_matrix.columns):
            for j, asset2 in enumerate(correlation_matrix.columns):
                if i < j:
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) >= threshold:
                        # Weight is absolute correlation
                        # Distance is 1 - abs(correlation) for algorithms
                        G.add_edge(
                            asset1,
                            asset2,
                            weight=abs(corr),
                            correlation=corr,
                            distance=1 - abs(corr)
                        )
                        
        self.network = G
        logger.info(f"Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Build MST if enabled
        if self.config.mst_enabled:
            self.mst = self._build_minimum_spanning_tree(correlation_matrix)
            
        # Store edges
        self._extract_edges(correlation_matrix)
        
        return G
        
    def detect_clusters(
        self,
        method: Optional[str] = None
    ) -> Dict[int, CorrelationCluster]:
        """
        Detect clusters in correlation network
        
        Args:
            method: Clustering method
            
        Returns:
            Dictionary of clusters
        """
        if self.network is None:
            raise ValueError("Network not built. Call build_network first.")
            
        method = method or self.config.clustering_method
        
        if method == 'spectral':
            clusters = self._spectral_clustering()
        elif method == 'hierarchical':
            clusters = self._hierarchical_clustering()
        elif method == 'louvain':
            clusters = self._louvain_clustering()
        else:
            raise ValueError(f"Unknown clustering method: {method}")
            
        # Analyze each cluster
        self.clusters = {}
        for cluster_id, assets in clusters.items():
            self.clusters[cluster_id] = self._analyze_cluster(
                cluster_id,
                assets
            )
            
        logger.info(f"Detected {len(self.clusters)} clusters")
        
        return self.clusters
        
    def calculate_centrality(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate various centrality measures
        
        Returns:
            Dictionary of centrality scores by asset
        """
        if self.network is None:
            raise ValueError("Network not built. Call build_network first.")
            
        centrality_scores = {}
        
        if 'degree' in self.config.centrality_measures:
            centrality_scores['degree'] = nx.degree_centrality(self.network)
            
        if 'betweenness' in self.config.centrality_measures:
            centrality_scores['betweenness'] = nx.betweenness_centrality(
                self.network,
                weight='distance'
            )
            
        if 'eigenvector' in self.config.centrality_measures:
            try:
                centrality_scores['eigenvector'] = nx.eigenvector_centrality(
                    self.network,
                    weight='weight',
                    max_iter=1000
                )
            except:
                logger.warning("Eigenvector centrality failed to converge")
                centrality_scores['eigenvector'] = {}
                
        if 'closeness' in self.config.centrality_measures:
            centrality_scores['closeness'] = nx.closeness_centrality(
                self.network,
                distance='distance'
            )
            
        # Create node objects
        self._create_network_nodes(centrality_scores)
        
        return centrality_scores
        
    def identify_key_assets(
        self,
        top_n: int = 10
    ) -> Dict[str, List[str]]:
        """
        Identify key assets in the network
        
        Args:
            top_n: Number of top assets to identify
            
        Returns:
            Dictionary with hub, bridge, and peripheral assets
        """
        if not self.nodes:
            self.calculate_centrality()
            
        # Identify hubs (high degree centrality)
        degree_scores = {
            node.asset: node.centrality_scores.get('degree', 0)
            for node in self.nodes.values()
        }
        hubs = sorted(degree_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Identify bridges (high betweenness)
        betweenness_scores = {
            node.asset: node.centrality_scores.get('betweenness', 0)
            for node in self.nodes.values()
        }
        bridges = sorted(betweenness_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Identify peripheral assets (low centrality)
        peripheral = sorted(degree_scores.items(), key=lambda x: x[1])[:top_n]
        
        # Mark nodes
        for asset, _ in hubs[:5]:
            if asset in self.nodes:
                self.nodes[asset].is_hub = True
                
        for asset, _ in bridges[:5]:
            if asset in self.nodes:
                self.nodes[asset].is_bridge = True
                
        return {
            'hubs': [asset for asset, _ in hubs],
            'bridges': [asset for asset, _ in bridges],
            'peripheral': [asset for asset, _ in peripheral]
        }
        
    def detect_lead_lag_relationships(
        self,
        returns_data: pd.DataFrame,
        max_lag: Optional[int] = None
    ) -> List[LeadLagRelationship]:
        """
        Detect lead-lag relationships between assets
        
        Args:
            returns_data: Returns time series
            max_lag: Maximum lag to test
            
        Returns:
            List of lead-lag relationships
        """
        max_lag = max_lag or self.config.lead_lag_max
        relationships = []
        
        # Calculate cross-correlations for different lags
        assets = returns_data.columns
        lead_lag_matrix = pd.DataFrame(
            index=assets,
            columns=assets,
            dtype=object
        )
        
        for asset1 in assets:
            for asset2 in assets:
                if asset1 != asset2:
                    # Find optimal lag
                    best_lag, best_corr = self._find_optimal_lag(
                        returns_data[asset1],
                        returns_data[asset2],
                        max_lag
                    )
                    
                    if abs(best_corr) > self.config.correlation_threshold:
                        # Test Granger causality
                        granger_stat = self._granger_causality_test(
                            returns_data[asset1],
                            returns_data[asset2],
                            best_lag
                        )
                        
                        if best_lag > 0:  # asset1 leads asset2
                            rel = LeadLagRelationship(
                                leader=asset1,
                                follower=asset2,
                                lag=best_lag,
                                correlation=best_corr,
                                significance=self._calculate_significance(best_corr, len(returns_data)),
                                granger_causality=granger_stat
                            )
                            relationships.append(rel)
                            
                        lead_lag_matrix.loc[asset1, asset2] = (best_lag, best_corr)
                        
        self.lead_lag_matrix = lead_lag_matrix
        
        # Filter significant relationships
        significant_relationships = [
            rel for rel in relationships
            if rel.significance < 0.05
        ]
        
        logger.info(f"Found {len(significant_relationships)} significant lead-lag relationships")
        
        return significant_relationships
        
    def calculate_network_metrics(self) -> Dict[str, float]:
        """
        Calculate overall network metrics
        
        Returns:
            Dictionary of network metrics
        """
        if self.network is None:
            raise ValueError("Network not built. Call build_network first.")
            
        metrics = {}
        
        # Basic metrics
        metrics['n_nodes'] = self.network.number_of_nodes()
        metrics['n_edges'] = self.network.number_of_edges()
        metrics['density'] = nx.density(self.network)
        
        # Connectivity
        metrics['is_connected'] = nx.is_connected(self.network)
        metrics['n_components'] = nx.number_connected_components(self.network)
        
        # Average path length (for largest component if disconnected)
        if metrics['is_connected']:
            metrics['avg_path_length'] = nx.average_shortest_path_length(
                self.network,
                weight='distance'
            )
        else:
            largest_cc = max(nx.connected_components(self.network), key=len)
            subgraph = self.network.subgraph(largest_cc)
            metrics['avg_path_length'] = nx.average_shortest_path_length(
                subgraph,
                weight='distance'
            )
            
        # Clustering coefficient
        metrics['avg_clustering'] = nx.average_clustering(
            self.network,
            weight='weight'
        )
        
        # Transitivity
        metrics['transitivity'] = nx.transitivity(self.network)
        
        # Assortativity (correlation between node degrees)
        metrics['degree_assortativity'] = nx.degree_assortativity_coefficient(
            self.network,
            weight='weight'
        )
        
        # Network diameter (for largest component)
        if metrics['is_connected']:
            metrics['diameter'] = nx.diameter(self.network)
        else:
            largest_cc = max(nx.connected_components(self.network), key=len)
            subgraph = self.network.subgraph(largest_cc)
            metrics['diameter'] = nx.diameter(subgraph)
            
        # Small-world coefficient
        metrics['small_world_coefficient'] = self._calculate_small_world_coefficient()
        
        return metrics
        
    def calculate_systemic_risk(self) -> Dict[str, float]:
        """
        Calculate systemic risk measures from network
        
        Returns:
            Dictionary of systemic risk metrics
        """
        if self.network is None:
            raise ValueError("Network not built. Call build_network first.")
            
        risk_metrics = {}
        
        # Network concentration (based on degree distribution)
        degrees = dict(self.network.degree(weight='weight'))
        degree_values = list(degrees.values())
        
        # Herfindahl index of degree distribution
        total_degree = sum(degree_values)
        if total_degree > 0:
            degree_shares = [d / total_degree for d in degree_values]
            risk_metrics['degree_concentration'] = sum(s**2 for s in degree_shares)
        else:
            risk_metrics['degree_concentration'] = 0
            
        # Contagion risk (average neighbor degree)
        risk_metrics['avg_neighbor_degree'] = np.mean([
            np.mean([degrees[n] for n in self.network.neighbors(node)])
            for node in self.network.nodes()
            if list(self.network.neighbors(node))
        ])
        
        # Core-periphery structure
        core_nodes = self._identify_core_nodes()
        risk_metrics['core_size'] = len(core_nodes) / self.network.number_of_nodes()
        
        # Fragility (impact of removing top nodes)
        risk_metrics['fragility'] = self._calculate_network_fragility()
        
        # Correlation concentration in clusters
        if self.clusters:
            cluster_sizes = [len(c.assets) for c in self.clusters.values()]
            total_assets = sum(cluster_sizes)
            cluster_concentration = sum((s/total_assets)**2 for s in cluster_sizes)
            risk_metrics['cluster_concentration'] = cluster_concentration
            
        return risk_metrics
        
    def get_network_summary(self) -> pd.DataFrame:
        """
        Get summary of network analysis
        
        Returns:
            DataFrame with network summary
        """
        summary_data = []
        
        # Node summary
        for asset, node in self.nodes.items():
            summary_data.append({
                'asset': asset,
                'cluster': node.cluster_id,
                'degree_centrality': node.centrality_scores.get('degree', 0),
                'betweenness_centrality': node.centrality_scores.get('betweenness', 0),
                'is_hub': node.is_hub,
                'is_bridge': node.is_bridge,
                'n_connections': len(node.connections),
                'strength': node.strength
            })
            
        return pd.DataFrame(summary_data)
        
    # Private helper methods
    
    def _build_minimum_spanning_tree(
        self,
        correlation_matrix: pd.DataFrame
    ) -> nx.Graph:
        """Build minimum spanning tree from correlation matrix"""
        # Convert to distance matrix
        distance_matrix = 1 - correlation_matrix.abs()
        
        # Create sparse matrix
        sparse_matrix = csr_matrix(distance_matrix.values)
        
        # Calculate MST
        mst_sparse = minimum_spanning_tree(sparse_matrix)
        
        # Convert to NetworkX graph
        mst_graph = nx.Graph()
        mst_graph.add_nodes_from(correlation_matrix.columns)
        
        # Add edges from MST
        mst_array = mst_sparse.toarray()
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                if mst_array[i, j] > 0 or mst_array[j, i] > 0:
                    mst_graph.add_edge(
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        weight=correlation_matrix.iloc[i, j],
                        distance=distance_matrix.iloc[i, j]
                    )
                    
        return mst_graph
        
    def _extract_edges(self, correlation_matrix: pd.DataFrame):
        """Extract edge information"""
        self.edges = []
        
        for i, asset1 in enumerate(correlation_matrix.columns):
            for j, asset2 in enumerate(correlation_matrix.columns):
                if i < j:
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) >= self.config.correlation_threshold:
                        # Check if edge is in MST
                        is_mst = False
                        if self.mst and self.mst.has_edge(asset1, asset2):
                            is_mst = True
                            
                        edge = NetworkEdge(
                            asset1=asset1,
                            asset2=asset2,
                            correlation=corr,
                            distance=1 - abs(corr),
                            is_mst=is_mst,
                            lead_lag=None
                        )
                        self.edges.append(edge)
                        
    def _spectral_clustering(self) -> Dict[int, List[str]]:
        """Perform spectral clustering"""
        # Get adjacency matrix
        adjacency = nx.adjacency_matrix(self.network, weight='weight')
        
        # Spectral clustering
        clustering = SpectralClustering(
            n_clusters=self.config.n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        labels = clustering.fit_predict(adjacency.toarray())
        
        # Group assets by cluster
        clusters = {}
        for i, node in enumerate(self.network.nodes()):
            cluster_id = labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(node)
            
        return clusters
        
    def _hierarchical_clustering(self) -> Dict[int, List[str]]:
        """Perform hierarchical clustering"""
        # Get distance matrix
        nodes = list(self.network.nodes())
        n = len(nodes)
        distance_matrix = np.ones((n, n))
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if self.network.has_edge(node1, node2):
                    distance_matrix[i, j] = self.network[node1][node2]['distance']
                    
        # Hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method='ward')
        labels = fcluster(linkage_matrix, self.config.n_clusters, criterion='maxclust')
        
        # Group assets by cluster
        clusters = {}
        for i, node in enumerate(nodes):
            cluster_id = labels[i] - 1
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(node)
            
        return clusters
        
    def _louvain_clustering(self) -> Dict[int, List[str]]:
        """Perform Louvain community detection"""
        # Use NetworkX community detection
        import networkx.algorithms.community as nx_comm
        
        communities = nx_comm.louvain_communities(
            self.network,
            weight='weight',
            resolution=self.config.community_resolution,
            seed=42
        )
        
        # Convert to dictionary format
        clusters = {}
        for i, community in enumerate(communities):
            clusters[i] = list(community)
            
        return clusters
        
    def _analyze_cluster(
        self,
        cluster_id: int,
        assets: List[str]
    ) -> CorrelationCluster:
        """Analyze a single cluster"""
        # Calculate internal correlations
        subgraph = self.network.subgraph(assets)
        
        # Average correlation within cluster
        internal_correlations = []
        for edge in subgraph.edges(data=True):
            internal_correlations.append(edge[2].get('correlation', 0))
            
        avg_correlation = np.mean(internal_correlations) if internal_correlations else 0
        
        # Internal density
        possible_edges = len(assets) * (len(assets) - 1) / 2
        actual_edges = subgraph.number_of_edges()
        internal_density = actual_edges / possible_edges if possible_edges > 0 else 0
        
        # Find centroid (most central node in cluster)
        if assets:
            subgraph_centrality = nx.degree_centrality(subgraph)
            centroid = max(subgraph_centrality.items(), key=lambda x: x[1])[0]
        else:
            centroid = None
            
        # External connections
        external_connections = {}
        for asset in assets:
            for neighbor in self.network.neighbors(asset):
                if neighbor not in assets:
                    # Find which cluster the neighbor belongs to
                    neighbor_cluster = self._find_asset_cluster(neighbor)
                    if neighbor_cluster is not None and neighbor_cluster != cluster_id:
                        if neighbor_cluster not in external_connections:
                            external_connections[neighbor_cluster] = 0
                        external_connections[neighbor_cluster] += 1
                        
        return CorrelationCluster(
            cluster_id=cluster_id,
            assets=assets,
            centroid=centroid,
            average_correlation=avg_correlation,
            internal_density=internal_density,
            external_connections=external_connections
        )
        
    def _create_network_nodes(self, centrality_scores: Dict[str, Dict[str, float]]):
        """Create NetworkNode objects"""
        self.nodes = {}
        
        for node in self.network.nodes():
            # Get centrality scores for this node
            node_centrality = {
                measure: scores.get(node, 0)
                for measure, scores in centrality_scores.items()
            }
            
            # Get connections
            connections = list(self.network.neighbors(node))
            
            # Calculate strength (sum of edge weights)
            strength = sum(
                self.network[node][neighbor].get('weight', 0)
                for neighbor in connections
            )
            
            # Determine cluster
            cluster_id = self._find_asset_cluster(node) if self.clusters else -1
            
            self.nodes[node] = NetworkNode(
                asset=node,
                cluster_id=cluster_id,
                centrality_scores=node_centrality,
                is_hub=False,
                is_bridge=False,
                connections=connections,
                strength=strength
            )
            
    def _find_asset_cluster(self, asset: str) -> Optional[int]:
        """Find which cluster an asset belongs to"""
        for cluster_id, cluster in self.clusters.items():
            if asset in cluster.assets:
                return cluster_id
        return None
        
    def _find_optimal_lag(
        self,
        series1: pd.Series,
        series2: pd.Series,
        max_lag: int
    ) -> Tuple[int, float]:
        """Find optimal lag between two series"""
        best_lag = 0
        best_corr = 0
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # series1 lags series2
                corr = series1.iloc[:lag].corr(series2.iloc[-lag:])
            elif lag > 0:
                # series1 leads series2
                corr = series1.iloc[:-lag].corr(series2.iloc[lag:])
            else:
                # No lag
                corr = series1.corr(series2)
                
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
                
        return best_lag, best_corr
        
    def _granger_causality_test(
        self,
        series1: pd.Series,
        series2: pd.Series,
        lag: int
    ) -> float:
        """Simplified Granger causality test"""
        # This is a simplified version
        # In production, use statsmodels.tsa.stattools.grangercausalitytests
        
        # Create lagged variables
        if lag > 0:
            x_lag = series1.shift(lag).dropna()
            y = series2.iloc[lag:]
            
            # Simple F-test for significance
            corr = x_lag.corr(y)
            n = len(y)
            
            # F-statistic approximation
            f_stat = (corr**2 * (n - 2)) / (1 - corr**2)
            
            return f_stat
        
        return 0.0
        
    def _calculate_significance(self, correlation: float, n: int) -> float:
        """Calculate p-value for correlation"""
        if n <= 2:
            return 1.0
            
        # Fisher transformation
        z = 0.5 * np.log((1 + correlation) / (1 - correlation))
        se = 1 / np.sqrt(n - 3)
        
        # Two-tailed test
        p_value = 2 * (1 - stats.norm.cdf(abs(z) / se))
        
        return p_value
        
    def _calculate_small_world_coefficient(self) -> float:
        """Calculate small-world coefficient"""
        # Small-world: high clustering, low path length
        
        # Get actual metrics
        actual_clustering = nx.average_clustering(self.network, weight='weight')
        
        # Generate random graph with same degree sequence
        degree_sequence = [d for n, d in self.network.degree()]
        random_graph = nx.configuration_model(degree_sequence, seed=42)
        random_graph = nx.Graph(random_graph)  # Remove multi-edges
        
        # Random graph metrics
        random_clustering = nx.average_clustering(random_graph)
        
        # Small-world coefficient
        if random_clustering > 0:
            sigma = actual_clustering / random_clustering
        else:
            sigma = 0
            
        return sigma
        
    def _identify_core_nodes(self, threshold: float = 0.8) -> Set[str]:
        """Identify core nodes using k-core decomposition"""
        # Get k-core
        k_core = nx.k_core(self.network)
        
        # Alternative: use degree threshold
        degree_threshold = np.percentile(
            [d for n, d in self.network.degree()],
            threshold * 100
        )
        
        core_nodes = {
            node for node, degree in self.network.degree()
            if degree >= degree_threshold
        }
        
        return core_nodes
        
    def _calculate_network_fragility(self, top_n: int = 5) -> float:
        """Calculate network fragility by removing top nodes"""
        if self.network.number_of_nodes() <= top_n:
            return 1.0
            
        # Get top nodes by degree
        top_nodes = sorted(
            self.network.degree(weight='weight'),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        # Create copy and remove top nodes
        test_graph = self.network.copy()
        for node, _ in top_nodes:
            test_graph.remove_node(node)
            
        # Measure fragmentation
        if test_graph.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(test_graph), key=len)
            fragility = 1 - len(largest_cc) / self.network.number_of_nodes()
        else:
            fragility = 1.0
            
        return fragility