import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class KDNode:
    """Node in KD tree containing point data and split information"""
    point: torch.Tensor            # The point stored at this node
    left: Optional['KDNode']      # Left child
    right: Optional['KDNode']     # Right child
    split_dim: int               # Dimension used for splitting


class KDTree:
    """KD Tree implementation specialized for state-action space exploration"""
    
    def __init__(self, points: torch.Tensor, leaf_size: int = 10):
        self.points = points
        self.leaf_size = leaf_size
        self.n_dims = points.shape[1]
        self.root = self._build_tree(points)
        
    def _build_tree(self, points: torch.Tensor, depth: int = 0) -> Optional[KDNode]:
        """Recursively build the KD tree"""

        if points is None or len(points) == 0:
            return None
            
        # Convert points to tensor if not already
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        
        # Require at least 2 points to build a meaningful tree
        if len(points) < 2:
            if len(points) == 1:
                return KDNode(
                    point=points[0],
                    left=None,
                    right=None,
                    split_dim=-1
                )
            return None
            
        # If we have fewer points than leaf_size, make a leaf node
        if len(points) <= self.leaf_size:
            return KDNode(
                point=points[0],
                left=None,
                right=None,
                split_dim=-1
            )
        
        # Choose splitting dimension based on maximum variance
        variances = torch.var(points, dim=0)
        split_dim = torch.argmax(variances).item()
        
        # Sort points along the splitting dimension
        sorted_indices = torch.argsort(points[:, split_dim])
        sorted_points = points[sorted_indices]
        
        # Find median index
        median_idx = len(points) // 2
        
        # Create node with median point
        node = KDNode(
            point=sorted_points[median_idx],
            left=None,
            right=None,
            split_dim=split_dim
        )
        
        # Recursively build left and right subtrees
        left_points = sorted_points[:median_idx]
        right_points = sorted_points[median_idx + 1:]
        
        if len(left_points) > 0:
            node.left = self._build_tree(left_points, depth + 1)
        if len(right_points) > 0:
            node.right = self._build_tree(right_points, depth + 1)
        
        return node
        
    def _knn_search(self, node: Optional[KDNode], query: torch.Tensor, k: int,
                    best_dists: List[float], best_points: List[torch.Tensor]) -> None:
        """Recursive k-nearest neighbor search"""
        if node is None:
            return
            
        # If leaf node, compute distances to all points
        if node.split_dim == -1:
            dist = torch.norm(query - node.point)
            
            # Update best points if this is closer
            if len(best_dists) < k or dist < best_dists[-1]:
                # Insert new distance and point in sorted order
                for i in range(len(best_dists)):
                    if i == len(best_dists) or dist < best_dists[i]:
                        best_dists.insert(i, dist.item())
                        best_points.insert(i, node.point)
                        break
                        
                # Keep only k closest
                if len(best_dists) > k:
                    best_dists.pop()
                    best_points.pop()
            return
            
        # Compare splitting dimension
        split_dim = node.split_dim
        split_val = node.point[split_dim]
        query_val = query[split_dim]
        
        # Determine which child to search first
        if query_val < split_val:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left
            
        # Check this node's point
        dist = torch.norm(query - node.point)
        if len(best_dists) < k or dist < best_dists[-1]:
            # Insert new distance and point in sorted order
            for i in range(len(best_dists)):
                if i == len(best_dists) or dist < best_dists[i]:
                    best_dists.insert(i, dist.item())
                    best_points.insert(i, node.point)
                    break
                    
            # Keep only k closest
            if len(best_dists) > k:
                best_dists.pop()
                best_points.pop()
        
        # Search first subtree
        self._knn_search(first, query, k, best_dists, best_points)
        
        # Check if we need to search other subtree
        if len(best_dists) < k or abs(query_val - split_val) < best_dists[-1]:
            self._knn_search(second, query, k, best_dists, best_points)
      
    def query(self, query_point: torch.Tensor, k: int = 1) -> torch.Tensor:
        """Find k nearest neighbors to query point."""

        if len(self.points) == 0:
            # Return infinite distance if tree is empty
            return torch.tensor([float('inf')], device=query_point.device)
            
        # Ensure query point has correct shape
        if query_point.dim() == 1:
            query_point = query_point.unsqueeze(0)
            
        # Calculate distances to all points
        distances = torch.norm(self.points - query_point, dim=1)
        
        # Get k smallest distances
        if k == 1:
            min_dist = torch.min(distances)
            return torch.tensor([min_dist.item()], device=query_point.device)
        else:
            k = min(k, len(distances))  # Don't try to get more neighbors than points
            values, _ = torch.topk(distances, k, largest=False)
            return values
        
    def update(self, new_points: Union[torch.Tensor, np.ndarray]) -> None:
        """
        Update tree with new points by rebuilding it.
        Future optimization: implement insertion instead of rebuild.
        
        Args:
            new_points: New points to add to the tree
        """
        if isinstance(new_points, np.ndarray):
            new_points = torch.from_numpy(new_points).float()
            
        self.points = torch.cat([self.points, new_points])
        self.root = self._build_tree(self.points)
