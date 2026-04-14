import pickle
from pathlib import Path

import networkx as nx
import pytest
from geopandas.testing import assert_geodataframe_equal

from falcomchain.helper import (
    add_to_saved_geodata,
    load_pickle,
    load_tree_class,
    remove_from_saved_geodata,
    save_pickle,
    save_tree_class,
)
from falcomchain.tree.tree import SpanningTree


def test_save_tree_class(spanningtree_with_forced_root):

    tree = spanningtree_with_forced_root.graph

    path_1 = "/Users/kirtisoglu/Documents/Documents/GitHub/FalcomChain/tests/test_data/test_tree.pkl"
    path_2 = "/Users/kirtisoglu/Documents/Documents/GitHub/FalcomChain/tests/test_data/test_tree_attributes.pkl"
    
    # save them using the function that will be tested
    save_tree_class(tree, path_1, path_2)
    
    # load them back
    with open(path_1, "rb") as file:
        loaded_tree = pickle.load(file)
    with open(path_2, "rb") as file:
        attr = pickle.load(file)

    # check whether everything is saved safely
    assert attr["root"] == tree.root
    assert attr["ideal_pop"] == tree.ideal_pop
    assert attr["n_teams"] == tree.n_teams
    assert attr["epsilon"] == tree.epsilon
    assert attr["supertree"] == tree.supertree
    assert attr["two_sided"] == tree.two_sided
    assert attr["tot_candidates"] == tree.total_cand
    assert attr["capacity_level"] == tree.capacity_level

    assert (
        nx.is_isomorphic(
            tree.graph,
            loaded_tree,
            node_match=nx.algorithms.isomorphism.categorical_node_match([], []),
            edge_match=nx.algorithms.isomorphism.categorical_edge_match([], []),
        )
        == True
    )


def test_load_tree_class(spanningtree_with_forced_root):

    tree = spanningtree_with_forced_root

    path_1 = "/Users/kirtisoglu/Documents/Documents/GitHub/FalcomChain/tests/test_data/test_tree.pkl"
    path_2 = "/Users/kirtisoglu/Documents/Documents/GitHub/FalcomChain/tests/test_data/test_tree_attributes.pkl"

    # They are already saved by the previous test function. Load them back.
    loaded_tree = load_tree_class(path_1, path_2)

    # check whether everything is loaded safely
    assert loaded_tree.root == tree.root
    assert loaded_tree.ideal_pop == tree.ideal_pop
    assert loaded_tree.n_teams == tree.n_teams
    assert loaded_tree.n_teams == tree.epsilon
    assert loaded_tree.supertree == tree.supertree
    assert loaded_tree.two_sided == tree.two_sided
    assert loaded_tree.tot == tree.tot_candidates
    assert loaded_tree.capacity_level == tree.capacity_level




    
def test_save_pickle(geodata_without_candidates):

    gdf = geodata_without_candidates
    path = "/Users/kirtisoglu/Documents/Documents/GitHub/FalcomChain/tests/test_data/test_geodata_no_candidates.pkl"
    
    # save it using the function that will be tested
    save_pickle(gdf, path)
    
    # load geodata back
    with open(path, "rb") as file:
        loaded_gdf = pickle.load(file)
    
    assert assert_geodataframe_equal(gdf, loaded_gdf, check_dtype=True, check_index_type='equiv', 
                                     check_column_type='equiv', check_frame_type=True, check_like=False, 
                                     check_less_precise=False, check_geom_type=True, check_crs=True, normalize=False)
    
    

def test_load_pickle(geodata_without_candidates):
    
    gdf = geodata_without_candidates
    path = "/Users/kirtisoglu/Documents/Documents/GitHub/FalcomChain/tests/test_data/test_geodata_no_candidates.pkl"
    
    # it is already saved with the previous test function. Load it directly
    loaded_gdf = load_pickle(path)
    
    assert assert_geodataframe_equal(gdf, loaded_gdf, check_dtype=True, check_index_type='equiv', 
                                     check_column_type='equiv', check_frame_type=True, check_like=False, 
                                     check_less_precise=False, check_geom_type=True, check_crs=True, normalize=False)
    

    
def test_add_saved_geodata():
    
    assert


def test_remove_from_saved_geodata():
    
    assert