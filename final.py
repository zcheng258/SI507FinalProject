import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
from datetime import datetime


def load_data(file_path='supermarketpage.csv'):
    """Load and clean the grocery store data.

    Reads the CSV file containing grocery store data.
    Cleans column names and formats data for analysis.
    Filters out entries with missing state information.

    Parameters
    ----------
    file_path: str
        path to the CSV file containing the grocery store data

    Returns
    -------
    pandas.DataFrame
        cleaned dataframe with store information
    """
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)

    # Clean and filter data
    data = data[['Name', '# of\nlocations', 'States']]
    data.columns = ['Store Name', 'Store Count', 'States']
    data.dropna(subset=['States'], inplace=True)

    print(f"Loaded data for {len(data)} grocery chains.")
    return data


def build_graph(data):
    """Build a bipartite graph with stores and states as nodes.

    Creates a NetworkX graph with stores and states as different node types.
    Adds edges between stores and the states they operate in.
    Stores additional attributes like store location count.

    Parameters
    ----------
    data: pandas.DataFrame
        cleaned dataframe with store information

    Returns
    -------
    networkx.Graph
        bipartite graph with stores and states as nodes
    """
    print("Building graph...")
    G = nx.Graph()

    # Add nodes and edges
    for _, row in data.iterrows():
        store = row['Store Name']
        count = row['Store Count']
        states_str = row['States'] if isinstance(row['States'], str) else ""
        states = [state.strip() for state in states_str.split(',')]

        # Store node with attributes
        G.add_node(store, type='store', locations=count)

        # State nodes and edges
        for state in states:
            if state:  # Only add non-empty state nodes
                G.add_node(state, type='state')
                G.add_edge(store, state)

    print(f"Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    return G


def check_node(G, node):
    """Helper function that verify a node exists in the graph.

    Parameters
    ----------
    G: networkx.Graph
        graph containing store and state nodes
    node: str
        name of the node to check

    Raises
    ------
    ValueError
        if the node is not found in the graph
    """
    if node not in G:
        raise ValueError(f"Node '{node}' not found in the data.")


def find_most_related_states(G, state, limit=5):
    """Find states that share the most store chains with the given state.

    Identifies store chains operating in the specified state.
    Counts which other states have the most overlap in store chains.
    Returns states sorted by number of shared store chains.

    Parameters
    ----------
    G: networkx.Graph
        graph containing store and state nodes
    state: str
        state abbreviation to find related states for
    limit: int
        maximum number of related states to return

    Returns
    -------
    list or str
        list of tuples (state_name, shared_count) or error message
    """
    try:
        check_node(G, state)
    except ValueError as e:
        return str(e)

    # Get all store chains in the given state
    stores_in_state = [node for node in G.neighbors(state) if G.nodes[node]['type'] == 'store']

    # Find all states connected to these stores
    state_connections = {}
    for store in stores_in_state:
        connected_states = [node for node in G.neighbors(store) if G.nodes[node]['type'] == 'state' and node != state]
        for connected_state in connected_states:
            state_connections[connected_state] = state_connections.get(connected_state, 0) + 1

    # Sort by number of shared stores
    sorted_states = sorted(state_connections.items(), key=lambda x: x[1], reverse=True)

    return sorted_states[:limit] if sorted_states else f"No states share store chains with {state}."


def find_most_common_store_in_state(G, state, limit=5):
    """Find the store chain that has most presence in the given state.

    Identifies store chains operating in the specified state.
    Sorts stores by their number of locations.

    Parameters
    ----------
    G: networkx.Graph
        graph containing store and state nodes
    state: str
        state abbreviation to find common stores for
    limit: int
        maximum number of stores to return

    Returns
    -------
    list or str
        list of tuples (store_name, location_count) or error message
    """
    try:
        check_node(G, state)
    except ValueError as e:
        return str(e)

    stores_in_state = [node for node in G.neighbors(state) if G.nodes[node]['type'] == 'store']

    # Sort stores by number of locations (if available)
    stores_with_locations = [(store, G.nodes[store].get('locations', 0)) for store in stores_in_state]
    sorted_stores = sorted(stores_with_locations, key=lambda x: x[1], reverse=True)

    return sorted_stores[:limit] if sorted_stores else f"No stores found in {state}."


def find_shortest_path(G, node1, node2):
    """Find the shortest path between two nodes.

    Uses NetworkX shortest_path algorithm to find connections.
    Path may go through states to stores to other states.

    Parameters
    ----------
    G: networkx.Graph
        graph containing store and state nodes
    node1: str
        starting node (store or state)
    node2: str
        ending node (store or state)

    Returns
    -------
    list or str
        ordered list of nodes in the path or error message
    """
    try:
        check_node(G, node1)
        check_node(G, node2)
    except ValueError as e:
        return str(e)

    # Uses NetworkX shortest_path to find the shortest path
    try:
        path = nx.shortest_path(G, node1, node2)
        return path
    except nx.NetworkXNoPath:
        return f"No path exists between {node1} and {node2}."


def get_node_stats(G, node):
    """Get statistics about a node.

    Provides different statistics based on node type (store or state).
    For stores: locations count, states count, states list.
    For states: store chains count, store chains list.

    Parameters
    ----------
    G: networkx.Graph
        graph containing store and state nodes
    node: str
        name of the node to get statistics for

    Returns
    -------
    dict or str
        dictionary with node statistics or error message
    """
    try:
        check_node(G, node)
    except ValueError as e:
        return str(e)

    # Get the node type
    node_type = G.nodes[node].get('type', 'unknown')

    # If the node is a store, return its data regarding the number of locations,
    # states and a list of states that it is in
    if node_type == 'store':
        locations = G.nodes[node].get('locations', 'Unknown')
        states_count = len([n for n in G.neighbors(node) if G.nodes[n]['type'] == 'state'])
        states_list = [n for n in G.neighbors(node) if G.nodes[n]['type'] == 'state']

        return {
            'name': node,
            'type': 'Store Chain',
            'total_locations': locations,
            'states_count': states_count,
            'states': sorted(states_list)
        }

    # If the node is a state, return its data about the stores in that state
    elif node_type == 'state':
        stores_count = len([n for n in G.neighbors(node) if G.nodes[n]['type'] == 'store'])
        stores_list = [n for n in G.neighbors(node) if G.nodes[n]['type'] == 'store']

        return {
            'name': node,
            'type': 'State',
            'total_store_chains': stores_count,
            'store_chains': sorted(stores_list)
        }

    return f"Unable to get stats for node '{node}'."


def find_most_connected_nodes(G, node_type=None, limit=10):
    """Find the most connected nodes in the graph.

    Identifies nodes with the most edges (connections).
    Can filter by node type (store or state).

    Parameters
    ----------
    G: networkx.Graph
        graph containing store and state nodes
    node_type: str or None
        type of nodes to consider ('store', 'state', or None for all)
    limit: int
        maximum number of nodes to return

    Returns
    -------
    list
        list of tuples (node_name, connection_count) sorted by connections
    """
    # Return the node with the most connections depending on its type
    if node_type:
        nodes_of_type = [node for node, data in G.nodes(data=True) if data.get('type') == node_type]
        node_connections = [(node, len(list(G.neighbors(node)))) for node in nodes_of_type]
    else:
        node_connections = [(node, len(list(G.neighbors(node)))) for node in G.nodes()]

    # Sort it by most connected node
    sorted_nodes = sorted(node_connections, key=lambda x: x[1], reverse=True)
    return sorted_nodes[:limit]


def visualize_graph(G, node_type=None, specific_node=None):
    """Visualize the graph or a subset of it.

    Creates a visualization of the network using matplotlib.
    Can show the full network, specific node connections, or nodes of specific type.
    Saves the visualization as a PNG file.

    Parameters
    ----------
    G: networkx.Graph
        graph containing store and state nodes
    node_type: str or None
        type of nodes to visualize ('store', 'state', or None for all)
    specific_node: str or None
        specific node and its connections to visualize

    Returns
    -------
    str
        filename of the saved visualization
    """
    plt.figure(figsize=(12, 8))

    # Create a subgraph based on parameters
    if specific_node and specific_node in G:

        # Get the node and its neighbors
        neighbors = list(G.neighbors(specific_node))
        nodes_to_include = [specific_node] + neighbors
        subgraph = G.subgraph(nodes_to_include)

    else:
        subgraph = G

    # Color nodes by type
    node_colors = []
    for node in subgraph:
        if G.nodes[node].get('type') == 'store':
            node_colors.append('lightblue')
        else:  # state
            node_colors.append('lightgreen')

    # Draw the graph
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=300, alpha=0.8)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5)
    nx.draw_networkx_labels(subgraph, pos, font_size=8)

    title = "Complete Network"
    if specific_node:
        title = f"Network for {specific_node}"
    elif node_type:
        title = f"Network for node type: {node_type}"

    plt.title(title)
    plt.axis('off')

    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"graph_viz_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Graph visualization saved as {filename}")
    return filename


def compare_nodes(G, node1, node2):
    """Compare two nodes (states or stores) side by side.

    Provides comparison statistics between two nodes of the same type.
    For states: common stores, unique stores for each.
    For stores: common states, unique states for each.

    Parameters
    ----------
    G: networkx.Graph
        graph containing store and state nodes
    node1: str
        first node to compare
    node2: str
        second node to compare

    Returns
    -------
    dict or str
        dictionary with comparison data or error message
    """
    try:
        check_node(G, node1)
        check_node(G, node2)
    except ValueError as e:
        return str(e)

    node1_type = G.nodes[node1].get('type', 'unknown')
    node2_type = G.nodes[node2].get('type', 'unknown')

    # Ensure that we are comparing the same node type
    if node1_type != node2_type:
        return f"Cannot compare {node1_type} with {node2_type}. Please select nodes of the same type."

    comparison = {'node1': {}, 'node2': {}}

    # Compare states
    if node1_type == 'state':
        node1_stores = [n for n in G.neighbors(node1) if G.nodes[n]['type'] == 'store']
        node2_stores = [n for n in G.neighbors(node2) if G.nodes[n]['type'] == 'store']

        common_stores = set(node1_stores).intersection(set(node2_stores))

        comparison['node1'] = {
            'name': node1,
            'total_stores': len(node1_stores),
            'unique_stores': len(set(node1_stores) - set(node2_stores))
        }

        comparison['node2'] = {
            'name': node2,
            'total_stores': len(node2_stores),
            'unique_stores': len(set(node2_stores) - set(node1_stores))
        }

        comparison['common'] = {
            'count': len(common_stores),
            'stores': sorted(list(common_stores))
        }

    # Compare store chains
    elif node1_type == 'store':
        node1_states = [n for n in G.neighbors(node1) if G.nodes[n]['type'] == 'state']
        node2_states = [n for n in G.neighbors(node2) if G.nodes[n]['type'] == 'state']

        common_states = set(node1_states).intersection(set(node2_states))

        comparison['node1'] = {
            'name': node1,
            'total_states': len(node1_states),
            'locations': G.nodes[node1].get('locations', 'Unknown'),
            'unique_states': len(set(node1_states) - set(node2_states))
        }

        comparison['node2'] = {
            'name': node2,
            'total_states': len(node2_states),
            'locations': G.nodes[node2].get('locations', 'Unknown'),
            'unique_states': len(set(node2_states) - set(node1_states))
        }

        comparison['common'] = {
            'count': len(common_states),
            'states': sorted(list(common_states))
        }

    return comparison


def export_graph_data(G, nodes_path='graph_nodes.csv', edges_path='graph_edges.csv', force=False):
    """Export the graph data to CSV files if they don't exist or if force is True.

    Creates two CSV files: one for nodes with their attributes,
    and one for edges representing connections.

    Parameters
    ----------
    G: networkx.Graph
        graph containing store and state nodes
    nodes_path: str
        file path for nodes CSV output
    edges_path: str
        file path for edges CSV output
    force: bool
        whether to overwrite existing files

    Returns
    -------
    bool
        True if files were exported, False if skipped
    """
    # Check if files already exist
    nodes_exist = os.path.exists(nodes_path)
    edges_exist = os.path.exists(edges_path)

    if nodes_exist and edges_exist and not force:
        print(f"Export files already exist. Skipping export.")
        return False

    # Convert nodes to DataFrame
    nodes_data = []
    for node, attrs in G.nodes(data=True):
        node_dict = {'Node': node, 'Type': attrs.get('type', '')}
        if 'locations' in attrs:
            node_dict['Locations'] = attrs['locations']
        nodes_data.append(node_dict)

    nodes_df = pd.DataFrame(nodes_data)

    # Convert edges to DataFrame
    edges_df = pd.DataFrame(G.edges(), columns=['Node1', 'Node2'])

    # Export to CSV
    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    print(f"Graph data exported to '{nodes_path}' and '{edges_path}'")
    return True


def get_integer_input(prompt, default, min_val=1, max_val=100):
    """Get integer input from user with validation.

    Prompts for input and validates it's an integer within specified range.
    Returns default value if user enters empty string.

    Parameters
    ----------
    prompt: str
        text to display when asking for input
    default: int
        default value to use if input is empty
    min_val: int
        minimum acceptable value
    max_val: int
        maximum acceptable value

    Returns
    -------
    int
        validated integer input
    """
    while True:
        try:
            user_input = input(prompt)
            if not user_input:
                return default
            value = int(user_input)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid number.")


def interactive_mode(G):
    """Run an interactive mode for the user to explore the data.

    Presents a menu of analysis options to the user.
    Calls appropriate functions based on user selections.
    Loops until the user chooses to exit.

    Parameters
    ----------
    G: networkx.Graph
        graph containing store and state nodes
    """
    print("\n" + "=" * 50)
    print("GROCERY CHAIN ANALYSIS - INTERACTIVE MODE")
    print("=" * 50)

    while True:
        print("\nOptions:")
        print("1. Find states that share the most store chains with a given state")
        print("2. Find the most common store chains in a state")
        print("3. Find the shortest path between two nodes")
        print("4. Get statistics about a node")
        print("5. Find the most connected states or store chains")
        print("6. Export graph data to CSV")
        print("7. Visualize the network")
        print("8. Compare two states or store chains")
        print("9. Exit")

        choice = input("\nEnter your choice (1-10): ")

        if choice == '1':
            state = input("Enter state abbreviation (e.g., CA, MI): ")
            limit = get_integer_input("How many results to show? (default: 5): ", 5)
            result = find_most_related_states(G, state, limit)
            print("\nStates sharing the most store chains with", state, ":")
            if isinstance(result, list):
                for state_name, count in result:
                    print(f"  {state_name}: {count} shared store chains")
            else:
                print(result)

        elif choice == '2':
            state = input("Enter state abbreviation (e.g., CA, MI): ")
            limit = get_integer_input("How many results to show? (default: 5): ", 5)
            result = find_most_common_store_in_state(G, state, limit)
            print("\nMost common store chains in", state, ":")
            if isinstance(result, list):
                for store, locations in result:
                    print(f"  {store}: {locations} locations")
            else:
                print(result)

        elif choice == '3':
            node1 = input("Enter first node (state abbreviation or store name): ")
            node2 = input("Enter second node (state abbreviation or store name): ")
            path = find_shortest_path(G, node1, node2)
            print("\nShortest path:")
            if isinstance(path, list):
                print(" -> ".join(path))
                print(f"Path length: {len(path) - 1}")
            else:
                print(path)

        elif choice == '4':
            node = input("Enter node name (state abbreviation or store name): ")
            stats = get_node_stats(G, node)
            print("\nNode statistics:")
            if isinstance(stats, dict):
                for key, value in stats.items():
                    if isinstance(value, list) and len(value) > 10:
                        display_limit = get_integer_input(f"Show how many {key}? (default: 10): ", 10)
                        print(f"  {key}: {', '.join(value[:display_limit])}... (and {len(value) - display_limit} more)")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(stats)

        elif choice == '5':
            node_type = input("Enter node type (state, store, or leave blank for all): ").lower()
            if node_type not in ['state', 'store', '']:
                print("Invalid node type. Please enter 'state', 'store', or leave blank.")
                continue

            node_type = node_type if node_type else None
            limit = get_integer_input("How many results to show? (default: 10): ", 10)
            most_connected = find_most_connected_nodes(G, node_type, limit)

            print("\nMost connected nodes:")
            for node, connections in most_connected:
                node_type_str = G.nodes[node].get('type', 'unknown')
                print(f"  {node} ({node_type_str}): {connections} connections")

        elif choice == '6':
            force = input("Force export even if files already exist? (y/n, default: n): ").lower() == 'y'
            exported = export_graph_data(G, force=force)
            if not exported:
                print("Export skipped. Use force option to overwrite existing files.")

        elif choice == '7':
            print("\nVisualization options:")
            print("1. Visualize entire network")
            print("2. Visualize specific node and its connections")

            viz_choice = input("Enter choice (1-2): ")

            if viz_choice == '1':
                filename = visualize_graph(G)
                print(f"Full network visualization saved as {filename}")
            elif viz_choice == '2':
                node = input("Enter node name to visualize: ")
                if node in G:
                    filename = visualize_graph(G, specific_node=node)
                    print(f"Network visualization for {node} saved as {filename}")
                else:
                    print(f"Node '{node}' not found in the data.")
            else:
                print("Invalid choice.")

        elif choice == '8':
            print("\nNode comparison options:")
            print("1. Compare two states")
            print("2. Compare two store chains")

            comp_choice = input("Enter choice (1-2): ")

            if comp_choice == '1':
                node_type = 'state'
                print("\nEnter two state abbreviations to compare:")
            elif comp_choice == '2':
                node_type = 'store'
                print("\nEnter two store chain names to compare:")
            else:
                print("Invalid choice.")
                continue

            node1 = input("Enter first node name: ")
            node2 = input("Enter second node name: ")

            result = compare_nodes(G, node1, node2)

            if isinstance(result, dict):
                print(f"\nComparison between {node1} and {node2}:")
                print(f"\n{node1}:")
                for key, value in result['node1'].items():
                    print(f"  {key}: {value}")

                print(f"\n{node2}:")
                for key, value in result['node2'].items():
                    print(f"  {key}: {value}")

                print("\nIn common:")
                print(f"  Number of common items: {result['common']['count']}")

                common_list = result['common'].get('stores' if node_type == 'state' else 'states', [])
                if common_list:
                    display_limit = get_integer_input(f"Show how many common items? (default: 10): ", 10)
                    if display_limit >= len(common_list):
                        print(f"  Common items: {', '.join(common_list)}")
                    else:
                        print(
                            f"  Common items: {', '.join(common_list[:display_limit])}... (and {len(common_list) - display_limit} more)")
            else:
                print(result)

        elif choice == '9':
            print("Exiting interactive mode.")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 7.")


def main():
    """Main function to run the grocery chain analysis program.

    Loads data, builds the graph, and launches interactive mode.
    Exports graph data to CSV if not already exported.
    """
    # Load data
    data = load_data()

    # Build graph
    G = build_graph(data)

    # Check if export files already exist
    nodes_path = 'graph_nodes.csv'
    edges_path = 'graph_edges.csv'
    nodes_exist = os.path.exists(nodes_path)
    edges_exist = os.path.exists(edges_path)

    if nodes_exist and edges_exist:
        print(f"Export files already exist. Skipping automatic export.")
    else:
        export_graph_data(G)

    # Run interactive mode
    interactive_mode(G)


if __name__ == "__main__":
    main()