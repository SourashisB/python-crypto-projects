import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
import os

def fetch_wikipedia_page(prompt_word):
    """
    Fetch the Wikipedia page for the given prompt word.
    Returns the HTML content of the page if it exists, or raises an exception if not found.
    """
    url = f"https://en.wikipedia.org/wiki/{prompt_word}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.text, url
    else:
        raise Exception(f"Error: Wikipedia page for '{prompt_word}' does not exist.")

def scrape_wikipedia_content(html_content):
    """
    Scrape the main content of the Wikipedia page using BeautifulSoup.
    Returns a list of (heading, level) tuples representing the hierarchy of headings.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract the main content of the page
    content_div = soup.find('div', {'id': 'mw-content-text'})
    if not content_div:
        raise Exception("Error: Unable to locate main content on the Wikipedia page.")
    
    # Collect headings and their levels
    headings = []
    for heading in content_div.find_all(['h2', 'h3']):
        heading_text = heading.get_text().strip()
        heading_text = heading_text.replace("[edit]", "")  # Clean up edit links
        if heading.name == 'h2':
            level = 2
        elif heading.name == 'h3':
            level = 3
        else:
            continue
        headings.append((heading_text, level))
    
    return headings

def build_graph_from_headings(headings):
    """
    Build a graph data structure from the extracted headings.
    Each heading is a node, and subheadings (h3) are connected to their parent headings (h2).
    """
    G = nx.DiGraph()
    parent_node = None  # To track the current parent node for subheadings

    for heading, level in headings:
        G.add_node(heading)  # Add the heading as a node in the graph
        if level == 2:  # Major heading
            parent_node = heading  # Update the current parent node
        elif level == 3 and parent_node:  # Subheading, connect it to the parent
            G.add_edge(parent_node, heading)
    
    return G

def visualize_graph(G, filename):
    """
    Visualize the graph using matplotlib and networkx, and save it as an image.
    """
    if G.number_of_nodes() == 0:
        print("Warning: The graph is empty. No visualization will be generated.")
        return

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)  # Layout for better visualization
    nx.draw(
        G, pos, with_labels=True, node_size=2000, font_size=10,
        node_color="lightblue", font_weight="bold", edge_color="gray"
    )
    plt.title("Wikipedia Page Graph", fontsize=16)
    
    # Save the graph as an image
    plt.savefig(filename, format="png")
    print(f"Graph image saved as: {filename}")
    plt.show()

def main():
    try:
        # Step 1: Get input from the user
        prompt_word = input("Enter a prompt word to search on Wikipedia: ").strip()
        
        # Step 2: Fetch the Wikipedia page
        html_content, url = fetch_wikipedia_page(prompt_word)
        print(f"Successfully fetched page: {url}")
        
        # Step 3: Scrape the content
        headings = scrape_wikipedia_content(html_content)
        if not headings:
            print("No headings found on the page.")
            return
        
        print(f"Headings found: {headings}")
        
        # Step 4: Build the graph
        G = build_graph_from_headings(headings)
        
        # Debugging: Print the graph nodes and edges
        print(f"Graph Nodes: {list(G.nodes)}")
        print(f"Graph Edges: {list(G.edges)}")
        
        # Step 5: Visualize the graph and save the image
        output_filename = os.path.join(os.getcwd(), f"{prompt_word}_graph.png")
        visualize_graph(G, output_filename)
    
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    main()