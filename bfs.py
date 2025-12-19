import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import heapq

st.set_page_config("Search Algorithm Analytics", layout="wide")
st.title("🧠 Visual Analytics of Search Algorithms")

st.markdown("""
This dashboard helps you **visually and analytically understand**  
**BFS, DFS, Greedy Best-First, and A\*** algorithms.
""")

# ---------------- DATA ----------------
edges = [
    ("A", "B", 2), ("A", "C", 4),
    ("B", "D", 7), ("B", "E", 3),
    ("C", "F", 5),
    ("D", "G", 1), ("E", "G", 6),
    ("F", "G", 2)
]

heuristic = {
    "A": 7, "B": 6, "C": 4,
    "D": 1, "E": 3, "F": 2, "G": 0
}

G = nx.Graph()
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

pos = nx.spring_layout(G, seed=10)

# ---------------- ALGORITHMS ----------------
def bfs(start, goal):
    queue, visited, steps = deque([[start]]), set(), []
    while queue:
        path = queue.popleft()
        node = path[-1]
        steps.append(node)
        if node == goal:
            return path, steps
        if node not in visited:
            visited.add(node)
            for n in G.neighbors(node):
                queue.append(path + [n])

def dfs(start, goal):
    stack, visited, steps = [[start]], set(), []
    while stack:
        path = stack.pop()
        node = path[-1]
        steps.append(node)
        if node == goal:
            return path, steps
        if node not in visited:
            visited.add(node)
            for n in G.neighbors(node):
                stack.append(path + [n])

def greedy(start, goal):
    pq, visited, steps = [(heuristic[start], [start])], set(), []
    while pq:
        _, path = heapq.heappop(pq)
        node = path[-1]
        steps.append(node)
        if node == goal:
            return path, steps
        if node not in visited:
            visited.add(node)
            for n in G.neighbors(node):
                heapq.heappush(pq, (heuristic[n], path + [n]))

def astar(start, goal):
    pq, visited, steps = [(heuristic[start], 0, [start])], set(), []
    while pq:
        f, g, path = heapq.heappop(pq)
        node = path[-1]
        steps.append(node)
        if node == goal:
            return path, steps, g
        if node not in visited:
            visited.add(node)
            for n in G.neighbors(node):
                cost = g + G[node][n]["weight"]
                heapq.heappush(pq, (cost + heuristic[n], cost, path + [n]))

# ---------------- SIDEBAR ----------------
algo = st.sidebar.selectbox(
    "Choose Algorithm",
    ["BFS", "DFS", "Greedy", "A*"]
)

start = st.sidebar.selectbox("Start Node", sorted(G.nodes))
goal = st.sidebar.selectbox("Goal Node", sorted(G.nodes), index=6)

# ---------------- RUN ----------------
if algo == "BFS":
    path, steps = bfs(start, goal)
    cost = len(path) - 1
elif algo == "DFS":
    path, steps = dfs(start, goal)
    cost = len(path) - 1
elif algo == "Greedy":
    path, steps = greedy(start, goal)
    cost = sum(G[path[i]][path[i+1]]["weight"] for i in range(len(path)-1))
else:
    path, steps, cost = astar(start, goal)

# ---------------- ANIMATION SLIDER ----------------
st.subheader("🎞️ Step-by-Step Animation")
step = st.slider("Algorithm Step", 1, len(steps), 1)

highlight_nodes = steps[:step]

plt.figure(figsize=(7, 6))
nx.draw(G, pos, with_labels=True, node_size=1800)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=highlight_nodes,
    node_color="orange",
    node_size=2200
)
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels={(u, v): d["weight"] for u, v, d in G.edges(data=True)}
)
st.pyplot(plt)

# ---------------- FINAL PATH ----------------
st.subheader("🗺️ Final Path Found")

plt.figure(figsize=(7, 6))
nx.draw(G, pos, with_labels=True, node_size=1800)
nx.draw_networkx_edges(
    G, pos,
    edgelist=list(zip(path, path[1:])),
    edge_color="red",
    width=4
)
st.pyplot(plt)

# ---------------- ANALYTICS ----------------
col1, col2, col3 = st.columns(3)
col1.metric("Algorithm", algo)
col2.metric("Nodes Expanded", len(steps))
col3.metric("Total Cost", cost)

# ---------------- DATA TABLE ----------------
st.subheader("📋 Node Expansion Analytics")

df = pd.DataFrame({
    "Step": range(1, len(steps)+1),
    "Expanded Node": steps,
    "Heuristic h(n)": [heuristic[n] for n in steps]
})

st.dataframe(df, use_container_width=True)

# ---------------- COMPARISON CHART ----------------
st.subheader("📊 Algorithm Comparison (Analytics View)")

comparison = pd.DataFrame({
    "Algorithm": ["BFS", "DFS", "Greedy", "A*"],
    "Nodes Expanded": [
        len(bfs(start, goal)[1]),
        len(dfs(start, goal)[1]),
        len(greedy(start, goal)[1]),
        len(astar(start, goal)[1])
    ]
})

st.bar_chart(comparison.set_index("Algorithm"))

# ---------------- INSIGHTS ----------------
st.info("""
### 📌 Learning Insights

• **BFS** → Best for shortest step workflows  
• **DFS** → Useful for deep pattern discovery  
• **Greedy** → Fast decisions, may be wrong  
• **A\\*** → Best cost-optimal analytics  

🎯 **A\\*** is preferred in logistics, navigation, AI planning
""")
