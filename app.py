import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Search & Fuzzy", layout="wide")

# ===========================================================
# TRIANGLE MEMBERSHIP FUNCTION (ANTI DIVISION BY ZERO)
# ===========================================================
def triangle(x, a, b, c):
    """Triangular membership function with special handling for shoulder triangles."""
    if b == a:  # left-shoulder (e.g., (0,0,5))
        if x <= a:
            return 1
        elif a < x < c:
            return (c - x) / (c - b) if (c - b) != 0 else 0
        else:
            return 0

    if b == c:  # right-shoulder (e.g., (5,10,10))
        if x >= c:
            return 1
        elif a < x < c:
            return (x - a) / (b - a) if (b - a) != 0 else 0
        else:
            return 0

    # normal triangle
    if a < x < b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    elif x == b:
        return 1
    else:
        return 0


# ===========================================================
# SEARCH ALGORITHMS: BFS, DFS, A*
# ===========================================================
def bfs(graph, start, goal):
    queue = [(start, [start])]
    visited = set()

    while queue:
        node, path = queue.pop(0)

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path

        for neighbor in graph[node]:
            queue.append((neighbor, path + [neighbor]))
    return None


def dfs(graph, start, goal):
    stack = [(start, [start])]
    visited = set()

    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path

        for neighbor in graph[node]:
            stack.append((neighbor, path + [neighbor]))

    return None


def a_star(graph, start, goal, heuristic):
    open_list = [(start, [start], 0)]
    visited = set()

    while open_list:
        open_list.sort(key=lambda x: x[2] + heuristic[x[0]])
        node, path, cost = open_list.pop(0)

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path, cost

        for next_node, weight in graph[node]:
            open_list.append((next_node, path + [next_node], cost + weight))

    return None, None


# ===========================================================
# STREAMLIT UI MENU
# ===========================================================
st.title("🤖 AI Search & Fuzzy Logic System")

menu = st.sidebar.radio("Menu", ["Fuzzy Tip", "BFS / DFS", "A* Search"])

# ===========================================================
# 1. FUZZY TIP CALCULATOR (TAMPILAN SEPERTI GAMBAR)
# ===========================================================
if menu == "Fuzzy Tip":

    st.markdown("## 🍽 Fuzzy Logic – Restaurant Tip Calculator")
    st.write("Masukkan **Food Quality** dan **Service Quality** (0–10) untuk menghitung tip menggunakan *Fuzzy Mamdani*.")

    col_input, col_plot = st.columns([1, 1.2])

    # ========================
    # INPUT SLIDER
    # ========================
    with col_input:
        food = st.slider("Food Quality (0–10)", 0, 10, 7)
        service = st.slider("Service Quality (0–10)", 0, 10, 3)

        hitung = st.button("Hitung Tip")

        # Membership values
        fq_bad = triangle(food, 0, 0, 5)
        fq_good = triangle(food, 5, 10, 10)

        sq_poor = triangle(service, 0, 0, 5)
        sq_excellent = triangle(service, 5, 10, 10)

        # Rules
        rule_low = max(sq_poor, fq_bad)      # OR
        rule_high = min(sq_excellent, fq_good)  # AND

        # Defuzzification
        tip_low = 5
        tip_high = 20
        total = rule_low + rule_high

        tip_result = 10 if total == 0 else (rule_low * tip_low + rule_high * tip_high) / total

        if hitung:
            st.success(f"### Tip Direkomendasikan: **{tip_result:.2f}%**")

        # Detail Membership
        with st.expander("Detail Membership"):
            st.write(f"- Food Bad: **{fq_bad:.3f}**")
            st.write(f"- Food Good: **{fq_good:.3f}**")
            st.write(f"- Service Poor: **{sq_poor:.3f}**")
            st.write(f"- Service Excellent: **{sq_excellent:.3f}**")

        with st.expander("Aktivasi Rules"):
            st.write(f"- Rule Low (max Poor/Bad): **{rule_low:.3f}**")
            st.write(f"- Rule High (min Excellent/Good): **{rule_high:.3f}**")

    # ========================
    # GRAPH MEMBERSHIP
    # ========================
    with col_plot:
        x = np.linspace(0, 10, 200)

        # FOOD MEMBERSHIP GRAPH
        fig, ax = plt.subplots(figsize=(6, 4))
        food_bad_curve = [triangle(i, 0, 0, 5) for i in x]
        food_good_curve = [triangle(i, 5, 10, 10) for i in x]

        ax.plot(x, food_bad_curve, label="Bad")
        ax.plot(x, food_good_curve, label="Good")

        ax.fill_between(x, 0, food_bad_curve, alpha=0.15)
        ax.fill_between(x, 0, food_good_curve, alpha=0.15)

        ax.set_title("Food Membership")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Food Quality")

        st.pyplot(fig)

        # SERVICE MEMBERSHIP GRAPH
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        service_poor_curve = [triangle(i, 0, 0, 5) for i in x]
        service_excellent_curve = [triangle(i, 5, 10, 10) for i in x]

        ax2.plot(x, service_poor_curve, label="Poor")
        ax2.plot(x, service_excellent_curve, label="Excellent")

        ax2.fill_between(x, 0, service_poor_curve, alpha=0.15)
        ax2.fill_between(x, 0, service_excellent_curve, alpha=0.15)

        ax2.set_title("Service Membership")
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel("Service Quality")

        st.pyplot(fig2)


# ===========================================================
# 2. BFS / DFS
# ===========================================================
if menu == "BFS / DFS":

    st.header("🧭 Blind Search: BFS & DFS")

    graph = {
        "A": ["B", "C"],
        "B": ["D", "E"],
        "C": ["F"],
        "D": [],
        "E": ["F"],
        "F": []
    }

    start = st.selectbox("Start", list(graph.keys()), index=0)
    goal = st.selectbox("Goal", list(graph.keys()), index=5)

    if st.button("Run BFS"):
        path = bfs(graph, start, goal)
        st.info(f"BFS Path: **{path}**")

    if st.button("Run DFS"):
        path = dfs(graph, start, goal)
        st.info(f"DFS Path: **{path}**")


# ===========================================================
# 3. A* SEARCH
# ===========================================================
if menu == "A* Search":

    st.header("⭐ A* Heuristic Search")

    graph = {
        "A": [("B", 1), ("C", 4)],
        "B": [("D", 2), ("E", 5)],
        "C": [("F", 3)],
        "D": [],
        "E": [("F", 1)],
        "F": []
    }

    heuristic = {
        "A": 6,
        "B": 4,
        "C": 2,
        "D": 4,
        "E": 1,
        "F": 0
    }

    start = st.selectbox("Start Node", list(graph.keys()))
    goal = st.selectbox("Goal Node", list(graph.keys()), index=5)

    if st.button("Run A*"):
        path, cost = a_star(graph, start, goal, heuristic)
        st.success(f"Path: **{path}**, Cost = **{cost}**")
