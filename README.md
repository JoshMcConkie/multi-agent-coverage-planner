Topic: Submodular Set Maximization/Path Planning
## Problem Structure
We have $K$ agents on a square grid. The goal is the maximize the total coverage of the grid given the following constraints:
- A set path length for each agent
- The grid is comprised of discrete integer coordinates
- Each coordinate is worth 1 point, with overlapping providing no extra benefit
>[!info] Scope
>Currently, we only consider the case where all agents share the same initial position.
### Objective
We seek paths such that the coverage of the grid is maximized. More explicitly, we follow an expansion of Nemhauser et al.'s submodular set maximization problem.
Let
- $K$ be the number of agents, where $\text{agent}_{k}$ chooses before $\text{agent}_{k+1}$ for $1\le k < K$
-  $D$ be the length of the path to be selected (same for all agents)
- $\Omega$ be an $m\times n$ utility matrix
	- $m$ is the number of possible grid locations
	- $n$ is the cumulative number of paths of length $D$ available to any agent
	- $\mathcal C$ be the row index set (encodes all coordinates)
	- $\mathcal P$ be the columns index set (encodes all possible paths)
	- $\omega_{cp}\in \Omega$ with $c\in\mathcal C$ and $p\in\mathcal P$.
- $\mathcal P_k\subseteq \mathcal P$ containing the indices for all available paths for $\text{agent}_{k}$
- $S_{k-1}\subseteq \mathcal P$ contains the path indices of all previous allocations at sequential choice $k$
- $\text{Coverage}(S)$ encode the utility score of a selection of paths. Let $I$ be the set of row indices, where each row represents a possible location to visit. $$\text{Coverage(S)} = \sum_{c\in \mathcal C} \max_{p\in S}\{\omega_{cp}\}.$$
- $e\in \mathcal P_k$, be a possible path for $\text{agent}_{k}$ with $e_k$ marking the final path for agent $k$. The selection is based by maximizing marginal gain, i.e.$$e_k\in \arg\max_{e\in\mathcal P_k}\Big\{\text{Coverage}\big(S_{k-1}\cup\{e\}\big) - \text{Coverage}(S_{k-1})\Big\}$$ with $e_k$ found prior to $e_{k+1}$.

Thus, our objective is selecting paths to maximize coverage,$$\max_{e\in\mathcal P}\text{Coverage}\bigg[\bigcup_{k=1}^K\{e_k\}\bigg].$$
>[!note]
>For clarification, here is an sample of a possible $\Omega$ coordinate-path utility matrix:
> $$
> \Omega =
> \begin{array}{c|ccc}
> & p=0 & p=1 & ... & p=n \\
> \hline
> (0,0) & 1 & 0 & ... & 0\\
> (0,1) & 0 & 1 & ... & 1\\
> (0,2) & 1 & 1 & ... & 0\\
> \vdots & \vdots & \vdots & \ddots & \vdots\\
> \end{array}
> $$
## Approach
The above structure outlines the greedy approach, maximizing marginal utility of path coverage for all agents, iterating over all paths of length $D$ available to that agent. This increases exponentially as $D$ increases. 

We attempt to divide the problem in rounds that is, instead of $e_k$ encoding a path of length $D$, we provide a smaller, maximum "chunk size" each round will plan for. Given some chunk size $0<d< D$, partial paths are planned in much the same manner as the full paths, maintained that each round carries over all previously chosen path indices. If $d\mid D$, then $D_r=d$ for each round  $r$. If $d \nmid D$ $D_r=d$ for all but the final round, where $D_{r_{final}} = D(\mod d)$.
	
In each round $r$, the starting position of $\text{agent}_k$ will be the final position in round $r-1$.

### Limitations
- Not randomized direction-filtering order
- grid size/agent constraints
- Only considering single point start

