# Graph error propagation theorem

**Status:** Classical deterministic graph-Laplacian consequence, not a novel neural learning theorem.

For a connected undirected comparison graph, define prediction errors e_i at action nodes, incidence matrix B, Laplacian L=BᵀB, and d=unit_t-unit_s. Define R_eff(s,t)=dᵀL⁺d.

**Claim:** |e_t| ≤ |e_s| + sqrt(R_eff(s,t)) sqrt(Σ_(i,j)∈E (e_i-e_j)²).

**Proof:** Because s and t are connected, d lies in range(L), hence d=LL⁺d. Therefore e_t-e_s=dᵀe=(BL⁺d)ᵀBe. Cauchy–Schwarz gives |e_t-e_s|≤||BL⁺d||₂||Be||₂. The first squared norm equals dᵀL⁺LL⁺d=dᵀL⁺d=R_eff; the second squared norm is Σ(e_i-e_j)². Finally use |e_t|≤|e_s|+|e_t-e_s|. QED.

**Limitations:** Requires true error differences; a pairwise training loss does not guarantee small error differences. It is not a convergence theorem, an RGB identifiability guarantee, or evidence of superiority over OpenWAM/VPP.

**Counterexample:** Identical rendered objects with hidden thresholds c=.5 versus c=.6 produce identical stationary outcomes for all observed pushes a≤.4, but differ for a=.55. Without additional observations or prior constraints, the counterfactual is not identifiable.
