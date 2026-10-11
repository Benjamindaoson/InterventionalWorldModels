# Mathematical compendium — professor-facing audit

**Status:** The following are conditional results; none establishes a novel general-RGB world-model learning theorem.

## A. Counterexample
Two physically identical-looking worlds have sliding thresholds c=0.5 and c=0.6. Every probe a<=0.4 yields identical stationary video. Action a=0.55 differs. Under equal prior over worlds, a forced prediction has error >= 1/2, absent additional evidence.

## B. Anchor sufficiency
Let p(h,a) be a vector of future outcome probabilities. Assume known coefficients w_j(a), independent of history, such that p(h,a)=sum_j w_j(a)p(h,a_j) for all histories and target actions. Define Z(h)=(p(h,a_1),...,p(h,a_m)). Then p(h,a) is a function of Z and a, proving action sufficiency. This assumption is strong and not established for contact-rich RGB video.

## C. Proper scoring and learning
For one-hot observed outcome e_Y and anchor predictor q_j(h), conditional expected squared loss is ||q_j-p_j||²+1-||p_j||². Therefore excess population risk equals sum_j E||q_j-p_j||². If the model class is realizable and the population optimum is reached, anchor predictions are correct almost everywhere. This does not establish neural optimizer convergence.

## D. Extrapolation bound
If p_a=sum_j w_j p_j+r_a, with ||r_a||<=rho and ||w||_2<=C, then ||hat p_a-p_a|| <= rho + C sqrt(sum_j ||hat p_j-p_j||²). Proof: subtract, use triangle inequality, then Cauchy-Schwarz.

## E. Graph comparison certificate
Let B be the incidence matrix of a connected action-comparison graph, L=B^T B, d=unit_target-unit_reference. Since d is in range(L), d=L L^+ d. Thus e_target-e_reference=(B L^+ d)^T B e. Cauchy-Schwarz gives |e_target|<=|e_reference|+sqrt(d^T L^+ d) ||B e||. This is an algebraic bound on true errors, not a training guarantee.

## F. Guard margin
For L-Lipschitz guard g, observed y_i with |y_i-g(a_i)|<=eta and nearest probe within r, |y_i-g(a)|<=eta+Lr. Event sign is certified only outside this uncertainty margin. Assumes measurable continuous guard response, not just RGB labels.

## G. Active threshold
For deterministic monotone event E(a)=1[a>=c], n bisection queries reduce feasible interval width to 2^-n. Binary query trees have <=2^n leaves, so this is minimax in the restricted model.

## H. Nonlinear local factorization
For C1 maps Phi and g with locally constant rank(DPhi), on a sufficiently small connected constant-rank coordinate neighborhood, g factors through Phi iff ker(DPhi) is contained in ker(Dg). In constant-rank coordinates Phi(u,v)=(u,0), the kernel condition is partial_v g=0, so g depends only on u.

## I. Hybrid event-time sensitivity
For G(s(tau,theta),theta,tau)=0 with transverse crossing, implicit differentiation yields d tau/d theta = -(G_s s_theta+G_theta)/(G_t+G_s f_minus). Near grazing, the denominator can approach zero.

## Open research obligations
Not proved: identifiability of general RGB counterfactual outcomes from sparse interventions; optimization guarantees for a learned video model; performance gains over OpenWAM/VPP; calibration under physical distribution shift. These cannot be described as completed theorems.
