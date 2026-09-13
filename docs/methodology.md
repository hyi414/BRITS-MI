# BRITS-MI methodology

## 1. Model hierarchy

BRITS-MI is a hierarchy rather than a new recurrent cell:

1. A gated recurrent unit (GRU) updates the nonlinear hidden state.
2. Recurrent Imputation for Time Series (RITS) surrounds the GRU with elapsed-time decay, temporal prediction, zero-diagonal cross-marker regression, and observed-value substitution.
3. Bidirectional Recurrent Imputation for Time Series (BRITS) runs RITS forward and backward and reconciles the two predictions.
4. BRITS-MI adds artificial holdouts, trajectory-summary and downstream losses, stochastic completed datasets, optional estimating-score calibration, downstream refitting, and Rubin pooling.

The outcome may influence training. It is not an input when a frozen imputer completes a new patient's trajectory. Outcome-inclusive score calibration is a separate explanatory-analysis operation.

## 2. Irregular longitudinal input

For subject `i`, visit `t`, and marker vector `Y_it`, let `M_it` be the source-observation mask, `R_it` an artificial holdout sampled only where `M_it=1`, and `Delta_it` the marker-specific elapsed times. During training,

```math
M^{in}=M\odot(1-R),
\qquad
u_{it}=[Y^{in}_{it}\odot M^{in}_{it},M^{in}_{it},\Delta_{it},H^{past}_{it},S_i,V_i,\tau_{it}].
```

`S_i` contains static covariates, `V_i` optional image context, and `H^{past}` directional history summaries. An unavailable standardized value is stored as zero, but the mask distinguishes that placeholder from a measured zero.

## 3. One RITS-GRU direction

For direction `d` in `{F,B}`, elapsed time decays the old state:

```math
\gamma^d_{it}=\exp[-\operatorname{ReLU}(W^d_\gamma\Delta^d_{it}+b^d_\gamma)],
\qquad
\bar h^d_{i,t-1}=\gamma^d_{it}\odot h^d_{i,t-1}.
```

The temporal and zero-diagonal cross-marker estimates are

```math
\widehat Y^{hist,d}_{it}=W_h^d\bar h^d_{i,t-1}+b_h^d,
```

```math
\widehat Y^{feat,d}_{it}=[W_f^d\odot(\mathbf 1-I_p)]Y^{prov,d}_{it}+b_f^d.
```

A mask-and-gap gate combines them and the observed-value lock forms the completed visit:

```math
a^d_{it}=\sigma\{W_a^d[M^{in,d}_{it},\Delta^d_{it}]+b_a^d\},
```

```math
\widehat Y^d_{it}=a^d_{it}\odot\widehat Y^{feat,d}_{it}
+(1-a^d_{it})\odot\widehat Y^{hist,d}_{it},
```

```math
Y^d_{it}=M^{in,d}_{it}\odot Y^{in,d}_{it}
+(1-M^{in,d}_{it})\odot\widehat Y^d_{it}.
```

With recurrent input `x_it^d=[Y_it^d,M_it^{in,d},Delta_it^d,H_it^d,S_i,V_i,tau_it^d]`, the GRU update is

```math
r^d_{it}=\sigma(W_r^dx^d_{it}+U_r^d\bar h^d_{i,t-1}+b_r^d),
```

```math
z^d_{it}=\sigma(W_z^dx^d_{it}+U_z^d\bar h^d_{i,t-1}+b_z^d),
```

```math
n^d_{it}=\tanh\{W_n^dx^d_{it}+b_n^d+r^d_{it}\odot(U_n^d\bar h^d_{i,t-1}+c_n^d)\},
```

```math
h^d_{it}=(1-z^d_{it})\odot n^d_{it}+z^d_{it}\odot\bar h^d_{i,t-1}.
```

The reset gate chooses which decayed history enters the candidate state; the update gate chooses how much old state is retained. Because the imputed visit enters `x_it^d`, one missing-cell estimate changes the state used at later visits in that direction.

## 4. BRITS reconciliation

The released code combines the two directional predictions using a gate based on the mask and elapsed gaps:

```math
q_{it}=\sigma\{W_q[M^{in}_{it},\Delta_{it}]+b_q\},
```

```math
\mu_{\theta,it}=q_{it}\odot\widehat Y^F_{it}
+(1-q_{it})\odot\widehat Y^B_{it},
```

where `theta=(theta_F,theta_B,theta_q)`. Measured values are then restored:

```math
\widetilde Y_i=M_i\odot Y_i+(1-M_i)\odot\mu_{\theta,i}.
```

## 5. Composite training objective

For downstream parameters `psi`, minibatch `B`, and artificial mask `R`, BRITS-MI minimizes

```math
\mathcal L_{\mathcal B,R}(\theta,\psi)=
\lambda_{rec}\mathcal L_{rec}
+\lambda_{cons}\mathcal L_{cons}
+\lambda_{sum}\mathcal L_{sum}
+\lambda_{down}\mathcal L_{down}.
```

`L_rec` evaluates only hidden-but-known measured cells; `L_cons` penalizes forward-backward disagreement; `L_sum` preserves prespecified trajectory summaries; and `L_down` is the association-score or classification target. Nonnegative loss weights are selected on artificial validation holdouts and then frozen.

## 6. Exact downstream gradient into BRITS-MI

Stack each subject's cells and define the missing-cell selector `D_i=diag(1-m_i)`. For a multiclass downstream head,

```math
g_i=G(\widetilde y_i,S_i,V_i),
\qquad
\pi_i=\operatorname{softmax}\{f_\psi(g_i)\}.
```

The cross-entropy gradient reaching the recurrent imputer is

```math
\nabla_\theta\mathcal L_{down}
=|\mathcal B|^{-1}\sum_{i\in\mathcal B}
J_\theta\mu_{\theta,i}^{\mathsf T}
D_iJ_G(\widetilde y_i)^{\mathsf T}
J_{f_\psi}(g_i)^{\mathsf T}(\pi_i-e_{O_i}).
```

The factors respectively represent BRITS sensitivity, the missing-cell lock, trajectory/fusion-feature sensitivity, downstream-head sensitivity, and prediction error. Because `D_i` is zero at measured cells, the downstream loss cannot change a source-observed value.

The BRITS Jacobian further decomposes as

```math
J_\theta\mu_{\theta,it}
=D(q_{it})J_\theta\widehat Y^F_{it}
+D(1-q_{it})J_\theta\widehat Y^B_{it}
+D(\widehat Y^F_{it}-\widehat Y^B_{it})J_\theta q_{it}.
```

Hence downstream error updates the forward and backward RITS-GRU parameters and the reconciliation gate. Back-propagation through time also carries the signal across visits through products of GRU state-transition Jacobians.

## 7. Association target and coefficient-control result

For a reference coefficient `beta*`, define a standardized estimating-score loss

```math
\mathcal L_{score}
=(2d)^{-1}\|D_s^{-1}U(\beta^*;\widetilde Y)\|_2^2.
```

Its recurrent-parameter gradient is

```math
\nabla_\theta\mathcal L_{score}
=d^{-1}J_\theta\widetilde y^{\mathsf T}
J_U(\widetilde y)^{\mathsf T}D_s^{-2}U(\beta^*;\widetilde Y).
```

If the downstream estimate solves `U(beta_hat;Y_tilde)=0` and the averaged score Jacobian `A_n` between `beta*` and `beta_hat` is nonsingular, the mean-value identity gives

```math
\widehat\beta-\beta^*=A_n^{-1}U(\beta^*;\widetilde Y),
```

and therefore

```math
\|\widehat\beta-\beta^*\|_2
\leq\|A_n^{-1}D_s\|_2\sqrt{2d\mathcal L_{score}}.
```

This bound explains why reducing the score discrepancy can preserve a coefficient even when cell RMSE changes little. It is conditional on a locally identifiable, well-conditioned outcome model and does not prove that naturally missing values are identified.

## 8. Stochastic completion, optional calibration, and pooling

After selecting and freezing a validation checkpoint, marker-specific residual scales are estimated on separate artificial holdouts. Completion `q` draws only missing cells:

```math
\widetilde Y_{itj}^{(q)}=M_{itj}Y_{itj}
+(1-M_{itj})\{\mu_{\widehat\theta,itj}+\kappa\widehat\sigma_j\epsilon_{itj}^{(q)}\},
\qquad \epsilon_{itj}^{(q)}\sim N(0,1).
```

For an explanatory association analysis, the optional bounded calibration step is

```math
y_{mis}^{(\ell+1)}=\Pi_{\mathcal C}
\{y_{mis}^{(\ell)}-\rho_\ell P_\ell\nabla_{y_{mis}}\mathcal L_{score}\}.
```

Backtracking accepts only a non-increasing score loss up to numerical tolerance, and observed cells are restored after every proposal. The downstream model is then refitted in every completed dataset. Estimates and variances are combined with Rubin's rules.

## 9. Optimization guarantee and its scope

Let `vartheta=(theta,psi)` and `F(vartheta)=E_{B,R}[L_{B,R}(vartheta)]`. Under a lower-bounded smooth objective, a Lipschitz gradient with constant `L_F`, and an unbiased stochastic gradient with conditional variance at most `sigma_g^2`, stochastic gradient descent with `eta<=1/L_F` obeys

```math
K^{-1}\sum_{k=0}^{K-1}\mathbb E\|\nabla F(\vartheta_k)\|_2^2
\leq\frac{2\{F(\vartheta_0)-F_{inf}\}}{\eta K}
+\eta L_F\sigma_g^2.
```

This is convergence toward a first-order stationary point of the smooth composite objective, not a global optimum. The implementation uses finite-epoch AdamW, gradient clipping, and validation checkpointing, so the theorem is an objective-level result rather than a guarantee for every fitted checkpoint. Nominal multiple-imputation coverage additionally requires an identified conditional model, proper stochastic draws, congenial downstream analysis, and valid within- and between-imputation variance.

See `src/brits_mi/model.py`, `src/brits_mi/imputer.py`, `src/brits_mi/calibration.py`, and `src/brits_mi/analysis.py` for the corresponding implementation.
