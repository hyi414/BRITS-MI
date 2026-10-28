# Unified recurrent BRITS-MI

This specifies `UnifiedBRITSMI` in `src/brits_mi/unified.py` and
the new comparisons in `simulation/unified/`. Earlier implementations
remain available in [the legacy specification](methodology_legacy.md);
their results are not results from the new recurrent implementation.

## Analysis target and observed information

Let $Y_{itj}$ denote biomarker $j$ at time $\tau_{it}$ for subject $i$,
$S_i$ observed static covariates, $C_i$ the outcome, and $V_i=\phi(I_i)$
optional image features. The source-observation indicator is $M_{itj}$.
In a masking experiment, $A_{itj}=1$ denotes a measured cell hidden before
any imputer is fitted. The imputer receives $O=M\odot(1-A)$ and values
only where $O=1$. A separate training mask $R$ is sampled within $O=1$,
giving $Q=O\odot(1-R)$. Structural padding is neither missing data nor an
evaluation target.

The feature map $G(Y_i,\tau_i,S_i)$ defines the downstream model. Clinical
association analysis uses binary logistic regression with an intercept and
13 terms: late AST, ALT, and platelet levels; their three terminal slopes;
diabetes, hypertension, and sex; AST level by diabetes; AST slope by
diabetes; platelet slope by hypertension; and hypertension by sex.
For each marker, the late level is the mean of the last three visits and
the terminal slope is the last-minus-third-last value divided by elapsed time.

The shared engine supports two distinct information settings:

- **Association analysis:** the observed outcome enters the imputation
  context, as in outcome-inclusive MICE. Its one-hot encoding is not
  directly supplied to the downstream head, which receives the planned
  clinical design. New association imputations require an observed outcome.
- **Prediction:** training labels supervise the head, but held-out labels
  never enter the imputer. New-patient inputs consist of observed biomarkers,
  masks, times, covariates, and optional images.

## RITS, GRU, and BRITS

Recurrent Imputation for Time Series (RITS) combines elapsed-time decay,
history regression, cross-marker regression, and a gated recurrent unit
(GRU). Bidirectional Recurrent Imputation for Time Series (BRITS) combines
forward and backward RITS. Original BRITS already supports downstream
supervision [1]; adding classification cross-entropy alone is not the
contribution here. The extension targets stochastic multiple imputation
and recovery of prespecified regression associations as well as prediction.

In each direction, $\Delta_{it}$ is time since the previous observed value.
Times are divided by the median valid follow-up duration in inner-training
subjects; padding is excluded. The decayed state and history estimate are

$$\gamma_{it}=\exp\{-\operatorname{ReLU}(W_\gamma\Delta_{it}+b_\gamma)\},\qquad \bar h_{i,t-1}=\gamma_{it}\odot h_{i,t-1},$$
$$a_{it}=W_h\bar h_{i,t-1}+b_h.$$

Restore observations in $Y^p_{it}=Q_{it}\odot Y_{it}+(1-Q_{it})\odot a_{it}$.
The zero-diagonal cross-marker estimate is
$b_{it}=(W_f\odot(1-I))Y^p_{it}+b_f$. A mask-and-gap gate combines estimates:

$$\alpha_{it}=\sigma(W_\alpha[Q_{it},\Delta_{it}]+b_\alpha),\qquad \widehat Y_{it}=\alpha_{it}\odot b_{it}+(1-\alpha_{it})\odot a_{it},$$
$$Y^c_{it}=Q_{it}\odot Y_{it}+(1-Q_{it})\odot\widehat Y_{it}.$$

The GRU input $u_{it}$ contains $Y^c_{it}$, masks, time gaps, static and image
context, time, and directional observed-history summaries. The summaries
(mean, SD, most recent value, first-to-most-recent slope, and observed/missing
proportions) use only measurements preceding that update. The PyTorch GRU
convention is

$$r_{it}=\sigma(W_ru_{it}+U_r\bar h_{i,t-1}+b_r),\qquad z_{it}=\sigma(W_zu_{it}+U_z\bar h_{i,t-1}+b_z),$$
$$n_{it}=\tanh\{W_nu_{it}+b_n+r_{it}\odot(U_n\bar h_{i,t-1}+c_n)\},$$
$$h_{it}=(1-z_{it})\odot n_{it}+z_{it}\odot\bar h_{i,t-1}.$$

The reset gate controls history entering the candidate state; the update
gate controls retention of the previous state. Forward and backward passes
have separate parameters. A context network adds a subject-specific shift
to their predictions. With a learned mask-and-gap gate $q_{it}$,

$$\mu_{\theta,it}=q_{it}\odot\widehat Y^F_{it}+(1-q_{it})\odot\widehat Y^B_{it}+s_\theta(S_i,V_i,C_i).$$

$C_i$ is omitted in prediction mode. Image information therefore changes
current-cell predictions directly and later states through the GRU input.
The network predicts a conditional mean, not a learned residual scale.

## Objective and downstream gradient

Observed biomarkers are standardized using inner-training observations
only. The fixed development objective is

$$\mathcal L(\theta,\psi)=\mathcal L_{rec}+0.1\mathcal L_{cons}+0.2\mathcal L_{sum}+0.2\mathcal L_{down}.$$

- $\mathcal L_{rec}$ is standardized squared error on artificially hidden
  measured training cells.
- $\mathcal L_{cons}$ penalizes forward-backward disagreement on valid
  cells with $Q=0$.
- $\mathcal L_{sum}$ is squared error in standardized downstream features.
  Its reference restores observed training values and uses detached current
  predictions where values are unavailable. Neither hidden simulation truth
  nor full-data coefficients enter this reference.
- $\mathcal L_{down}$ is binary cross-entropy for the clinical logistic head
  or multiclass cross-entropy for the neural fusion head. Only known training
  outcomes contribute.

Let $D_i=\operatorname{diag}(1-Q_i)$, $g_i=G(\widetilde Y_i,\tau_i,S_i)$,
$\eta_i=f_\psi(g_i,V_i)$, and $p_i$ the outcome probability vector. For the
subjects with known training outcomes, $\mathcal B_C$,

$$\nabla_\theta\mathcal L_{down}=\frac1{|\mathcal B_C|}\sum_{i\in\mathcal B_C}(J_\theta\mu_i)^T D_i(J_YG_i)^T(J_gf_\psi)^T(p_i-e_{C_i}).$$

$J$ denotes a Jacobian; in binary logistic regression the last factor is
$p_i-C_i$. This connects outcome error through the planned features and
imputed cells to both RITS-GRU directions, their gates, and the context
network. The observation lock gives zero direct gradient through retained
measurements. Shared parameters still learn from observed training cells.

Training uses AdamW (learning rate 0.002, weight decay 0.0001), hidden
dimension 48, minibatches of 128, gradient clipping at norm 5, up to 120
epochs, and patience 25. Twenty percent of subject groups form the internal
validation set. Training masks 25% of available cells. A fixed validation
mask and the lowest composite validation loss select the checkpoint.
The new implementation does not use a true-coefficient score loss,
a missForest conditional-mean anchor, or deterministic outcome correction.

## Stochastic imputation and refitting

Development runs compared independent residual draws, correlated residual
draws, and correlated draws with subject-bootstrap refitting. The third
scheme is carried forward: five networks fit bootstrap samples of
inner-training subjects, with one fixed validation partition and shared
training-based preprocessing. This approximates parameter-estimation
uncertainty; it is not an exact parameter posterior.

For each selected network, six new artificial validation masks provide
residuals. A cell-pair residual second-moment matrix is shrunk halfway to
its diagonal, projected to positive definiteness, and rescaled to preserve
the estimated diagonal. Denote it $\widehat\Sigma_b$ in standardized units.
Imputation $q$ selects a network $b_q$ uniformly and draws

$$\varepsilon_i^{(q)}\sim N(0,\widehat\Sigma_{b_q}),\qquad Y_i^{(q)}=O_i\odot Y_i+(1-O_i)\odot\{\mu_{\theta_{b_q},i}+\varepsilon_i^{(q)}\}.$$

The residual multiplier is one. Residuals are independent between subjects
and correlated across cells within a subject. They are added jointly after
a recurrent mean pass, not fed back through the recurrent states.
Observed measurements remain fixed and padding is excluded. There are
$M=20$ imputations in the new benchmark.

Refit the planned outcome model in each imputed dataset. With estimates
$\widehat\beta^{(q)}$ and covariance matrices $U^{(q)}$, use

$$\bar\beta=M^{-1}\sum_q\widehat\beta^{(q)},\qquad \bar U=M^{-1}\sum_q U^{(q)},$$
$$B=(M-1)^{-1}\sum_q(\widehat\beta^{(q)}-\bar\beta)(\widehat\beta^{(q)}-\bar\beta)^T,\qquad T=\bar U+(1+M^{-1})B.$$

Scalar intervals use Barnard-Rubin degrees of freedom [2]. One common
reference SD for each predictor transforms all fitted coefficients and
covariances. Simulation reference SDs come from an independent 20,000-subject
synthetic reference and never enter training. Every requested downstream
fit must be valid. Rank deficiency, separation, invalid covariance, and
nonconvergence are recorded, not pooled or silently discarded.

## Recurrent clinical-image extension

The alternative-ultrasound generator produces synthetic 24 by 24 grayscale
images. Seventeen fixed descriptors summarize intensity, texture, spatial
contrast, and edges; `image_features` in `_image_reference.py`
defines them. This benchmark does not use a CNN or actual ultrasound scans.
The same vector $V_i$ enters both RITS directions, current-cell predictions,
and the fusion head.

Clinical fusion features include eight summaries per biomarker: mean, SD,
late mean, late-minus-early mean, mean absolute increment, mean absolute
second difference, total change, and terminal slope. They and observed
clinical covariates are concatenated with $V_i$. A 48-unit ReLU layer and
three-class softmax output form the fusion model. The same architecture
is refitted for each imputation from each method; 20% of labeled training
subjects select its checkpoint.

Seventy-five percent of subjects supply training labels; 25% are held out.
Default missForest has no inductive transform. This benchmark therefore
allows every imputer to use observed unlabeled test covariates:
it is **transductive, held-out-label prediction**, not external validation.
True test labels never enter imputers, scalers, or checkpoint selection.
They are missing in the MICE/missForest input frame and excluded from
BRITS-MI supervision. All three methods receive the same image descriptors.

Average the $M$ probability vectors first, then calculate macro
one-versus-rest AUROC, macro average precision (reported as AUPRC), and
log loss. Rubin's coefficient-pooling rules are not applied to AUROC/AUPRC.

## Association preservation: interpretation and conditions

Let $U(\beta;Y,S,C)$ be the full-data outcome-model score. Suppose the
imputation distribution equals the identified conditional distribution
$p(Y_{mis}\mid Y_{obs},S,C,M)$ and $E\{U(\beta_0;Y,S,C)\}=0$. Then

$$E[E\{U(\beta_0;Y^{imp},S,C)\mid Y_{obs},S,C,M\}]=E\{U(\beta_0;Y,S,C)\}=0.$$

The result follows from equality of the conditional distributions and
iterated expectation. It explains why outcome conditioning matters for
association estimation. The fitted recurrent losses are an estimation
strategy, not proof that this equality holds. Residual noise and a finite
bootstrap ensemble require empirical evaluation of bias, coefficient MSE,
model and empirical SE, and interval coverage. They do not guarantee 95%
coverage or identify an unrestricted nonignorable missingness mechanism.
Optimization, conditional-model adequacy, and interval calibration are
separate properties; no generic SGD theorem is claimed for finite-epoch
AdamW with clipping and checkpoint selection.

## Comparators and real-data masking

MICE uses R package conditional defaults, including predictive mean matching
for continuous biomarkers, five iterations, and 20 imputations [3].
The observed outcome is included for association analysis. Default R
missForest supplies one imputed dataset; its model SE is conditional on that
imputation. Forest random seeds are not presented as proper MI.
Mean imputation, complete-case analysis, and full data are separate baselines.

For the TE application, AST/ALT/platelet masking precedes all training and
preprocessing. Pre-mask estimates and predictor SDs are evaluation references
only. Interval inclusion of a reference estimate is not coverage of a known
population parameter. Classification evaluated on the fitting cohort is
reported as apparent performance. Naturally unavailable cells remain unknown.
The user has authorized the older source for the new run: 3,995 TE participants,
of whom 2,209 have at least two recorded bins for each liver marker and complete
required covariates. The reference-estimate masking analysis uses that subset.
It is not relabeled as the manuscript's 3,977-participant cohort. Each new
repeat masks observed cells before all model fitting and preprocessing.

## Sources

1. Cao W, et al. BRITS: Bidirectional Recurrent Imputation for Time Series.
   NeurIPS 2018. [Paper](https://papers.nips.cc/paper_files/paper/2018/file/734e6bfcd358e25ac1db0a4241b95651-Paper.pdf).
2. [Rubin pooling and Barnard-Rubin degrees of freedom](https://amices.org/mice/reference/pool.scalar.html).
3. [MICE conditional defaults and predictor specification](https://amices.org/mice/reference/mice.html).
