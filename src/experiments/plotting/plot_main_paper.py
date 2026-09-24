"""Compact main-paper figures from canonical saved rows; no training."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.experiments.plotting.plot_paper_figures import style, save, PROCESS, OUTCOME, BASELINE

ROOT = Path(__file__).resolve().parents[3]
COHERENT = '#b42318'
#: register-machine transition state is the pair (x, y) in Z_17 x Z_17, so the
#: population threshold of Cor. 1 uses K = 289.  The reported answer is x
#: alone, so chance accuracy uses 17.  These are different quantities.
K_REGISTER_ANSWER = 17
K_REGISTER_STATE = 17 * 17
K_REGISTER = K_REGISTER_ANSWER  # back-compat: chance lines only


def plot_reliability_structure():
    style()
    data = pd.read_csv(ROOT / 'results/paper/boolean_reliability_canonical.csv')
    stats = data[data.condition != 'outcome'].groupby('rho').answer_accuracy.agg(['mean', 'std'])
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 2.45), layout='constrained')
    ax = axes[0]
    ax.fill_between(stats.index, stats['mean'] - stats['std'], stats['mean'] + stats['std'],
                    color=OUTCOME, alpha=.20, linewidth=0)
    ax.plot(stats.index, stats['mean'], 'o-', color=OUTCOME, ms=4, label='Process targets')
    ax.axhline(data[data.condition == 'outcome'].answer_accuracy.mean(), color=BASELINE,
               ls='-.', lw=1.4, label='Outcome-only')
    ax.axhline(1/16, color=BASELINE, ls=':', lw=1.2, label='Chance')
    ax.set(title='(a) Boolean circuit, depth 8', xlabel=r'Trace reliability $\rho$',
           ylabel='Answer accuracy', xlim=(-.02, 1.02), ylim=(-.03, 1.02))
    ax.legend(loc='upper left', fontsize=8)
    ax = axes[1]
    for law, color, marker, label in [('symmetric', PROCESS, 'o', 'Symmetric'),
                                      ('coherent', '#b42318', 's', 'Coherent wrong rule')]:
        data = pd.read_csv(ROOT / f'results/paper/matched_corruption/register_machine_16_{law}.csv')
        data = data[data.step == 8000]
        stats = data.groupby('rho').answer_accuracy.agg(['mean', 'std'])
        ax.errorbar(stats.index, stats['mean'], yerr=stats['std'], color=color,
                    marker=marker, ms=4, capsize=3, label=label)
    ax.axhline(1/17, color=BASELINE, ls=':', lw=1.2, label='Chance')
    ax.set(title='(b) Register machine, 16 steps', xlabel=r'Trace reliability $\rho$',
           ylabel='Answer accuracy', xlim=(.25, .75), ylim=(-.03, 1.02), xticks=[.3, .5, .7])
    ax.legend(loc='lower right', fontsize=8)
    save(fig, 'reliability_structure')


def plot_register_corruption_thresholds():
    """Appendix companion to Fig. 1b / Table 6: same trained-Transformer
    register-machine rows, widened so both population thresholds
    (mu_pop=1/K for symmetric, mu_pop=1/2 for one coherent wrong rule) are
    visible alongside the three tested reliabilities.

    K here is the size of the transition state space, (x, y) in Z_17 x Z_17, so
    K = 289 and the symmetric threshold is 1/289.  That is NOT the chance
    accuracy: the reported answer is x alone, so chance is 1/17.  The two were
    the same number in an earlier version of this figure, which was wrong."""
    style()
    fig, ax = plt.subplots(figsize=(5.0, 3.0), layout='constrained')
    for law, color, marker, label in [('symmetric', PROCESS, 'o', r'Symmetric: $\mu_{\mathrm{pop}}=1/K$'),
                                      ('coherent', COHERENT, '^', r'One coherent wrong rule: $\mu_{\mathrm{pop}}=1/2$')]:
        data = pd.read_csv(ROOT / f'results/paper/matched_corruption/register_machine_16_{law}.csv')
        data = data[data.step == 8000]
        stats = data.groupby('rho').answer_accuracy.agg(['mean', 'std'])
        ax.errorbar(stats.index, stats['mean'], yerr=stats['std'], color=color,
                    marker=marker, ms=5, lw=1.8, capsize=3, label=label)
    ax.axvline(1 / K_REGISTER_STATE, color=PROCESS, ls='--', lw=1.0)
    ax.axvline(0.5, color=COHERENT, ls='--', lw=1.0)
    ax.axhline(1 / K_REGISTER_ANSWER, color=BASELINE, ls=':', lw=1.0, label='Chance')
    ax.set(title='Register machine: tested reliabilities against both thresholds',
           xlabel=r'Trace reliability $\rho$', ylabel='Answer accuracy',
           xlim=(-.02, 0.82), ylim=(-.03, 1.02), xticks=[1 / K_REGISTER_STATE, .3, .5, .7])
    ax.set_xticklabels(['1/289', '.3', '.5', '.7'])
    ax.legend(loc='upper left', fontsize=8)
    save(fig, 'register_corruption_thresholds')


def plot_gradient_alignment():
    style()
    data = pd.read_csv(ROOT / 'results/gradient_alignment_archived_3seed/gradient_alignment.csv')
    fig, ax = plt.subplots(figsize=(5.0, 1.95), layout='constrained')
    for column, color, label in [('proc_cosine', PROCESS, 'Process'),
                                 ('out_cosine', OUTCOME, 'Outcome (counterfactual)')]:
        stats = data.groupby('step')[column].agg(['mean', 'std'])
        ax.errorbar(stats.index/1000, stats['mean'], yerr=stats['std'],
                    color=color, marker='o', ms=4, capsize=3, label=label)
    ax.axhline(0, color=BASELINE, ls=':', lw=1)
    ax.set(xlabel='Training steps (thousands)', ylabel='Cosine with local gradient',
           xticks=[0, 2, 4, 8], ylim=(-.22, 1.12))
    ax.legend(loc='center right', fontsize=8)
    save(fig, 'gradient_alignment_compact')


def plot_zero_init_softmax_table():
    """Process escapes Z=0; outcome does not (Thm. thm:complete-mixing, as an
    optimization-landscape statement). P_g(Z) = softmax_row(Z_g) --
    app:shared-kernel-embedding's own construction at its proven C -> infinity
    limit -- trained by full-batch AdamW from a small random perturbation of
    Z=0 (src/experiments/circuit_match/zero_init_softmax_table.py). Same
    circuits, same Z, two losses: process (per-step teacher-forced CE) vs.
    outcome (CE on the D-fold composed belief, no teacher forcing). Three
    seeds.
    """
    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.0), layout='constrained')
    data = pd.read_csv(ROOT / 'results/zero_init_softmax_table_random_init/metrics.csv')
    for mode, color, marker, label in [
        ('process', PROCESS, 'o', 'Process'), ('outcome', OUTCOME, 's', 'Outcome'),
    ]:
        stats = data[data['mode'] == mode].groupby('step')['test_answer_accuracy'].agg(['mean', 'std'])
        ax.fill_between(stats.index, stats['mean'] - stats['std'], stats['mean'] + stats['std'],
                         color=color, alpha=.18, linewidth=0)
        ax.plot(stats.index, stats['mean'], '-', color=color, marker=marker, ms=3,
                markevery=max(1, len(stats) // 15), label=label)
    ax.axhline(1 / 16, color=BASELINE, ls=':', lw=1.2, label='Chance (1/16)')
    ax.set(xlabel='Training steps (full-batch AdamW)', ylabel='Test answer accuracy',
           ylim=(-0.03, 1.03))
    ax.legend(loc='center right', fontsize=8)
    save(fig, 'zero_init_softmax_table')


def plot_circuit_match_gradient_polarity():
    """What a wrong trace's gradient does to a trained network: geometry vs competence.

    A random-init copy of the exact-construction architecture is trained on
    clean process supervision, then probed (never trained) on a matched wrong
    process trace -- same prompt and gold answer, corrupted intermediate states
    (src/experiments/circuit_match/handcoded_circuit_match.py).

    (a) beta_r^- = <-g_r^-, (theta*_r - theta_r)> / ||theta*_r - theta_r||, the
        projection onto the direction that actually shortens the distance to the
        constructed solution.  (An earlier version projected onto theta*_r
        itself, which does not measure movement toward it.)
    (b) the functional test: change in CLEAN state-token loss after one step
        restricted to the MLP group, built from the clean trace (control) or
        from the wrong trace.  Negative means the step improves clean
        next-state prediction.  Plotted up to step 600, after which the clean
        loss is numerically solved and the probe is uninformative.
    """
    style()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.6), layout='constrained')
    frames = {law: pd.read_csv(ROOT / f'results/handcoded_circuit_match_{law}_n20k/grad_match.csv')
              for law in ('symmetric', 'coherent')}

    ax = axes[0]
    for group, color, marker in [('MLP', PROCESS, 'o'), ('OV', OUTCOME, 's'), ('QK', COHERENT, '^')]:
        data = pd.concat(frames.values())
        stats = data.groupby('step')[f'beta_{group}_minus'].agg(['mean', 'std'])
        ax.fill_between(stats.index, stats['mean'] - stats['std'], stats['mean'] + stats['std'],
                        color=color, alpha=.16, linewidth=0)
        ax.plot(stats.index, stats['mean'], '-', color=color, marker=marker, ms=3,
                markevery=5, label=group)
    ax.axhline(0, color=BASELINE, ls=':', lw=1.2)
    ax.set(xlabel='Training steps',
           ylabel=r'$\beta^-=\langle -g^-,\widehat{\theta^\star-\theta}\rangle$',
           title='(a) Movement toward the constructed circuit')
    ax.legend(loc='upper left', fontsize=8, ncol=3, title='Parameter group', title_fontsize=8)

    ax = axes[1]
    data = pd.concat(frames.values())
    data = data[data.step <= 600]
    for tag, color, marker, label in [('plus', PROCESS, 'o', 'Clean trace (control)'),
                                      ('minus', COHERENT, '^', 'Wrong trace')]:
        stats = data.groupby('step')[f'dLclean_MLP_{tag}_eta001'].agg(['mean', 'std'])
        ax.fill_between(stats.index, stats['mean'] - stats['std'], stats['mean'] + stats['std'],
                        color=color, alpha=.16, linewidth=0)
        ax.plot(stats.index, stats['mean'], '-', color=color, marker=marker, ms=3,
                markevery=3, label=label)
    ax.axhline(0, color=BASELINE, ls=':', lw=1.2)
    ax.set(xlabel='Training steps', ylabel=r'$\Delta$ clean state loss',
           title='(b) Effect of one MLP step on clean competence')
    ax.legend(loc='lower left', fontsize=8)
    save(fig, 'circuit_match_gradient_polarity')


def plot_depth_control():
    """Aggregate the archived matched outcome/clean-process runs at each depth."""
    style()
    frames = []
    for depth in (2, 4, 6, 8):
        data = pd.read_csv(
            ROOT / f'results/paper/depth_reliability/boolean_circuit_{depth}_outcome.csv'
        )
        data = data[(data.step == 8000) & data.condition.isin(['outcome', 'rho=1.00'])].copy()
        for condition in ('outcome', 'rho=1.00'):
            seeds = sorted(data.loc[data.condition == condition, 'seed'].tolist())
            if seeds != [2001, 2002, 2003]:
                raise ValueError(f'Depth {depth}, {condition}: expected one row per archived seed')
        data['depth'] = depth
        frames.append(data)
    data = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(5.8, 2.30), layout='constrained')
    for condition, color, marker, label in [
        ('rho=1.00', PROCESS, 'o', r'Process ($\rho=1$)'),
        ('outcome', OUTCOME, 's', 'Outcome'),
    ]:
        stats = data[data.condition == condition].groupby('depth').answer_accuracy.agg(['mean', 'std'])
        ax.errorbar(stats.index, 100 * stats['mean'], yerr=100 * stats['std'],
                    color=color, marker=marker, ms=5, capsize=3, label=label)
    ax.axhline(100 / 16, color=BASELINE, ls=':', lw=1.2, label='Chance (6.25%)')
    ax.set(xlabel=r'Composition depth $D$', ylabel='Answer accuracy (%)',
           xticks=[2, 4, 6, 8], yticks=[0, 25, 50, 75, 100],
           xlim=(1.7, 8.3), ylim=(0, 105))
    ax.legend(loc='center right', fontsize=9)
    save(fig, 'depth_control')


def _credit_chain(K, eps, depth, seed=0):
    """Near-uniform shared transition model of Thm. 1: P_g = (1-eps)U + eps T_g.

    Returns the exact forward signals q_j, the backward signals b_j, the realized
    state path, and p_y, so the hero figure draws measured quantities.
    """
    rng = np.random.default_rng(seed)
    uniform = np.full((K, K), 1.0 / K)
    perms = [rng.permutation(K) for _ in range(depth)]
    slices = [(1 - eps) * uniform + eps * np.eye(K)[p] for p in perms]
    states = [0]
    for p in perms:
        states.append(int(p[states[-1]]))
    forward = [np.eye(K)[states[0]]]
    for slice_ in slices:
        forward.append(slice_.T @ forward[-1])
    backward = [None] * (depth + 1)
    backward[depth] = np.eye(K)[states[depth]]
    for j in range(depth - 1, -1, -1):
        backward[j] = slices[j] @ backward[j + 1]
    terminal = slices[0]
    for slice_ in slices[1:]:
        terminal = terminal @ slice_
    return slices, states, forward, backward, terminal[states[0], states[depth]]


def _credit_norms(K, eps, depth, occ=None, seed=0):
    """Exact projected outcome and process rule-credit norms at occurrence `occ`."""
    occ = max(1, (depth + 1) // 2) if occ is None else occ
    slices, states, forward, backward, p_y = _credit_chain(K, eps, depth, seed)
    center = np.eye(K) - np.full((K, K), 1.0 / K)
    outcome = np.linalg.norm(
        np.outer(center @ forward[occ - 1], center @ backward[occ])) / p_y
    process = (1 - 1 / K) / slices[occ - 1][states[occ - 1], states[occ]]
    return outcome, process


def plot_credit_geometry():
    """Exact occurrence credit, drawn at the paper's 5.5-inch text width.

    The schematic elides intermediate transitions, but all four outcome
    profiles share a probability scale and come from the same depth-5 chain.
    Only the two endpoints of the selected process target are emphasized.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # Keep this figure's typography and white plotting field local: the other
    # empirical figures continue to use the common paper style.
    with plt.rc_context():
        style(**{
            'font.size': 8,
            'mathtext.fontset': 'dejavusans',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#9da4ac',
            'axes.linewidth': .6,
            'axes.grid': False,
            'axes.labelsize': 8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'savefig.pad_inches': .035,
        })
        K, EPS, DEPTH, OCC = 16, .45, 5, 3
        _, states, forward, backward, _ = _credit_chain(K, EPS, DEPTH)
        out_norm, proc_norm = _credit_norms(K, EPS, DEPTH, OCC)
        ink, muted, hairline = '#222831', '#616a75', '#cbd0d6'
        fig = plt.figure(figsize=(5.5, 3.05), facecolor='white')
        ax_out = fig.add_axes([.012, .555, .575, .335])
        ax_proc = fig.add_axes([.012, .065, .575, .335])
        ax_dec = fig.add_axes([.73, .19, .258, .665])

        for ax in (ax_out, ax_proc):
            ax.set(xlim=(0, 6.5), ylim=(0, 2.1))
            ax.axis('off')

        def heading(x, y, letter, title, color=ink):
            fig.text(x, y, letter, fontsize=9, weight='bold', color=ink, va='top')
            fig.text(x + .043, y, title, fontsize=9, weight='bold', color=color, va='top')

        heading(.012, .982, '(a)', 'Outcome supervision', OUTCOME)
        heading(.012, .492, '(b)', 'Process supervision', PROCESS)
        heading(.655, .982, '(c)', 'Exact loss along the true-rule path')
        fig.text(.055, .922, 'Only the prompt and answer are observed',
                 fontsize=7.2, color=muted, va='top')
        fig.text(.055, .432, 'A local target names both endpoints',
                 fontsize=7.2, color=muted, va='top')
        fig.text(.698, .922, r'$K=16$',
                 fontsize=7.3, color=muted, va='top')

        # Four profiles: two observed boundary states and the two distributions
        # at the differentiated occurrence. Arrows stand for the omitted gates.
        width, height, base = .88, .55, .94
        prompt_x, source_x, rule_x, target_x, answer_x = .49, 2.04, 3.25, 4.46, 6.01

        def profile(ax, vec, x, color):
            edges = np.linspace(x - width / 2, x + width / 2, K + 1)
            ax.stairs(base + height * vec, edges, baseline=base, fill=True,
                      facecolor=color, alpha=.18, linewidth=0, zorder=2)
            ax.stairs(base + height * vec, edges, baseline=None,
                      color=color, linewidth=.9, zorder=3)
            ax.plot([edges[0], edges[-1]], [base, base], color=hairline, lw=.6)
            # The dotted uniform level makes attenuation interpretable without
            # normalizing small profiles to the same apparent peak height.
            ax.plot([edges[0], edges[-1]], [base + height / K] * 2,
                    color=muted, lw=.5, ls=(0, (1.5, 2)), zorder=4)

        def arrow(ax, x0, x1, y, color, label=None):
            ax.add_patch(FancyArrowPatch(
                (x0, y), (x1, y), arrowstyle='-|>', mutation_scale=7,
                linewidth=.9, color=color, shrinkA=0, shrinkB=0,
            ))
            if label:
                ax.text((x0 + x1) / 2, y + .11, label, ha='center',
                        va='bottom', fontsize=7, color=color)

        def rule(ax):
            ax.add_patch(FancyBboxPatch(
                (rule_x - .38, base + .015), .76, .56,
                boxstyle='round,pad=0.025,rounding_size=0.045',
                facecolor='#f5f6f7', edgecolor=ink, linewidth=.85, zorder=5,
            ))
            ax.text(rule_x, base + .295, r'$\mathbf{P}_{g_t}$',
                    ha='center', va='center', fontsize=9, color=ink, zorder=6)
            ax.text(rule_x, base - .12, r'$t=3$', ha='center',
                    va='top', fontsize=7, color=muted)

        for vec, x in [(forward[0], prompt_x), (forward[OCC - 1], source_x),
                       (backward[OCC], target_x), (backward[DEPTH], answer_x)]:
            profile(ax_out, vec, x, OUTCOME)
        arrow(ax_out, prompt_x, source_x, 1.64, OUTCOME, r'$t-1$ steps')
        arrow(ax_out, answer_x, target_x, 1.64, OUTCOME, r'$D-t$ steps')
        rule(ax_out)
        for x, label in [(prompt_x, r'$s_0$'), (source_x, r'$q_{t-1}$'),
                         (target_x, r'$b_t$'), (answer_x, r'$y$')]:
            ax_out.text(x, base - .12, label, ha='center', va='top', fontsize=8, color=ink)
        ax_out.text(3.25, .015,
                    r'$\|\bar q_{t-1}\bar b_t^{\mathsf{T}}\|_F\,/\,p_y = %.2f$' % out_norm,
                    ha='center', va='bottom', fontsize=8.3, color=OUTCOME)

        for j, x in [(OCC - 1, source_x), (OCC, target_x)]:
            profile(ax_proc, np.eye(K)[states[j]], x, PROCESS)
        rule(ax_proc)
        arrow(ax_proc, source_x + width / 2 + .05, rule_x - .43, base + .29, PROCESS)
        arrow(ax_proc, rule_x + .43, target_x - width / 2 - .05, base + .29, PROCESS)
        for x, name, state in [(source_x, 'source', r'$s_{t-1}$'),
                                (target_x, 'successor', r'$s_t$')]:
            ax_proc.text(x, 1.78, name, ha='center', va='bottom', fontsize=7.3, color=PROCESS)
            ax_proc.text(x, base - .12, state, ha='center', va='top', fontsize=8, color=ink)
        ax_proc.text(3.25, .015,
                     r'$\frac{1-1/K}{(P_{g_t})_{s_{t-1},s_t}} = %.2f$' % proc_norm,
                     ha='center', va='bottom', fontsize=8.3, color=PROCESS)

        # Exact loss along the true-rule path P_g = U + gamma*(T_g - U), not a
        # fitted or bounded quantity: L_proc(gamma) = log K - log(1+(K-1)gamma)
        # is depth-independent and moves at first order; L_out(gamma) = log K
        # - log(1+(K-1)gamma^D) is flat to order D-1 near gamma=0 and steepens
        # with D. This replaces the earlier geometric-decay-vs-depth panel,
        # which visually implied the near-mixing norm bound was the headline
        # result; the exact order-D flatness (Thm. thm:high-order-flatness) is.
        gammas = np.linspace(0, 1, 400)
        L_proc_gamma = np.log(K) - np.log(1 + (K - 1) * gammas)
        ax_dec.set(xlim=(0, 1), ylim=(-0.08, np.log(K) + 0.08), xticks=[0, .5, 1],
                   xlabel=r'True-rule progress $\gamma$')
        ax_dec.set_ylabel(r'Loss $L(\gamma)$', labelpad=3)
        ax_dec.spines[['top', 'right']].set_visible(False)
        ax_dec.tick_params(which='major', length=2.5, width=.6, pad=2, color=muted)
        ax_dec.grid(axis='y', color='#e4e7eb', linewidth=.55)
        ax_dec.plot(gammas, L_proc_gamma, color=PROCESS, lw=1.5, zorder=3)
        for d, alpha, label_x in zip((2, 4, 8), (1.0, 0.65, 0.4), (0.62, 0.78, 0.90)):
            L_out_gamma = np.log(K) - np.log(1 + (K - 1) * gammas ** d)
            ax_dec.plot(gammas, L_out_gamma, color=OUTCOME, lw=1.3, alpha=alpha, zorder=3)
            label_y = np.log(K) - np.log(1 + (K - 1) * label_x ** d)
            ax_dec.text(label_x, label_y + .12, f'$D{{=}}{d}$', fontsize=6.3, color=OUTCOME,
                        alpha=alpha, ha='center', va='bottom')
        ax_dec.text(.32, np.log(K) - np.log(1 + (K - 1) * .32) + .18, 'Process',
                    fontsize=7.5, color=PROCESS, ha='center')
        ax_dec.text(.68, np.log(K) - .35, 'Outcome', fontsize=7.5, color=OUTCOME, ha='center')
        save(fig, 'credit_geometry')


if __name__ == '__main__':
    plot_reliability_structure()
    plot_register_corruption_thresholds()
    plot_gradient_alignment()
    plot_zero_init_softmax_table()
    plot_circuit_match_gradient_polarity()
    plot_credit_geometry()
    plot_depth_control()
