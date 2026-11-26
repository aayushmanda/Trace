"""
Authentic 3Blue1Brown-style 1080p60 Manim animations (with native Computer Modern LaTeX) for:
"Silent Routing: How Step Supervision Trains Transformers Where Outcome Supervision Stalls"

Scenes:
  1. Scene1_OutcomePlateau     — Parity chain, shared expressivity, and the ε²-canceling Ω(λ^-n) energy barrier
  2. Scene2_SilentRouting      — Curved causal attention arcs + prior calibration silencing n-1 distractors
  3. Scene3_StatelessBlockRMS  — Hyperbolic (q, k) saddle trajectories + 0-state Block-RMS normalization
"""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
import numpy as np

import manim.utils.tex_file_writing as tfw
from manim import (
    Scene,
    VGroup,
    VMobject,
    Tex,
    SingleStringMathTex,
    TexTemplate,
    Circle,
    Dot,
    Line,
    DashedLine,
    Arrow,
    CubicBezier,
    Rectangle,
    RoundedRectangle,
    Brace,
    Axes,
    NumberLine,
    ParametricFunction,
    ValueTracker,
    always_redraw,
    FadeIn,
    FadeOut,
    Create,
    Write,
    GrowArrow,
    GrowFromCenter,
    Transform,
    ReplacementTransform,
    Circumscribe,
    ShowPassingFlash,
    LaggedStart,
    AnimationGroup,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UR,
    DL,
    DR,
    ORIGIN,
    WHITE,
    BLACK,
    GREY_B,
    GREY_C,
    GREY_D,
    config,
    rate_functions,
)

# ------------------------------------------------------------------------------
# Neutral Google-Style Sans-Serif (Roboto + newtxsf) -> Vector SVG via pdftocairo
# ------------------------------------------------------------------------------
_tex_template = TexTemplate(tex_compiler="pdflatex", output_format=".pdf")
_tex_template.add_to_preamble(
    r"\usepackage[T1]{fontenc}"
    r"\usepackage{amsmath,amssymb,mathtools}"
    r"\usepackage{xcolor}"
    r"\usepackage[sfdefault]{roboto}"
    r"\usepackage[cmintegrals,nosymbolsc]{newtxsf}"
    r"\renewcommand{\familydefault}{\sfdefault}"
)
config.tex_template = _tex_template


_STROKE_RULE_RE = re.compile(
    r'<path\s+fill="none"\s+stroke-width="(?P<sw>[0-9.eE+-]+)"[^>]*?'
    r'stroke="(?P<col>[^"]+)"[^>]*?'
    r'd="M\s+(?P<x1>[0-9.eE+-]+)\s+(?P<y1>[0-9.eE+-]+)\s+L\s+(?P<x2>[0-9.eE+-]+)\s+(?P<y2>[0-9.eE+-]+)\s*"'
    r'(?P<tf>\s+transform="[^"]+")?\s*/>'
)


def _stroke_rule_to_filled_rect(match: re.Match) -> str:
    sw = float(match.group("sw"))
    col = match.group("col")
    x1, y1, x2, y2 = (float(match.group(k)) for k in ("x1", "y1", "x2", "y2"))
    tf = match.group("tf") or ""
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return match.group(0)
    hw = max(sw * 0.56, 0.24)
    nx, ny = -dy / length * hw, dx / length * hw
    d_rect = (
        f"M {x1 + nx:.6f} {y1 + ny:.6f} "
        f"L {x2 + nx:.6f} {y2 + ny:.6f} "
        f"L {x2 - nx:.6f} {y2 - ny:.6f} "
        f"L {x1 - nx:.6f} {y1 - ny:.6f} Z"
    )
    return f'<path fill-rule="nonzero" fill="{col}" fill-opacity="1" d="{d_rect}"{tf}/>'


def _pdftocairo_convert_to_svg(dvi_file: Path, extension: str, page: int = 1) -> Path:
    result = dvi_file.with_suffix(".svg")
    if not result.exists():
        pdf_file = dvi_file
        if dvi_file.suffix.lower() != ".pdf":
            pdf_file = dvi_file.with_suffix(".pdf")
            subprocess.run(
                ["dvipdf", str(dvi_file), str(pdf_file)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        subprocess.run(
            ["pdftocairo", "-svg", "-f", str(page), "-l", str(page), str(pdf_file), str(result)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        svg_txt = result.read_text(encoding="utf-8")
        svg_fixed = _STROKE_RULE_RE.sub(_stroke_rule_to_filled_rect, svg_txt)
        if svg_fixed != svg_txt:
            result.write_text(svg_fixed, encoding="utf-8")
    return result


tfw.convert_to_svg = _pdftocairo_convert_to_svg

# ------------------------------------------------------------------------------
# Authentic 3Blue1Brown Color Palette on Pure Black (#000000)
# ------------------------------------------------------------------------------
BG_BLACK = "#000000"
B_BLUE = "#58C4DD"      # 3b1b BLUE_C
B_TEAL = "#5CD0B3"      # 3b1b TEAL_C
B_GREEN = "#83C167"     # 3b1b GREEN_C
B_YELLOW = "#FFFF00"    # 3b1b YELLOW
B_GOLD = "#F0AC5F"      # 3b1b GOLD_C
B_RED = "#FC6255"       # 3b1b RED_C
B_MAROON = "#C55F73"    # 3b1b MAROON_B
B_PURPLE = "#9A72AC"    # 3b1b PURPLE_B
B_GREY = "#888888"
B_DARK = "#11161F"


def mtex(expr: str, font_size: int = 36, color: str | None = WHITE) -> SingleStringMathTex:
    obj = SingleStringMathTex(expr, font_size=font_size, tex_template=_tex_template)
    if color is not None:
        obj.set_color(color)
    return obj


def ttex(expr: str, font_size: int = 34, color: str | None = WHITE) -> SingleStringMathTex:
    obj = SingleStringMathTex(rf"\textnormal{{{expr}}}", font_size=font_size, tex_template=_tex_template)
    if color is not None:
        obj.set_color(color)
    if obj.width > 13.2:
        obj.scale_to_fit_width(13.2)
    return obj


def make_3b1b_title(main_tex: str, sub_tex: str | None = None) -> VGroup:
    t = ttex(main_tex, font_size=38, color=WHITE).to_edge(UP, buff=0.28)
    ul = Line(LEFT * 6.3, RIGHT * 6.3, stroke_width=1.4, color=B_GREY).next_to(t, DOWN, buff=0.12)
    if sub_tex is None:
        return VGroup(t, ul)
    sub = ttex(sub_tex, font_size=24, color=GREY_B).next_to(ul, DOWN, buff=0.11)
    return VGroup(t, ul, sub)


def make_tex_token(tex_label: str, border_col: str = B_BLUE, fill_col: str = "#09131C", w: float = 0.92, h: float = 0.62, fsize: int = 28) -> VGroup:
    box = RoundedRectangle(
        corner_radius=0.09, width=w, height=h,
        fill_color=fill_col, fill_opacity=0.92,
        stroke_color=border_col, stroke_width=2.2,
    )
    lbl = mtex(tex_label, font_size=fsize, color=WHITE).move_to(box)
    return VGroup(box, lbl)


# ==============================================================================
# SCENE 1: The Outcome Plateau Paradox & ε²-Canceling Energy Barrier
# ==============================================================================
class Scene1_OutcomePlateau(Scene):
    def construct(self):
        self.camera.background_color = BG_BLACK

        title = make_3b1b_title(
            r"\textbf{Why Outcome Supervision Stalls Exponentially}",
            r"Same two-block Transformer expresses both executors --- the bottleneck is purely dynamical",
        )
        self.play(Write(title[0]), Create(title[1]), FadeIn(title[2], shift=DOWN * 0.1), run_time=1.4)

        # --- Act 1: Live Markov Parity Chain ---
        tok_specs = [
            (r"s_0\!=\!0", B_BLUE, "#091824"),
            (r"w^\star\!=\!\text{\textbf{A}}", B_PURPLE, "#180F24"),
            (r"w_1\!=\!\text{\textbf{B}}", B_GREY, "#101216"),
            (r"w_2\!=\!\text{\textbf{A}}", B_YELLOW, "#211E08"),
            (r"w_3\!=\!\text{\textbf{C}}", B_GREY, "#101216"),
            (r"w_4\!=\!\text{\textbf{A}}", B_YELLOW, "#211E08"),
            (r"w_5\!=\!\text{\textbf{D}}", B_GREY, "#101216"),
            (r"w_6\!=\!\text{\textbf{A}}", B_YELLOW, "#211E08"),
        ]
        chain = VGroup(*[
            make_tex_token(lbl, bc, fc, w=1.34, h=0.62, fsize=25)
            for lbl, bc, fc in tok_specs
        ]).arrange(RIGHT, buff=0.15).move_to(UP * 1.15)

        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.18) for t in chain], lag_ratio=0.07), run_time=1.3)

        # Curved state-hop arcs above the context tokens
        state_vals = [0, 1, 1, 0, 0, 1]
        flips = [False, True, False, True, False, True]
        hop_group = VGroup()
        prev_pt = chain[0].get_top() + UP * 0.04
        for idx, (sv, fl) in enumerate(zip(state_vals, flips)):
            cur_tok = chain[idx + 2]
            cur_pt = cur_tok.get_top() + UP * 0.04
            mid = 0.5 * (prev_pt + cur_pt) + UP * (0.48 if idx == 0 else 0.38)
            arc = CubicBezier(
                prev_pt, prev_pt + UP * 0.34, cur_pt + UP * 0.34, cur_pt,
                color=B_GREEN if fl else B_GREY, stroke_width=2.6 if fl else 1.6,
            )
            node = Circle(
                radius=0.20,
                fill_color="#0D2618" if sv == 1 else "#0B1926",
                fill_opacity=1.0,
                stroke_color=B_GREEN if sv == 1 else B_BLUE,
                stroke_width=2.0,
            ).move_to(cur_pt + UP * 0.28)
            nlbl = mtex(rf"s_{{{idx+1}}}\!=\!{sv}", font_size=18, color=B_GREEN if sv == 1 else B_BLUE).move_to(node)
            # Small badge instead of cramped circle
            pill = RoundedRectangle(
                corner_radius=0.10, width=0.82, height=0.34,
                fill_color="#0D2618" if sv == 1 else "#0B1926", fill_opacity=1.0,
                stroke_color=B_GREEN if sv == 1 else B_BLUE, stroke_width=1.8,
            ).move_to(cur_pt + UP * 0.30)
            nlbl.move_to(pill)
            hop_group.add(VGroup(arc, pill, nlbl))
            prev_pt = cur_pt

        rule_tex = mtex(
            r"\textcolor[HTML]{58C4DD}{s_t = s_{t-1} + \mathbf{1}[w_t = w^\star] \pmod 2}"
            r"\qquad\Longleftrightarrow\qquad"
            r"\textcolor[HTML]{FFFF00}{x_t = x_{t-1}\chi_t},\quad \lambda := |\mathbb{E}[\chi_t]| = |1-2p|",
            font_size=28, color=None,
        ).next_to(chain, DOWN, buff=0.28)

        self.play(LaggedStart(*[FadeIn(h, shift=UP * 0.08) for h in hop_group], lag_ratio=0.14), run_time=1.6)
        self.play(Write(rule_tex), run_time=1.4)
        self.wait(0.6)

        # Move chain compactly up to make open black stage below
        top_ensemble = VGroup(chain, hop_group)
        self.play(
            FadeOut(title[2]),
            top_ensemble.animate.scale(0.84).move_to(UP * 1.88),
            FadeOut(rule_tex, shift=UP * 0.2),
            run_time=0.9,
        )

        # --- Act 2: Shared Expressivity vs. Parameter Ball Energy Barrier ---
        # Left side: Parameter-space ball B_epsilon with trapped gradient flow particle
        ball_center = LEFT * 3.80 + DOWN * 1.12
        ball_circle = Circle(
            radius=1.46, color=B_BLUE, stroke_width=2.4,
            fill_color=B_BLUE, fill_opacity=0.07,
        ).move_to(ball_center)
        theta0_dot = Dot(ball_center, radius=0.065, color=WHITE)
        theta0_lbl = mtex(r"\theta_0", font_size=22, color=WHITE).next_to(theta0_dot, DL, buff=0.06)

        rad_line = Line(ball_center, ball_center + RIGHT * 1.46, color=B_GOLD, stroke_width=2.4)
        rad_lbl = mtex(r"R = r_0\varepsilon", font_size=21, color=B_GOLD).next_to(rad_line, UP, buff=0.06).shift(RIGHT * 0.24)
        ball_tag = ttex(r"\textbf{Outcome Flow Trapped in } $\mathcal{B}_\varepsilon$", font_size=22, color=B_BLUE)
        ball_tag.next_to(ball_circle, UP, buff=0.24).shift(LEFT * 0.20)

        # Distant exact executors clearly outside the ball on the upper-right!
        th_out_dot = Dot(ball_center + RIGHT * 1.88 + UP * 1.12, radius=0.08, color=B_GREEN)
        th_out_lbl = mtex(r"\theta^\star_{\mathrm{out}}\text{ (Prop.\ 1)}", font_size=21, color=B_GREEN).next_to(th_out_dot, RIGHT, buff=0.09)

        # Right side: Step-by-step LaTeX derivation & ε² cancellation!
        kl_eq = mtex(
            r"\mathcal{L}_{\mathrm{out}}(\theta) = \log 2 + \mathbb{E}\bigl[\mathrm{D}_{\mathrm{KL}}(\bar y \,\|\, \hat y_\theta)\bigr] - \textcolor[HTML]{FC6255}{C_{\mathrm{out}}(\theta)}",
            font_size=26, color=None,
        ).move_to(RIGHT * 2.85 + UP * 0.35)

        credit_eq = mtex(
            r"\sup_{\theta \in \mathcal{B}_\varepsilon} |C_{\mathrm{out}}(\theta)| \;\le\; \Delta \;=\; C_2\,\textcolor[HTML]{FC6255}{\varepsilon^2}\,\lambda^n",
            font_size=27, color=None,
        ).next_to(kl_eq, DOWN, buff=0.34)

        barrier_eq_1 = mtex(
            r"T_{\mathrm{exit}} \;\ge\; \frac{R^2}{\Delta} \;=\; \frac{r_0^2\,\textcolor[HTML]{F0AC5F}{\varepsilon^2}}{C_2\,\textcolor[HTML]{FC6255}{\varepsilon^2}\,\lambda^n}",
            font_size=33, color=None,
        ).next_to(credit_eq, DOWN, buff=0.38)

        barrier_eq_2 = mtex(
            r"T_{\mathrm{exit}} \;\ge\; \frac{r_0^2}{C_2}\,\textcolor[HTML]{FFFF00}{\lambda^{-n}} \;=\; \textcolor[HTML]{FFFF00}{\Omega(\lambda^{-n})}",
            font_size=35, color=None,
        ).move_to(barrier_eq_1)

        self.play(
            Create(ball_circle), FadeIn(theta0_dot), Write(theta0_lbl),
            GrowFromCenter(rad_line), Write(rad_lbl), FadeIn(ball_tag),
            FadeIn(th_out_dot, scale=1.4), Write(th_out_lbl),
            run_time=1.5,
        )

        # Animate sluggish trajectory spiraling near theta_0 while writing equations
        t_spiral = ValueTracker(0.0)
        traj_curve = always_redraw(lambda: ParametricFunction(
            lambda s: ball_center + np.array([
                0.70 * (s / 6.0) * math.cos(3.3 * s),
                0.70 * (s / 6.0) * math.sin(3.3 * s),
                0.0,
            ]),
            t_range=[0.0, max(0.02, t_spiral.get_value())],
            color=B_RED, stroke_width=3.0,
        ))
        traj_dot = always_redraw(lambda: Dot(
            ball_center + np.array([
                0.70 * (t_spiral.get_value() / 6.0) * math.cos(3.3 * t_spiral.get_value()),
                0.70 * (t_spiral.get_value() / 6.0) * math.sin(3.3 * t_spiral.get_value()),
                0.0,
            ]),
            radius=0.065, color=B_YELLOW,
        ))
        self.add(traj_curve, traj_dot)

        self.play(
            t_spiral.animate.set_value(2.0),
            Write(kl_eq),
            run_time=1.6,
        )
        self.play(
            t_spiral.animate.set_value(4.0),
            Write(credit_eq),
            # Dim unread context letters to show λ^(n-m) contraction
            *[chain[j][0].animate.set_stroke(B_RED, width=2.4) for j in [2, 4, 5, 6, 7]],
            run_time=1.6,
        )
        self.play(
            t_spiral.animate.set_value(5.5),
            Write(barrier_eq_1),
            run_time=1.5,
        )
        self.wait(0.5)

        # Morph fraction canceling ε² / ε² into Ω(λ^-n)!
        self.play(
            t_spiral.animate.set_value(6.0),
            ReplacementTransform(barrier_eq_1, barrier_eq_2),
            run_time=1.2,
        )
        self.play(Circumscribe(barrier_eq_2, color=B_YELLOW, stroke_width=2.5, time_width=1.4), run_time=1.3)

        foot = ttex(
            r"Shrinking $\varepsilon$ slows gradient speed and shrinks $\mathcal{B}_\varepsilon$ equally --- $\Omega(\lambda^{-n})$ holds on \emph{one fixed rule}.",
            font_size=23, color=GREY_B,
        ).to_edge(DOWN, buff=0.26)
        self.play(FadeIn(foot, shift=UP * 0.12), run_time=0.9)
        self.wait(2.2)


# ==============================================================================
# SCENE 2: Silent Routing — Curved Attention Arcs & Prior Cancellation
# ==============================================================================
class Scene2_SilentRouting(Scene):
    def construct(self):
        self.camera.background_color = BG_BLACK

        title = make_3b1b_title(
            r"\textbf{Silent Routing on the Flat-Loss Plateau (Theorem 2)}",
        )
        self.play(Write(title[0]), Create(title[1]), run_time=1.1)

        # --- Top Stage: Full Causal Prefix with Glowing Curved Attention Arcs ---
        prefix_labels = [
            (r"s_0", B_GREY),
            (r"w^\star", B_PURPLE),
            (r"w_1", B_GOLD),
            (r"w_2", B_GOLD),
            (r"w_3", B_GOLD),
            (r"w_t", B_BLUE),       # index 5 = true target letter at lag δ* = n!
            (r"w_5", B_GOLD),
            (r"w_6", B_GOLD),
            (r"\mathrm{SEP}", B_GREY),
            (r"s_1", B_GREY),
            (r"s_{t-1}", B_YELLOW), # index 10 = query state token!
        ]
        tok_row = VGroup(*[
            make_tex_token(
                lbl, col,
                fill_col="#0A222E" if idx == 5 else ("#242008" if idx == 10 else "#0E1218"),
                w=0.96 if idx in (5, 8, 10) else 0.82,
                h=0.54,
                fsize=23,
            )
            for idx, (lbl, col) in enumerate(prefix_labels)
        ]).arrange(RIGHT, buff=0.12).move_to(UP * 1.55)

        target_brace = Brace(tok_row[5], DOWN, buff=0.06, color=B_BLUE)
        target_tag = mtex(r"\delta^\star = n", font_size=21, color=B_BLUE).next_to(target_brace, DOWN, buff=0.04)
        query_brace = Brace(tok_row[10], DOWN, buff=0.06, color=B_YELLOW)
        query_tag = ttex(r"query", font_size=20, color=B_YELLOW).next_to(query_brace, DOWN, buff=0.04)

        self.play(
            LaggedStart(*[FadeIn(t, shift=UP * 0.12) for t in tok_row], lag_ratio=0.05),
            FadeIn(target_brace), Write(target_tag),
            FadeIn(query_brace), Write(query_tag),
            run_time=1.4,
        )

        # Progress tracker u in [0, 1] driving BOTH the curved arcs AND the plots!
        prog = ValueTracker(0.0)

        def loss_fn(u: float) -> float:
            p1 = 1.0 - 0.19 * (1.0 - math.exp(-u / 0.055))
            gate_drop = 0.79 / (1.0 + math.exp(-(u - 0.83) / 0.033))
            return max(0.02, p1 - gate_drop)

        def prior_fn(u: float) -> float:
            return 1.0 - math.exp(-u / 0.055)

        def route_fn(u: float) -> float:
            if u < 0.18:
                return 0.167 + 0.015 * (u / 0.18)
            surge = 0.74 / (1.0 + math.exp(-(u - 0.45) / 0.062))
            tail = 0.06 / (1.0 + math.exp(-(u - 0.84) / 0.04))
            return min(0.98, 0.182 + surge + tail)

        def acc_fn(u: float) -> float:
            base = 0.50 + 0.02 * min(1.0, u / 0.12)
            jump = 0.48 / (1.0 + math.exp(-(u - 0.82) / 0.028))
            return min(1.0, base + jump)

        def edist_fn(u: float) -> float:
            return math.exp(-u / 0.055)

        # 3Blue1Brown-style curved Bezier attention arcs from query s_{t-1} (idx 10) to letters (idx 2..7)
        def draw_attention_arcs():
            u = prog.get_value()
            pi_star = route_fn(u)
            distractor_w = (1.0 - pi_star) / 5.0
            q_pt = tok_row[10].get_top() + UP * 0.03
            arcs = VGroup()
            letter_indices = [2, 3, 4, 5, 6, 7]
            for j_idx in letter_indices:
                k_pt = tok_row[j_idx].get_top() + UP * 0.03
                span = abs(q_pt[0] - k_pt[0])
                arch_h = 0.32 + 0.095 * span
                is_target = (j_idx == 5)
                wt = pi_star if is_target else distractor_w
                col = B_BLUE if is_target else B_GOLD
                sw = (1.5 + 7.5 * wt) if is_target else (0.6 + 6.5 * wt)
                op = min(1.0, 0.25 + 0.85 * wt) if is_target else max(0.08, min(0.85, 3.2 * wt))
                arc = CubicBezier(
                    q_pt,
                    q_pt + UP * arch_h + LEFT * (0.18 * span),
                    k_pt + UP * arch_h + RIGHT * (0.18 * span),
                    k_pt,
                    color=col,
                    stroke_width=sw,
                    stroke_opacity=op,
                )
                arcs.add(arc)
            return arcs

        attn_arcs = always_redraw(draw_attention_arcs)
        self.play(FadeIn(attn_arcs), run_time=0.9)

        # --- Bottom Left: Clean LaTeX Axes (No clunky dashboard boxes!) ---
        ax = Axes(
            x_range=[0, 1.0, 0.25],
            y_range=[0, 1.08, 0.25],
            x_length=5.35,
            y_length=2.62,
            axis_config={"color": B_GREY, "stroke_width": 1.8, "include_tip": True, "tip_width": 0.16, "tip_height": 0.16},
        ).move_to(LEFT * 3.25 + DOWN * 1.40)

        ax_x_lbl = mtex(r"\text{Training time }\tau", font_size=22, color=GREY_B).next_to(ax.x_axis, DOWN, buff=0.12)
        hp_line = DashedLine(ax.c2p(0, 0.81), ax.c2p(0.98, 0.81), color=B_GOLD, stroke_width=1.5, dash_length=0.09)
        hp_tex = mtex(r"H(p)\text{ plateau}", font_size=20, color=B_GOLD).next_to(ax.c2p(0.45, 0.81), DOWN, buff=0.08)

        curve_key = VGroup(
            mtex(r"\textcolor[HTML]{FC6255}{\mathcal{L}_{\mathrm{step}}}", font_size=22, color=None),
            mtex(r"\textcolor[HTML]{9A72AC}{c / c^\star}", font_size=22, color=None),
            mtex(r"\textcolor[HTML]{58C4DD}{\pi_t^\star\text{ (lag }n\text{)}}", font_size=22, color=None),
            mtex(r"\textcolor[HTML]{83C167}{\text{Rollout}}", font_size=22, color=None),
        ).arrange(RIGHT, buff=0.32).next_to(ax, UP, buff=0.14)

        # --- Bottom Right: Computer Modern Margin Decomposition & Live Cancellation ---
        margin_eq = mtex(
            r"M_t = \chi_t\Bigl(\underbrace{\textcolor[HTML]{9A72AC}{c}}_{\text{prior}} + "
            r"\underbrace{\textcolor[HTML]{58C4DD}{g D_t}}_{\text{gate}}\cdot"
            r"\underbrace{\textcolor[HTML]{83C167}{\rho_t}}_{\text{routed}}\Bigr)",
            font_size=29, color=None,
        ).move_to(RIGHT * 3.15 + DOWN * 0.42)

        edist_eq = mtex(
            r"E_{\mathrm{dist}}(c) = (1 - 2p)\bigl(\sigma(-c) - p\bigr)",
            font_size=26, color=WHITE,
        ).move_to(RIGHT * 3.15 + DOWN * 1.42)

        cstar_eq = mtex(
            r"c^\star = \log\frac{1-p}{p} \;\;\Longrightarrow\;\; \textcolor[HTML]{83C167}{E_{\mathrm{dist}}(c^\star) = 0}",
            font_size=26, color=None,
        ).move_to(RIGHT * 3.15 + DOWN * 2.12)

        # Precompile static LaTeX objects outside always_redraw so 60fps loop runs at memory speed
        d_lbl_static = mtex(r"E_{\mathrm{dist}}:", font_size=22, color=B_RED).move_to(np.array([1.27, -2.90, 0.0]))
        t_lbl_static = mtex(r"E_{\mathrm{true}}:", font_size=22, color=B_GREEN).move_to(np.array([4.50, -2.90, 0.0]))
        b_phase1 = ttex(r"\textbf{Phase 1:} Prior calibrates ($c \to c^\star$), cancelling all $n-1$ distractors", font_size=22, color=B_PURPLE).to_edge(DOWN, buff=0.18)
        b_phase2 = ttex(r"\textbf{Phase 2 (Silent Routing):} $\mathcal{L}_{\mathrm{step}} \approx H(p)$ sits flat while lag-$n$ attention $\pi_t^\star \to 1$", font_size=22, color=B_BLUE).to_edge(DOWN, buff=0.18)
        b_phase3 = ttex(r"\textbf{Phase 3:} Query--key gate opens ($g D_t \uparrow$), driving $\mathcal{L}_{\mathrm{step}} \to 0$ and Rollout $\to 100\%$", font_size=22, color=B_GREEN).to_edge(DOWN, buff=0.18)

        def draw_live_curves_and_vectors():
            u = prog.get_value()
            grp = VGroup()

            u_max = max(0.015, u)
            xs = np.linspace(0, u_max, max(8, int(140 * u_max)))
            for fn, col in [(loss_fn, B_RED), (prior_fn, B_PURPLE), (route_fn, B_BLUE), (acc_fn, B_GREEN)]:
                pts = [ax.c2p(float(xv), float(fn(float(xv)))) for xv in xs]
                c = VMobject(color=col, stroke_width=3.4)
                c.set_points_smoothly(pts)
                grp.add(c, Dot(pts[-1], radius=0.055, color=col))

            ed = edist_fn(u)
            vec_y = -2.90
            d_len = max(0.05, 1.95 * ed)
            d_vec = Arrow(
                np.array([1.90, vec_y, 0.0]), np.array([1.90 + d_len, vec_y, 0.0]),
                buff=0, color=B_RED if ed > 0.12 else B_GREY, stroke_width=4.2,
                max_tip_length_to_length_ratio=0.35,
            )
            t_vec = Arrow(
                np.array([5.13, vec_y, 0.0]), np.array([6.45, vec_y, 0.0]),
                buff=0, color=B_GREEN, stroke_width=4.5,
                max_tip_length_to_length_ratio=0.22,
            )

            banner = b_phase1.copy() if u < 0.25 else (b_phase2.copy() if u < 0.72 else b_phase3.copy())
            grp.add(d_lbl_static.copy(), d_vec, t_lbl_static.copy(), t_vec, banner)
            return grp

        live_dyn = always_redraw(draw_live_curves_and_vectors)

        self.play(
            Create(ax), Write(ax_x_lbl), Create(hp_line), FadeIn(hp_tex), FadeIn(curve_key),
            Write(margin_eq), Write(edist_eq), FadeIn(live_dyn),
            run_time=1.6,
        )

        # Phase 1: Prior calibration c -> c*
        self.play(prog.animate.set_value(0.25), run_time=3.2, rate_func=rate_functions.ease_in_out_sine)
        self.play(Write(cstar_eq), run_time=1.1)
        self.wait(0.5)

        # Phase 2: Silent routing on the flat H(p) plateau!
        self.play(prog.animate.set_value(0.70), run_time=5.4, rate_func=rate_functions.linear)
        self.play(Circumscribe(tok_row[5][0], color=B_YELLOW, stroke_width=3.0), run_time=1.0)

        # Phase 3: Gate escape & exact rollout!
        self.play(prog.animate.set_value(1.00), run_time=3.2, rate_func=rate_functions.ease_in_out_sine)
        self.wait(2.2)


# ==============================================================================
# SCENE 3: Hyperbolic Query-Key Saddle & Stateless Block-RMS Escape
# ==============================================================================
class Scene3_StatelessBlockRMS(Scene):
    def construct(self):
        self.camera.background_color = BG_BLACK

        title = make_3b1b_title(
            r"\textbf{Breaking the $\Theta(\varepsilon^{-5})$ Query--Key Saddle with Zero State}",
        )
        self.play(Write(title[0]), Create(title[1]), run_time=1.1)

        # --- Left Side: Authentic 3Blue1Brown Hyperbolic Saddle Phase Portrait in (q, k) ---
        q_ax = Axes(
            x_range=[-2.2, 2.2, 1.0],
            y_range=[-2.2, 2.2, 1.0],
            x_length=4.05,
            y_length=4.05,
            axis_config={"color": B_GREY, "stroke_width": 1.5, "include_tip": True, "tip_width": 0.15, "tip_height": 0.15},
        ).move_to(LEFT * 3.65 + DOWN * 0.62)
        q_lbl = mtex(r"q", font_size=26, color=B_BLUE).next_to(q_ax.x_axis.get_end(), RIGHT, buff=0.08)
        k_lbl = mtex(r"k", font_size=26, color=B_TEAL).next_to(q_ax.y_axis.get_end(), RIGHT, buff=0.10)
        phase_hdr = ttex(r"\textbf{Query--Key Saddle } $a \propto \langle q, k\rangle / \sqrt{d_{\mathrm{head}}}$", font_size=22, color=WHITE)
        phase_hdr.next_to(q_ax, UP, buff=0.25)

        # Hyperbolic level curves q*k = const (classic 3b1b saddle contours!)
        contours = VGroup()
        for c_val in [0.22, 0.65, 1.35]:
            for sgn in [1.0, -1.0]:
                # Positive product branches (blue/teal) & negative branches (dim maroon)
                pos_c = ParametricFunction(
                    lambda x, cv=c_val, sg=sgn: q_ax.c2p(sg * x, sg * (cv / x)),
                    t_range=[c_val / 2.05, 2.05],
                    color=B_BLUE, stroke_width=1.2, stroke_opacity=0.35,
                )
                neg_c = ParametricFunction(
                    lambda x, cv=c_val, sg=sgn: q_ax.c2p(sg * x, -sg * (cv / x)),
                    t_range=[c_val / 2.05, 2.05],
                    color=B_MAROON, stroke_width=1.0, stroke_opacity=0.22,
                )
                contours.add(pos_c, neg_c)

        diag_line = DashedLine(q_ax.c2p(-2.0, -2.0), q_ax.c2p(2.0, 2.0), color=B_GREEN, stroke_width=1.3, stroke_opacity=0.45)

        self.play(
            Create(q_ax), Write(q_lbl), Write(k_lbl), FadeIn(phase_hdr),
            LaggedStart(*[Create(c) for c in contours], lag_ratio=0.04),
            Create(diag_line),
            run_time=1.6,
        )

        # --- Right Top: LaTeX Saddle Equation & Algorithm 1 Transformation ---
        saddle_tex = mtex(
            r"\dot{a} \propto \frac{\varepsilon^5}{n d_{\mathrm{head}}^{1/4}}\,a^{3/2}"
            r"\;\;\Longrightarrow\;\;"
            r"T_{\mathrm{step}} = \widetilde{O}\!\left(n\sqrt{d_{\mathrm{head}}}\,\textcolor[HTML]{FC6255}{\varepsilon^{-5}}\right)",
            font_size=25, color=None,
        ).move_to(RIGHT * 2.95 + UP * 1.95)

        alg_tex = mtex(
            r"\textbf{Algorithm 1:}\quad \theta_B \leftarrow \theta_B - \eta\,\frac{G_B}{\mathrm{RMS}(G_B) + \iota}"
            r"\quad\textcolor[HTML]{83C167}{\text{(0 state!)}}",
            font_size=24, color=None,
        ).move_to(RIGHT * 2.95 + UP * 1.18)

        # --- Right Middle: Rollout Race Axes ---
        ax_r = Axes(
            x_range=[0, 3.0, 1.0],
            y_range=[45, 104, 25],
            x_length=5.25,
            y_length=2.25,
            axis_config={"color": B_GREY, "stroke_width": 1.7, "include_tip": True, "tip_width": 0.15, "tip_height": 0.15},
        ).move_to(RIGHT * 3.05 + DOWN * 0.42)
        rx_lbl = mtex(r"\text{Updates }(\times 10^3)", font_size=20, color=GREY_B).next_to(ax_r.x_axis, DOWN, buff=0.10)
        ry_lbl = mtex(r"\text{Rollout }\%", font_size=20, color=GREY_B).next_to(ax_r.y_axis, LEFT, buff=0.08)

        r_leg = VGroup(
            mtex(r"\textcolor[HTML]{83C167}{\blacksquare\;\text{Stateless Block-RMS (5/5, }0\text{ state)}}", font_size=17, color=None),
            mtex(r"\textcolor[HTML]{58C4DD}{\blacksquare\;\text{AdamW (5/5, }2|\theta|\text{ state)}}", font_size=17, color=None),
            mtex(r"\textcolor[HTML]{FC6255}{\blacksquare\;\text{Tuned SGD (0/5, stuck at }H(p)\text{)}}", font_size=17, color=None),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06).move_to(ax_r.c2p(1.94, 58.0))

        self.play(Write(saddle_tex), Create(ax_r), FadeIn(rx_lbl), FadeIn(ry_lbl), FadeIn(r_leg), run_time=1.4)
        self.play(Write(alg_tex), run_time=1.2)

        # Simultaneous 60fps race:
        #   - On left (q, k) saddle: Red SGD particle barely crawls near origin; Green Block-RMS hyperbola shoots to (1.9, 1.9)!
        #   - On right axes: Tuned SGD stalls at 75%, while Block-RMS & AdamW surge to 100%!
        race = ValueTracker(0.02)

        def sgd_acc(t: float) -> float:
            return 50.0 + 24.5 * (1.0 - math.exp(-t / 0.18))

        def brms_acc(t: float) -> float:
            base = 50.0 + 24.5 * (1.0 - math.exp(-t / 0.15))
            esc = 25.4 / (1.0 + math.exp(-(t - 1.22) / 0.11))
            return min(100.0, base + esc)

        def adam_acc(t: float) -> float:
            base = 50.0 + 24.5 * (1.0 - math.exp(-t / 0.15))
            esc = 25.4 / (1.0 + math.exp(-(t - 1.46) / 0.12))
            return min(100.0, base + esc)

        def brms_qk(s: float) -> np.ndarray:
            # Starts asymmetric (q0=0.42, k0=0.10), aligns rapidly to q=k under Block-RMS, then grows linearly!
            q = 0.42 * math.exp(-1.6 * s) + (0.18 + 0.58 * s)
            k = 0.10 * math.exp(-1.6 * s) + (0.18 + 0.58 * s)
            return q_ax.c2p(min(1.95, q), min(1.95, k))

        def sgd_qk(s: float) -> np.ndarray:
            # Crawls glacially near saddle origin!
            q = 0.42 + 0.04 * s
            k = 0.10 + 0.05 * s
            return q_ax.c2p(q, k)

        def draw_synchronized_race():
            tm = max(0.04, race.get_value())
            grp = VGroup()

            # Saddle trajectories on left
            ss = np.linspace(0, tm, max(8, int(90 * tm / 3.0)))
            b_pts = [brms_qk(float(sv)) for sv in ss]
            s_pts = [sgd_qk(float(sv)) for sv in ss]

            b_path = VMobject(color=B_GREEN, stroke_width=4.0)
            b_path.set_points_smoothly(b_pts)
            s_path = VMobject(color=B_RED, stroke_width=3.5)
            s_path.set_points_smoothly(s_pts)

            grp.add(
                b_path, Dot(b_pts[-1], radius=0.08, color=B_GREEN),
                s_path, Dot(s_pts[-1], radius=0.075, color=B_RED),
            )

            # Rollout curves on right
            for fn, col in [(sgd_acc, B_RED), (adam_acc, B_BLUE), (brms_acc, B_GREEN)]:
                pts = [ax_r.c2p(float(sv), float(fn(float(sv)))) for sv in ss]
                rc = VMobject(color=col, stroke_width=3.4)
                rc.set_points_smoothly(pts)
                grp.add(rc, Dot(pts[-1], radius=0.055, color=col))

            return grp

        sync_dyn = always_redraw(draw_synchronized_race)
        self.add(sync_dyn)

        self.play(race.animate.set_value(3.0), run_time=4.8, rate_func=rate_functions.linear)

        # Bottom Theorem 3 LaTeX result with genuine Computer Modern underbraces!
        thm3_tex = mtex(
            r"\textbf{Theorem 3 (Unconditional):}\quad "
            r"T_{\mathrm{RMS}} \le C\Biggl("
            r"\underbrace{\textcolor[HTML]{FFFF00}{\frac{1 + |c^\star|}{\varepsilon^2}}}_{\text{warm-up}}"
            r" + "
            r"\underbrace{\textcolor[HTML]{58C4DD}{\frac{n^2\log(8n)}{p(1-p)}}}_{\text{prefix routing}}"
            r"\Biggr)",
            font_size=26, color=None,
        ).to_edge(DOWN, buff=0.22)

        self.play(Write(thm3_tex), run_time=1.6)
        self.play(Circumscribe(thm3_tex, color=B_YELLOW, stroke_width=2.4, time_width=1.4), run_time=1.3)
        self.wait(2.2)
