"""Figure 1 of the paper as a vector drawing: paper/figures/overview.svg and overview.pdf (and a PNG preview).

Usage: make_overview.py [--png]
The maps, robots, task glyphs and the schedule are drawn here; the small UI icons of panels (b) and (c) are Tabler
icons (MIT licence, paper/figures/icons/tabler/LICENSE). Text is set in Linux Biolinum O (the sans face of the ACM
template). The PDF is converted with cairosvg (CLI), which supports everything used here (no filters).
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
from itertools import pairwise
from pathlib import Path

FIG = Path(__file__).resolve().parents[1] / "figures"
ICONS = FIG / "icons" / "tabler"
W, H = 2400, 680
MAP_W, MAP_H = 600, 450
FONT = "Linux Biolinum O"
# Okabe-Ito palette and neutrals
BLUE, ORANGE, GREEN, VERM, PINK, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"
INK, GREY, MUTED, LIGHT = "#2F3A45", "#8C949C", "#55606B", "#F4F6F8"
SKILL = {"cam": BLUE, "grip": ORANGE, "arm": GREEN, "sense": PINK}  # skill colours (dots)
DRONE_SK, ROVER_SK, LEG_SK = ("cam", "sense"), ("grip", "cam"), ("arm", "sense")
FADE = 0.42  # unchanged content of the dynamic map

out: list[str] = []


def add(s: str) -> None:
    out.append(s)


def text(x, y, s, size=28, weight="normal", anchor="start", fill=INK, style="normal") -> None:
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'font-style="{style}" text-anchor="{anchor}" fill="{fill}">{s}</text>')


def tabler(name: str, cx, cy, size, color, width=2.0, filled=False) -> None:
    """A Tabler icon centred at (cx, cy), ``size`` px wide, stroked (or filled) in ``color``."""
    src = (ICONS / (f"filled-{name}.svg" if filled else f"{name}.svg")).read_text()
    inner = re.search(r"<svg[^>]*>(.*)</svg>", src, re.DOTALL).group(1)
    inner = re.sub(r'<path stroke="none" d="M0 0h24v24H0z" fill="none"\s*/>', "", inner)
    k = size / 24
    paint = f'fill="{color}" stroke="none"' if filled else \
        f'fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"'
    add(f'<g transform="translate({cx - size / 2:.1f},{cy - size / 2:.1f}) scale({k:.4f})" {paint}>{inner}</g>')


def defs() -> None:
    add("<defs>")
    for name, col in (("blue", BLUE), ("orange", ORANGE), ("green", GREEN), ("grey", "#6B7580"), ("verm", VERM)):
        add(f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="14" markerHeight="14" '
            f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 L3,5 z" fill="{col}"/></marker>')
    add('<pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<rect width="8" height="8" fill="#F1F3F5"/><path d="M0,0 L0,8" stroke="#9AA3AD" stroke-width="3"/></pattern>')
    add("</defs>")


def arrow(d, color, name, width=5, dash=None) -> None:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round"{dash_attr} '
        f'marker-end="url(#ah-{name})"/>')


def dots(cx, cy, skills, r=7) -> None:
    n = len(skills)
    for k, sk in enumerate(skills):
        x = cx + (k - (n - 1) / 2) * (2 * r + 5)
        add(f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="{r}" fill="{SKILL[sk]}" stroke="white" stroke-width="2"/>')


# --- robots -----------------------------------------------------------------------------------------------------

def drone(cx, cy, s=1.0, color=BLUE, light="#A9D2EE") -> None:
    add(f'<g transform="translate({cx:.1f},{cy:.1f}) scale({s})">')
    add(f'<path d="M-30,-22 L30,22 M-30,22 L30,-22" stroke="{INK}" stroke-width="6" stroke-linecap="round"/>')
    for x, y in ((-30, -22), (30, 22), (-30, 22), (30, -22)):
        add(f'<ellipse cx="{x}" cy="{y - 4}" rx="21" ry="6.5" fill="{light}" stroke="{INK}" stroke-width="2.5"/>')
        add(f'<circle cx="{x}" cy="{y}" r="4" fill="{INK}"/>')
    add(f'<rect x="-16" y="-12" width="32" height="24" rx="7" fill="{color}" stroke="{INK}" stroke-width="2.5"/>')
    add(f'<circle cx="0" cy="15" r="5.5" fill="{INK}"/>')
    add("</g>")


def rover(cx, cy, s=1.0, color=ORANGE, light="#F5CB6B", broken=False) -> None:
    if broken:
        color, light = "#A7AEB5", "#C9CED3"
    add(f'<g transform="translate({cx:.1f},{cy:.1f}) scale({s})">')
    add(f'<path d="M14,-26 L22,-46" stroke="{INK}" stroke-width="4" stroke-linecap="round"/>')
    add(f'<circle cx="23" cy="-48" r="6" fill="{INK}"/>')
    add(f'<rect x="-24" y="-30" width="38" height="16" rx="5" fill="{light}" stroke="{INK}" stroke-width="2.5"/>')
    add(f'<rect x="-40" y="-16" width="80" height="26" rx="8" fill="{color}" stroke="{INK}" stroke-width="2.5"/>')
    for x in (-27, 0, 27):
        add(f'<circle cx="{x}" cy="16" r="12" fill="{INK}"/><circle cx="{x}" cy="16" r="4.5" fill="#B8BEC4"/>')
    add("</g>")


def legged(cx, cy, s=1.0, color=GREEN, light="#6CC5A7") -> None:
    add(f'<g transform="translate({cx:.1f},{cy:.1f}) scale({s})">')
    for hx, kx, fx in ((-26, -34, -28), (-12, -4, -10), (10, 2, 8), (24, 32, 28)):
        add(f'<path d="M{hx},0 L{kx},18 L{fx},36" fill="none" stroke="{INK}" stroke-width="5.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')
        add(f'<circle cx="{fx}" cy="37" r="3.5" fill="{INK}"/>')
    add(f'<rect x="-36" y="-16" width="70" height="22" rx="9" fill="{color}" stroke="{INK}" stroke-width="2.5"/>')
    add(f'<path d="M28,-12 L50,-22 L54,-6 L32,0 z" fill="{light}" stroke="{INK}" stroke-width="2.5" '
        f'stroke-linejoin="round"/>')
    add(f'<circle cx="47" cy="-12" r="3" fill="{INK}"/>')
    add("</g>")


def pad(cx, cy, color, rx=48, ry=15) -> None:
    add(f'<ellipse cx="{cx:.1f}" cy="{cy + 3:.1f}" rx="{rx}" ry="{ry}" fill="#000" opacity="0.08"/>')
    add(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx}" ry="{ry}" fill="{LIGHT}" stroke="{color}" stroke-width="3.5"/>')
    add(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx - 11}" ry="{ry - 5}" fill="none" stroke="{color}" '
        f'stroke-width="1.8" opacity="0.5"/>')


# --- task glyphs and map furniture --------------------------------------------------------------------------------

def rubble(cx, cy, s=1.0) -> None:
    add(f'<g transform="translate({cx:.1f},{cy:.1f}) scale({s})">')
    add('<ellipse cx="0" cy="26" rx="52" ry="9" fill="#000" opacity="0.10"/>')
    add(f'<path d="M-46,24 L-30,-4 L-14,10 L-2,-22 L16,-6 L28,-18 L46,24 z" fill="#A9B0B7" stroke="{INK}" '
        f'stroke-width="2.5" stroke-linejoin="round"/>')
    add(f'<path d="M-30,24 L-18,6 L-4,24 z M4,24 L14,4 L26,24 z" fill="#7E878F" stroke="{INK}" stroke-width="2" '
        f'stroke-linejoin="round"/>')
    add('<path d="M-8,-14 L22,-34 M4,-2 L34,-14" stroke="#8A6E52" stroke-width="5" stroke-linecap="round"/>')
    add("</g>")


def flame(cx, cy, s=1.0) -> None:
    add(f'<g transform="translate({cx:.1f},{cy:.1f}) scale({s})">')
    add(f'<path d="M0,26 C-26,24 -30,0 -14,-18 C-12,-4 -4,-2 -2,-8 C-4,-22 4,-34 10,-40 C10,-24 28,-14 26,4 '
        f'C26,18 14,26 0,26 z" fill="{VERM}" stroke="{INK}" stroke-width="2.5" stroke-linejoin="round"/>')
    add(f'<path d="M0,22 C-14,20 -16,6 -6,-4 C-4,4 4,4 6,-4 C14,4 16,14 10,18 C7,21 4,22 0,22 z" '
        f'fill="{ORANGE}"/>')
    add('<path d="M0,20 C-6,19 -7,12 -2,8 C0,12 4,12 5,9 C7,13 5,19 0,20 z" fill="#F6D55C"/>')
    add("</g>")


def fire_site(cx, cy, s=1.0) -> None:
    rubble(cx, cy + 18 * s, 0.8 * s)
    flame(cx - 2 * s, cy - 10 * s, 0.95 * s)


def ruin(cx, cy, s=1.0) -> None:
    add(f'<g transform="translate({cx:.1f},{cy:.1f}) scale({s})">')
    add('<ellipse cx="0" cy="34" rx="50" ry="9" fill="#000" opacity="0.10"/>')
    add(f'<path d="M-30,34 L-30,-30 L-16,-38 L-6,-24 L6,-40 L18,-26 L30,-34 L30,34 z" fill="#B3BAC1" '
        f'stroke="{INK}" stroke-width="2.5" stroke-linejoin="round"/>')
    for x in (-20, -4, 12):
        for y in (-18, -2, 14):
            add(f'<rect x="{x}" y="{y}" width="9" height="10" rx="1.5" fill="#5D6873"/>')
    add(f'<path d="M-44,34 L-34,20 L-24,34 z M24,34 L36,16 L48,34 z" fill="#8E979F" stroke="{INK}" '
        f'stroke-width="2" stroke-linejoin="round"/>')
    add("</g>")


def burst(cx, cy, r=32, label="NEW") -> None:
    pts = []
    for k in range(28):
        rad = r if k % 2 == 0 else r * 0.72
        a = math.pi * k / 14 - math.pi / 2
        pts.append(f"{cx + rad * math.cos(a):.1f},{cy + rad * math.sin(a):.1f}")
    add(f'<polygon points="{" ".join(pts)}" fill="{VERM}" stroke="white" stroke-width="2.5"/>')
    text(cx, cy + 7, label, size=19, weight="bold", anchor="middle", fill="white")


def hourglass(cx, cy, s=1.0) -> None:
    add(f'<g transform="translate({cx:.1f},{cy:.1f}) scale({s})">')
    add(f'<path d="M-15,-24 L15,-24 L2,0 L15,24 L-15,24 L-2,0 z" fill="white" stroke="{INK}" stroke-width="3" '
        f'stroke-linejoin="round"/>')
    add(f'<path d="M-9,21 L9,21 L2,10 L-2,10 z M-7,-18 L7,-18 L1,-8 L-1,-8 z" fill="{ORANGE}"/>')
    add(f'<path d="M-19,-26 L19,-26 M-19,26 L19,26" stroke="{INK}" stroke-width="5" stroke-linecap="round"/>')
    add("</g>")


def tower(cx, cy, s=1.0) -> None:
    add(f'<g transform="translate({cx:.1f},{cy:.1f}) scale({s})" stroke="{INK}" stroke-linecap="round" fill="none">')
    add('<path d="M-18,34 L0,-22 L18,34 M-12,16 L12,16 M-7,0 L7,0 M-12,16 L7,0 M12,16 L-7,0 M-18,34 L12,16 '
        'M18,34 L-12,16" stroke-width="3.5"/>')
    add(f'<circle cx="0" cy="-24" r="5" fill="{INK}" stroke="none"/>')
    for rr in (13, 22):
        add(f'<path d="M{-rr * 0.7:.1f},{-24 - rr * 0.7:.1f} A{rr},{rr} 0 0 1 {rr * 0.7:.1f},{-24 - rr * 0.7:.1f}" '
            f'stroke="{BLUE}" stroke-width="3.5"/>')
    add("</g>")


def warning(cx, cy, s=1.0) -> None:
    add(f'<g transform="translate({cx:.1f},{cy:.1f}) scale({s})">')
    add(f'<path d="M0,-24 L24,18 L-24,18 z" fill="{VERM}" stroke="white" stroke-width="3" stroke-linejoin="round"/>')
    text(0, 13, "!", size=30, weight="bold", anchor="middle", fill="white")
    add("</g>")


def cross(cx, cy, r=9) -> None:
    add(f'<path d="M{cx - r:.1f},{cy - r:.1f} L{cx + r:.1f},{cy + r:.1f} M{cx - r:.1f},{cy + r:.1f} '
        f'L{cx + r:.1f},{cy - r:.1f}" stroke="{VERM}" stroke-width="5.5" stroke-linecap="round"/>')


def link(x1, y1, x2, y2, lost=False) -> None:
    """Radio link; a lost one is drawn faint with a red cross."""
    col, op = ("#3E6F9C", 1.0) if not lost else ("#8FAFCB", 0.55)
    add(f'<path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" stroke="{col}" stroke-width="3.5" stroke-dasharray="1 8" '
        f'stroke-linecap="round" opacity="{op}"/>')
    if lost:
        cross((x1 + x2) / 2, (y1 + y2) / 2)


def tree(x, y, r=9) -> None:
    add(f'<rect x="{x - 2:.1f}" y="{y + r - 3:.1f}" width="4" height="7" fill="#CFC6BA"/>')
    add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#C9DDCC"/><circle cx="{x - r * 0.3:.1f}" '
        f'cy="{y - r * 0.3:.1f}" r="{r * 0.45:.1f}" fill="#D8E7DA"/>')


def house(x, y) -> None:
    add(f'<rect x="{x - 11}" y="{y - 5}" width="22" height="16" fill="#F5F2EE" stroke="#D6CEC4" stroke-width="2"/>')
    add(f'<path d="M{x - 14},{y - 3} L{x},{y - 16} L{x + 14},{y - 3} z" fill="#E2D9CF" stroke="#D6CEC4" '
        f'stroke-width="2" stroke-linejoin="round"/>')


def halo(cx, cy, r) -> None:
    """White disc behind an object so it separates from the map."""
    add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="white" opacity="0.92" stroke="#D5DBD2" '
        f'stroke-width="2"/>')


# --- panel (a): the two maps --------------------------------------------------------------------------------------

TREES = [(28, 190), (50, 205), (150, 70), (175, 58), (330, 40), (360, 55), (560, 150), (575, 175), (40, 300),
         (160, 300), (185, 318), (395, 420), (430, 405), (560, 300), (20, 420), (455, 250), (480, 262),
         (285, 205), (130, 430), (590, 420)]
HOUSES = [(440, 330), (470, 350), (150, 250), (395, 120)]


def terrain(ox, oy, cid) -> None:
    """Muted background (land, river, roads, relief, houses, trees) so that robots and tasks stand out."""
    w, h = MAP_W, MAP_H
    add(f'<clipPath id="{cid}"><rect x="{ox}" y="{oy}" width="{w}" height="{h}" rx="18"/></clipPath>')
    add(f'<g clip-path="url(#{cid})"><g transform="translate({ox},{oy})">')
    add(f'<rect width="{w}" height="{h}" fill="#F4F6F2"/>')
    for bx, by, br in ((170, 140, 110), (460, 330, 130), (520, 80, 90)):
        add(f'<circle cx="{bx}" cy="{by}" r="{br}" fill="#EDF1EA"/>')
    river = "M-20,130 C90,95 170,170 270,185 S450,165 500,235 S560,330 620,350"
    add(f'<path d="{river}" fill="none" stroke="#D3E6F0" stroke-width="40" stroke-linecap="round"/>')
    add(f'<path d="{river}" fill="none" stroke="#E2EFF6" stroke-width="22" stroke-linecap="round"/>')
    for rd in ("M-20,265 C120,235 200,290 320,262 S500,215 620,245", "M215,-20 C235,90 175,170 215,240 S260,380 240,470"):
        add(f'<path d="{rd}" fill="none" stroke="#E5E8EB" stroke-width="17" stroke-linecap="round"/>')
        add(f'<path d="{rd}" fill="none" stroke="#FAFBFC" stroke-width="2.5" stroke-dasharray="11 10"/>')
    for bx, by, ang in ((222, 176, -12), (478, 206, 55)):
        add(f'<rect x="{bx - 20}" y="{by - 10}" width="40" height="20" rx="3" fill="#DADEE2" stroke="#C6CBD0" '
            f'stroke-width="2" transform="rotate({ang},{bx},{by})"/>')
    add('<path d="M470,70 L520,8 L570,70 z" fill="#DDE1E5"/><path d="M506,26 L520,8 L534,26 L525,22 L516,28 z" '
        'fill="white"/><path d="M535,70 L575,25 L615,70 z" fill="#E4E7EA"/>')
    for x, y in HOUSES:
        house(x, y)
    for x, y in TREES:
        tree(x, y)
    add("</g></g>")
    add(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" rx="18" fill="none" stroke="#C5CDC2" stroke-width="3"/>')


def scene(ox, oy, dynamic: bool) -> None:
    """One map frame at (ox, oy) (600 x 450). In the dynamic frame, what did not change is faded."""
    def p(x, y):
        return ox + x, oy + y
    tx, ty = p(300, 120)
    fade = f'<g opacity="{FADE}">' if dynamic else "<g>"
    add(fade)
    for (x, y, r) in ((170, 205, 44), (500, 190, 44), (320, 330, 52)):
        halo(*p(x, y), r)
    pad(*p(72, 72), BLUE)
    pad(*p(80, 402), ORANGE)
    pad(*p(522, 402), GREEN)
    link(*p(100, 70), tx - 14, ty - 8)
    link(tx + 8, ty + 4, *p(478, 176))
    if not dynamic:
        link(tx - 8, ty + 14, *p(100, 375))
    link(tx + 6, ty + 20, *p(500, 372))
    halo(tx, ty - 2, 30)
    tower(tx, ty, 0.85)
    fire_site(*p(170, 200), 0.8)
    dots(*p(170, 240), ("cam", "sense"))
    ruin(*p(500, 186), 0.75)
    dots(*p(500, 224), ("cam", "sense"))
    rubble(*p(320, 325), 0.95)
    dots(*p(320, 364), ("grip", "arm"))
    hourglass(*p(378, 304), 0.8)
    arrow("M{:.0f},{:.0f} C{:.0f},{:.0f} {:.0f},{:.0f} {:.0f},{:.0f}".format(*p(492, 380), *p(440, 370),
                                                                           *p(400, 355), *p(362, 342)), GREEN, "green")
    if not dynamic:
        arrow("M{:.0f},{:.0f} C{:.0f},{:.0f} {:.0f},{:.0f} {:.0f},{:.0f}".format(*p(92, 96), *p(110, 140),
                                                                               *p(130, 170), *p(146, 185)), BLUE, "blue")
        arrow("M{:.0f},{:.0f} C{:.0f},{:.0f} {:.0f},{:.0f} {:.0f},{:.0f}".format(*p(118, 385), *p(190, 380),
                                                                               *p(240, 360), *p(276, 342)),
              ORANGE, "orange")
    legged(*p(516, 386), 0.72)
    dots(*p(522, 428), LEG_SK)
    dots(*p(72, 98), DRONE_SK)
    dots(*p(80, 428), ROVER_SK)
    if not dynamic:
        drone(*p(72, 60), 0.72)
        rover(*p(80, 388), 0.68)
    add("</g>")
    if dynamic:  # what changed, at full strength
        link(tx - 8, ty + 14, *p(100, 375), lost=True)
        halo(*p(470, 88), 42)
        ruin(*p(470, 84), 0.62)
        dots(*p(470, 118), ("cam", "sense"))
        burst(*p(522, 58), 30)
        arrow("M{:.0f},{:.0f} C{:.0f},{:.0f} {:.0f},{:.0f} {:.0f},{:.0f}".format(*p(112, 58), *p(220, 20),
                                                                               *p(340, 40), *p(426, 76)),
              BLUE, "blue", width=5.5, dash="13 9")
        text(*p(236, 78), "replan", size=25, weight="bold", fill=BLUE)
        drone(*p(72, 60), 0.72)
        rover(*p(80, 388), 0.68, broken=True)
        warning(*p(130, 352), 0.85)


def legend(x0, y0) -> None:
    """Symbols of panel (a), in two rows."""
    y1, y2 = y0 + 16, y0 + 66
    pad(x0 + 30, y1, BLUE, rx=26, ry=9)
    text(x0 + 64, y1 + 9, "station", size=25)
    tower(x0 + 185, y1 - 2, 0.6)
    text(x0 + 208, y1 + 9, "relay", size=25)
    link(x0 + 290, y1, x0 + 350, y1)
    text(x0 + 360, y1 + 9, "radio link", size=25)
    dots(x0 + 520, y1, ("cam", "grip", "arm", "sense"))
    text(x0 + 565, y1 + 9, "skills (robot has / task needs)", size=25)
    hourglass(x0 + 30, y2, 0.7)
    text(x0 + 64, y2 + 9, "coalition task: starts when all members are there", size=25)
    burst(x0 + 625, y2, 22, label="")
    text(x0 + 655, y2 + 9, "new task", size=25)
    warning(x0 + 790, y2, 0.7)
    text(x0 + 816, y2 + 9, "failure", size=25)
    link(x0 + 930, y2, x0 + 990, y2, lost=True)
    text(x0 + 1002, y2 + 9, "link lost", size=25)


# --- panels (b) and (c) ------------------------------------------------------------------------------------------

def card(x, y, w, h, dashed=False) -> None:
    if not dashed:
        add(f'<rect x="{x + 3}" y="{y + 4}" width="{w}" height="{h}" rx="12" fill="#000" opacity="0.07"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="white" stroke="#AEB6BE" stroke-width="2.5"'
        + (' stroke-dasharray="8 7"/>' if dashed else "/>"))


def gantt(x0, y0) -> None:
    """Schedule of the plan: each robot in the global order; the coalition task starts when both have arrived."""
    t0, k = x0 + 70, 45.0  # x of time 0, px per time unit
    rows = [(drone, BLUE, 0.4), (rover, ORANGE, 0.36), (legged, GREEN, 0.38)]
    ys = [y0 + 22, y0 + 64, y0 + 106]
    fire_c, rub_c, ruin_c = "#F3B08A", "#C3C9CF", "#A9B5C2"

    def bar(y, a, b, col, label=None):
        add(f'<rect x="{t0 + a * k:.1f}" y="{y - 13}" width="{(b - a) * k:.1f}" height="26" rx="5" fill="{col}" '
            f'stroke="{INK}" stroke-width="1.8"/>')
        if label:
            text(t0 + (a + b) / 2 * k, y + 7, label, size=19, anchor="middle")

    def travel(y, a, b):
        add(f'<path d="M{t0 + a * k:.1f},{y} L{t0 + b * k:.1f},{y}" stroke="#8C949C" stroke-width="2.5"/>')

    for (fn, _, s), y in zip(rows, ys):
        fn(x0 + 28, y + (6 if fn is legged else 2), s)
    travel(ys[0], 0, 1.5)
    bar(ys[0], 1.5, 3.4, fire_c, "fire")
    travel(ys[0], 3.4, 5.0)
    bar(ys[0], 5.0, 6.8, ruin_c, "ruin")
    travel(ys[0], 6.8, 8.2)
    travel(ys[1], 0, 3.2)
    bar(ys[1], 3.2, 6.0, rub_c, "rubble")
    travel(ys[1], 6.0, 7.4)
    travel(ys[2], 0, 1.8)
    add(f'<rect x="{t0 + 1.8 * k:.1f}" y="{ys[2] - 11}" width="{1.4 * k:.1f}" height="22" rx="4" fill="url(#hatch)" '
        f'stroke="#9AA3AD" stroke-width="1.5"/>')
    text(t0 + 2.5 * k, ys[2] + 42, "waits", size=19, anchor="middle", fill=MUTED)
    bar(ys[2], 3.2, 6.0, rub_c, "rubble")
    travel(ys[2], 6.0, 7.0)
    add(f'<path d="M{t0 + 3.2 * k:.1f},{ys[1] - 22} L{t0 + 3.2 * k:.1f},{ys[2] + 26}" stroke="{VERM}" '
        f'stroke-width="3" stroke-dasharray="6 5"/>')
    text(t0 + 3.35 * k, ys[2] + 42, "start together", size=19, fill=VERM)
    add(f'<path d="M{t0 + 8.2 * k:.1f},{ys[0] - 16} L{t0 + 8.2 * k:.1f},{ys[2] + 16}" stroke="{INK}" '
        f'stroke-width="3"/>')
    text(t0 + 8.2 * k, ys[2] + 42, "makespan", size=19, anchor="middle", fill=INK, weight="bold")
    add(f'<path d="M{t0},{ys[2] + 22} L{t0 + 9.4 * k:.1f},{ys[2] + 22}" stroke="#AEB6BE" stroke-width="2" '
        f'marker-end="url(#ah-grey)"/>')


def panel_b(x0, y0) -> None:
    text(x0, y0 + 34, "(b) Our search: ALNS", size=36, weight="bold")
    text(x0, y0 + 76, "plan = one global task order + a minimal coalition per task", size=24, fill=MUTED,
         style="italic")
    cy = y0 + 130
    xs = [x0 + 60 + k * 130 for k in range(4)]
    for k, (x, glyph, sk) in enumerate(zip(xs, (fire_site, rubble, ruin, None),
                                           (("cam", "sense"), ("grip", "arm"), ("cam", "sense"), None))):
        if glyph is None:
            card(x - 44, cy - 42, 88, 84, dashed=True)
            text(x, cy + 10, "…", size=36, anchor="middle", fill=GREY)
        else:
            card(x - 44, cy - 42, 88, 84)
            glyph(x, cy - 8, 0.52)
            dots(x, cy + 28, sk, r=6)
        if k:
            add(f'<path d="M{xs[k - 1] + 48},{cy} L{x - 52},{cy}" stroke="#6B7580" stroke-width="4" '
                f'marker-end="url(#ah-grey)"/>')
    gantt(x0, y0 + 208)
    text(x0, y0 + 396, "robots follow the order: no deadlock, exact schedule", size=24, fill=MUTED, style="italic")
    # the search loop, one line per step
    steps = [("scissors", BLUE, "remove", "take a few tasks out of the plan"),
             ("puzzle", ORANGE, "re-insert", "best position + coalition, scored exactly in O(|C|)"),
             ("temperature", PINK, "accept", "simulated annealing (sometimes accept worse)"),
             ("adjustments-horizontal", GREEN, "adapt", "favour operators that found better plans")]
    ly = [y0 + 440 + k * 66 for k in range(4)]
    for (icon, col, name, desc), y in zip(steps, ly):
        add(f'<circle cx="{x0 + 50}" cy="{y}" r="27" fill="{col}"/>')
        tabler(icon, x0 + 50, y, 32, "white", width=2.3)
        text(x0 + 92, y - 3, name, size=27, weight="bold")
        text(x0 + 92, y + 23, desc, size=22, fill=MUTED)
    add(f'<path d="M{x0 + 16},{ly[3] - 4} C{x0 - 4},{ly[3] - 40} {x0 - 4},{ly[0] + 40} {x0 + 18},{ly[0] + 6}" '
        f'fill="none" stroke="#6B7580" stroke-width="3.5" marker-end="url(#ah-grey)"/>')
    for a, b in pairwise(ly):
        add(f'<path d="M{x0 + 50},{a + 30} L{x0 + 50},{b - 32}" stroke="#6B7580" stroke-width="3.5" '
            f'marker-end="url(#ah-grey)"/>')


def panel_c(x0, y0) -> None:
    text(x0, y0 + 34, "(c) Fair test", size=36, weight="bold")
    # protocol timeline
    stages = [("dev", "tune all", BLUE), ("val", "dry run", ORANGE), ("freeze", "prereg", GREEN),
              ("test", "one run", PINK)]
    sx = [x0 + 54 + k * 122 for k in range(4)]
    for k, ((a, b, col), x) in enumerate(zip(stages, sx)):
        add(f'<rect x="{x - 52}" y="{y0 + 58}" width="104" height="62" rx="10" fill="{col}" opacity="0.15"/>')
        add(f'<rect x="{x - 52}" y="{y0 + 58}" width="104" height="62" rx="10" fill="none" stroke="{col}" '
            f'stroke-width="2.5"/>')
        if a == "freeze":
            tabler("lock", x - 34, y0 + 78, 21, col, width=2.4)
            text(x + 10, y0 + 85, a, size=23, weight="bold", anchor="middle")
        else:
            text(x, y0 + 85, a, size=23, weight="bold", anchor="middle")
        text(x, y0 + 110, b, size=21, anchor="middle", fill=MUTED)
        if k:
            add(f'<path d="M{sx[k - 1] + 54},{y0 + 89} L{x - 57},{y0 + 89}" stroke="#6B7580" stroke-width="3.5" '
                f'marker-end="url(#ah-grey)"/>')
    # methods, all run the same way
    rows = [("search", BLUE, "ALNS (ours)", ""), ("affiliate", PINK, "RL policy", "published, samples rollouts"),
            ("settings", GREEN, "CP-SAT", "LNS, parallel LNS, full model"), ("sum", ORANGE, "MILP", "CTAS-D"),
            ("dice-5", "#6B7580", "restarts", "8 randomized constructions")]
    ry = [y0 + 172 + k * 56 for k in range(5)]
    for (icon, col, name, desc), y in zip(rows, ry):
        add(f'<rect x="{x0 + 6}" y="{y - 22}" width="44" height="44" rx="10" fill="{col}" opacity="0.16"/>')
        tabler(icon, x0 + 28, y, 32, col, width=2.3)
        text(x0 + 62, y + (8 if not desc else 1), name, size=25, weight="bold")
        if desc:
            text(x0 + 62, y + 23, desc, size=20, fill=MUTED)
    # every method goes through the same protocol
    ya = ry[4] + 36
    add(f'<path d="M{x0 + 8},{ya} L{x0 + 8},{ya + 12} L{x0 + 462},{ya + 12} L{x0 + 462},{ya}" fill="none" '
        f'stroke="#8C949C" stroke-width="3"/>')
    boxes = [("stopwatch", INK, "same 8 cores", "same budget"), ("device-desktop-check", GREEN, "simulator",
                                                                  "scores all plans"),
             ("chart-dots", BLUE, "paired ratios", "Holm tests")]
    bxs, by = [x0 + 78 + k * 157 for k in range(3)], ya + 118
    add(f'<path d="M{x0 + 235},{ya + 12} L{x0 + 235},{ya + 26}" stroke="#8C949C" stroke-width="3"/>')
    add(f'<path d="M{x0 + 235},{ya + 26} L{bxs[0]},{ya + 26} L{bxs[0]},{by - 62}" fill="none" stroke="#6B7580" '
        f'stroke-width="3.5" marker-end="url(#ah-grey)"/>')
    for (icon, col, a, b), x in zip(boxes, bxs):
        add(f'<rect x="{x - 72}" y="{by - 56}" width="144" height="112" rx="12" fill="{LIGHT}" stroke="#AEB6BE" '
            f'stroke-width="2.5"/>')
        tabler(icon, x, by - 24, 34, col, width=2.2)
        text(x, by + 18, a, size=20, weight="bold", anchor="middle")
        text(x, by + 42, b, size=19, anchor="middle", fill=MUTED)
    for a, b in pairwise(bxs):
        add(f'<path d="M{a + 74},{by} L{b - 78},{by}" stroke="#6B7580" stroke-width="3.5" '
            f'marker-end="url(#ah-grey)"/>')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png", action="store_true", help="also write a PNG preview")
    args = ap.parse_args()
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    defs()
    add(f'<rect width="{W}" height="{H}" fill="white"/>')
    top, lx, rx = 96, 16, 16 + MAP_W + 64
    text(lx, 36, "(a) Static vs. dynamic", size=36, weight="bold")
    text(lx + MAP_W / 2, 84, "static: all tasks known at start", size=26, weight="bold", anchor="middle",
         fill="#1F5F99")
    text(rx + MAP_W / 2, 84, "dynamic: tasks appear, robots fail", size=26, weight="bold", anchor="middle",
         fill=VERM)
    terrain(lx, top, "mapA")
    scene(lx, top, dynamic=False)
    mid = top + MAP_H / 2
    ax = lx + MAP_W + 10
    add(f'<path d="M{ax},{mid - 22} L{ax + 22},{mid - 22} L{ax + 22},{mid - 40} L{ax + 44},{mid} L{ax + 22},'
        f'{mid + 40} L{ax + 22},{mid + 22} L{ax},{mid + 22} z" fill="#B3BAC1"/>')
    terrain(rx, top, "mapB")
    scene(rx, top, dynamic=True)
    legend(lx, top + MAP_H + 22)
    add(f'<path d="M1298,16 L1298,{H - 16}" stroke="#C9CED3" stroke-width="3" stroke-dasharray="6 10"/>')
    panel_b(1322, 2)
    add(f'<path d="M1886,16 L1886,{H - 16}" stroke="#C9CED3" stroke-width="3" stroke-dasharray="6 10"/>')
    panel_c(1902, 2)
    add("</svg>")
    svg = FIG / "overview.svg"
    svg.write_text("\n".join(out))
    subprocess.run(["cairosvg", str(svg), "-o", str(FIG / "overview.pdf")], check=True)
    if args.png:
        subprocess.run(["cairosvg", str(svg), "-o", str(FIG / "overview_preview.png"), "--output-width", "2400"],
                       check=True)
    print(f"wrote {svg}, overview.pdf" + (", overview_preview.png" if args.png else ""))


if __name__ == "__main__":
    main()
