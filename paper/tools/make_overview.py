"""Figure 1 of the paper as a vector drawing: paper/figures/overview.svg and overview.pdf (and a PNG preview).

Usage: make_overview.py [--png]
The map, robots and task glyphs are drawn here; the small UI icons of panels (b) and (c) are Tabler icons (MIT
licence, paper/figures/icons/tabler/LICENSE). Text is set in Linux Biolinum O (the sans face of the ACM template).
The PDF is converted with cairosvg (CLI), which supports everything used here (no filters).
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
from pathlib import Path

FIG = Path(__file__).resolve().parents[1] / "figures"
ICONS = FIG / "icons" / "tabler"
W, H = 2400, 864
MAP_W, MAP_H = 740, 745  # map frames; scene coordinates are given on a 740 x 830 grid
KY = MAP_H / 830
FONT = "Linux Biolinum O"
# Okabe-Ito palette and neutrals
BLUE, ORANGE, GREEN, VERM, PINK, SKY, YELLOW = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"
INK, GREY, LIGHT = "#2F3A45", "#8C949C", "#F4F6F8"
SKILL = {"cam": BLUE, "grip": ORANGE, "arm": GREEN, "sense": PINK}  # skill colours (dots)
DRONE_SK, ROVER_SK, LEG_SK = ("cam", "sense"), ("grip", "cam"), ("arm", "sense")

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


def markers() -> None:
    add("<defs>")
    for name, col in (("blue", BLUE), ("orange", ORANGE), ("green", GREEN), ("grey", "#6B7580"), ("verm", VERM)):
        add(f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="16" markerHeight="16" '
            f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,5 L0,10 L3,5 z" fill="{col}"/></marker>')
    add("</defs>")


def arrow(d, color, name, width=6, dash=None, opacity=1.0) -> None:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round"{dash_attr} '
        f'opacity="{opacity}" marker-end="url(#ah-{name})"/>')


def dots(cx, cy, skills, r=9) -> None:
    n = len(skills)
    for k, sk in enumerate(skills):
        x = cx + (k - (n - 1) / 2) * (2 * r + 6)
        add(f'<circle cx="{x:.1f}" cy="{cy:.1f}" r="{r}" fill="{SKILL[sk]}" stroke="white" stroke-width="2.5"/>')


# --- robots -----------------------------------------------------------------------------------------------------

def drone(cx, cy, s=1.0, color=BLUE, light="#A9D2EE") -> None:
    add(f'<g transform="translate({cx},{cy}) scale({s})">')
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
    add(f'<g transform="translate({cx},{cy}) scale({s})">')
    add(f'<path d="M14,-26 L22,-46" stroke="{INK}" stroke-width="4" stroke-linecap="round"/>')
    add(f'<circle cx="23" cy="-48" r="6" fill="{INK}"/>')
    add(f'<rect x="-24" y="-30" width="38" height="16" rx="5" fill="{light}" stroke="{INK}" stroke-width="2.5"/>')
    add(f'<rect x="-40" y="-16" width="80" height="26" rx="8" fill="{color}" stroke="{INK}" stroke-width="2.5"/>')
    for x in (-27, 0, 27):
        add(f'<circle cx="{x}" cy="16" r="12" fill="{INK}"/><circle cx="{x}" cy="16" r="4.5" fill="#B8BEC4"/>')
    add("</g>")


def legged(cx, cy, s=1.0, color=GREEN, light="#6CC5A7") -> None:
    add(f'<g transform="translate({cx},{cy}) scale({s})">')
    for hx, kx, fx in ((-26, -34, -28), (-12, -4, -10), (10, 2, 8), (24, 32, 28)):
        add(f'<path d="M{hx},0 L{kx},18 L{fx},36" fill="none" stroke="{INK}" stroke-width="5.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')
        add(f'<circle cx="{fx}" cy="37" r="3.5" fill="{INK}"/>')
    add(f'<rect x="-36" y="-16" width="70" height="22" rx="9" fill="{color}" stroke="{INK}" stroke-width="2.5"/>')
    add(f'<path d="M28,-12 L50,-22 L54,-6 L32,0 z" fill="{light}" stroke="{INK}" stroke-width="2.5" '
        f'stroke-linejoin="round"/>')
    add(f'<circle cx="47" cy="-12" r="3" fill="{INK}"/>')
    add("</g>")


def pad(cx, cy, color, rx=62, ry=20) -> None:
    add(f'<ellipse cx="{cx}" cy="{cy + 4}" rx="{rx}" ry="{ry}" fill="#000" opacity="0.08"/>')
    add(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{LIGHT}" stroke="{color}" stroke-width="4"/>')
    add(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx - 14}" ry="{ry - 7}" fill="none" stroke="{color}" '
        f'stroke-width="2" opacity="0.5"/>')


# --- task glyphs and map furniture --------------------------------------------------------------------------------

def rubble(cx, cy, s=1.0) -> None:
    add(f'<g transform="translate({cx},{cy}) scale({s})">')
    add('<ellipse cx="0" cy="26" rx="52" ry="9" fill="#000" opacity="0.10"/>')
    add(f'<path d="M-46,24 L-30,-4 L-14,10 L-2,-22 L16,-6 L28,-18 L46,24 z" fill="#A9B0B7" stroke="{INK}" '
        f'stroke-width="2.5" stroke-linejoin="round"/>')
    add(f'<path d="M-30,24 L-18,6 L-4,24 z M4,24 L14,4 L26,24 z" fill="#7E878F" stroke="{INK}" stroke-width="2" '
        f'stroke-linejoin="round"/>')
    add('<path d="M-8,-14 L22,-34 M4,-2 L34,-14" stroke="#8A6E52" stroke-width="5" stroke-linecap="round"/>')
    add("</g>")


def flame(cx, cy, s=1.0) -> None:
    add(f'<g transform="translate({cx},{cy}) scale({s})">')
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
    add(f'<g transform="translate({cx},{cy}) scale({s})">')
    add('<ellipse cx="0" cy="34" rx="50" ry="9" fill="#000" opacity="0.10"/>')
    add(f'<path d="M-30,34 L-30,-30 L-16,-38 L-6,-24 L6,-40 L18,-26 L30,-34 L30,34 z" fill="#B3BAC1" '
        f'stroke="{INK}" stroke-width="2.5" stroke-linejoin="round"/>')
    for x in (-20, -4, 12):
        for y in (-18, -2, 14):
            add(f'<rect x="{x}" y="{y}" width="9" height="10" rx="1.5" fill="#5D6873"/>')
    add(f'<path d="M-44,34 L-34,20 L-24,34 z M24,34 L36,16 L48,34 z" fill="#8E979F" stroke="{INK}" '
        f'stroke-width="2" stroke-linejoin="round"/>')
    add("</g>")


def burst(cx, cy, r=42, label="NEW") -> None:
    pts = []
    for k in range(28):
        rad = r if k % 2 == 0 else r * 0.72
        a = math.pi * k / 14 - math.pi / 2
        pts.append(f"{cx + rad * math.cos(a):.1f},{cy + rad * math.sin(a):.1f}")
    add(f'<polygon points="{" ".join(pts)}" fill="{VERM}" stroke="white" stroke-width="3"/>')
    text(cx, cy + 9, label, size=25, weight="bold", anchor="middle", fill="white")


def hourglass(cx, cy, s=1.0) -> None:
    add(f'<g transform="translate({cx},{cy}) scale({s})">')
    add(f'<path d="M-15,-24 L15,-24 L2,0 L15,24 L-15,24 L-2,0 z" fill="white" stroke="{INK}" stroke-width="3" '
        f'stroke-linejoin="round"/>')
    add(f'<path d="M-9,21 L9,21 L2,10 L-2,10 z M-7,-18 L7,-18 L1,-8 L-1,-8 z" fill="{ORANGE}"/>')
    add(f'<path d="M-19,-26 L19,-26 M-19,26 L19,26" stroke="{INK}" stroke-width="5" stroke-linecap="round"/>')
    add("</g>")


def tower(cx, cy, s=1.0) -> None:
    add(f'<g transform="translate({cx},{cy}) scale({s})" stroke="{INK}" stroke-linecap="round" fill="none">')
    add('<path d="M-18,34 L0,-22 L18,34 M-12,16 L12,16 M-7,0 L7,0 M-12,16 L7,0 M12,16 L-7,0 M-18,34 L12,16 '
        'M18,34 L-12,16" stroke-width="3.5"/>')
    add(f'<circle cx="0" cy="-24" r="5" fill="{INK}" stroke="none"/>')
    for rr in (13, 22):
        add(f'<path d="M{-rr * 0.7:.1f},{-24 - rr * 0.7:.1f} A{rr},{rr} 0 0 1 {rr * 0.7:.1f},{-24 - rr * 0.7:.1f}" '
            f'stroke="{BLUE}" stroke-width="3.5"/>')
    add("</g>")


def warning(cx, cy, s=1.0) -> None:
    add(f'<g transform="translate({cx},{cy}) scale({s})">')
    add(f'<path d="M0,-24 L24,18 L-24,18 z" fill="{VERM}" stroke="white" stroke-width="3" stroke-linejoin="round"/>')
    text(0, 13, "!", size=30, weight="bold", anchor="middle", fill="white")
    add("</g>")


def cross(cx, cy, r=11) -> None:
    add(f'<path d="M{cx - r},{cy - r} L{cx + r},{cy + r} M{cx - r},{cy + r} L{cx + r},{cy - r}" stroke="{VERM}" '
        f'stroke-width="6" stroke-linecap="round"/>')


def link(x1, y1, x2, y2, broken=False) -> None:
    col = "#3E6F9C"
    add(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{col}" stroke-width="4" stroke-dasharray="1 9" '
        f'stroke-linecap="round"/>')
    if broken:
        cross((x1 + x2) / 2, (y1 + y2) / 2)


def tree(x, y, r=12) -> None:
    add(f'<rect x="{x - 2.5}" y="{y + r - 4}" width="5" height="9" fill="#CFC6BA"/>')
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#C9DDCC"/><circle cx="{x - r * 0.3:.1f}" cy="{y - r * 0.3:.1f}" '
        f'r="{r * 0.45:.1f}" fill="#D8E7DA"/>')


def house(x, y) -> None:
    add(f'<rect x="{x - 14}" y="{y - 6}" width="28" height="20" fill="#F5F2EE" stroke="#D6CEC4" stroke-width="2"/>')
    add(f'<path d="M{x - 18},{y - 4} L{x},{y - 20} L{x + 18},{y - 4} z" fill="#E2D9CF" stroke="#D6CEC4" '
        f'stroke-width="2" stroke-linejoin="round"/>')


# --- panel (a): the two maps --------------------------------------------------------------------------------------

TREES = [(40, 330), (70, 360), (205, 170), (235, 150), (300, 90), (455, 110), (480, 140), (690, 250), (660, 280),
         (45, 600), (80, 630), (220, 690), (250, 720), (300, 810), (470, 780), (520, 800), (700, 560), (720, 640),
         (560, 520), (590, 540), (200, 520), (170, 500), (420, 430), (690, 810), (30, 800), (365, 150)]
HOUSES = [(560, 640), (600, 670), (230, 380), (470, 250)]


def terrain(ox, oy, w, h, cid) -> None:
    """Muted background (land, river, roads, relief, houses, trees) so that robots and tasks stand out."""
    add(f'<clipPath id="{cid}"><rect x="{ox}" y="{oy}" width="{w}" height="{h}" rx="22"/></clipPath>')
    add(f'<g clip-path="url(#{cid})">')
    add(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" fill="#F4F6F2"/>')
    for bx, by, br in ((200, 250, 150), (560, 620, 170), (620, 150, 120)):
        add(f'<circle cx="{ox + bx}" cy="{oy + by * KY:.1f}" r="{br}" fill="#EDF1EA"/>')
    add(f'<g transform="translate({ox},{oy}) scale(1,{KY:.4f})">')
    river = "M-30,250 C110,190 230,300 350,335 S560,300 630,420 S700,610 780,650"
    add(f'<path d="{river}" fill="none" stroke="#D3E6F0" stroke-width="54" stroke-linecap="round"/>')
    add(f'<path d="{river}" fill="none" stroke="#E2EFF6" stroke-width="30" stroke-linecap="round"/>')
    roads = ("M-20,520 C150,470 260,570 400,520 S620,430 780,480",
             "M270,-20 C300,150 210,300 270,430 S330,700 300,880")
    for rd in roads:
        add(f'<path d="{rd}" fill="none" stroke="#E5E8EB" stroke-width="22" stroke-linecap="round"/>')
        add(f'<path d="{rd}" fill="none" stroke="#FAFBFC" stroke-width="3" stroke-dasharray="14 12"/>')
    for bx, by, ang in ((290, 318, -20), (560, 330, 60)):
        add(f'<rect x="{bx - 26}" y="{by - 13}" width="52" height="26" rx="4" fill="#DADEE2" stroke="#C6CBD0" '
            f'stroke-width="2" transform="rotate({ang},{bx},{by})"/>')
    add('<path d="M560,110 L625,20 L690,110 z" fill="#DDE1E5"/><path d="M606,46 L625,20 L644,46 L632,40 L620,48 z" '
        'fill="white"/><path d="M640,110 L690,45 L740,110 z" fill="#E4E7EA"/>')
    add("</g>")
    add(f'<g transform="translate({ox},{oy})">')
    for x, y in HOUSES:
        house(x, y * KY)
    for x, y in TREES:
        tree(x, y * KY)
    add("</g></g>")
    add(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" rx="22" fill="none" stroke="#C5CDC2" stroke-width="3"/>')


def halo(cx, cy, r) -> None:
    """White disc behind an object so it separates from the map."""
    add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="white" opacity="0.92" stroke="#D5DBD2" '
        f'stroke-width="2"/>')


def scene(ox, oy, dynamic: bool) -> None:
    """One map frame at (ox, oy); coordinates below are relative to the frame (740 x 830)."""
    def p(x, y):
        return ox + x, oy + y * KY
    # stations with their robots
    pad(*p(115, 118), BLUE)
    pad(*p(125, 752), ORANGE)
    pad(*p(628, 752), GREEN)
    # radio links (behind everything else)
    tx, ty = p(372, 250)
    halo(*p(135, 425), 62)
    halo(*p(612, 425), 62)
    halo(*p(410, 612), 72)
    link(*p(150, 110), tx, ty - 20)
    link(tx + 8, ty, *p(600, 405))
    link(tx - 8, ty + 10, *p(150, 715), broken=dynamic)
    link(tx + 4, ty + 34, *p(598, 712))
    halo(tx, ty - 2, 40)
    tower(tx, ty, 1.15)
    # tasks
    fire_site(*p(135, 420), 1.05)
    dots(*p(135, 482), ("cam", "sense"))
    ruin(*p(612, 418), 1.05)
    dots(*p(612, 482), ("cam", "sense"))
    rubble(*p(385, 612), 1.2)
    dots(*p(385, 668), ("grip", "arm"))
    hourglass(*p(462, 585), 1.05)
    # routes and robots
    if not dynamic:
        arrow("M{},{} C{},{} {},{} {},{}".format(*p(118, 150), *p(95, 230), *p(120, 300), *p(133, 372)), BLUE, "blue")
        arrow("M{},{} C{},{} {},{} {},{}".format(*p(172, 722), *p(250, 700), *p(300, 660), *p(338, 632)),
              ORANGE, "orange")
        arrow("M{},{} C{},{} {},{} {},{}".format(*p(585, 722), *p(520, 700), *p(470, 670), *p(435, 636)),
              GREEN, "green")
        drone(*p(115, 100), 0.95)
        rover(*p(125, 730), 0.9)
    else:
        halo(*p(588, 215), 54)
        ruin(*p(588, 205), 0.8)
        burst(*p(655, 158), 40)
        dots(*p(588, 258), ("cam", "sense"))
        arrow("M{},{} C{},{} {},{} {},{}".format(*p(170, 105), *p(300, 60), *p(430, 150), *p(540, 190)), BLUE, "blue",
              dash="16 10")
        text(*p(340, 95), "replan", size=30, weight="bold", fill=BLUE)
        arrow("M{},{} C{},{} {},{} {},{}".format(*p(585, 722), *p(520, 700), *p(470, 670), *p(435, 636)),
              GREEN, "green")
        drone(*p(115, 100), 0.95)
        rover(*p(125, 730), 0.9, broken=True)
        warning(*p(180, 690), 1.0)
    legged(*p(622, 728), 0.95)
    dots(*p(115, 158), DRONE_SK, r=8)
    dots(*p(125, 790), ROVER_SK, r=8)
    dots(*p(628, 790), LEG_SK, r=8)


# --- panels (b) and (c) ------------------------------------------------------------------------------------------

def card(x, y, w, h) -> None:
    add(f'<rect x="{x + 3}" y="{y + 5}" width="{w}" height="{h}" rx="14" fill="#000" opacity="0.07"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="white" stroke="#AEB6BE" stroke-width="2.5"/>')


def panel_b(x0, y0) -> None:
    text(x0, y0 + 36, "(b) Our search: ALNS", size=40, weight="bold")
    cy = y0 + 124
    xs = [x0 + 80 + k * 190 for k in range(4)]
    glyphs = [(fire_site, ("cam", "sense"), [("drone",)]), (rubble, ("grip", "arm"), [("rover",), ("legged",)]),
              (ruin, ("cam", "sense"), [("drone",)])]
    for k, x in enumerate(xs):
        if k < 3:
            card(x - 62, cy - 62, 124, 124)
            fn, sk, _ = glyphs[k]
            fn(x, cy - 12, 0.72)
            dots(x, cy + 42, sk, r=8)
        else:
            add(f'<rect x="{x - 62}" y="{cy - 62}" width="124" height="124" rx="14" fill="none" stroke="#AEB6BE" '
                f'stroke-width="2.5" stroke-dasharray="8 8"/>')
            text(x, cy + 12, "…", size=48, anchor="middle", fill=GREY)
        if k:
            add(f'<path d="M{xs[k - 1] + 70},{cy} L{x - 74},{cy}" stroke="#6B7580" stroke-width="5" '
                f'marker-end="url(#ah-grey)"/>')
    drone(xs[0], cy + 100, 0.62)
    rover(xs[1] - 34, cy + 106, 0.55)
    legged(xs[1] + 34, cy + 100, 0.58)
    drone(xs[2], cy + 100, 0.62)
    # the search loop
    ly = y0 + 352
    steps = [("scissors", BLUE, "remove"), ("puzzle", ORANGE, "re-insert"), ("temperature", PINK, "accept"),
             ("adjustments-horizontal", GREEN, "adapt")]
    lx = [x0 + 95 + k * 175 for k in range(4)]
    for k, ((icon, col, lab), x) in enumerate(zip(steps, lx)):
        add(f'<circle cx="{x}" cy="{ly + 4}" r="44" fill="#000" opacity="0.08"/>')
        add(f'<circle cx="{x}" cy="{ly}" r="44" fill="{col}"/>')
        tabler(icon, x, ly, 50, "white", width=2.2)
        text(x, ly + 78, lab, size=29, anchor="middle")
        if k < 3:
            add(f'<path d="M{x + 56},{ly} L{lx[k + 1] - 60},{ly}" stroke="#6B7580" stroke-width="5" '
                f'marker-end="url(#ah-grey)"/>')
    add(f'<path d="M{lx[3]},{ly - 50} C{lx[3]},{ly - 92} {lx[0]},{ly - 92} {lx[0]},{ly - 55}" fill="none" '
        f'stroke="#6B7580" stroke-width="5" marker-end="url(#ah-grey)"/>')


def panel_c(x0, y0) -> None:
    text(x0, y0 + 36, "(c) Fair test", size=40, weight="bold")
    rows = [("search", BLUE, "ALNS"), ("affiliate", PINK, "RL policy"), ("settings", GREEN, "CP-SAT"),
            ("sum", ORANGE, "MILP"), ("dice-5", "#6B7580", "restarts")]
    sx, sy = x0 + 420, y0 + 205
    for k, (icon, col, lab) in enumerate(rows):
        y = y0 + 90 + k * 58
        add(f'<rect x="{x0 + 8}" y="{y - 27}" width="54" height="54" rx="12" fill="{col}" opacity="0.16"/>')
        tabler(icon, x0 + 35, y, 40, col, width=2.2)
        text(x0 + 76, y + 10, lab, size=30, weight="bold" if k == 0 else "normal")
        add(f'<path d="M{x0 + 215},{y} C{x0 + 300},{y} {sx - 110},{sy} {sx - 58},{sy}" fill="none" stroke="#8C949C" '
            f'stroke-width="3.5"/>')
    add(f'<circle cx="{sx}" cy="{sy}" r="54" fill="{LIGHT}" stroke="#AEB6BE" stroke-width="2.5"/>')
    tabler("stopwatch", sx, sy, 70, INK, width=1.8)
    text(sx, sy + 88, "same budget", size=29, anchor="middle")
    mx = sx + 205
    add(f'<path d="M{sx + 62},{sy} L{mx - 70},{sy}" stroke="#6B7580" stroke-width="5" marker-end="url(#ah-grey)"/>')
    add(f'<circle cx="{mx}" cy="{sy}" r="54" fill="{LIGHT}" stroke="#AEB6BE" stroke-width="2.5"/>')
    tabler("device-desktop-check", mx, sy, 70, GREEN, width=1.8)
    text(mx, sy + 88, "simulator", size=29, anchor="middle")
    # pre-registration badge over the whole comparison
    bx, by = mx - 30, y0 + 50
    add(f'<rect x="{bx - 150}" y="{by - 34}" width="300" height="60" rx="30" fill="{GREEN}" opacity="0.14"/>')
    tabler("lock", bx - 112, by - 4, 40, GREEN, width=2.3)
    text(bx - 84, by + 6, "pre-registered", size=30, weight="bold", fill="#1E6B55")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png", action="store_true", help="also write a PNG preview")
    args = ap.parse_args()
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    markers()
    add(f'<rect width="{W}" height="{H}" fill="white"/>')
    top = 104
    text(18, 42, "(a) Static vs. dynamic", size=40, weight="bold")
    text(18 + MAP_W / 2, 90, "static: all tasks known at start", size=31, weight="bold", anchor="middle",
         fill="#1F5F99")
    text(840 + MAP_W / 2, 90, "dynamic: tasks appear, robots fail", size=31, weight="bold", anchor="middle",
         fill=VERM)
    terrain(18, top, MAP_W, MAP_H, "mapA")
    scene(18, top, dynamic=False)
    mid = top + MAP_H / 2
    add(f'<path d="M772,{mid - 30} L800,{mid - 30} L800,{mid - 52} L826,{mid} L800,{mid + 52} L800,{mid + 30} '
        f'L772,{mid + 30} z" fill="#B3BAC1"/>')
    terrain(840, top, MAP_W, MAP_H, "mapB")
    scene(840, top, dynamic=True)
    add(f'<path d="M1610,18 L1610,{H - 18}" stroke="#C9CED3" stroke-width="3" stroke-dasharray="6 10"/>')
    panel_b(1640, 4)
    add('<path d="M1640,494 L2385,494" stroke="#C9CED3" stroke-width="3" stroke-dasharray="6 10"/>')
    panel_c(1640, 500)
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
