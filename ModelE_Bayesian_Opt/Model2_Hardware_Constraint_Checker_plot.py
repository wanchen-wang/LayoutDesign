import numpy as np
import matplotlib.pyplot as plt

from Model2_Hardware_Constraint_Checker import build_model2_data


def _annotate_contour_edge(ax, cs, label, color='k', fontsize=10):
    """在等值线最右端旁边添加标注，不截断曲线本身。"""
    for segs in cs.allsegs:
        for seg in segs:
            if len(seg) == 0:
                continue
            idx = int(np.argmax(seg[:, 0]))
            ax.annotate(
                label,
                xy=(seg[idx, 0], seg[idx, 1]),
                xytext=(6, 0),
                textcoords='offset points',
                fontsize=fontsize,
                color=color,
                va='center',
                clip_on=False,
            )


def _shade_contour_enclosure(ax, cs, x_left, x_right, y_bottom, y_top, color, alpha):
    """按等值线真实曲线填充围合区域，并在 y_bottom 与 y_top 之间裁切。"""
    if not cs.allsegs or not cs.allsegs[0]:
        ax.fill_betweenx([y_bottom, y_top], min(x_left, x_right), max(x_left, x_right), color=color, alpha=alpha, zorder=0)
        return

    target_left = min(x_left, x_right)
    target_right = max(x_left, x_right)

    chosen_segment = None
    chosen_span = -np.inf
    for segment in cs.allsegs[0]:
        if len(segment) < 2:
            continue
        segment_left = np.min(segment[:, 0])
        segment_right = np.max(segment[:, 0])
        overlap_left = max(segment_left, target_left)
        overlap_right = min(segment_right, target_right)
        overlap = overlap_right - overlap_left
        if overlap > chosen_span:
            chosen_span = overlap
            chosen_segment = segment

    if chosen_segment is None or chosen_span <= 0:
        ax.fill_betweenx([y_bottom, y_top], target_left, target_right, color=color, alpha=alpha, zorder=0)
        return

    ordered = chosen_segment[np.argsort(chosen_segment[:, 0])]
    unique_x, unique_indices = np.unique(ordered[:, 0], return_index=True)
    unique_y = ordered[unique_indices, 1]
    x_samples = np.linspace(target_left, target_right, 500)
    y_curve = np.interp(x_samples, unique_x, unique_y)
    y_curve = np.clip(y_curve, y_bottom, y_top)
    ax.fill_between(x_samples, y_bottom, y_curve, color=color, alpha=alpha, zorder=0)


def _contour_line_intersections(cs, *, x_fixed=None, y_fixed=None):
    """返回等值线与水平/竖直直线的所有交点候选。"""
    if (x_fixed is None) == (y_fixed is None):
        raise ValueError("x_fixed 和 y_fixed 只能传入一个")

    intersections = []
    for segs in cs.allsegs:
        for segment in segs:
            if len(segment) < 2:
                continue
            for p1, p2 in zip(segment[:-1], segment[1:]):
                x1, y1 = p1
                x2, y2 = p2
                if x_fixed is not None:
                    if np.isclose(x1, x2):
                        if np.isclose(x1, x_fixed):
                            intersections.extend([(x_fixed, y1), (x_fixed, y2)])
                        continue
                    if (x1 - x_fixed) * (x2 - x_fixed) > 0:
                        continue
                    t = (x_fixed - x1) / (x2 - x1)
                    if 0.0 <= t <= 1.0:
                        intersections.append((x_fixed, y1 + t * (y2 - y1)))
                else:
                    if np.isclose(y1, y2):
                        if np.isclose(y1, y_fixed):
                            intersections.extend([(x1, y_fixed), (x2, y_fixed)])
                        continue
                    if (y1 - y_fixed) * (y2 - y_fixed) > 0:
                        continue
                    t = (y_fixed - y1) / (y2 - y1)
                    if 0.0 <= t <= 1.0:
                        intersections.append((x1 + t * (x2 - x1), y_fixed))

    unique_points = []
    for point in intersections:
        if not any(np.allclose(point, existing, atol=1e-6) for existing in unique_points):
            unique_points.append(point)
    return unique_points


def _pick_intersection(points, *, x_bounds=None, y_bounds=None, prefer_x=None, prefer_y=None):
    if not points:
        return None

    filtered = []
    for x, y in points:
        if x_bounds is not None and not (x_bounds[0] - 1e-9 <= x <= x_bounds[1] + 1e-9):
            continue
        if y_bounds is not None and not (y_bounds[0] - 1e-9 <= y <= y_bounds[1] + 1e-9):
            continue
        filtered.append((x, y))

    candidates = filtered if filtered else points

    def score(point):
        x, y = point
        total = 0.0
        if prefer_x is not None:
            total += abs(x - prefer_x)
        if prefer_y is not None:
            total += abs(y - prefer_y)
        return total

    return min(candidates, key=score)


def _draw_intersection_chord(ax, cs, *, x_fixed, y_fixed, label_prefix, line_color='black'):
    """标出等值线与参考线的两个交点，并画过这两点的直线。"""
    on_y = _pick_intersection(_contour_line_intersections(cs, y_fixed=y_fixed), prefer_y=y_fixed)
    on_x = _pick_intersection(
        _contour_line_intersections(cs, x_fixed=x_fixed),
        y_bounds=(0.0, y_fixed),
        prefer_x=x_fixed,
        prefer_y=y_fixed / 2.0,
    )

    if on_y is None or on_x is None:
        return None

    ax.scatter([on_y[0], on_x[0]], [on_y[1], on_x[1]], s=46, c=['deepskyblue', 'crimson'], edgecolors='black', zorder=6)
    ax.annotate(f"{label_prefix}A\n({on_y[0]:.2f}, {on_y[1]:.2f})", xy=on_y, xytext=(8, 10), textcoords='offset points', fontsize=9, color='deepskyblue', clip_on=False)
    ax.annotate(f"{label_prefix}B\n({on_x[0]:.2f}, {on_x[1]:.2f})", xy=on_x, xytext=(8, -18), textcoords='offset points', fontsize=9, color='crimson', clip_on=False)

    ax.plot([on_y[0], on_x[0]], [on_y[1], on_x[1]], color=line_color, linestyle='-.', linewidth=1.6, zorder=5)

    if np.isclose(on_y[0], on_x[0]):
        eq_text = f"x = {on_y[0]:.3f}"
    else:
        slope = (on_x[1] - on_y[1]) / (on_x[0] - on_y[0])
        intercept = on_y[1] - slope * on_y[0]
        eq_text = f"y = {slope:.4f}x {intercept:+.4f}"

    ax.text(
        0.03,
        0.96,
        eq_text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor=line_color),
    )
    return on_y, on_x, eq_text


def main():
    data = build_model2_data()

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    print(f"下潜阶段：θ=-45° → ζ={data['zeta_theta45_dive']:.2f}°，θ=-15° → ζ={data['zeta_theta15_dive']:.2f}°")
    print(f"上浮阶段：θ=+15° → ζ={data['zeta_theta15_climb']:.2f}°，θ=+45° → ζ={data['zeta_theta45_climb']:.2f}°")

    fig1 = plt.figure(figsize=(15, 6))
    fig1.suptitle('第一组图：下潜阶段 (滑翔角 < 0)', fontsize=16)

    ax1 = fig1.add_subplot(121, projection='3d')
    ax1.plot_surface(data['ZETA_D'], data['V_D'], data['RP1_D'], cmap='autumn', alpha=0.8)
    ax1.plot_surface(data['ZETA_D'], data['V_D'], np.full_like(data['ZETA_D'], -2.0), color='lightgray', alpha=0.5)
    ax1.contour(data['ZETA_D'], data['V_D'], data['RP1_D'], levels=[-1.9], colors='k', linewidths=2.0)
    ax1.set_title('子图一：所需电池包位移 $r_{p1}$', fontsize=14)
    ax1.set_xlabel('滑翔角 $\zeta$ (°)')
    ax1.set_ylabel('滑翔速度 $V$ (m/s)')
    ax1.set_zlabel('电池位移 $r_{p1}$ (cm)')

    ax2 = fig1.add_subplot(122, projection='3d')
    ax2.plot_surface(data['ZETA_D'], data['V_D'], data['DELTA_B_D'], cmap='autumn', alpha=0.8)
    ax2.plot_surface(data['ZETA_D'], data['V_D'], np.full_like(data['ZETA_D'], -5.5), color='lightgray', alpha=0.5)
    ax2.contour(data['ZETA_D'], data['V_D'], data['DELTA_B_D'], levels=[-5.4], colors='k', linewidths=2.0)
    ax2.set_title('子图二：所需净浮力 $\Delta B$', fontsize=14)
    ax2.set_xlabel('滑翔角 $\zeta$ (°)')
    ax2.set_ylabel('滑翔速度 $V$ (m/s)')
    ax2.set_zlabel('净浮力 $\Delta B$ (N)')

    plt.tight_layout()
    fig1.subplots_adjust(top=0.88)

    fig2 = plt.figure(figsize=(15, 6))
    fig2.suptitle('第二组图：上浮阶段 (滑翔角 > 0)', fontsize=16)

    ax3 = fig2.add_subplot(121, projection='3d')
    ax3.plot_surface(data['ZETA_C'], data['V_C'], data['RP1_C'], cmap='winter', alpha=0.8)
    ax3.plot_surface(data['ZETA_C'], data['V_C'], np.full_like(data['ZETA_C'], 2.0), color='lightgray', alpha=0.5)
    ax3.contour(data['ZETA_C'], data['V_C'], data['RP1_C'], levels=[2.1], colors='k', linewidths=2.0)
    ax3.set_title('子图一：所需电池包位移 $r_{p1}$', fontsize=14)
    ax3.set_xlabel('滑翔角 $\zeta$ (°)')
    ax3.set_ylabel('滑翔速度 $V$ (m/s)')
    ax3.set_zlabel('电池位移 $r_{p1}$ (cm)')

    ax4 = fig2.add_subplot(122, projection='3d')
    ax4.plot_surface(data['ZETA_C'], data['V_C'], data['DELTA_B_C'], cmap='winter', alpha=0.8)
    ax4.plot_surface(data['ZETA_C'], data['V_C'], np.full_like(data['ZETA_C'], 5.5), color='lightgray', alpha=0.5)
    ax4.contour(data['ZETA_C'], data['V_C'], data['DELTA_B_C'], levels=[5.6], colors='k', linewidths=2.0)
    ax4.set_title('子图二：所需净浮力 $\Delta B$', fontsize=14)
    ax4.set_xlabel('滑翔角 $\zeta$ (°)')
    ax4.set_ylabel('滑翔速度 $V$ (m/s)')
    ax4.set_zlabel('净浮力 $\Delta B$ (N)')

    plt.tight_layout()
    fig2.subplots_adjust(top=0.88)

    fig3, axes2d = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('硬件极限截面交界线（二维投影）', fontsize=16)

    ax2d1 = axes2d[0, 0]
    cs1 = ax2d1.contour(data['ZETA_D'], data['V_D'], data['RP1_D'], levels=[-2.0], colors='k', linewidths=2.0, linestyles='--')
    _annotate_contour_edge(ax2d1, cs1, '-2.0 cm', color='k')
    _shade_contour_enclosure(ax2d1, cs1, data['zeta_theta45_dive'], data['zeta_theta15_dive'], 0.0, 0.5, '#6b9ac4', 0.35)
    ax2d1.axvline(data['zeta_theta45_dive'], color='purple', linestyle='-', linewidth=1.5, label=f"theta=-45deg (zeta={data['zeta_theta45_dive']:.1f}deg)")
    ax2d1.axvline(data['zeta_theta15_dive'], color='purple', linestyle='-', linewidth=1.2, label=f"theta=-15deg (zeta={data['zeta_theta15_dive']:.1f}deg)")
    ax2d1.axhline(0.0, color='goldenrod', linestyle='-', linewidth=1.2, label='V=0 m/s')
    ax2d1.axhline(0.5, color='gold', linestyle='-', linewidth=1.5, label='V=0.5 m/s')
    _draw_intersection_chord(ax2d1, cs1, x_fixed=data['zeta_theta15_dive'], y_fixed=0.5, label_prefix='D1-')
    ax2d1.legend(fontsize=9)
    ax2d1.set_title('下潜阶段：电池位移极限线 (-2.0 cm)', fontsize=12)
    ax2d1.set_xlabel('滑翔角 zeta (deg)')
    ax2d1.set_ylabel('滑翔速度 V (m/s)')
    ax2d1.set_ylim(bottom=0.0)
    ax2d1.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)
    ax2d1.minorticks_on()
    ax2d1.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)

    ax2d2 = axes2d[0, 1]
    cs2 = ax2d2.contour(data['ZETA_D'], data['V_D'], data['DELTA_B_D'], levels=[-5.5], colors='k', linewidths=2.0, linestyles='--')
    _annotate_contour_edge(ax2d2, cs2, '-5.5 N', color='r')
    _shade_contour_enclosure(ax2d2, cs2, data['zeta_theta45_dive'], data['zeta_theta15_dive'], 0.0, 0.5, '#6b9ac4', 0.35)
    ax2d2.axvline(data['zeta_theta45_dive'], color='purple', linestyle='-', linewidth=1.5, label=f"theta=-45deg (zeta={data['zeta_theta45_dive']:.1f}deg)")
    ax2d2.axvline(data['zeta_theta15_dive'], color='purple', linestyle='-', linewidth=1.2, label=f"theta=-15deg (zeta={data['zeta_theta15_dive']:.1f}deg)")
    ax2d2.axhline(0.0, color='goldenrod', linestyle='-', linewidth=1.2, label='V=0 m/s')
    ax2d2.axhline(0.5, color='gold', linestyle='-', linewidth=1.5, label='V=0.5 m/s')
    _draw_intersection_chord(ax2d2, cs2, x_fixed=data['zeta_theta15_dive'], y_fixed=0.5, label_prefix='D2-')
    ax2d2.legend(fontsize=9)
    ax2d2.set_title('下潜阶段：净浮力极限线 (-5.5 N)', fontsize=12)
    ax2d2.set_xlabel('滑翔角 zeta (deg)')
    ax2d2.set_ylabel('滑翔速度 V (m/s)')
    ax2d2.set_ylim(bottom=0.0)
    ax2d2.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)
    ax2d2.minorticks_on()
    ax2d2.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)

    ax2d3 = axes2d[1, 0]
    cs3 = ax2d3.contour(data['ZETA_C'], data['V_C'], data['RP1_C'], levels=[2.0], colors='k', linewidths=2.0, linestyles='--')
    _annotate_contour_edge(ax2d3, cs3, '+2.0 cm', color='k')
    _shade_contour_enclosure(ax2d3, cs3, data['zeta_theta15_climb'], data['zeta_theta45_climb'], 0.0, 0.5, '#f4a1c1', 0.45)
    ax2d3.axvline(data['zeta_theta15_climb'], color='purple', linestyle='-', linewidth=1.2, label=f"theta=+15deg (zeta={data['zeta_theta15_climb']:.1f}deg)")
    ax2d3.axvline(data['zeta_theta45_climb'], color='purple', linestyle='-', linewidth=1.5, label=f"theta=+45deg (zeta={data['zeta_theta45_climb']:.1f}deg)")
    ax2d3.axhline(0.0, color='goldenrod', linestyle='-', linewidth=1.2, label='V=0 m/s')
    ax2d3.axhline(0.5, color='gold', linestyle='-', linewidth=1.5, label='V=0.5 m/s')
    _draw_intersection_chord(ax2d3, cs3, x_fixed=data['zeta_theta15_climb'], y_fixed=0.5, label_prefix='C1-')
    ax2d3.legend(fontsize=9)
    ax2d3.set_title('上浮阶段：电池位移极限线 (+2.0 cm)', fontsize=12)
    ax2d3.set_xlabel('滑翔角 zeta (deg)')
    ax2d3.set_ylabel('滑翔速度 V (m/s)')
    ax2d3.set_ylim(bottom=0.0)
    ax2d3.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)
    ax2d3.minorticks_on()
    ax2d3.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)

    ax2d4 = axes2d[1, 1]
    cs4 = ax2d4.contour(data['ZETA_C'], data['V_C'], data['DELTA_B_C'], levels=[5.5], colors='k', linewidths=2.0, linestyles='--')
    _annotate_contour_edge(ax2d4, cs4, '+5.5 N', color='r')
    _shade_contour_enclosure(ax2d4, cs4, data['zeta_theta15_climb'], data['zeta_theta45_climb'], 0.0, 0.5, '#f4a1c1', 0.45)
    ax2d4.axvline(data['zeta_theta15_climb'], color='purple', linestyle='-', linewidth=1.2, label=f"theta=+15deg (zeta={data['zeta_theta15_climb']:.1f}deg)")
    ax2d4.axvline(data['zeta_theta45_climb'], color='purple', linestyle='-', linewidth=1.5, label=f"theta=+45deg (zeta={data['zeta_theta45_climb']:.1f}deg)")
    ax2d4.axhline(0.0, color='goldenrod', linestyle='-', linewidth=1.2, label='V=0 m/s')
    ax2d4.axhline(0.5, color='gold', linestyle='-', linewidth=1.5, label='V=0.5 m/s')
    _draw_intersection_chord(ax2d4, cs4, x_fixed=data['zeta_theta15_climb'], y_fixed=0.5, label_prefix='C2-')
    ax2d4.legend(fontsize=9)
    ax2d4.set_title('上浮阶段：净浮力极限线 (+5.5 N)', fontsize=12)
    ax2d4.set_xlabel('滑翔角 zeta (deg)')
    ax2d4.set_ylabel('滑翔速度 V (m/s)')
    ax2d4.set_ylim(bottom=0.0)
    ax2d4.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.7)
    ax2d4.minorticks_on()
    ax2d4.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    fig3.subplots_adjust(top=0.92)
    plt.show()


if __name__ == '__main__':
    main()